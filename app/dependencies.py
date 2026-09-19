"""预约时序依赖（reservation dependency）核心逻辑：有向间隔约束、环检测与改期/取消重校验。

一条依赖是预约之间的有向约束::

    predecessor.end + minimum_gap_minutes <= successor.start

（最小间隔可以为 0，此时首尾相接恰好满足。）全体依赖构成一张有向图，
系统保证其始终为 **DAG**：

- **自环拒绝**：``predecessor_id == successor_id`` 在 API 层 422 拦截，
  核心层再防御一次；
- **环拒绝**：新增 ``predecessor -> successor`` 前沿 successor 方向 BFS，
  若 successor 已可到达 predecessor，加入该边即成环
  → :class:`DependencyCycleError`（409，响应给出闭合链路）。

依赖只在**建立时**做自环/环校验，不校验当前排班是否已满足间隔
（约束建立在既有预约之间，建立后由改期/取消路径负责维护）。
约束只对**双方均为 active** 的边生效：一端已取消的边视为休眠，
不参与重校验（取消是终态，休眠边也参与环检测，使复活式语义下
图的无环性永远成立）。

幂等：``dependency_request_id`` 唯一。相同请求 ID + 相同规范化内容
（predecessor_id、successor_id、minimum_gap_minutes）重放
→ 返回原约束（``created=False``，API 层 200）；同 ID 内容不同
→ :class:`DependencyRequestConflictError`（409）。
同一对预约之间至多一条边（``UNIQUE(predecessor_id, successor_id)``）：
同对不同最小间隔 → :class:`DependencyPairConflictError`（409）。

改期 / 取消 / 时段交换在其**同一个 ``BEGIN IMMEDIATE`` 事务**内调用
:func:`evaluate_reachable`，重新校验与被变更预约**可达**（忽略方向、
逐边相连）的全部依赖；任一不满足 → 事务回滚、**不写入任何部分变更**，
异常的 ``violations`` 给出从被变更预约出发的完整链路（409）。
成功改期/取消/交换沿用现有版本递增与哈希链审计事件，依赖本身不产生审计事件。
"""

from __future__ import annotations

import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

from . import db

# 复用 reservations 的定宽 UTC 文本规则（字典序即时间序，
# 形如 2026-03-09T13:00:00.000000Z，本身即合法的 ISO 8601）
from .reservations import _canon, _parse

_DEP_COLUMNS = (
    "id, dependency_request_id, predecessor_id, successor_id,"
    " minimum_gap_minutes, created_at"
)


class DependencySelfLoopError(ValueError):
    """predecessor 与 successor 是同一条预约（自环）。"""

    def __init__(self, reservation_id: int) -> None:
        self.reservation_id = reservation_id
        super().__init__(
            f"reservation dependency cannot reference the same reservation "
            f"#{reservation_id} as both predecessor and successor"
        )


class DependencyEndpointNotFoundError(Exception):
    """依赖端点（前驱或后继预约）不存在。"""

    def __init__(self, role: str, reservation_id: int) -> None:
        self.role = role  # predecessor / successor
        self.reservation_id = reservation_id
        super().__init__(f"{role} reservation #{reservation_id} not found")


class DependencyRequestConflictError(Exception):
    """``dependency_request_id`` 已被内容不同的依赖占用。"""

    def __init__(self, dependency_id: int) -> None:
        self.dependency_id = dependency_id
        super().__init__(
            f"dependency_request_id is already used by reservation dependency "
            f"#{dependency_id} with different content"
        )


class DependencyPairConflictError(Exception):
    """同一对预约之间已存在不同最小间隔的约束。"""

    def __init__(self, dependency_id: int) -> None:
        self.dependency_id = dependency_id
        super().__init__(
            f"a dependency between the same two reservations already exists "
            f"(reservation dependency #{dependency_id});"
            " changing the minimum gap requires removing the existing one"
        )


class DependencyCycleError(Exception):
    """新增边会在依赖图中形成有向环。"""

    def __init__(self, cycle: list[int], existing_path: list[int]) -> None:
        # cycle：闭合链路 [successor, ..., predecessor, successor]
        # existing_path：已存在的 successor -> predecessor 有向路径
        self.cycle = cycle
        self.existing_path = existing_path
        pretty = " -> ".join(str(r) for r in cycle)
        super().__init__(
            f"reservation dependency would create a directed cycle: {pretty}"
        )

    @property
    def payload(self) -> dict[str, object]:
        return {"cycle": self.cycle, "path": self.existing_path}


class DependencyViolationError(Exception):
    """重校验时至少一条可达时序依赖不满足。"""

    def __init__(self, violations: list[dict[str, object]]) -> None:
        self.violations = violations
        first = violations[0]
        super().__init__(
            f"reservation dependency violated: reservation "
            f"#{first['predecessor_id']} must end at least "
            f"{first['minimum_gap_minutes']} minute(s) before reservation "
            f"#{first['successor_id']} starts"
        )


@dataclass(frozen=True)
class ReservationDependency:
    id: int
    dependency_request_id: str
    predecessor_id: int
    successor_id: int
    minimum_gap_minutes: int
    created_at: datetime


def _row_to_dependency(row: sqlite3.Row) -> ReservationDependency:
    return ReservationDependency(
        id=row["id"],
        dependency_request_id=row["dependency_request_id"],
        predecessor_id=row["predecessor_id"],
        successor_id=row["successor_id"],
        minimum_gap_minutes=row["minimum_gap_minutes"],
        created_at=_parse(row["created_at"]),
    )


def load_dependency(
    conn: sqlite3.Connection, dependency_id: int
) -> ReservationDependency | None:
    """在调用方已打开的连接/事务内按 ID 读取依赖，不存在返回 ``None``。"""
    row = conn.execute(
        f"SELECT {_DEP_COLUMNS} FROM reservation_dependencies WHERE id = ?",
        (dependency_id,),
    ).fetchone()
    return _row_to_dependency(row) if row is not None else None


def load_all_edges(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """读取全部依赖边（调用方须在事务内；图规模为预约数的边数）。"""
    return conn.execute(
        f"SELECT {_DEP_COLUMNS} FROM reservation_dependencies"
        " ORDER BY predecessor_id, successor_id, id"
    ).fetchall()


def _find_directed_path(
    adjacency: Mapping[int, list[int]], source: int, target: int
) -> list[int] | None:
    """沿 successor 方向的 BFS，返回 ``source -> ... -> target`` 的最短路。"""
    if source == target:
        return [source]
    queue: deque[int] = deque([source])
    parent: dict[int, int] = {source: source}
    while queue:
        node = queue.popleft()
        for nxt in adjacency.get(node, ()):
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == target:
                path = [nxt]
                while path[-1] != source:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    return None


def create_dependency(
    dependency_request_id: str,
    predecessor_id: int,
    successor_id: int,
    minimum_gap_minutes: int,
    db_path: str | None = None,
) -> tuple[ReservationDependency, bool]:
    """建立一条预约时序依赖，返回 ``(依赖, 是否新建)``。

    整个流程在单个 ``BEGIN IMMEDIATE`` 事务内完成，任何一步失败都回滚：

    1. 自环（两端为同一预约）→ :class:`DependencySelfLoopError`；
    2. 同一 ``dependency_request_id`` 已存在：规范化内容相同 →
       幂等返回原依赖（``created=False``），内容不同
       → :class:`DependencyRequestConflictError`（409）；
    3. 前驱/后继预约任一不存在 → :class:`DependencyEndpointNotFoundError`（404）；
    4. 同一对预约之间已有约束：最小间隔相同 → 幂等返回原约束，
       不同 → :class:`DependencyPairConflictError`（409）；
    5. 沿 successor 方向 BFS，successor 已可到达 predecessor（加入
       即成环）→ :class:`DependencyCycleError`（409 + 闭合链路）；
    6. 否则写入约束（``created=True``）。

    依赖不产生哈希链审计事件，也不校验当前排班是否已满足间隔
    （建立后由改期/取消/交换路径统一维护）。
    """
    if predecessor_id == successor_id:
        raise DependencySelfLoopError(predecessor_id)
    if minimum_gap_minutes < 0:
        # API 层以 422 拦截；核心层防御
        raise ValueError("minimum_gap_minutes must be non-negative")

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：存在性 + 幂等 + 环检测 + 写入对并发请求原子
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_s = _canon(datetime.now(timezone.utc))

            row = conn.execute(
                f"SELECT {_DEP_COLUMNS} FROM reservation_dependencies"
                " WHERE dependency_request_id = ?",
                (dependency_request_id,),
            ).fetchone()
            if row is not None:
                if (row["predecessor_id"] == predecessor_id
                        and row["successor_id"] == successor_id
                        and row["minimum_gap_minutes"] == minimum_gap_minutes):
                    conn.execute("COMMIT")
                    return _row_to_dependency(row), False
                raise DependencyRequestConflictError(row["id"])

            for role, endpoint_id in (
                ("predecessor", predecessor_id),
                ("successor", successor_id),
            ):
                exists = conn.execute(
                    "SELECT 1 FROM reservations WHERE id = ?",
                    (endpoint_id,),
                ).fetchone()
                if exists is None:
                    raise DependencyEndpointNotFoundError(role, endpoint_id)

            pair = conn.execute(
                f"SELECT {_DEP_COLUMNS} FROM reservation_dependencies"
                " WHERE predecessor_id = ? AND successor_id = ?",
                (predecessor_id, successor_id),
            ).fetchone()
            if pair is not None:
                if pair["minimum_gap_minutes"] == minimum_gap_minutes:
                    # 同对同间隔的重复建立：视为幂等重放返回原约束
                    conn.execute("COMMIT")
                    return _row_to_dependency(pair), False
                raise DependencyPairConflictError(pair["id"])

            edges = load_all_edges(conn)
            adjacency: dict[int, list[int]] = {}
            for edge in edges:
                adjacency.setdefault(edge["predecessor_id"], []).append(
                    edge["successor_id"])
            existing_path = _find_directed_path(
                adjacency, successor_id, predecessor_id)
            if existing_path is not None:
                # 闭合链路：successor 已能沿现有边到达 predecessor，
                # 再补 predecessor -> successor 即成环
                raise DependencyCycleError(
                    cycle=[*existing_path, successor_id],
                    existing_path=existing_path,
                )

            cursor = conn.execute(
                "INSERT INTO reservation_dependencies"
                " (dependency_request_id, predecessor_id, successor_id,"
                "  minimum_gap_minutes, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (dependency_request_id, predecessor_id, successor_id,
                 minimum_gap_minutes, now_s),
            )
            dependency_id = int(cursor.lastrowid)
            conn.execute("COMMIT")
            return ReservationDependency(
                id=dependency_id,
                dependency_request_id=dependency_request_id,
                predecessor_id=predecessor_id,
                successor_id=successor_id,
                minimum_gap_minutes=minimum_gap_minutes,
                created_at=_parse(now_s),
            ), True
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


# ------------------------------------------------- 改期 / 取消 / 交换重校验


def _load_node_states(
    conn: sqlite3.Connection, node_ids: Iterable[int]
) -> dict[int, sqlite3.Row]:
    """批量读取预约节点行（id -> reservations 行）。"""
    node_ids = tuple(set(node_ids))
    if not node_ids:
        return {}
    placeholders = ",".join("?" for _ in node_ids)
    return {
        row["id"]: row
        for row in conn.execute(
            f"SELECT id, start_utc, end_utc, status FROM reservations"
            f" WHERE id IN ({placeholders})",
            node_ids,
        )
    }


def _chain_node(
    node_id: int, nodes: Mapping[int, sqlite3.Row]
) -> dict[str, object]:
    row = nodes.get(node_id)
    if row is None:
        # 防御：被删除的预约（当前仅软删除，不会进入这里）
        return {"reservation_id": node_id, "missing": True}
    return {
        "reservation_id": node_id,
        "start_utc": row["start_utc"],
        "end_utc": row["end_utc"],
        "status": row["status"],
    }


def evaluate_reachable(
    conn: sqlite3.Connection,
    *,
    roots: Iterable[int],
    overrides: Mapping[int, Mapping[str, str]] | None = None,
) -> None:
    """在调用方的写事务内重校验与 ``roots`` 可达的全部时序依赖。

    - 可达性按**忽略方向、逐边相连**理解（前驱链路与后继链路都会受
      被变更预约影响），枚举其连通分量内的每条边；
    - 一端已取消的边**休眠**：跳过，不参与校验（取消是终态）；
    - 节点时刻优先取 ``overrides[rid]``（形如
      ``{"start_utc": ..., "end_utc": ..., "status": ...}``，定宽 UTC 文本），
      用于改期/交换/取消在**尚未 COMMIT 的同一事务**内校验其待写入状态；
      未覆盖的节点从数据库读取当前状态；
    - 任一 active 边违反 ``predecessor.end + gap <= successor.start``
      → :class:`DependencyViolationError`，``violations`` 给出
      从被变更预约出发的完整链路（含各节点时刻、实际/要求间隔），
      调用方事务随之回滚，不写入任何部分变更。
    """
    overrides = overrides or {}
    roots = tuple(dict.fromkeys(roots))
    if not roots:
        return

    edges = load_all_edges(conn)
    node_ids: set[int] = set()
    undirected: dict[int, list[int]] = {}
    directed: dict[int, list[int]] = {}
    for edge in edges:
        p, s = edge["predecessor_id"], edge["successor_id"]
        node_ids.update((p, s))
        undirected.setdefault(p, []).append(s)
        undirected.setdefault(s, []).append(p)
        directed.setdefault(p, []).append(s)

    # 与任一 root 忽略方向逐边相连的连通节点集合
    reachable_nodes: set[int] = set()
    frontier: deque[int] = deque()
    for root in roots:
        if root not in reachable_nodes:
            reachable_nodes.add(root)
            frontier.append(root)
    while frontier:
        node = frontier.popleft()
        for nxt in undirected.get(node, ()):
            if nxt not in reachable_nodes:
                reachable_nodes.add(nxt)
                frontier.append(nxt)

    nodes = _load_node_states(conn, reachable_nodes)

    def _state(node_id: int) -> Mapping[str, str] | None:
        override = overrides.get(node_id)
        if override is not None:
            return override
        row = nodes.get(node_id)
        if row is None:
            return None
        return row

    def _shortest_path(source: int, target: int,
                       adjacency: Mapping[int, list[int]]) -> list[int] | None:
        if source == target:
            return [source]
        queue: deque[int] = deque([source])
        parent: dict[int, int] = {source: source}
        while queue:
            node = queue.popleft()
            for nxt in adjacency.get(node, ()):
                if nxt in parent:
                    continue
                parent[nxt] = node
                if nxt == target:
                    found = [nxt]
                    while found[-1] != source:
                        found.append(parent[found[-1]])
                    found.reverse()
                    return found
                queue.append(nxt)
        return None

    violations: list[dict[str, object]] = []
    for edge in edges:
        p, s = edge["predecessor_id"], edge["successor_id"]
        if p not in reachable_nodes:
            continue
        pred_state = _state(p)
        succ_state = _state(s)
        if pred_state is None or succ_state is None:
            continue
        # 一端已取消的边休眠：约束只约束两条 active 预约
        if pred_state["status"] != "active" or succ_state["status"] != "active":
            continue

        pred_end = _parse(pred_state["end_utc"])
        succ_start = _parse(succ_state["start_utc"])
        actual_gap_minutes = (
            succ_start - pred_end
        ).total_seconds() / 60.0
        gap = edge["minimum_gap_minutes"]
        if actual_gap_minutes + 1e-9 >= gap:
            continue

        # 给出从被变更预约出发、沿 successor 方向包含被违反边 p -> s 的
        # 完整链路。单次只移动一个预约的改期/取消场景链路为 [p, s]；
        # 交换同时移动双方，被违反边可能经由另一条腿多跳可达。
        chain_ids: list[int] = [p, s]
        if not any(root in (p, s) for root in roots):
            for root in roots:
                via_pred = _shortest_path(root, p, directed)
                if via_pred is not None:
                    # root -> ... -> p -> s
                    chain_ids = [*via_pred, s]
                    break
                via_succ = _shortest_path(s, root, directed)
                if via_succ is not None:
                    # p -> s -> ... -> root（via_succ 首元素即 s）
                    chain_ids = [p, *via_succ]
                    break
            else:
                # 防御：忽略方向的连通路兜底（DAG 正常路径下不会到达）
                chain_ids = _shortest_path(p, s, undirected) or [p, s]

        chain = [_chain_node(node_id, nodes)
                 for node_id in dict.fromkeys(chain_ids)]
        # override 节点的时刻/状态以待写入值为准（BFS 用的是库里的旧行）
        for item in chain:
            node_id = item["reservation_id"]
            if node_id in overrides:
                ov = overrides[node_id]
                item["start_utc"] = ov.get("start_utc")
                item["end_utc"] = ov.get("end_utc")
                item["status"] = ov.get("status")

        violations.append({
            "dependency_id": edge["id"],
            "predecessor_id": p,
            "successor_id": s,
            "minimum_gap_minutes": gap,
            "actual_gap_minutes": actual_gap_minutes,
            "chain": chain,
        })

    if violations:
        # 确定的顺序：按依赖 id，错误可复现、便于客户端比对
        violations.sort(key=lambda v: v["dependency_id"])
        raise DependencyViolationError(violations)
