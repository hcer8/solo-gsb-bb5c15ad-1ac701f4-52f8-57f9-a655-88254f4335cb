"""预约时序依赖测试：约束建立、自环/环拒绝、幂等重放、改期/取消/交换重校验。"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import dependencies as dep_core
from app import reservations as res_core
from app.main import app


@pytest.fixture()
def db_path(tmp_path, monkeypatch) -> str:
    """每个测试用独立的 SQLite 文件。"""
    path = str(tmp_path / "reservations.db")
    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    return path


@pytest.fixture()
def client(db_path) -> TestClient:
    return TestClient(app)


def _res(payload: dict, client: TestClient) -> int:
    resp = client.post("/api/v1/reservations", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["reservation_id"]


def _dep(client: TestClient, request_id: str, predecessor: int, successor: int,
         gap: int = 30) -> dict:
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": request_id,
            "predecessor_id": predecessor,
            "successor_id": successor,
            "minimum_gap_minutes": gap,
        },
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()


def _version(client: TestClient, reservation_id: int) -> int:
    # 从历史接口的最新快照拿当前版本，避免额外的 GET 端点
    resp = client.get(f"/api/v1/reservations/{reservation_id}/history")
    assert resp.status_code == 200, resp.text
    return resp.json()["events"][-1]["snapshot"]["version"]


@pytest.fixture()
def chain_of_three(client) -> tuple[int, int, int]:
    """三条首尾间隔 30 分钟的预约：#1 10:00-11:00 / #2 11:30-12:00 / #3 12:30-13:00。"""
    a = _res({
        "request_id": "req-a",
        "start_utc": "2026-03-09T10:00:00Z",
        "end_utc": "2026-03-09T11:00:00Z",
        "participant_ids": ["alice"],
    }, client)
    b = _res({
        "request_id": "req-b",
        "start_utc": "2026-03-09T11:30:00Z",
        "end_utc": "2026-03-09T12:00:00Z",
        "participant_ids": ["bob"],
    }, client)
    c = _res({
        "request_id": "req-c",
        "start_utc": "2026-03-09T12:30:00Z",
        "end_utc": "2026-03-09T13:00:00Z",
        "participant_ids": ["carol"],
    }, client)
    return a, b, c


# ------------------------------------------------------------------ 建立约束


def test_create_dependency_returns_201(client, chain_of_three):
    a, b, _ = chain_of_three
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-1",
            "predecessor_id": a,
            "successor_id": b,
            "minimum_gap_minutes": 30,
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["dependency_id"] == 1
    assert data["dependency_request_id"] == "dep-1"
    assert data["predecessor_id"] == a
    assert data["successor_id"] == b
    assert data["minimum_gap_minutes"] == 30
    assert data["created_at"].endswith("Z")


def test_create_dependency_self_loop_rejected_422(client, chain_of_three):
    a, _, _ = chain_of_three
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-loop",
            "predecessor_id": a,
            "successor_id": a,
            "minimum_gap_minutes": 0,
        },
    )
    assert resp.status_code == 422, resp.text


def test_core_self_loop_rejected(db_path, chain_of_three):
    a, _, _ = chain_of_three
    with pytest.raises(dep_core.DependencySelfLoopError):
        dep_core.create_dependency(
            dependency_request_id="dep-core-loop",
            predecessor_id=a, successor_id=a, minimum_gap_minutes=0,
            db_path=db_path,
        )


def test_negative_gap_rejected_422(client, chain_of_three):
    a, b, _ = chain_of_three
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-neg",
            "predecessor_id": a,
            "successor_id": b,
            "minimum_gap_minutes": -1,
        },
    )
    assert resp.status_code == 422


def test_endpoint_not_found_returns_404(client, chain_of_three):
    a, _, _ = chain_of_three
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-missing",
            "predecessor_id": a,
            "successor_id": 999,
            "minimum_gap_minutes": 0,
        },
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["role"] == "successor"
    assert resp.json()["reservation_id"] == 999

    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-missing-2",
            "predecessor_id": 998,
            "successor_id": a,
            "minimum_gap_minutes": 0,
        },
    )
    assert resp.status_code == 404
    assert resp.json()["role"] == "predecessor"


def test_cycle_rejected_with_closed_chain(client, chain_of_three):
    a, b, c = chain_of_three
    _dep(client, "dep-ab", a, b)
    _dep(client, "dep-bc", b, c)
    # c -> a 成环：c 已无路径到 a……a->b->c 存在，闭合需 c->a
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-ca",
            "predecessor_id": c,
            "successor_id": a,
            "minimum_gap_minutes": 0,
        },
    )
    assert resp.status_code == 409, resp.text
    data = resp.json()
    # 已有 successor(a) -> predecessor(c) 的有向路径即 a -> b -> c，
    # 闭合链路为 [a, b, c, a]
    assert data["path"] == [a, b, c]
    assert data["cycle"] == [a, b, c, a]
    assert "directed cycle" in data["detail"]


def test_cycle_two_node_reverse_edge_rejected(client, chain_of_three):
    a, b, _ = chain_of_three
    _dep(client, "dep-ab", a, b)
    # b -> a 与 a -> b 形成 2 元环
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-ba",
            "predecessor_id": b,
            "successor_id": a,
            "minimum_gap_minutes": 0,
        },
    )
    assert resp.status_code == 409
    assert resp.json()["cycle"] == [a, b, a]
    assert resp.json()["path"] == [a, b]


def test_failed_dependency_is_not_persisted(client, chain_of_three):
    a, b, c = chain_of_three
    _dep(client, "dep-ab", a, b)
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-ba",
            "predecessor_id": b,
            "successor_id": a,
            "minimum_gap_minutes": 0,
        },
    )
    assert resp.status_code == 409
    # 未成环的其他边仍可建立（证明回滚后写锁/状态干净）
    _dep(client, "dep-ac", a, c, gap=0)


def test_dependency_can_reference_canceled_endpoint(client, chain_of_three):
    a, b, _ = chain_of_three
    # 先取消 a，再建立引用它的依赖：允许建立（边休眠，不约束 active 排班）
    assert client.delete(
        f"/api/v1/reservations/{a}?expected_version=1").status_code == 200
    dep = _dep(client, "dep-dormant", a, b, gap=30)
    assert dep["predecessor_id"] == a
    # 休眠边不阻止 b 任意改期
    resp = client.put(
        f"/api/v1/reservations/{b}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T08:00:00Z",
            "end_utc": "2026-03-09T08:30:00Z",
            "participant_ids": ["bob"],
        },
    )
    assert resp.status_code == 200, resp.text


# ------------------------------------------------------------------ 幂等重放


def test_idempotent_replay_same_content_returns_200(client, chain_of_three):
    a, b, _ = chain_of_three
    first = _dep(client, "dep-replay", a, b, gap=15)
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-replay",
            "predecessor_id": a,
            "successor_id": b,
            "minimum_gap_minutes": 15,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == first


def test_same_request_id_different_content_returns_409(client, chain_of_three):
    a, b, c = chain_of_three
    _dep(client, "dep-x", a, b, gap=10)
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-x",
            "predecessor_id": a,
            "successor_id": c,  # 不同后继
            "minimum_gap_minutes": 10,
        },
    )
    assert resp.status_code == 409
    assert resp.json()["existing_dependency_id"] == 1

    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-x",
            "predecessor_id": a,
            "successor_id": b,
            "minimum_gap_minutes": 25,  # 不同间隔
        },
    )
    assert resp.status_code == 409


def test_same_pair_different_gap_returns_409(client, chain_of_three):
    a, b, _ = chain_of_three
    _dep(client, "dep-pair-1", a, b, gap=10)
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-pair-2",
            "predecessor_id": a,
            "successor_id": b,
            "minimum_gap_minutes": 20,
        },
    )
    assert resp.status_code == 409
    assert resp.json()["existing_dependency_id"] == 1


def test_same_pair_same_gap_different_request_id_is_idempotent(client, chain_of_three):
    a, b, _ = chain_of_three
    first = _dep(client, "dep-pair-a", a, b, gap=10)
    resp = client.post(
        "/api/v1/reservation-dependencies",
        json={
            "dependency_request_id": "dep-pair-b",
            "predecessor_id": a,
            "successor_id": b,
            "minimum_gap_minutes": 10,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == first


# ------------------------------------------------------------ 改期重校验


def test_reschedule_successor_too_early_returns_409(client, chain_of_three):
    from app.schemas import ReservationDependencyViolationOut

    a, b, _ = chain_of_three
    _dep(client, "dep-ab", a, b, gap=30)
    # b 改到 11:15 开始：a 11:00 结束 + 30 <= 11:15 不成立（仅 15 分钟）
    resp = client.put(
        f"/api/v1/reservations/{b}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T11:15:00Z",
            "end_utc": "2026-03-09T11:45:00Z",
            "participant_ids": ["bob"],
        },
    )
    assert resp.status_code == 409, resp.text
    data = resp.json()
    assert data["violations"]
    violation = data["violations"][0]
    assert violation["predecessor_id"] == a
    assert violation["successor_id"] == b
    assert violation["minimum_gap_minutes"] == 30
    assert violation["actual_gap_minutes"] == 15
    chain_ids = [node["reservation_id"] for node in violation["chain"]]
    assert chain_ids == [a, b]
    # 链上给出各节点重校验时刻（ISO 8601 带 Z，与其他接口一致）
    by_id = {node["reservation_id"]: node for node in violation["chain"]}
    assert by_id[b]["start_utc"] == "2026-03-09T11:15:00Z"
    assert by_id[a]["end_utc"] == "2026-03-09T11:00:00Z"
    # 响应结构可由文档化的输出模型校验
    validated = ReservationDependencyViolationOut.model_validate(violation)
    assert [node.reservation_id for node in validated.chain] == [a, b]


def test_reschedule_predecessor_too_late_returns_409(client, chain_of_three):
    a, b, _ = chain_of_three
    _dep(client, "dep-ab", a, b, gap=30)
    # a 改到 11:20 结束：11:20 + 30 = 11:50 <= b 11:30 不成立
    resp = client.put(
        f"/api/v1/reservations/{a}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T10:20:00Z",
            "end_utc": "2026-03-09T11:20:00Z",
            "participant_ids": ["alice"],
        },
    )
    assert resp.status_code == 409
    violation = resp.json()["violations"][0]
    assert (violation["predecessor_id"], violation["successor_id"]) == (a, b)
    assert violation["actual_gap_minutes"] == 10


def test_reschedule_exact_gap_boundary_allowed(client, chain_of_three):
    a, b, _ = chain_of_three
    _dep(client, "dep-ab", a, b, gap=30)
    # b 改到 11:30 开始（恰好 a 11:00 + 30）：等号成立，允许
    resp = client.put(
        f"/api/v1/reservations/{b}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T11:30:00Z",
            "end_utc": "2026-03-09T12:00:00Z",
            "participant_ids": ["bob"],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["version"] == 2


def test_gap_zero_means_adjacent_allowed(client, chain_of_three):
    a, b, _ = chain_of_three
    _dep(client, "dep-ab0", a, b, gap=0)
    # b 改到 a 结束即开始：11:00 相接，gap 0 满足
    resp = client.put(
        f"/api/v1/reservations/{b}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T11:00:00Z",
            "end_utc": "2026-03-09T11:30:00Z",
            "participant_ids": ["bob"],
        },
    )
    assert resp.status_code == 200, resp.text
    # 再改成重叠：gap < 0 违反
    resp = client.put(
        f"/api/v1/reservations/{b}",
        json={
            "expected_version": 2,
            "start_utc": "2026-03-09T10:30:00Z",
            "end_utc": "2026-03-09T11:00:00Z",
            "participant_ids": ["bob"],
        },
    )
    assert resp.status_code == 409
    assert resp.json()["violations"][0]["actual_gap_minutes"] == -30


def test_reschedule_unreachable_dependency_not_checked(client, chain_of_three):
    a, b, c = chain_of_three
    # 只在 b -> c 之间有依赖；改 a 不影响它，即使把 a 任意移动也不校验 b->c
    _dep(client, "dep-bc", b, c, gap=30)
    resp = client.put(
        f"/api/v1/reservations/{a}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T20:00:00Z",
            "end_utc": "2026-03-09T21:00:00Z",
            "participant_ids": ["alice"],
        },
    )
    assert resp.status_code == 200, resp.text


def test_transitive_dependency_revalidated_with_full_chain(client, chain_of_three):
    a, b, c = chain_of_three
    # a -> b gap 30（11:00 -> 11:30 恰好），b -> c gap 30（12:00 -> 12:30 恰好）
    _dep(client, "dep-ab", a, b, gap=30)
    _dep(client, "dep-bc", b, c, gap=30)
    # 把 a 延长到 11:30 结束：a->b 直接失败（b 11:30 开始，gap 0 < 30）
    resp = client.put(
        f"/api/v1/reservations/{a}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T10:30:00Z",
            "end_utc": "2026-03-09T11:30:00Z",
            "participant_ids": ["alice"],
        },
    )
    assert resp.status_code == 409
    # 移动前驱 a 时，后继链路 a->b、b->c 全部可达；a->b 已被违反。
    # （b->c 仍满足，不在 violations 中）
    violations = resp.json()["violations"]
    pairs = [(v["predecessor_id"], v["successor_id"]) for v in violations]
    assert pairs == [(a, b)]

    # 现在只移动 b：b 提前到 10:45-11:00。a->b：11:00 + 30 <= 10:45 失败；
    # b->c：11:00 + 30 <= 12:30 满足。只有 a->b 违规。
    resp = client.put(
        f"/api/v1/reservations/{b}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T10:45:00Z",
            "end_utc": "2026-03-09T11:00:00Z",
            "participant_ids": ["bob"],
        },
    )
    assert resp.status_code == 409
    pairs = [(v["predecessor_id"], v["successor_id"])
             for v in resp.json()["violations"]]
    assert pairs == [(a, b)]
    chain_ids = [node["reservation_id"]
                 for node in resp.json()["violations"][0]["chain"]]
    assert chain_ids == [a, b]


def test_failed_reschedule_is_fully_rolled_back(client, chain_of_three):
    a, b, _ = chain_of_three
    _dep(client, "dep-ab", a, b, gap=30)
    resp = client.put(
        f"/api/v1/reservations/{b}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T11:15:00Z",
            "end_utc": "2026-03-09T11:45:00Z",
            "participant_ids": ["bob"],
        },
    )
    assert resp.status_code == 409
    # 版本未推进、时段未变化、没有新审计事件
    history = client.get(
        f"/api/v1/reservations/{b}/history").json()["events"]
    assert len(history) == 1
    assert history[0]["action"] == "created"
    assert history[0]["snapshot"]["start_utc"].startswith("2026-03-09T11:30")

    # 原版本仍可成功改期到满足约束的时段
    resp = client.put(
        f"/api/v1/reservations/{b}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T11:30:00Z",
            "end_utc": "2026-03-09T12:00:00Z",
            "participant_ids": ["bob"],
        },
    )
    assert resp.status_code == 200


def test_successful_reschedule_appends_audit_event_as_before(client, chain_of_three):
    a, b, _ = chain_of_three
    _dep(client, "dep-ab", a, b, gap=30)
    resp = client.put(
        f"/api/v1/reservations/{b}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T12:00:00Z",
            "end_utc": "2026-03-09T12:15:00Z",
            "participant_ids": ["bob"],
        },
    )
    assert resp.status_code == 200
    history = client.get(
        f"/api/v1/reservations/{b}/history").json()["events"]
    assert [e["action"] for e in history] == ["created", "rescheduled"]
    assert history[-1]["snapshot"]["version"] == 2
    # 哈希链仍可由服务端重验（GET 历史返回 200 即已重验）


# ------------------------------------------------------------ 取消重校验


def test_cancel_does_not_block_when_incident_edge_dormant(client, chain_of_three):
    a, b, _ = chain_of_three
    _dep(client, "dep-ab", a, b, gap=30)
    # 取消前驱 a：a->b 一端已取消，边休眠，取消成功
    resp = client.delete(
        f"/api/v1/reservations/{a}?expected_version=1")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "canceled"
    assert resp.json()["version"] == 2

    # 取消后 b 可自由改期（休眠边不再约束）
    resp = client.put(
        f"/api/v1/reservations/{b}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T08:00:00Z",
            "end_utc": "2026-03-09T08:30:00Z",
            "participant_ids": ["bob"],
        },
    )
    assert resp.status_code == 200, resp.text


def test_cancel_in_the_middle_keeps_other_edges_enforced(client, chain_of_three):
    a, b, c = chain_of_three
    # a -> b -> c 链；另外 a -> c 直连一条 gap 90（11:00 + 90 = 12:30 恰好）
    _dep(client, "dep-ab", a, b, gap=30)
    _dep(client, "dep-bc", b, c, gap=30)
    _dep(client, "dep-ac", a, c, gap=90)
    # 取消 b：a->b、b->c 休眠；a->c 仍为 active-active，必须继续满足
    resp = client.delete(f"/api/v1/reservations/{b}?expected_version=1")
    assert resp.status_code == 200, resp.text
    # b 取消后把 a 延长到 11:30 结束：a->c 要求 11:30 + 90 <= 12:30 不成立
    resp = client.put(
        f"/api/v1/reservations/{a}",
        json={
            "expected_version": 1,
            "start_utc": "2026-03-09T10:30:00Z",
            "end_utc": "2026-03-09T11:30:00Z",
            "participant_ids": ["alice"],
        },
    )
    assert resp.status_code == 409
    violation = resp.json()["violations"][0]
    assert (violation["predecessor_id"], violation["successor_id"]) == (a, c)
    assert violation["minimum_gap_minutes"] == 90
    assert violation["actual_gap_minutes"] == 60


def test_failed_cancel_is_rolled_back_without_partial_write(client, chain_of_three):
    a, b, c = chain_of_three
    _dep(client, "dep-ab", a, b, gap=30)
    _dep(client, "dep-bc", b, c, gap=30)
    _dep(client, "dep-ac", a, c, gap=90)
    # 构造 a->c 已被违反的状态不可达（改期会被拦），取消路径本身不会违反；
    # 版本不符时取消回滚，依赖与预约状态保持原样。
    resp = client.delete(f"/api/v1/reservations/{a}?expected_version=5")
    assert resp.status_code == 409
    history = client.get(
        f"/api/v1/reservations/{a}/history").json()["events"]
    assert [e["action"] for e in history] == ["created"]


# --------------------------------------------------------------- 交换重校验


def test_swap_violating_dependency_returns_409(client, chain_of_three):
    a, b, c = chain_of_three
    # a 10-11, b 11:30-12, c 12:30-13；a->b gap30 满足（11:00 -> 11:30）
    _dep(client, "dep-ab", a, b, gap=30)
    # 需要两条可交换的预约：另建 d 08:00-09:00（参与者 dave，无冲突）
    d = _res({
        "request_id": "req-d",
        "start_utc": "2026-03-09T08:00:00Z",
        "end_utc": "2026-03-09T09:00:00Z",
        "participant_ids": ["dave"],
    }, client)
    # 交换 a 与 d：a 落到 08-09，d 落到 10-11。
    # a->b 变为 09:00 + 30 <= 11:30 满足；d 无依赖。故交换应成功。
    resp = client.post(
        "/api/v1/reservations/swap",
        json={
            "first": {"reservation_id": a, "expected_version": 1},
            "second": {"reservation_id": d, "expected_version": 1},
        },
    )
    assert resp.status_code == 200, resp.text

    # 再交换 b 与 c（当前 b 11:30-12, c 12:30-13，b->c gap30 满足）：
    # 交换后 b 为 12:30-13, c 为 11:30-12。
    # a（现在 08-09）-> b：09:00 + 30 <= 12:30 满足；
    # b -> c：13:00 + 30 <= 11:30 不成立 → 409
    _dep(client, "dep-bc2", b, c, gap=30)
    resp = client.post(
        "/api/v1/reservations/swap",
        json={
            "first": {"reservation_id": b, "expected_version": 1},
            "second": {"reservation_id": c, "expected_version": 1},
        },
    )
    assert resp.status_code == 409, resp.text
    violations = resp.json()["violations"]
    assert any(
        (v["predecessor_id"], v["successor_id"]) == (b, c)
        for v in violations
    )
    # 链路含完整节点信息，且被移动节点取待写入时刻
    violation = next(
        v for v in violations
        if (v["predecessor_id"], v["successor_id"]) == (b, c)
    )
    by_id = {node["reservation_id"]: node for node in violation["chain"]}
    assert by_id[b]["start_utc"].startswith("2026-03-09T12:30")
    assert by_id[c]["end_utc"].startswith("2026-03-09T12:00")

    # 失败交换整体回滚：双方版本仍为 1（a、d 已各 +1，b、c 未变）
    assert _version(client, b) == 1
    assert _version(client, c) == 1


def test_swap_satisfying_dependencies_succeeds_and_bumps_versions(
    client, chain_of_three
):
    a, b, _ = chain_of_three
    _dep(client, "dep-ab", a, b, gap=30)
    # d 07:00-08:00：与 a 交换后 d 占 10-11，a 占 07-08；
    # a->b：08:00 + 30 <= 11:30 满足
    d = _res({
        "request_id": "req-d",
        "start_utc": "2026-03-09T07:00:00Z",
        "end_utc": "2026-03-09T08:00:00Z",
        "participant_ids": ["dave"],
    }, client)
    resp = client.post(
        "/api/v1/reservations/swap",
        json={
            "first": {"reservation_id": a, "expected_version": 1},
            "second": {"reservation_id": d, "expected_version": 1},
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["first"]["version"] == 2
    assert resp.json()["second"]["version"] == 2


# --------------------------------------------------------------- 并发原子性


def test_concurrent_cycle_attempts_single_winner(
    db_path, client, chain_of_three
):
    a, b, c = chain_of_three
    _dep(client, "dep-ab", a, b)
    barrier = threading.Barrier(2)

    def worker(request_id, predecessor, successor, gap):
        barrier.wait()
        return dep_core.create_dependency(
            dependency_request_id=request_id,
            predecessor_id=predecessor,
            successor_id=successor,
            minimum_gap_minutes=gap,
            db_path=db_path,
        )

    # 并发建立 b->c 与 c->a：无论谁先，后者必然因成环被拒
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_bc = pool.submit(worker, "dep-bc", b, c, 30)
        fut_ca = pool.submit(worker, "dep-ca", c, a, 0)
        futures = (fut_bc, fut_ca)
        outcomes = []
        for fut in futures:
            try:
                _, created = fut.result()
                outcomes.append(("created", created))
            except dep_core.DependencyCycleError:
                outcomes.append(("cycle", None))
    assert sorted(outcomes) == [("created", True), ("cycle", None)]


def test_concurrent_dependency_replay_creates_one_row(db_path, chain_of_three):
    a, b, _ = chain_of_three
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        return dep_core.create_dependency(
            dependency_request_id="dep-same",
            predecessor_id=a,
            successor_id=b,
            minimum_gap_minutes=30,
            db_path=db_path,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        # 先全部提交（max_workers=4 保证四个 worker 都能到达 barrier），
        # 再统一收集结果
        futures = [pool.submit(worker) for _ in range(4)]
        results = [fut.result() for fut in futures]
    assert sum(created for _, created in results) == 1
    assert len({dep.id for dep, _ in results}) == 1


def test_concurrent_reschedules_violating_and_keeping_gap(
    db_path, client, chain_of_three
):
    a, b, _ = chain_of_three
    _dep(client, "dep-ab", a, b, gap=30)
    barrier = threading.Barrier(2)

    def move(start: str, end: str):
        barrier.wait()
        return res_core.reschedule_reservation(
            reservation_id=b,
            expected_version=1,
            start_utc=datetime.fromisoformat(start.replace("Z", "+00:00")),
            end_utc=datetime.fromisoformat(end.replace("Z", "+00:00")),
            participant_ids=("bob",),
            db_path=db_path,
        )

    # 两个并发改期 b：一个满足（11:30），一个违反（11:15）。
    # 满足者先提交 → 违反者随后因依赖违规或版本过期失败；
    # 违反者先到 → 409 回滚，满足者正常成功。任何交错下最终时段都必须
    # 满足依赖，且至多一次写入。
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_ok = pool.submit(
            move, "2026-03-09T11:30:00Z", "2026-03-09T12:00:00Z")
        fut_bad = pool.submit(
            move, "2026-03-09T11:15:00Z", "2026-03-09T11:45:00Z")
        bad_error: Exception | None = None
        try:
            fut_ok.result()
            good_won = True
        except Exception as exc:  # noqa: BLE001 - 违反者先到时满足者必成功，这里理论上不可达
            good_won = False
            bad_error = exc
        try:
            fut_bad.result()
        except (dep_core.DependencyViolationError,
                res_core.VersionConflictError) as exc:
            bad_error = exc
    assert good_won, "satisfying reschedule must always succeed"
    assert isinstance(
        bad_error,
        (dep_core.DependencyViolationError, res_core.VersionConflictError),
    )
    conn = db_module.connect(db_path)
    try:
        final = res_core.load_reservation(conn, b)
    finally:
        conn.close()
    assert final.start_utc.isoformat() == "2026-03-09T11:30:00+00:00"
    assert final.version == 2
