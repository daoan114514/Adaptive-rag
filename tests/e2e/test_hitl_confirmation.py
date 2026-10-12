"""HITL: a write runs only after the user confirms the exact action the server proposed."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config.settings import clear_settings_cache
from app.main import create_app


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    # Each test gets its own database: cancelling ORD10003 must not leak into the
    # shared test.db that other tests reuse across runs.
    path = tmp_path / "hitl.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{path.as_posix()}")
    clear_settings_cache()
    yield path
    clear_settings_cache()


@pytest.fixture
def client(db_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def _scalar(db_path: Path, sql: str, *params: str):
    with closing(sqlite3.connect(db_path)) as con:
        return con.execute(sql, params).fetchone()[0]


def _status(db_path: Path, order_id: str) -> str:
    return _scalar(db_path, "select status from orders where order_id = ?", order_id)


def _audit_count(db_path: Path, action: str) -> int:
    return _scalar(db_path, "select count(*) from audit_logs where action = ?", action)


def _chat(client: TestClient, message: str, **extra: str):
    return client.post("/api/v1/chat", json={"message": message, "user_id": "u_demo", **extra})


def _propose_cancel(client: TestClient) -> dict:
    body = _chat(client, "帮我取消订单 ORD10003").json()
    pending = body["pending_action"]
    assert pending is not None
    assert pending["tool_name"] == "cancel_order"
    assert pending["arguments"] == {"order_id": "ORD10003"}
    return body


def test_demo_orders_are_seeded_on_startup(client: TestClient, db_path: Path) -> None:
    assert _status(db_path, "ORD10001") == "shipped"
    assert _status(db_path, "ORD10002") == "delivered"
    assert _status(db_path, "ORD10003") == "paid"


def test_cancel_runs_only_after_confirming_the_stored_action(
    client: TestClient, db_path: Path
) -> None:
    proposed = _propose_cancel(client)
    assert _status(db_path, "ORD10003") == "paid"

    # The CLI confirms with just "确认"; the stored action must still run.
    confirmed = _chat(
        client,
        "确认",
        session_id=proposed["session_id"],
        confirm_action_id=proposed["pending_action"]["action_id"],
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["pending_action"] is None
    assert _status(db_path, "ORD10003") == "cancelled"

    replay = _chat(
        client,
        "确认",
        session_id=proposed["session_id"],
        confirm_action_id=proposed["pending_action"]["action_id"],
    )
    assert replay.status_code == 200
    assert _audit_count(db_path, "cancel_order") == 1


def test_unknown_confirm_action_id_is_rejected(client: TestClient, db_path: Path) -> None:
    response = _chat(client, "帮我取消订单 ORD10003", confirm_action_id="act_made_up_by_client")
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_REQUEST"
    assert _status(db_path, "ORD10003") == "paid"


def test_confirmation_is_bound_to_user_and_session(client: TestClient, db_path: Path) -> None:
    proposed = _propose_cancel(client)
    action_id = proposed["pending_action"]["action_id"]

    other_user = client.post(
        "/api/v1/chat",
        json={
            "message": "确认",
            "user_id": "someone_else",
            "session_id": proposed["session_id"],
            "confirm_action_id": action_id,
        },
    )
    assert other_user.status_code == 400

    other_session = _chat(client, "确认", confirm_action_id=action_id)
    assert other_session.status_code == 400

    assert _status(db_path, "ORD10003") == "paid"
