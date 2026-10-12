import pytest
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.agents.nodes.executor import executor_node
from app.config.settings import Settings
from app.core.enums import ToolRisk
from app.core.request_scope import request_session
from app.repositories.db import init_schema
from app.schemas.agent import AgentPlan, PlanStep, ToolResult
from app.schemas.chat import ActionProposal
from app.tools.base import Tool, ToolContext
from app.tools.registry import ToolRegistry


class _OrderInput(BaseModel):
    order_id: str = Field(default="")


class _RecordingWrite(Tool):
    risk = ToolRisk.DESTRUCTIVE
    input_model = _OrderInput

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name
        self.calls: list[str] = []

    async def execute(self, parsed: _OrderInput, context: ToolContext) -> ToolResult:
        self.calls.append(parsed.order_id)
        return ToolResult(name=self.name, success=True, output={"order_id": parsed.order_id})


@pytest.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    await init_schema(engine)
    yield async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    await engine.dispose()


def _state(plan: AgentPlan, approved: ActionProposal | None) -> dict:
    return {
        "trace_id": "t",
        "session_id": "s",
        "user_id": "u",
        "user_query": "确认",
        "plan": plan,
        "pending_action": None,
        "confirmed_action": approved,
        "step_count": 0,
    }


@pytest.mark.asyncio
async def test_approval_covers_only_the_confirmed_step(session_factory) -> None:
    cancel, refund = _RecordingWrite("cancel_order"), _RecordingWrite("create_refund_request")
    registry = ToolRegistry()
    registry.register(cancel)
    registry.register(refund)
    approved = ActionProposal(
        action_id="act_1",
        tool_name="cancel_order",
        arguments={"order_id": "ORD10003"},
        risk=ToolRisk.DESTRUCTIVE,
        summary="",
    )
    plan = AgentPlan(
        steps=[
            PlanStep(
                step_id=1, goal="a", tool_name="cancel_order", arguments={"order_id": "ORD10003"}
            ),
            PlanStep(
                step_id=2, goal="b", tool_name="cancel_order", arguments={"order_id": "ORD10001"}
            ),
            PlanStep(
                step_id=3,
                goal="c",
                tool_name="create_refund_request",
                arguments={"order_id": "ORD10003"},
            ),
        ]
    )
    async with request_session(session_factory):
        update = await executor_node(_state(plan, approved), registry=registry, settings=Settings())

    assert cancel.calls == ["ORD10003"]
    assert refund.calls == []
    assert [result.success for result in update["tool_results"]] == [True, False, False]


@pytest.mark.asyncio
async def test_writes_are_denied_without_a_verified_confirmation(session_factory) -> None:
    cancel = _RecordingWrite("cancel_order")
    registry = ToolRegistry()
    registry.register(cancel)
    plan = AgentPlan(
        steps=[
            PlanStep(step_id=1, goal="a", tool_name="cancel_order", arguments={"order_id": "X1"})
        ]
    )
    async with request_session(session_factory):
        update = await executor_node(_state(plan, None), registry=registry, settings=Settings())

    assert cancel.calls == []
    assert update["tool_results"][0].success is False
