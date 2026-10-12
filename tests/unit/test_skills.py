import pytest

from app.agents.nodes.planner import skill_to_plan
from app.skills.loader import match_skill
from app.tools.factory import build_tool_registry


def test_match_skill_by_trigger() -> None:
    skill = match_skill("这件衣服能退货吗", "faq_policy")
    assert skill is not None
    assert skill["name"] == "return_handling"
    assert "create_return_request" in skill["tools"]


def test_match_skill_by_intent_when_no_trigger() -> None:
    skill = match_skill("查一下 ORD10001", "order_inquiry")
    assert skill is not None
    assert skill["name"] == "order_tracking"


def test_no_skill_for_chitchat() -> None:
    assert match_skill("今天天气真好", "chitchat") is None


@pytest.mark.parametrize(
    ("query", "intent", "skill_name", "write_tool"),
    [
        ("帮我取消订单 ORD10003", "cancel_order", "order_cancel", "cancel_order"),
        ("不想要了，ORD10003 帮我取消", "cancel_order", "order_cancel", "cancel_order"),
        ("ORD10002 我要申请退款", "refund", "refund_handling", "create_refund_request"),
    ],
)
def test_write_intents_reach_their_write_tool(
    query: str, intent: str, skill_name: str, write_tool: str
) -> None:
    # The planner drops any step outside the matched skill's tools, so a write
    # tool missing here can never be proposed, let alone confirmed.
    skill = match_skill(query, intent)
    assert skill is not None
    assert skill["name"] == skill_name
    assert write_tool in skill["tools"]


def test_rule_plan_proposes_cancel_when_order_id_is_known() -> None:
    registry = build_tool_registry(retriever=None)
    skill = match_skill("帮我取消订单 ORD10003", "cancel_order")
    assert skill is not None
    plan = skill_to_plan(skill, "帮我取消订单 ORD10003", registry)
    cancel = [step for step in plan.steps if step.tool_name == "cancel_order"]
    assert cancel and cancel[0].arguments == {"order_id": "ORD10003"}


def test_rule_plan_skips_writes_it_cannot_fill() -> None:
    # No reason can be derived from the query, so proposing a return would only
    # yield a confirmation that fails argument validation.
    registry = build_tool_registry(retriever=None)
    skill = match_skill("ORD10002 这件衣服能退货吗", "return_exchange")
    assert skill is not None
    plan = skill_to_plan(skill, "ORD10002 这件衣服能退货吗", registry)
    assert "create_return_request" not in [step.tool_name for step in plan.steps]
    assert plan.steps
