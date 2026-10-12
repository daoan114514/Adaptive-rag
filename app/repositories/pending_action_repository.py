"""Server-side record of write actions awaiting human confirmation."""

import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import ToolRisk
from app.repositories.models import PendingActionRow
from app.schemas.chat import ActionProposal


class PendingActionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, proposal: ActionProposal, *, user_id: str, session_id: str) -> None:
        self._session.add(
            PendingActionRow(
                action_id=proposal.action_id,
                user_id=user_id,
                session_id=session_id,
                tool_name=proposal.tool_name,
                arguments_json=json.dumps(proposal.arguments, ensure_ascii=False, sort_keys=True),
                risk=proposal.risk.value,
                summary=proposal.summary,
            )
        )
        await self._session.flush()

    async def get_owned(
        self, action_id: str, *, user_id: str, session_id: str
    ) -> ActionProposal | None:
        """Return the proposal only to the user and session it was issued to.

        Unknown ids and ids issued elsewhere look the same, so a caller cannot
        probe for other users' pending actions.
        """
        row = await self._session.get(PendingActionRow, action_id)
        if row is None or row.user_id != user_id or row.session_id != session_id:
            return None
        return ActionProposal(
            action_id=row.action_id,
            tool_name=row.tool_name,
            arguments=json.loads(row.arguments_json),
            risk=ToolRisk(row.risk),
            summary=row.summary,
        )
