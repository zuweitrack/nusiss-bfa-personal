import asyncio
import logging
import time
from typing import Any

from app.agents.base import BaseAgent, StrictAgentModel
from app.llm.gateway import FunctionCallRecord, StructuredLLM
from app.monitoring.metrics import record_agent_run
from app.tools.registry import ToolRegistry

logger = logging.getLogger("app.agents.response")


class ResponseOutput(StrictAgentModel):
    message: str
    citations: list[str]
    reason_codes: list[str]


REFUSAL_MESSAGE = "I do not have enough approved facility information to answer that question."


class ResponseAgent(BaseAgent[ResponseOutput]):
    name = "response"
    output_schema = ResponseOutput

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        model: str,
        timeout_seconds: float,
        tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(
            llm,
            model=model,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
        )
        self.tools = tools
        self.tool_calls: list[FunctionCallRecord] = []

    async def run(self, payload: dict[str, Any]) -> ResponseOutput:
        # If tool calling is explicitly requested or when payload is an ungrounded query
        # and public tools (e.g. search_approved_knowledge, lookup_incident_status) are registered:
        use_tools = payload.get("use_tools", False) or (
            self.tools is not None
            and "retrieval_chunks" not in payload
            and "status_lookup" not in payload
            and "reference_code" not in payload
            and bool(
                self.tools.function_tools(
                    caller_role="PUBLIC",
                    names={"search_approved_knowledge", "lookup_incident_status"},
                )
            )
        )

        if not use_tools or self.tools is None:
            output = await super().run(payload)
            if output.message.strip() == REFUSAL_MESSAGE:
                # A refusal draws no fact from any chunk, so it can never cite one.
                output.citations = []
            return output

        registry = self.tools
        available_tools = registry.function_tools(
            caller_role="PUBLIC",
            names={"search_approved_knowledge", "lookup_incident_status"},
        )

        async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            result = await registry.execute(name, arguments, caller_role="PUBLIC")
            return result.model_dump(mode="json")

        start_time = time.perf_counter()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                result = await self.llm.generate_with_tools(
                    system_prompt=self.system_prompt,
                    user_payload=payload,
                    output_schema=self.output_schema,
                    tools=available_tools,
                    tool_executor=execute_tool,
                    model=self.model,
                    temperature=0.0,
                    timeout_seconds=self.timeout_seconds,
                    max_tool_calls=1,
                    **self._effort_kwargs(),
                )
            self.tool_calls = result.tool_calls
            output = result.output
            if output.message.strip() == REFUSAL_MESSAGE:
                output.citations = []
            duration = time.perf_counter() - start_time
            record_agent_run(self.name, status="success", duration_seconds=duration)
            return output
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.warning(
                "Agent '%s' tool-calling failed after %.2fs, executing fallback: %r",
                self.name,
                duration,
                exc,
            )
            record_agent_run(self.name, status="fallback", duration_seconds=duration)
            return self.fallback(payload, exc)

    def fallback(self, payload: dict[str, Any], error: Exception) -> ResponseOutput:
        del error
        reference = payload.get("reference_code")
        message = (
            f"Your report has been saved. Reference: {reference}."
            if reference
            else "The request needs manual review."
        )
        return ResponseOutput(message=message, citations=[], reason_codes=["CONTROLLED_TEMPLATE"])
