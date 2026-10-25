from typing import Any
import pytest

from app.agents.response import ResponseAgent
from app.llm.fake import FakeStructuredLLM
from app.llm.gateway import FunctionCallRecord, FunctionTool, ToolCallingResult
from app.rag.embeddings import FakeEmbeddings
from app.tools.registry import ToolRegistry
from tests.fakes import InMemoryIncidentRepository
from app.tools.incident_tools import register_incident_tools


@pytest.mark.asyncio
async def test_response_agent_executes_tool_when_available() -> None:
    incidents = InMemoryIncidentRepository()
    incident = incidents.create(description="Aircon leaking water", location="Level 3 Room 302")
    ref = incident.reference_code

    tools = ToolRegistry()
    register_incident_tools(tools, incidents, FakeEmbeddings(dim=1536))

    class ToolCallingLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def generate(self, **kwargs: Any) -> Any:
            raise AssertionError("generate() should not be called when tool calling is triggered")

        async def generate_with_tools(
            self,
            *,
            system_prompt: str,
            user_payload: dict[str, Any],
            output_schema: type[Any],
            tools: list[FunctionTool],
            tool_executor: Any,
            model: str,
            **kwargs: Any,
        ) -> Any:
            self.calls.append({"payload": user_payload, "tools": [t.name for t in tools]})
            # Execute tool: lookup_incident_status
            tool_result = await tool_executor("lookup_incident_status", {"reference_code": ref})
            return ToolCallingResult(
                output=output_schema(
                    message=f"Incident {ref} is currently {tool_result['data']['status']}.",
                    citations=[],
                    reason_codes=["STATUS_TOOL_CALL"],
                ),
                tool_calls=[
                    FunctionCallRecord(
                        call_id="call-lookup-1",
                        name="lookup_incident_status",
                        model_input=user_payload,
                        arguments={"reference_code": ref},
                        output=tool_result,
                    )
                ],
                model_calls=2,
                first_model_input=user_payload,
            )

    llm = ToolCallingLLM()
    agent = ResponseAgent(
        llm,  # type: ignore[arg-type]
        model="fake",
        timeout_seconds=5.0,
        tools=tools,
    )

    # Calling with use_tools=True
    output = await agent.run({"text": f"What is the status of {ref}?", "use_tools": True})

    assert output.reason_codes == ["STATUS_TOOL_CALL"]
    assert ref in output.message
    assert len(agent.tool_calls) == 1
    assert agent.tool_calls[0].name == "lookup_incident_status"
    assert agent.tool_calls[0].arguments == {"reference_code": ref}
    assert agent.tool_calls[0].output["success"] is True
    assert agent.tool_calls[0].output["data"]["reference_code"] == ref


@pytest.mark.asyncio
async def test_response_agent_backward_compat_without_tools() -> None:
    llm = FakeStructuredLLM(
        handlers={
            "ResponseOutput": {
                "message": "Report logged as BFA-001.",
                "citations": [],
                "reason_codes": ["INCIDENT_ACK"],
            }
        }
    )
    # When payload has reference_code (Mode A acknowledgement), it does not trigger tool calling
    agent = ResponseAgent(llm, model="fake", timeout_seconds=5.0, tools=None)
    output = await agent.run({"reference_code": "BFA-001", "assigned_team": "Plumbing", "priority": "P2"})

    assert output.message == "Report logged as BFA-001."
    assert agent.tool_calls == []
