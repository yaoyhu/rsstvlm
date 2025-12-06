import asyncio
from typing import Any

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import ToolOutput, ToolSelection
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai_like import OpenAILike

from rsstvlm.services.mcp.client import MCPClient
from rsstvlm.utils import deepseek


class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class AgentWorkflow(Workflow):
    def __init__(
        self,
        llm: OpenAILike,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mcp_client = MCPClient()
        self.llm = llm
        self.tools = None

    async def _setup_tools(self):
        """Asynchronously connect to the server and get the tools."""
        return await self.mcp_client.connect_to_server()

    @classmethod
    async def create(
        cls,
        llm: OpenAILike,
        *args: Any,
        **kwargs: Any,
    ):
        """Async factory to create an agent and initialize its tools."""
        agent = cls(llm, *args, **kwargs)
        agent.tools = await agent._setup_tools()
        return agent

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: StartEvent
    ) -> InputEvent:
        # clear sources
        await ctx.store.set("sources", [])

        # check if memory is setup
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # get chat history
        chat_history = memory.get()

        # update context
        await ctx.store.set("memory", memory)

        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input

        # stream the response
        response_stream = await self.llm.astream_chat_with_tools(
            tools=self.tools, chat_history=chat_history
        )
        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        # save the final response, which should have all content
        memory = await ctx.store.get("memory")
        memory.put(response.message)
        await ctx.store.set("memory", memory)

        # get tool calls
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            sources = await ctx.store.get("sources", default=[])
            return StopEvent(
                result={"response": response, "sources": [*sources]}
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> InputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []
        sources = await ctx.store.get("sources", default=[])

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                kwargs = tool_call.tool_kwargs
                if "kwargs" in kwargs and isinstance(kwargs["kwargs"], dict):
                    kwargs = kwargs["kwargs"]

                tool_output = tool(**kwargs)
                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        # update memory
        memory = await ctx.store.get("memory")
        for msg in tool_msgs:
            memory.put(msg)

        await ctx.store.set("sources", sources)
        await ctx.store.set("memory", memory)

        chat_history = memory.get()
        return InputEvent(input=chat_history)


async def main():
    agent = await AgentWorkflow.create(
        deepseek, timeout=120, verbose=True
    )
    ret = await agent.run(input="Querying the database, what does excessive NO2 cause?")
    print(ret["response"])


if __name__ == "__main__":
    asyncio.run(main())
