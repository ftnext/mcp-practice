# Build server: https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search#build
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, Protocol, TypedDict, get_args

from anthropic import Anthropic
from anthropic.types import ContentBlock, ToolUseBlock
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, Tool, stdio_client
from mcp.client.stdio import get_default_environment
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

load_dotenv()

MAX_TOKENS = 1000

logger = logging.getLogger("my_mcp_client")

SupportedModels = Literal["gpt-4o-mini", "gemini-2.0-flash", "claude-3-5-sonnet-latest"]
ServerName = str
ToolName = str


class MCPServerParameter(TypedDict):
    command: str
    args: list[str]
    env: NotRequired[list[str]]


class MCPClient:
    def __init__(self, llm_api_client):
        self.exit_stack = AsyncExitStack()
        self.llm_api_client = llm_api_client
        self.available_tools: list[Tool] = []
        self.tool2session: dict[ToolName, ClientSession] = {}

    async def connect_to_server(self, servers: Mapping[ServerName, MCPServerParameter]):
        sessions: dict[ServerName, ClientSession] = {}
        for name, parameters in servers.items():
            env = get_default_environment()
            for env_name in parameters.get("env", []):
                env[env_name] = os.getenv(env_name)
            server_params = StdioServerParameters(
                command=parameters["command"],
                args=parameters["args"],
                env=env,
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            sessions[name] = await self.exit_stack.enter_async_context(
                ClientSession(*stdio_transport)
            )

        for name, session in sessions.items():
            await session.initialize()
            response = await session.list_tools()
            self.available_tools.extend(response.tools)
            for tool in response.tools:
                self.tool2session[tool.name] = session
            logger.debug(
                "Session of server %s is initialized and tools are listed", name
            )

        logger.info(
            "Connection to server with tools: %s",
            [tool.name for tool in self.available_tools],
        )

    def completion(self, messages):
        return self.llm_api_client.completion(
            messages,
            max_tokens=MAX_TOKENS,
            tools=self.llm_api_client.format_tools(self.available_tools),
        )

    def get_text(self, message):
        return self.llm_api_client.get_text(message)

    def get_tool_calls(self, message):
        return self.llm_api_client.get_tool_calls(message)

    async def call_tools(self, tool_calls):
        return [await self.call_tool(tool_call) for tool_call in tool_calls]

    async def call_tool(self, tool_call):
        tool = self.llm_api_client.format_tool_call(tool_call)
        tool_result = await self.tool2session[tool.name].call_tool(tool.name, tool.args)
        logger.info(
            "[Calling tool %s with args %s, Got %s]", tool.name, tool.args, tool_result
        )
        return self.llm_api_client.format_tool_result(tool_result.content, tool)

    async def cleanup(self):
        await self.exit_stack.aclose()


class ChatSession:
    def __init__(self, client: MCPClient) -> None:
        self.mcp_client = client
        self.messages = []

    def print_empty_line(self):
        print()

    async def start(self) -> None:
        logger.info("MCP Client Started!")
        print("Type your queries or `quit` to exit.")

        while True:
            try:
                self.print_empty_line()
                query = input("Query: ").strip()
                if query == "":
                    continue
                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                self.print_empty_line()
                print(response)
            except Exception as e:
                logger.exception("Error: %r", e)
                self.print_empty_line()
                print("エラーが発生しました。終了します")
                break

    async def process_query(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})
        message = self.mcp_client.completion(self.messages)

        self.messages.append(message)
        while tool_calls := self.mcp_client.get_tool_calls(message):
            tool_call_messages = await self.mcp_client.call_tools(tool_calls)
            self.messages.extend(tool_call_messages)

            message = self.mcp_client.completion(self.messages)
            self.messages.append(message)

        return self.mcp_client.get_text(message)


async def main(llm_api_client: LlmWebApiClient) -> None:
    with open("servers.json") as f:
        servers = json.load(f)
    mcp_client = MCPClient(llm_api_client)
    try:
        await mcp_client.connect_to_server(servers["mcpServers"])
        chat_session = ChatSession(mcp_client)
        await chat_session.start()
    finally:
        await mcp_client.cleanup()


@dataclass
class GenericToolCall:
    id: str
    name: str
    args: dict[str, Any]


class LlmWebApiClientInterface(Protocol):
    def format_tools(self, tools: Iterable[Tool]): ...

    def completion(self, messages, max_tokens, tools): ...

    def get_text(self, message): ...

    def get_tool_calls(self, message): ...

    def format_tool_call(self, tool_call) -> GenericToolCall: ...

    def format_tool_result(self, result_content, tool_call: GenericToolCall) -> str: ...


class OpenAICompatibleApiClient:
    client: OpenAI
    model_name: str

    def format_tools(self, tools: Iterable[Tool]) -> list[ChatCompletionToolParam]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools
        ]

    def completion(
        self, messages: list[ChatCompletionMessage], max_tokens: int, tools
    ) -> ChatCompletionMessage:
        logger.debug(messages)
        response = self.client.chat.completions.create(
            model=self.model_name, max_tokens=max_tokens, messages=messages, tools=tools
        )
        logger.debug(response)

        assert len(response.choices) == 1
        return response.choices[0].message

    def get_text(self, message: ChatCompletionMessage) -> str:
        return message.content

    def get_tool_calls(self, message: ChatCompletionMessage) -> bool:
        return message.tool_calls

    def format_tool_call(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> GenericToolCall:
        return GenericToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            args=json.loads(tool_call.function.arguments),
        )

    def format_tool_result(self, content, tool_call: GenericToolCall):
        tool_result_contents = [content.model_dump() for content in content]
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.name,
            "content": tool_result_contents,
        }


class GPTApiClient(OpenAICompatibleApiClient):
    def __init__(self, model_name: str) -> None:
        self.client = OpenAI()
        self.model_name = model_name


class GeminiCompatibleOpenAIApiClient(OpenAICompatibleApiClient):
    def __init__(self, model_name: str) -> None:
        self.client = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/",
        )
        self.model_name = model_name


class ClaudeApiClient:
    def __init__(self, model_name: str) -> None:
        self.client = Anthropic()
        self.model_name = model_name

    def format_tools(self, tools: Iterable[Tool]):
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ]

    def completion(self, messages, max_tokens, tools) -> list[ContentBlock]:
        logger.debug(messages)
        response = self.client.messages.create(
            model=self.model_name, max_tokens=max_tokens, messages=messages, tools=tools
        )
        logger.debug(response)
        return {"role": "assistant", "content": response.content}

    def get_text(self, message) -> str:
        content = message["content"]
        assert len(content) == 1
        return content[0].text

    def get_tool_calls(self, message) -> list[ToolUseBlock]:
        contents: list[ContentBlock] = message["content"]
        return [content for content in contents if content.type == "tool_use"]

    def format_tool_call(self, tool_call: ToolUseBlock) -> GenericToolCall:
        return GenericToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.input,
        )

    def format_tool_result(self, content, tool_call: GenericToolCall) -> str:
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_call.id, "content": content}
            ],
        }


def LlmWebApiClient(model_name: str) -> OpenAI:
    if model_name.startswith("gpt"):
        return GPTApiClient(model_name)
    elif model_name.startswith("gemini"):
        return GeminiCompatibleOpenAIApiClient(model_name)
    elif model_name.startswith("claude"):
        return ClaudeApiClient(model_name)

    raise ValueError(f"Unsupported model name: {model_name}")


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=get_args(SupportedModels))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    my_mcp_client_logger = logging.getLogger("my_mcp_client")
    my_mcp_client_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    my_mcp_client_handler = logging.FileHandler("mcp-client.log")
    my_mcp_client_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )
    )
    my_mcp_client_logger.addHandler(my_mcp_client_handler)

    client = LlmWebApiClient(args.model_name)
    asyncio.run(main(client))
