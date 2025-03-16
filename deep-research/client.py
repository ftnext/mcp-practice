# Build server: https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search#build
import json
import logging
import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, Tool, stdio_client
from mcp.client.stdio import get_default_environment
from openai import OpenAI

load_dotenv()

MAX_TOKENS = 1000

logger = logging.getLogger("my_mcp_client")

SupportedModels = Literal["gpt-4o-mini", "gemini-2.0-flash"]
ServerName = str
ToolName = str


class MCPServerParameter(TypedDict):
    command: str
    args: list[str]
    env: NotRequired[list[str]]


class MCPClient:
    def __init__(self, openai: OpenAI, model_name: str):
        self.exit_stack = AsyncExitStack()
        self.openai = openai
        self.model_name = model_name
        self.available_tools: list[Tool] = []
        self.tool2session: dict[ToolName, ClientSession] = {}
        self.messages = []

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
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in self.available_tools
        ]

        logger.debug(messages)
        response = self.openai.chat.completions.create(
            model=self.model_name,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=available_tools,
        )
        logger.debug(response)

        assert len(response.choices) == 1
        return response.choices[0].message

    async def call_tool(self, tool_call):
        tool_name = tool_call.function.name
        tool_call_id = tool_call.id

        tool_args = json.loads(tool_call.function.arguments)
        tool_result = await self.tool2session[tool_name].call_tool(tool_name, tool_args)
        logger.info(
            "[Calling tool %s with args %s, Got %s]", tool_name, tool_args, tool_result
        )
        tool_result_contents = [content.model_dump() for content in tool_result.content]
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": tool_result_contents,
        }

    async def process_query(self, query: str) -> str:
        self.messages.append({"role": "user", "content": query})
        message = self.completion(self.messages)

        self.messages.append(message)
        while message.tool_calls:
            for tool_call in message.tool_calls:
                tool_message = await self.call_tool(tool_call)
                self.messages.append(tool_message)

                message = self.completion(self.messages)
                self.messages.append(message)

        return message.content

    async def cleanup(self):
        await self.exit_stack.aclose()


class ChatSession:
    def __init__(self, client: MCPClient) -> None:
        self.client = client

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

                response = await self.client.process_query(query)
                self.print_empty_line()
                print(response)
            except Exception as e:
                logger.exception("Error: %r", e)
                self.print_empty_line()
                print("エラーが発生しました。終了します")
                break


async def main(openai_client: OpenAI, model_name: SupportedModels) -> None:
    with open("servers.json") as f:
        servers = json.load(f)
    client = MCPClient(openai_client, model_name)
    try:
        await client.connect_to_server(servers["mcpServers"])
        chat_session = ChatSession(client)
        await chat_session.start()
    finally:
        await client.cleanup()


def OpenAIClient(model_name: SupportedModels) -> OpenAI:
    if model_name == "gpt-4o-mini":
        return OpenAI()
    return OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/",
    )


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=["gpt-4o-mini", "gemini-2.0-flash"])
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

    client = OpenAIClient(args.model_name)
    asyncio.run(main(client, args.model_name))
