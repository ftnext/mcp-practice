# Build server: https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search#build
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Literal

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, stdio_client
from openai import OpenAI

load_dotenv()

MAX_TOKENS = 1000

logger = logging.getLogger("my_mcp_client")

SupportedModels = Literal["gpt-4o-mini", "gemini-2.0-flash"]


class MCPClient:
    def __init__(self, openai: OpenAI, model_name: str):
        self.exit_stack = AsyncExitStack()
        self.openai = openai
        self.model_name = model_name
        self.sessions: dict[str, ClientSession] = {}
        self.available_tools = []
        self.tool2session = {}

    async def initialize(self):
        for session in self.sessions.values():
            await session.initialize()

    async def list_tools(self):
        for session in self.sessions.values():
            response = await session.list_tools()
            self.available_tools.extend(response.tools)
            for tool in response.tools:
                self.tool2session[tool.name] = session

    async def connect_to_server(self, servers):
        for name, parameters in servers.items():
            env = {
                env_name: os.getenv(env_name) for env_name in parameters.get("env", [])
            }
            if parameters["command"] == "docker":
                env["PATH"] = os.getenv("PATH")
            server_params = StdioServerParameters(
                command=parameters["command"],
                args=parameters["args"],
                env=env,
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.sessions[name] = await self.exit_stack.enter_async_context(
                ClientSession(*stdio_transport)
            )

        await self.initialize()

        await self.list_tools()
        logger.info(
            "Connection to server with tools: %s",
            [tool.name for tool in self.available_tools],
        )

    async def process_query(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
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
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content

        final_text = []
        messages.append(message)
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id

            tool_args = json.loads(tool_call.function.arguments)
            tool_result = await self.tool2session[tool_name].call_tool(
                tool_name, tool_args
            )
            logger.info(
                "[Calling tool %s with args %s, Got %s]",
                tool_name,
                tool_args,
                tool_result,
            )
            tool_result_contents = [
                content.model_dump() for content in tool_result.content
            ]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result_contents,
                }
            )

            logger.debug(messages)
            response = self.openai.chat.completions.create(
                model=self.model_name,
                max_tokens=MAX_TOKENS,
                messages=messages,
                tools=available_tools,
            )
            logger.debug(response)
            assert len(response.choices) == 1
            final_text.append(response.choices[0].message.content)

        return "\n".join(final_text)

    async def chat_loop(self) -> None:
        logger.info("MCP Client Started!")
        print("Type your queries or `quit` to exit.")

        while True:
            try:
                print()
                query = input("Query: ").strip()
                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print()
                print(response)
            except Exception as e:
                print()
                print(f"Error: {e!r}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main(openai_client: OpenAI, model_name: SupportedModels) -> None:
    with open("servers.json") as f:
        servers = json.load(f)
    client = MCPClient(openai_client, model_name)
    try:
        await client.connect_to_server(servers["mcpServers"])
        await client.chat_loop()
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
