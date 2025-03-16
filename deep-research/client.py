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

logger = logging.getLogger(__name__)

SupportedModels = Literal["gpt-4o-mini", "gemini-2.0-flash"]


class MCPClient:
    def __init__(self, openai: OpenAI, model_name: str):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.openai = openai
        self.model_name = model_name

    async def connect_to_server(self):
        server_params = StdioServerParameters(
            command="docker",
            args=["run", "-i", "--rm", "-e", "BRAVE_API_KEY", "mcp/brave-search"],
            env={
                "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY"),
                "PATH": os.getenv("PATH"),
            },
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        response = await self.session.list_tools()
        self.available_tools = response.tools
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

        response = self.openai.chat.completions.create(
            model=self.model_name,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=available_tools,
        )

        message = response.choices[0].message
        if not message.tool_calls:
            return message.content

        final_text = []
        messages.append(message)
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id

            tool_args = json.loads(tool_call.function.arguments)
            tool_result = await self.session.call_tool(tool_name, tool_args)
            tool_result_contents = [
                content.model_dump() for content in tool_result.content
            ]
            logger.info("[Calling tool %s with args %s]", tool_name, tool_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result_contents,
                }
            )

            response = self.openai.chat.completions.create(
                model=self.model_name,
                max_tokens=MAX_TOKENS,
                messages=messages,
                tools=available_tools,
            )
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
    client = MCPClient(openai_client, model_name)
    try:
        await client.connect_to_server()
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
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        level=logging.INFO,
        filename="mcp-client.log",
    )

    client = OpenAIClient(args.model_name)
    asyncio.run(main(client, args.model_name))
