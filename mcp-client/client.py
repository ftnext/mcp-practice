# https://modelcontextprotocol.io/quickstart/client
from collections.abc import Sequence
from contextlib import AsyncExitStack

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, stdio_client

load_dotenv()


class MCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        if not server_script_path.endswith(".py"):
            raise ValueError("Server script must be a .py file.")

        command = "python"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
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
        tools = response.tools
        print()
        print("Connection to server with tools:", [tool.name for tool in tools])

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main(argv: Sequence[str]) -> None:
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import asyncio
    import sys

    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    asyncio.run(main(sys.argv))
