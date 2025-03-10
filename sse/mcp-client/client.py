from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    async with sse_client("http://localhost:8000/sse") as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(tools)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
