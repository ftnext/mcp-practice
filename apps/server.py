# /// script
# requires-python = ">=3.15"
# dependencies = [
#     "mcp>=2.3.0",
# ]
# ///
from datetime import datetime
from pathlib import Path

from mcp.server.apps import Apps, client_supports_apps
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context

RESOURCE_URI = "ui://get-time/app.html"

apps = Apps()


@apps.tool(
    resource_uri=RESOURCE_URI, title="Get Time", description="Return the current time."
)
def get_time(ctx: Context) -> str:
    now = "2026-06-26T00:00:00Z"
    if not client_supports_apps(ctx):
        return f"The time is {now}."
    return str(datetime.now())


apps.add_html_resource(
    RESOURCE_URI, (Path(__file__).parent / "clock.html").read_text(), title="Clock"
)
server = MCPServer("apps-example", extensions=[apps])

if __name__ == "__main__":
    server.run("streamable-http", host="127.0.0.1", port=8000)
