# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio[mcp]",
#     "httpx",
# ]
# ///

# https://modelcontextprotocol.io/quickstart/server
from typing import Any
from urllib.parse import urljoin

import gradio as gr
import httpx

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url: str) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()


def format_alert(feature: dict[str, Any]) -> str:
    props = feature["properties"]
    return f"""\
Event: {props.get("event", "Unknown")}
Area: {props.get("areaDesc", "Unknown")}
Severity: {props.get("severity", "Unknown")}
Description: {props.get("description", "No description available")}
Instructions: {props.get("instructions", "No instructions available")}\
"""


async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = urljoin(NWS_API_BASE, f"/alerts/active/area/{state}")
    try:
        data = await make_nws_request(url)
    except Exception:
        return "Unable to fetch alerts."

    if "features" not in data:
        return "No alerts found."
    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


demo = gr.Interface(
    fn=get_alerts,
    inputs=["textbox"],
    outputs="text",
    title="Weather Alerts",
    description="Get weather alerts for a US state using the National Weather Service API.",
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
