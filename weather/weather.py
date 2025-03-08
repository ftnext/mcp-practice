# https://modelcontextprotocol.io/quickstart/server
from typing import Any
from urllib.parse import urljoin

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

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


@mcp.tool()
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


def format_forecast(period: dict[str, Any]) -> str:
    return f"""\
{period["name"]}:
Temperature: {period["temperature"]}°{period["temperatureUnit"]}
Wind: {period["windSpeed"]} {period["windDirection"]}
Forecast: {period["detailedForecast"]}\
"""


@mcp.tool()
async def get_forecasts(latitude: float, longitude: float) -> str:
    """Get weather forecasts for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    points_url = urljoin(NWS_API_BASE, f"/points/{latitude},{longitude}")
    try:
        points_data = await make_nws_request(points_url)
    except Exception:
        return "Unable to fetch forecast data for this location."

    forecast_url = points_data["properties"]["forecast"]
    try:
        forecast_data = await make_nws_request(forecast_url)
    except Exception:
        return "Unable to fetch detailed forecast."

    periods = forecast_data["properties"]["periods"]
    forecasts = [format_forecast(period) for period in periods[:5]]
    return "\n---\n".join(forecasts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
