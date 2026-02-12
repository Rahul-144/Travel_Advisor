import asyncio
import os
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

serpapi_key = os.getenv("SERP_API_KEY")


async def mcp_session():
    async with streamable_http_client(
        f"https://mcp.serpapi.com/{serpapi_key}/mcp"
    ) as transport:
        read = transport[0]
        write = transport[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

async def call_mcp_tool(tool_name: str, arguments: dict):
    async with mcp_session() as session:
        result = await session.call_tool(tool_name, arguments)
        return result.content

async def list_mcp_tools():
    async with mcp_session() as session:
        tools = await session.list_tools()
        return tools.tools