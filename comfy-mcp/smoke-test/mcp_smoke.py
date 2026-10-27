"""Drive comfy-mcp over stdio exactly as Claude Code will, and print each step."""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = "/home/meroe/docker-stacks/comfy-mcp"
HERE = sys.argv[1]
ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": f"{ROOT}/home",
    "COMFY_BIN": f"{ROOT}/.venv/bin/comfy",
    "COMFYUI_URL": "http://127.0.0.1:8188",
    "DO_NOT_TRACK": "1",
    "COMFY_NO_UPDATE_CHECK": "1",
}


def show(label, result, limit=900):
    text = "".join(getattr(c, "text", f"<{c.type}>") for c in result.content)
    flag = " [isError]" if result.is_error else ""
    print(f"=== {label}{flag} ===\n{text[:limit]}\n", flush=True)
    return text


async def main():
    params = StdioServerParameters(command=f"{ROOT}/.venv/bin/comfy-mcp", env=ENV)
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        init = await s.initialize()
        print("server:", init.server_info.name, init.server_info.version)
        tools = (await s.list_tools()).tools
        print(f"{len(tools)} tools:", " ".join(t.name for t in tools), "\n")
        show("server_info", await s.call_tool("server_info", {}), 1500)
        show("search_models z_image", await s.call_tool("search_models", {"query": "z_image"}))
        show("validate_workflow", await s.call_tool("validate_workflow", {"workflow_path": f"{HERE}/zimage_smoke_api.json"}))
        run = show("run_workflow", await s.call_tool(
            "run_workflow",
            {"workflow_path": f"{HERE}/zimage_smoke_api.json", "wait": True, "timeout_seconds": 240},
            read_timeout_seconds=300,
        ), 1500)
        pid = None
        try:
            pid = json.loads(run).get("prompt_id")
        except Exception:
            pass
        if pid:
            show("fetch_outputs", await s.call_tool("fetch_outputs", {"prompt_id": pid, "out_dir": f"{HERE}/out"}))


asyncio.run(main())
