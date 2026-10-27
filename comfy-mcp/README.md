# ComfyUI MCP server for Claude Code

Installed 2026-10-26. Gives Claude Code tools (`mcp__comfyui__*`) to drive the
Docker ComfyUI in `../comfy-ui` (127.0.0.1:8188).

- **Server:** `comfy-mcp` 0.10.0, Comfy-Org's official local MCP server
  (github.com/Comfy-Org/comfy-mcp, AGPL-3.0-or-later OR commercial)
- **Engine:** `comfy-cli` 1.23.0 (github.com/Comfy-Org/comfy-cli, GPL-3.0).
  comfy-mcp does all its I/O by calling this CLI.
- **Python env:** `.venv/` (system Python 3.12, nothing installed globally)

## How it is wired

Registered at **user scope** (every project), stored in `~/.claude.json`:

| Env var | Value | Why |
|---|---|---|
| `HOME` | `docker-stacks/comfy-mcp/home` | comfy-cli keeps config, job records and cache under `~/.config/comfy-cli`; this keeps them in this folder |
| `COMFY_BIN` | `.venv/bin/comfy` | MCP clients don't inherit your shell PATH |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | "Remote" mode: disables the gate that offers to kill whatever holds port 8188 (that is Docker's proxy here) and refuses host-side model downloads |
| `DO_NOT_TRACK` | `1` | comfy-cli analytics (Mixpanel/PostHog) forced off; they are opt-in anyway |
| `COMFY_NO_UPDATE_CHECK` | `1` | no daily PyPI version ping |

There is no comfy-cli *workspace* (no ComfyUI checkout on the host), on purpose.
Everything useful goes over ComfyUI's HTTP API: nodes, models, templates, validate,
run, jobs, upload, fetch outputs, system stats.

**Blocked tools** (`permissions.deny` in `~/.claude/settings.json`). They act on a
host-side ComfyUI install or spend money, so they are never right for this stack:
`launch_comfyui`, `stop_comfyui`, `restart_comfyui`, `update_comfyui`,
`switch_comfyui_version`, `install_node`, `download_model`, `auth_login`,
`partner_generate`, `emit_partner_workflow`.
Container lifecycle, node packs and models are handled through Docker and
`../comfy-ui` as before.

**Outputs:** ComfyUI writes to `../comfy-ui/basedir/output/`.
`fetch_outputs(prompt_id, out_dir)` copies them anywhere via `/view`.

## Smoke test

Runs the server over stdio exactly as Claude Code does. It lists tools and calls
server_info, search_models and validate_workflow, then generates one 1024² Z-Image
Turbo image and fetches it:

```bash
mkdir -p /tmp/comfy-mcp-smoke && cp smoke-test/zimage_smoke_api.json /tmp/comfy-mcp-smoke/
.venv/bin/python -I smoke-test/mcp_smoke.py /tmp/comfy-mcp-smoke
```

First run, 2026-10-26: 39 tools, valid, completed in 24.8 s including model load,
and the image was correct.

## Update

```bash
.venv/bin/pip install --only-binary=:all: -U comfy-mcp comfy-cli
```

Then re-run the smoke test. Read the changelogs first: comfy-cli is GPL tooling
that runs `pip`/`git` for some verbs.

## Uninstall

```bash
claude mcp remove comfyui -s user
# then delete the "deny" block from ~/.claude/settings.json
rm -rf /home/meroe/docker-stacks/comfy-mcp
```
