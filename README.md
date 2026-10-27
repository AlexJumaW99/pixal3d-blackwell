# pixal3d-blackwell

[Pixal3D](https://github.com/TencentARC/Pixal3D) (TencentARC's image-to-3D model) running on a 16 GB
RTX 5070 Ti (Blackwell, sm_120), and the tools built around it: single- and multi-view generation that fits in
16 GB, a GLB exporter without nvdiffrast, background matting, camera fitting for photos whose cameras are
unknown, and checks that score a generated model against the photos it came from.

## Layout

| Path | What it is |
|---|---|
| `Dockerfile` | The `pixal3d:blackwell` image: Ubuntu 24.04, torch 2.11 + CUDA 13.0, the Pixal3D authors' prebuilt extension wheels for sm_120. No flash-attn: runs with `ATTN_BACKEND=sdpa`. |
| `src/` | Upstream TencentARC/Pixal3D as a git submodule, pinned to `f7cf384` ("feat: support multi-view image input"), unmodified. |
| `tools/` | Scripts that run inside the container (table below). |
| `blender/` | Headless Blender renders of a GLB: one view, a silhouette, a turntable. |
| `gltf/` | `pack.mjs`: a web-weight, meshopt-compressed GLB (Node, glTF-Transform). |
| `mv/` | Multi-view experiments through a local ComfyUI (Qwen-Image-Edit-2511). |
| `comfy-mcp/` | Notes and a smoke test for the ComfyUI MCP server used alongside. |
| `fetch-comfy.sh` | Downloads the ComfyUI models `mv/` uses. |

Not in the repo: model weights (`hf/`, about 46 GB), generated outputs (`out/`, `tut/`, `mv/*/`), logs and
virtual environments. `fetch-comfy.sh`, `mv/` and `comfy-mcp/` carry this machine's absolute paths; the
Pixal3D tools use container paths only.

## Build and fetch the weights

```bash
git clone --recurse-submodules https://github.com/AlexJumaW99/pixal3d-blackwell
cd pixal3d-blackwell
docker build -t pixal3d:blackwell .
mkdir -p hf out
# DINOv3 is gated: accept Meta's licence on its Hugging Face page, then pass a token
docker run --rm -e HF_TOKEN -v "$PWD/hf":/hf -v "$PWD/tools":/tools:ro pixal3d:blackwell python /tools/fetch.py
docker run --rm -e HF_TOKEN -v "$PWD/hf":/hf -v "$PWD/tools":/tools:ro pixal3d:blackwell python /tools/fetch_mv.py
```

## Run

Every tool expects the same mounts:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
  -v "$PWD/src":/work -v "$PWD/tools":/tools:ro -v "$PWD/out":/out -v "$PWD/hf":/hf \
  pixal3d:blackwell python /tools/run_single.py --image /out/in/photo_rgba.png --out /out/sv --seeds 7 --resolution 1024
```

- Inputs carry their own alpha. The pipeline's background remover (RMBG-2.0, CC BY-NC) is replaced by a stub
  that fails if it is ever called; `tools/birefnet_mask.py` (BiRefNet, MIT) makes masks.
- `/work` must be writable: Pixal3D writes its kernel autotune cache there.
- NAF is loaded with `torch.hub`, which asks GitHub for the repo's default branch even when its weights are
  cached, so runs need network; `HF_HUB_OFFLINE=1` keeps Hugging Face offline.
- Multi-view: `run_mv.py --views <dir> --out <dir> --seeds 7,42,1 --resolution 1024`, where `<dir>` holds
  `transforms.json` and one RGBA image per frame, frame 0 the canonical front view.

## Memory on a 16 GB card

- NAF upsampling runs in channel slices (`tools/naf_chunked.py`): the same result in a fraction of the memory.
- Two-view generation at 1024 peaked at 12.1 GiB (PyTorch's own count): 45–65 s per seed to generate,
  85–165 s with the GLB export. Single view at 1024: 88 s to generate, 205 s in all with model loading.
- Free the GPU before a run. Anything else holding 2–3 GB (a browser's GPU process, an idle ComfyUI) can push
  a multi-seed run out of memory partway through.

## Tools

| Tool | What it does |
|---|---|
| `run_single.py` | Single-view generation; inputs must carry alpha; DINOv3 from Meta's own repo; GLB baked by `uvbake.py`. |
| `run_mv.py` | Multi-view generation from a `transforms.json` input, same substitutions. |
| `uvbake.py` | GLB export without nvdiffrast (NVIDIA's licence is non-commercial): its own UV rasterizer in PyTorch. |
| `naf_chunked.py` | NAF feature upsampling in channel chunks, so the 1024 stages fit 16 GB. |
| `fetch.py`, `fetch_mv.py` | Download the single-view weights (with MoGe-2 and DINOv3) and the multi-view weights. |
| `birefnet_mask.py` | BiRefNet foreground masks on the CPU. |
| `bg_swap.py` | Puts a photo on another photo's backdrop: fits the old backdrop, solves the edge alpha against it, removes its spill, keeps every opaque pixel unchanged; also writes bust-only RGBA views. |
| `fit_view.py` | Two-view input for photos with unknown cameras: Pixal3D-style crops, the second camera from landmarks plus silhouette, `transforms.json`, and scoring of generations against both photos. |
| `frames.py` | A generation's frames found from its geometry (voxel occupancy, nearest-surface distance), not from bounding extents. |
| `render_check.py` | Clay and albedo renders of generations through each input camera, beside the photos. |
| `glb_render.py` | Renders of an exported GLB with its texture through the input cameras, and a turntable. |
| `glb_finish.py` | Drops loose fragments, measures the texture's colour against the photos (and, with `--grade`, removes a cast), re-exports with lossless textures and checks the round trip. |
| `mv_prep.py`, `mv_compare.py` | Multi-view input built from generated views of a single-view run, and silhouette comparisons. |
| `views.py`, `webmesh.py`, `to_raw.py`, `bake.py`, `fallback.py`, `debug_tex.py`, `align_check.py`, `check_naf.py` | Rasterizer and clay renders, web-weight meshes, texture baking from photos and views, and checks used along the way. |

## Two photos, unknown cameras

The workflow behind `bg_swap.py`, `fit_view.py` and the finishing tools, for two photos of one object:

1. `birefnet_mask.py`: masks (CPU, in the container).
2. `bg_swap.py`: refined alpha and bust-only RGBA views (host Python: numpy, SciPy, OpenCV, Pillow).
3. `fit_view.py crop`: square crops framed the way Pixal3D frames its input.
4. `run_single.py` on the front view: a mesh and MoGe-2's camera.
5. `fit_view.py landmarks`: the second camera from points read in both photos (`landmarks.json`) and the
   silhouette. The silhouette alone could not tell a camera 10° above a head from one 10° below it.
6. `fit_view.py build`, then `run_mv.py`.
7. `fit_view.py score` and `render_check.py` to choose a seed; `glb_finish.py` and `glb_render.py` for the GLB.

## Licences

The code in this repository is MIT (`LICENSE`). Some tools are adapted from Pixal3D (Tencent, MIT) and from
TRELLIS.2's `o_voxel` (Microsoft, MIT); `NOTICE` lists which, with their copyright notices. Pixal3D itself, in
`src/`, is MIT (see `src/LICENSE` and `src/NOTICE`). The models keep their own licences: DINOv3 (Meta, gated), MoGe-2, BiRefNet (MIT). RMBG-2.0 (CC BY-NC) is deliberately not used.
The Dockerfile installs NVIDIA's nvdiffrast wheel with the authors' other wheels; `run_single.py` and
`run_mv.py` bake the GLB with `uvbake.py` instead.
