#!/bin/bash
# the multi-view models for ComfyUI (public Hugging Face repos), resumable
set -u
get() { curl -sSL --fail --retry 5 -C - -o "$2" "$1" && echo "OK $(basename $2) $(stat -c %s $2)" || echo "FAIL $2"; }
get https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/qwen-image-edit-2511-multiple-angles-lora.safetensors /home/meroe/docker-stacks/comfy-ui/basedir/models/loras/qwen-image-edit-2511-multiple-angles-lora.safetensors
get https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors /home/meroe/docker-stacks/comfy-ui/basedir/models/loras/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors
get https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors /home/meroe/docker-stacks/comfy-ui/basedir/models/vae/qwen_image_vae.safetensors
get https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors /home/meroe/docker-stacks/comfy-ui/basedir/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
get https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors /home/meroe/docker-stacks/comfy-ui/basedir/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors
echo DONE
