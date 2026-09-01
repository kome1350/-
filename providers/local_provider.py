"""
自分のPCのGPUを使って画像・動画を生成するローカルプロバイダー。

- 完全無料・無制限（電気代のみ）で、クラウドのクレジット上限を気にせず使えます
- 事前に GPU 版 PyTorch（CUDA対応）のインストールが必要です。詳しくは README.md の
  「🖥️ ローカルGPUで生成する」を参照してください
- 初回実行時にモデルファイル（数GB）が自動ダウンロードされ、~/.cache/huggingface/ に保存されます
  （2回目以降はキャッシュされるので高速です）
- 8GB VRAM 程度のGPUでも動くように、比較的軽量なモデルをデフォルトにしています
"""

import io
import os
import tempfile
from typing import Optional

import streamlit as st

from .base import ImageProvider, VideoProvider

# --- 画像 -------------------------------------------------------------
# SDXL-Turbo: 1〜4ステップで高速に生成でき、8GB VRAM でも動作しやすい
DEFAULT_LOCAL_IMAGE_MODEL = "stabilityai/sdxl-turbo"
LOCAL_IMAGE_MODEL_CHOICES = [
    "stabilityai/sdxl-turbo",
    "runwayml/stable-diffusion-v1-5",
]

# --- 動画 -------------------------------------------------------------
# AnimateDiff: SD1.5ベースの軽量な動画生成手法。8GB VRAM でも動作しやすい
DEFAULT_LOCAL_VIDEO_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch がインストールされていません。README.md の「ローカルGPUで生成する」の"
            "手順に従って、CUDA対応版PyTorchをインストールしてください。"
        ) from exc
    return torch


def is_local_generation_available() -> bool:
    """ローカル生成に必要なライブラリが揃っているか（簡易チェック）。"""
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError:
        return False
    return True


def has_cuda_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@st.cache_resource(show_spinner="画像生成モデルを読み込んでいます（初回はダウンロードのため数分かかります）...")
def _load_image_pipeline(model_id: str):
    torch = _require_torch()
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe, device


@st.cache_resource(show_spinner="画像編集モデルを読み込んでいます（初回はダウンロードのため数分かかります）...")
def _load_image_to_image_pipeline(model_id: str):
    torch = _require_torch()
    from diffusers import AutoPipelineForImage2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe, device


@st.cache_resource(show_spinner="動画生成モデルを読み込んでいます（初回はダウンロードのため数分かかります）...")
def _load_video_pipeline(base_model_id: str, motion_adapter_id: str):
    torch = _require_torch()
    from diffusers import AnimateDiffPipeline, EulerAncestralDiscreteScheduler, MotionAdapter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=dtype)
    pipe = AnimateDiffPipeline.from_pretrained(base_model_id, motion_adapter=adapter, torch_dtype=dtype)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    pipe = pipe.to(device)
    if device == "cuda":
        # 8GB前後のVRAMでも動きやすくするための省メモリ設定
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
    return pipe, device


class LocalImageProvider(ImageProvider):
    name = "ローカルGPU（無料・無制限・要セットアップ）"
    requires_key = False

    def __init__(self, model_id: str = DEFAULT_LOCAL_IMAGE_MODEL):
        self.model_id = model_id

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        reference_image: Optional[bytes] = None,
        strength: float = 0.6,
        **kwargs,
    ) -> bytes:
        torch = _require_torch()
        model_id = model or self.model_id

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

        # SDXL-Turbo はガイダンス無し・少ステップでの生成が推奨されている
        is_turbo = "turbo" in model_id.lower()

        if reference_image is not None:
            from PIL import Image

            pipe, device = _load_image_to_image_pipeline(model_id)
            init_image = Image.open(io.BytesIO(reference_image)).convert("RGB")
            init_image = init_image.resize((width, height))

            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=init_image,
                strength=strength,
                num_inference_steps=2 if is_turbo else 30,
                guidance_scale=0.0 if is_turbo else 7.0,
                generator=generator,
            )
        else:
            pipe, device = _load_image_pipeline(model_id)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=1 if is_turbo else 30,
                guidance_scale=0.0 if is_turbo else 7.0,
                generator=generator,
            )

        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


class LocalVideoProvider(VideoProvider):
    name = "ローカルGPU（無料・無制限・要セットアップ）"
    requires_key = False

    def __init__(
        self,
        base_model_id: str = DEFAULT_LOCAL_VIDEO_BASE_MODEL,
        motion_adapter_id: str = DEFAULT_MOTION_ADAPTER,
    ):
        self.base_model_id = base_model_id
        self.motion_adapter_id = motion_adapter_id

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        torch = _require_torch()
        from diffusers.utils import export_to_video

        pipe, device = _load_video_pipeline(self.base_model_id, self.motion_adapter_id)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_frames=num_frames or 16,
            num_inference_steps=num_inference_steps or 25,
            guidance_scale=7.5,
            generator=generator,
        )
        frames = result.frames[0]

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name
            export_to_video(frames, tmp_path, fps=8)
            with open(tmp_path, "rb") as f:
                video_bytes = f.read()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return video_bytes
