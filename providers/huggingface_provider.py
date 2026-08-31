"""
Hugging Face の Inference Providers（無料/低コストのAPIルーティング基盤）を
使った画像・動画生成プロバイダー。

- 利用には無料の Hugging Face アカウント + アクセストークンが必要
  （https://huggingface.co/settings/tokens で発行。
   「Make calls to Inference Providers」権限を有効にしてください）
- 新規アカウントには毎月無料クレジットが付与されるが、量は限られる
  （動画生成は特にコストが高いため、無料枠内で試せるのは数本程度が目安）
- 上限に達した場合や対応プロバイダーが混雑している場合はエラーになることがある
- 公式ドキュメント: https://huggingface.co/docs/inference-providers/
"""

import io
from typing import Optional

from huggingface_hub import InferenceClient

from .base import ImageProvider, VideoProvider

# text-to-image / text-to-video に対応した代表的な無料利用可能モデル。
# 利用可能なモデルは Hugging Face 側の対応状況により変化するため、
# UI からユーザーが自由に変更できるようにしている。
DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
IMAGE_MODEL_CHOICES = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
]

DEFAULT_VIDEO_MODEL = "Wan-AI/Wan2.2-TI2V-5B"
VIDEO_MODEL_CHOICES = [
    "Wan-AI/Wan2.2-TI2V-5B",
    "tencent/HunyuanVideo",
    "Lightricks/LTX-Video-0.9.8-13B-distilled",
]

DEFAULT_PROVIDER_ROUTE = "auto"  # HF に最適なプロバイダーを自動選択させる


class HuggingFaceImageProvider(ImageProvider):
    name = "Hugging Face（要トークン・高品質）"
    requires_key = True

    def __init__(self, api_key: str, provider: str = DEFAULT_PROVIDER_ROUTE):
        if not api_key:
            raise ValueError("Hugging Face のアクセストークンが設定されていません。")
        self.client = InferenceClient(api_key=api_key, provider=provider)

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        image = self.client.text_to_image(
            prompt,
            model=model or DEFAULT_IMAGE_MODEL,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            seed=seed,
        )
        # InferenceClient.text_to_image は PIL.Image を返す
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


class HuggingFaceVideoProvider(VideoProvider):
    name = "Hugging Face（無料クレジット・実験的）"
    requires_key = True

    def __init__(self, api_key: str, provider: str = "fal-ai"):
        if not api_key:
            raise ValueError("Hugging Face のアクセストークンが設定されていません。")
        self.client = InferenceClient(api_key=api_key, provider=provider)

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
        extra_kwargs = {}
        if num_frames:
            extra_kwargs["num_frames"] = num_frames
        if num_inference_steps:
            extra_kwargs["num_inference_steps"] = num_inference_steps
        if seed is not None:
            extra_kwargs["seed"] = seed
        if negative_prompt:
            extra_kwargs["negative_prompt"] = [negative_prompt]

        video_bytes = self.client.text_to_video(
            prompt,
            model=model or DEFAULT_VIDEO_MODEL,
            **extra_kwargs,
        )
        # InferenceClient.text_to_video はバイト列（MP4等）を返す
        return video_bytes
