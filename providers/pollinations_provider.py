"""
Pollinations.AI を使った画像生成プロバイダー。

- 登録・APIキー不要で無料利用できる（匿名利用は 15秒に1回程度のレート制限あり）
- エンドポイント: https://image.pollinations.ai/prompt/{prompt}
- 公式リポジトリ: https://github.com/pollinations/pollinations
- APIキーを取得すると（https://auth.pollinations.ai）レート制限が緩和される
"""

import urllib.parse
from typing import Optional

import requests

from .base import ImageProvider

DEFAULT_MODEL = "flux"
AVAILABLE_MODELS = ["flux", "turbo"]


class PollinationsImageProvider(ImageProvider):
    name = "Pollinations（無料・登録不要）"
    requires_key = False

    BASE_URL = "https://image.pollinations.ai/prompt/"

    def __init__(self, api_key: Optional[str] = None):
        # api_key は任意。指定するとレート制限が緩和される（無くても動作する）。
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        enhance: bool = True,
        **kwargs,
    ) -> bytes:
        encoded_prompt = urllib.parse.quote(prompt.strip())
        url = self.BASE_URL + encoded_prompt

        params = {
            "model": model or DEFAULT_MODEL,
            "width": width,
            "height": height,
            "nologo": "true",
        }
        if seed is not None:
            params["seed"] = seed
        if enhance:
            params["enhance"] = "true"
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        if self.api_key:
            params["key"] = self.api_key

        response = requests.get(url, params=params, timeout=180)
        if response.status_code == 429:
            raise RuntimeError(
                "Pollinations のレート制限に達しました。しばらく待ってから再度お試しください"
                "（匿名利用は約15秒に1回まで）。"
            )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            raise RuntimeError(
                f"画像以外のレスポンスが返されました（content-type: {content_type}）。"
                "プロンプトを変えて再度お試しください。"
            )
        return response.content
