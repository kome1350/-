"""
プロバイダーの共通インターフェース定義。

新しい生成エンジン（有料APIも含む）を追加したい場合は、
ImageProvider または VideoProvider を継承したクラスを作り、
generate() を実装するだけで app.py 側から利用できます。
"""

from abc import ABC, abstractmethod
from typing import Optional


class ImageProvider(ABC):
    """画像生成プロバイダーの基底クラス。"""

    #: UIに表示する名前
    name: str = "unknown"
    #: APIキーが必須かどうか
    requires_key: bool = False

    @abstractmethod
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
        **kwargs,
    ) -> bytes:
        """
        プロンプトから画像を生成し、画像バイナリ（PNG/JPEG等）を返す。

        reference_image が渡された場合、対応しているプロバイダーは
        画像編集（image-to-image）として、その画像を元に生成する。
        対応していないプロバイダーは無視するか例外を送出してよい。

        実装側は失敗時に例外を送出してよい（呼び出し側でキャッチしてUIに表示する）。
        """
        raise NotImplementedError


class VideoProvider(ABC):
    """動画生成プロバイダーの基底クラス。"""

    name: str = "unknown"
    requires_key: bool = True

    @abstractmethod
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
        """
        プロンプトから動画を生成し、動画バイナリ（MP4等）を返す。
        """
        raise NotImplementedError
