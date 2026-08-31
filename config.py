"""
APIキー等の設定を読み込む。

優先順位:
  1. Streamlit Community Cloud の Secrets (st.secrets) ※クラウドにデプロイした場合
  2. ローカルの .env ファイル（開発時）
  3. 未設定（空文字）→ アプリのサイドバーから都度入力すればOK

.env や Secrets が無くてもアプリは動作する
（画像生成は Pollinations が無料利用でき、他のキーはUI上から都度入力できるため）。
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _get_setting(key: str) -> str:
    # Streamlit Community Cloud にデプロイした場合、st.secrets にキーがあれば優先して使う
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        # ローカル実行等、secrets.toml が無い環境では st.secrets へのアクセス自体が例外になる
        pass
    return os.getenv(key, "")


# Hugging Face の無料アクセストークン（画像高品質生成・動画生成に使用）
DEFAULT_HF_TOKEN = _get_setting("HF_TOKEN")

# Pollinations の登録キー（任意。無くても画像生成は可能。あるとレート制限が緩和される）
DEFAULT_POLLINATIONS_KEY = _get_setting("POLLINATIONS_API_KEY")
