"""
AI画像・動画生成スタジオ

無料で使えるAIエンジンを使って、画像・動画を生成するStreamlitアプリ。

起動方法:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from config import DEFAULT_HF_TOKEN, DEFAULT_POLLINATIONS_KEY
from providers.huggingface_provider import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_VIDEO_MODEL,
    IMAGE_MODEL_CHOICES,
    VIDEO_MODEL_CHOICES,
    HuggingFaceImageProvider,
    HuggingFaceVideoProvider,
)
from providers.pollinations_provider import AVAILABLE_MODELS, PollinationsImageProvider
from utils.history import delete_generation, list_generations, save_generation

st.set_page_config(page_title="AI画像・動画生成スタジオ", page_icon="🎨", layout="wide")

st.title("🎨 AI画像・動画生成スタジオ")
st.caption("無料で使えるAIエンジンを組み合わせて、画像・動画を生成します。")

# ---------------------------------------------------------------------------
# サイドバー：APIキー設定
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 設定")

    hf_token = st.text_input(
        "Hugging Face アクセストークン（任意）",
        value=st.session_state.get("hf_token", DEFAULT_HF_TOKEN),
        type="password",
        help=(
            "huggingface.co で無料アカウントを作成し、"
            "Settings > Access Tokens から「Make calls to Inference Providers」"
            "権限付きのトークンを発行してください。"
        ),
    )
    st.session_state["hf_token"] = hf_token

    if hf_token:
        st.success("Hugging Face トークン設定済み")
    else:
        st.info("トークン未設定：画像生成（Pollinations）のみ利用できます")

    st.divider()
    st.markdown("**無料利用についての注意**")
    st.markdown(
        "- 🖼️ 画像生成（Pollinations）: 登録不要・完全無料\n"
        "  （匿名利用は約15秒に1回まで）\n"
        "- 🖼️ 画像生成（Hugging Face）: 無料トークン必須・高品質\n"
        "- 🎬 動画生成（Hugging Face）: 無料トークン必須。"
        "新規アカウントの無料クレジット範囲内でのみ利用可能で、"
        "生成に数分かかる・上限に達すると失敗することがあります。\n"
    )
    st.caption(
        "本アプリはプロバイダーを差し替えられる設計です。"
        "将来的に有料の高品質API（Runway、Stability AI等）を追加したい場合は、"
        "providers/ 以下に実装を追加するだけで拡張できます。"
    )

tab_image, tab_video, tab_history = st.tabs(["🖼️ 画像生成", "🎬 動画生成", "🗂️ 生成履歴"])

# ---------------------------------------------------------------------------
# 画像生成タブ
# ---------------------------------------------------------------------------
with tab_image:
    col_input, col_output = st.columns([1, 1])

    with col_input:
        image_engine = st.radio(
            "生成エンジン",
            ["Pollinations（無料・登録不要）", "Hugging Face（要トークン・高品質）"],
            key="image_engine",
        )
        image_prompt = st.text_area(
            "プロンプト（生成したい画像の説明）",
            height=100,
            placeholder="例: 夕焼けに照らされた富士山、油絵風、高解像度",
            key="image_prompt",
        )
        image_negative_prompt = st.text_input(
            "ネガティブプロンプト（除外したい要素・任意）",
            key="image_negative_prompt",
        )

        col_w, col_h = st.columns(2)
        with col_w:
            image_width = st.selectbox("幅", [512, 768, 1024, 1280], index=2, key="image_width")
        with col_h:
            image_height = st.selectbox("高さ", [512, 768, 1024, 1280], index=2, key="image_height")

        if image_engine.startswith("Pollinations"):
            image_model = st.selectbox("モデル", AVAILABLE_MODELS, index=0, key="image_model_pollinations")
        else:
            image_model = st.selectbox(
                "モデル（Hugging Face リポジトリ名）",
                IMAGE_MODEL_CHOICES,
                index=IMAGE_MODEL_CHOICES.index(DEFAULT_IMAGE_MODEL),
                key="image_model_hf",
            )
            image_model = st.text_input(
                "モデル名を直接入力する場合はこちら（上の選択より優先）",
                value=image_model,
                key="image_model_hf_custom",
            )

        image_generate_clicked = st.button("🎨 画像を生成", type="primary", use_container_width=True)

    with col_output:
        if image_generate_clicked:
            if not image_prompt.strip():
                st.warning("プロンプトを入力してください。")
            else:
                with st.spinner("画像を生成しています...（数秒〜数十秒）"):
                    try:
                        if image_engine.startswith("Pollinations"):
                            provider = PollinationsImageProvider(api_key=DEFAULT_POLLINATIONS_KEY or None)
                            provider_name = "pollinations"
                        else:
                            if not hf_token:
                                st.error("Hugging Face トークンをサイドバーから入力してください。")
                                st.stop()
                            provider = HuggingFaceImageProvider(api_key=hf_token)
                            provider_name = "huggingface"

                        image_bytes = provider.generate(
                            image_prompt,
                            model=image_model,
                            width=image_width,
                            height=image_height,
                            negative_prompt=image_negative_prompt or None,
                        )
                        record = save_generation(
                            "image", image_bytes, image_prompt, provider_name, image_model, "png"
                        )
                        st.session_state["last_image_file"] = str(record.path)
                        st.success("生成が完了しました！")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"生成に失敗しました: {exc}")

        last_image_file = st.session_state.get("last_image_file")
        if last_image_file and Path(last_image_file).exists():
            img_path = Path(last_image_file)
            st.image(str(img_path), use_container_width=True)
            st.download_button(
                "⬇️ 画像をダウンロード",
                data=img_path.read_bytes(),
                file_name=img_path.name,
                mime="image/png",
                use_container_width=True,
            )
        else:
            st.info("プロンプトを入力して「画像を生成」を押すと、ここに結果が表示されます。")

# ---------------------------------------------------------------------------
# 動画生成タブ
# ---------------------------------------------------------------------------
with tab_video:
    st.info(
        "動画生成は Hugging Face の無料クレジットを使用します。"
        "生成に数十秒〜数分かかることがあり、無料枠を使い切るとエラーになる場合があります。",
        icon="⏳",
    )

    col_input, col_output = st.columns([1, 1])

    with col_input:
        video_prompt = st.text_area(
            "プロンプト（生成したい動画の説明）",
            height=100,
            placeholder="例: 夜の街を歩く猫、シネマティック、スローモーション",
            key="video_prompt",
        )
        video_negative_prompt = st.text_input(
            "ネガティブプロンプト（除外したい要素・任意）",
            key="video_negative_prompt",
        )
        video_model = st.selectbox(
            "モデル",
            VIDEO_MODEL_CHOICES,
            index=VIDEO_MODEL_CHOICES.index(DEFAULT_VIDEO_MODEL),
            key="video_model",
        )

        with st.expander("詳細設定（任意）"):
            video_num_frames = st.number_input(
                "フレーム数（0=モデルのデフォルト）", min_value=0, max_value=200, value=0, step=1
            )
            video_steps = st.number_input(
                "推論ステップ数（0=モデルのデフォルト）", min_value=0, max_value=100, value=0, step=1
            )
            video_seed = st.number_input("シード値（0=ランダム）", min_value=0, value=0, step=1)

        video_generate_clicked = st.button("🎬 動画を生成", type="primary", use_container_width=True)

    with col_output:
        if video_generate_clicked:
            if not video_prompt.strip():
                st.warning("プロンプトを入力してください。")
            elif not hf_token:
                st.error("動画生成には Hugging Face トークンが必要です。サイドバーから入力してください。")
            else:
                with st.spinner("動画を生成しています...（数十秒〜数分かかります）"):
                    try:
                        provider = HuggingFaceVideoProvider(api_key=hf_token)
                        video_bytes = provider.generate(
                            video_prompt,
                            model=video_model,
                            negative_prompt=video_negative_prompt or None,
                            num_frames=video_num_frames or None,
                            num_inference_steps=video_steps or None,
                            seed=video_seed or None,
                        )
                        record = save_generation(
                            "video", video_bytes, video_prompt, "huggingface", video_model, "mp4"
                        )
                        st.session_state["last_video_file"] = str(record.path)
                        st.success("生成が完了しました！")
                    except Exception as exc:  # noqa: BLE001
                        st.error(
                            f"生成に失敗しました: {exc}\n\n"
                            "無料クレジットの上限や、モデルの混雑が原因の場合があります。"
                            "別のモデルを試すか、時間を置いて再度お試しください。"
                        )

        last_video_file = st.session_state.get("last_video_file")
        if last_video_file and Path(last_video_file).exists():
            vid_path = Path(last_video_file)
            st.video(str(vid_path))
            st.download_button(
                "⬇️ 動画をダウンロード",
                data=vid_path.read_bytes(),
                file_name=vid_path.name,
                mime="video/mp4",
                use_container_width=True,
            )
        else:
            st.info("プロンプトを入力して「動画を生成」を押すと、ここに結果が表示されます。")

# ---------------------------------------------------------------------------
# 生成履歴タブ
# ---------------------------------------------------------------------------
with tab_history:
    filter_kind = st.radio(
        "表示するもの", ["すべて", "画像のみ", "動画のみ"], horizontal=True, key="history_filter"
    )
    kind_map = {"すべて": None, "画像のみ": "image", "動画のみ": "video"}
    records = list_generations(kind=kind_map[filter_kind])

    if not records:
        st.info("まだ生成履歴がありません。画像・動画を生成すると、ここに表示されます。")
    else:
        cols = st.columns(3)
        for idx, record in enumerate(records):
            with cols[idx % 3]:
                st.caption(f"🕒 {record.timestamp}｜{record.provider}｜{record.model}")
                if record.kind == "image":
                    st.image(str(record.path), use_container_width=True)
                else:
                    st.video(str(record.path))
                st.caption(record.prompt[:80] + ("..." if len(record.prompt) > 80 else ""))
                dl_col, del_col = st.columns(2)
                with dl_col:
                    st.download_button(
                        "⬇️",
                        data=record.path.read_bytes(),
                        file_name=record.file,
                        key=f"dl_{record.file}",
                        use_container_width=True,
                    )
                with del_col:
                    if st.button("🗑️", key=f"del_{record.file}", use_container_width=True):
                        delete_generation(record)
                        st.rerun()
                st.divider()
