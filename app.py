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
    DEFAULT_IMAGE_TO_IMAGE_MODEL,
    DEFAULT_VIDEO_MODEL,
    IMAGE_MODEL_CHOICES,
    IMAGE_TO_IMAGE_MODEL_CHOICES,
    VIDEO_MODEL_CHOICES,
    HuggingFaceImageProvider,
    HuggingFaceVideoProvider,
)
from providers.local_provider import (
    DEFAULT_LOCAL_IMAGE_MODEL,
    LOCAL_IMAGE_MODEL_CHOICES,
    LocalImageProvider,
    LocalVideoProvider,
    has_cuda_gpu,
    is_local_generation_available,
)
from providers.pollinations_provider import AVAILABLE_MODELS, PollinationsImageProvider
from utils.history import delete_generation, list_generations, save_generation

ENGINE_LOCAL = "ローカルGPU（無料・無制限・要セットアップ）"

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

    local_available = is_local_generation_available()
    local_gpu = has_cuda_gpu() if local_available else False
    if local_gpu:
        st.success("🖥️ ローカルGPU生成: 利用可能（無料・無制限）")
    elif local_available:
        st.warning("🖥️ ローカルGPU生成: ライブラリはあるがGPU(CUDA)が未検出です")
    else:
        st.caption(
            "🖥️ ローカルGPU生成は未セットアップです。README.md の"
            "「ローカルGPUで生成する」を参照してください。"
        )

    st.divider()
    st.markdown("**無料利用についての注意**")
    st.markdown(
        "- 🖼️ 画像生成（Pollinations）: 登録不要・完全無料\n"
        "  （匿名利用は約15秒に1回まで）\n"
        "- 🖼️ 画像生成（Hugging Face）: 無料トークン必須・高品質\n"
        "- 🎬 動画生成（Hugging Face）: 無料トークン必須。"
        "新規アカウントの無料クレジット範囲内でのみ利用可能で、"
        "生成に数分かかる・上限に達すると失敗することがあります。\n"
        "- 🖥️ ローカルGPU生成: 完全無料・無制限（電気代のみ）。"
        "事前セットアップが必要で、初回はモデルのダウンロードに時間がかかります。\n"
    )
    st.caption(
        "本アプリはプロバイダーを差し替えられる設計です。"
        "将来的に有料の高品質API（Runway、Stability AI等）を追加したい場合は、"
        "providers/ 以下に実装を追加するだけで拡張できます。"
    )

tab_chat, tab_image, tab_video, tab_history = st.tabs(
    ["💬 チャットで生成", "🖼️ 画像生成（詳細設定）", "🎬 動画生成", "🗂️ 生成履歴"]
)

# ---------------------------------------------------------------------------
# チャット形式で画像生成
# ---------------------------------------------------------------------------
CHAT_ENGINES = ["Pollinations（無料・登録不要）", "Hugging Face（要トークン・高品質）", ENGINE_LOCAL]

with tab_chat:
    st.caption("チャットのように話しかけると、その場で画像を生成します。")

    chat_engine = st.selectbox("生成エンジン", CHAT_ENGINES, key="chat_engine")

    chat_reference_file = st.file_uploader(
        "参考画像（任意・添付すると、その画像を元に生成します）",
        type=["png", "jpg", "jpeg", "webp"],
        key="chat_reference_file",
    )
    chat_reference_usable = chat_reference_file is not None and not chat_engine.startswith("Pollinations")
    if chat_reference_file is not None:
        if chat_engine.startswith("Pollinations"):
            st.warning(
                "Pollinationsエンジンは参考画像に対応していません。"
                "「Hugging Face」または「ローカルGPU」を選ぶと使用できます（このまま送ると通常の生成になります）。"
            )
        else:
            st.image(chat_reference_file, caption="この画像を参考に生成します", width=160)

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # これまでのやり取りを表示
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            if msg["type"] == "image":
                st.image(msg["content"])
            else:
                st.write(msg["content"])

    chat_prompt = st.chat_input("生成したい画像を説明してください（例: 夕焼けに照らされた富士山、油絵風）")

    if chat_prompt:
        st.session_state["chat_messages"].append({"role": "user", "type": "text", "content": chat_prompt})
        with st.chat_message("user"):
            st.write(chat_prompt)

        with st.chat_message("assistant"):
            if chat_engine.startswith("Hugging Face") and not hf_token:
                error_text = "Hugging Face トークンをサイドバーから入力してください。"
                st.error(error_text)
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "type": "text", "content": f"⚠️ {error_text}"}
                )
            else:
                spinner_text = (
                    "初回はモデルのダウンロードのため数分かかることがあります..."
                    if chat_engine == ENGINE_LOCAL
                    else "画像を生成しています..."
                )
                with st.spinner(spinner_text):
                    try:
                        chat_reference_bytes = chat_reference_file.getvalue() if chat_reference_usable else None

                        if chat_engine.startswith("Pollinations"):
                            provider = PollinationsImageProvider(api_key=DEFAULT_POLLINATIONS_KEY or None)
                            provider_name, model_name = "pollinations", "flux"
                        elif chat_engine == ENGINE_LOCAL:
                            provider = LocalImageProvider()
                            provider_name, model_name = "local", DEFAULT_LOCAL_IMAGE_MODEL
                        else:
                            provider = HuggingFaceImageProvider(api_key=hf_token)
                            provider_name = "huggingface"
                            model_name = DEFAULT_IMAGE_TO_IMAGE_MODEL if chat_reference_bytes else DEFAULT_IMAGE_MODEL

                        image_bytes = provider.generate(
                            chat_prompt, model=model_name, reference_image=chat_reference_bytes
                        )
                        record = save_generation(
                            "image", image_bytes, chat_prompt, provider_name, model_name, "png"
                        )
                        st.image(str(record.path))
                        st.session_state["chat_messages"].append(
                            {"role": "assistant", "type": "image", "content": str(record.path)}
                        )
                    except Exception as exc:  # noqa: BLE001
                        error_text = f"生成に失敗しました: {exc}"
                        st.error(error_text)
                        st.session_state["chat_messages"].append(
                            {"role": "assistant", "type": "text", "content": f"⚠️ {error_text}"}
                        )

    if st.session_state["chat_messages"]:
        if st.button("🗑️ チャット履歴をクリア", key="clear_chat"):
            st.session_state["chat_messages"] = []
            st.rerun()

# ---------------------------------------------------------------------------
# 画像生成タブ（詳細設定）
# ---------------------------------------------------------------------------
with tab_image:
    col_input, col_output = st.columns([1, 1])

    with col_input:
        image_engine = st.radio(
            "生成エンジン",
            ["Pollinations（無料・登録不要）", "Hugging Face（要トークン・高品質）", ENGINE_LOCAL],
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

        image_reference_file = st.file_uploader(
            "参考画像（任意・添付すると画像編集/image-to-imageになります）",
            type=["png", "jpg", "jpeg", "webp"],
            key="image_reference_file",
        )
        image_reference_usable = image_reference_file is not None and not image_engine.startswith("Pollinations")
        image_strength = 0.6
        if image_reference_file is not None:
            if image_engine.startswith("Pollinations"):
                st.warning(
                    "Pollinationsエンジンは参考画像に対応していません。"
                    "「Hugging Face」または「ローカルGPU」を選んでください。"
                )
            else:
                st.image(image_reference_file, caption="この画像を参考にします", width=160)
                if image_engine == ENGINE_LOCAL:
                    image_strength = st.slider(
                        "変化の強さ（低いほど元画像に近く、高いほど大きく変化）",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.6,
                        step=0.05,
                        key="image_strength",
                    )

        col_w, col_h = st.columns(2)
        with col_w:
            image_width = st.selectbox("幅", [512, 768, 1024, 1280], index=2, key="image_width")
        with col_h:
            image_height = st.selectbox("高さ", [512, 768, 1024, 1280], index=2, key="image_height")

        if image_engine.startswith("Pollinations"):
            image_model = st.selectbox("モデル", AVAILABLE_MODELS, index=0, key="image_model_pollinations")
        elif image_engine == ENGINE_LOCAL:
            image_model = st.selectbox(
                "モデル（ローカル実行）",
                LOCAL_IMAGE_MODEL_CHOICES,
                index=LOCAL_IMAGE_MODEL_CHOICES.index(DEFAULT_LOCAL_IMAGE_MODEL),
                key="image_model_local",
                help="sdxl-turbo は高速・8GB VRAM でも動作しやすいモデルです。",
            )
        else:
            hf_model_choices = IMAGE_TO_IMAGE_MODEL_CHOICES if image_reference_usable else IMAGE_MODEL_CHOICES
            hf_default_model = DEFAULT_IMAGE_TO_IMAGE_MODEL if image_reference_usable else DEFAULT_IMAGE_MODEL
            image_model = st.selectbox(
                "モデル（Hugging Face リポジトリ名）",
                hf_model_choices,
                index=hf_model_choices.index(hf_default_model),
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
                spinner_text = (
                    "初回はモデルのダウンロードのため数分かかることがあります..."
                    if image_engine == ENGINE_LOCAL
                    else "画像を生成しています...（数秒〜数十秒）"
                )
                with st.spinner(spinner_text):
                    try:
                        if image_engine.startswith("Pollinations"):
                            provider = PollinationsImageProvider(api_key=DEFAULT_POLLINATIONS_KEY or None)
                            provider_name = "pollinations"
                        elif image_engine == ENGINE_LOCAL:
                            provider = LocalImageProvider()
                            provider_name = "local"
                        else:
                            if not hf_token:
                                st.error("Hugging Face トークンをサイドバーから入力してください。")
                                st.stop()
                            provider = HuggingFaceImageProvider(api_key=hf_token)
                            provider_name = "huggingface"

                        image_reference_bytes = (
                            image_reference_file.getvalue() if image_reference_usable else None
                        )
                        image_bytes = provider.generate(
                            image_prompt,
                            model=image_model,
                            width=image_width,
                            height=image_height,
                            negative_prompt=image_negative_prompt or None,
                            reference_image=image_reference_bytes,
                            strength=image_strength,
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
    video_engine = st.radio(
        "生成エンジン",
        ["Hugging Face（要トークン・無料クレジット）", ENGINE_LOCAL],
        key="video_engine",
        horizontal=True,
    )

    if video_engine == ENGINE_LOCAL:
        st.info(
            "ローカルGPUで動画を生成します。完全無料・無制限ですが、"
            "初回はモデルのダウンロードに時間がかかり、クラウド版より画質・解像度は控えめです。",
            icon="🖥️",
        )
    else:
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
        if video_engine != ENGINE_LOCAL:
            video_model = st.selectbox(
                "モデル",
                VIDEO_MODEL_CHOICES,
                index=VIDEO_MODEL_CHOICES.index(DEFAULT_VIDEO_MODEL),
                key="video_model",
            )
        else:
            video_model = "local-animatediff"
            st.caption("モデル: AnimateDiff（ローカル実行・8GB VRAM想定）")

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
            elif video_engine != ENGINE_LOCAL and not hf_token:
                st.error("動画生成には Hugging Face トークンが必要です。サイドバーから入力してください。")
            else:
                spinner_text = (
                    "初回はモデルのダウンロードのため数分かかることがあります..."
                    if video_engine == ENGINE_LOCAL
                    else "動画を生成しています...（数十秒〜数分かかります）"
                )
                with st.spinner(spinner_text):
                    try:
                        if video_engine == ENGINE_LOCAL:
                            provider = LocalVideoProvider()
                            provider_name = "local"
                        else:
                            provider = HuggingFaceVideoProvider(api_key=hf_token)
                            provider_name = "huggingface"

                        video_bytes = provider.generate(
                            video_prompt,
                            model=video_model,
                            negative_prompt=video_negative_prompt or None,
                            num_frames=video_num_frames or None,
                            num_inference_steps=video_steps or None,
                            seed=video_seed or None,
                        )
                        record = save_generation(
                            "video", video_bytes, video_prompt, provider_name, video_model, "mp4"
                        )
                        st.session_state["last_video_file"] = str(record.path)
                        st.success("生成が完了しました！")
                    except Exception as exc:  # noqa: BLE001
                        if video_engine == ENGINE_LOCAL:
                            st.error(
                                f"生成に失敗しました: {exc}\n\n"
                                "GPUメモリ不足（VRAM不足）や、PyTorch/diffusersのセットアップ不備が"
                                "原因の可能性があります。README.md のセットアップ手順を確認してください。"
                            )
                        else:
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
