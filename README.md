# 🎨 AI画像・動画生成スタジオ

Python + Streamlit で作った、無料で使えるAIエンジンによる画像・動画生成アプリです。

## 特徴

- 🖼️ **画像生成**: [Pollinations.AI](https://pollinations.ai/) を使えば **登録・APIキー不要で完全無料**
- 🖼️ **画像生成（高品質・任意）**: [Hugging Face](https://huggingface.co/) の無料トークンを使うと FLUX.1 等の高品質モデルも選べます
- 🎬 **動画生成**: Hugging Face の Inference Providers 経由（無料トークン必須）
- 🗂️ 生成履歴をギャラリー表示・ダウンロード・削除
- 🔌 プロバイダーを差し替え可能な設計（`providers/` に実装を足すだけで有料APIにも拡張できます）

## ⚠️ 「無料」について正直に

- **画像生成（Pollinations）は本当に無料・無制限**です（匿名利用は約15秒に1回のレート制限のみ）。まずはこれだけで十分遊べます。
- **動画生成に、恒久的に無料で高品質・無制限なAPIは現時点（2026年）で存在しません。** 本アプリは Hugging Face の Inference Providers を使い、新規アカウントに付与される**無料クレジットの範囲内**で動画生成を試せるようにしています。動画は生成コストが高いため、無料枠で試せるのは目安として月に数本程度です。上限に達すると失敗するので、その場合はエラーメッセージを確認してください。
- より高品質・大量に動画生成したい場合は、Runway・Stability AI・Replicate 等の有料APIを使う必要があります。`providers/` にプロバイダークラスを1つ追加するだけで、このアプリに組み込めるように設計してあります。

## セットアップ

```bash
# 1. 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 2. 依存パッケージをインストール
pip install -r requirements.txt

# 3. （任意）Hugging Face トークンを設定
cp .env.example .env
# .env を開いて HF_TOKEN=xxxx を記入
# 画像生成（Pollinations）だけ使うなら、この手順は省略してもOK

# 4. アプリを起動
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

### Hugging Face トークンの取得方法（無料）

1. https://huggingface.co/ で無料アカウントを作成
2. https://huggingface.co/settings/tokens を開き、「Create new token」
3. 権限で **「Make calls to Inference Providers」** を有効にして発行
4. `.env` の `HF_TOKEN` に貼り付け、またはアプリのサイドバーに直接入力

## 📱 スマホから・外出先からアクセスする（無料デプロイ）

自分のPCを起動しっぱなしにしなくても、どこからでもスマホでアクセスできるようにするには、
**Streamlit Community Cloud**（無料）にデプロイするのが一番簡単です。

### 手順

1. **GitHubにコードを置く**
   - GitHubアカウントを作成（無料）
   - 新しいリポジトリを作成し、この `ai_gen_studio` フォルダの中身をすべてプッシュする
     （`.env` は `.gitignore` で除外されるので、誤ってトークンをアップロードする心配はありません）
   - 非公開（Private）リポジトリでもデプロイ可能です

2. **Streamlit Community Cloud にデプロイ**
   - https://share.streamlit.io を開き、GitHubアカウントでログイン
   - 「Create app」→ 対象のリポジトリ・ブランチ・`app.py` を選択
   - 「Advanced settings」で **Secrets** を設定（`.env` の中身と同じ形式で、下記のように貼り付け）
     ```toml
     HF_TOKEN = "あなたのHugging Faceトークン"
     POLLINATIONS_API_KEY = ""
     ```
   - 「Deploy」を押すと数分でビルドされ、`https://xxxxx.streamlit.app` のような
     **公開URL** が発行されます

3. **スマホでアクセス**
   - 発行されたURLをスマホのブラウザで開くだけです（Wi-Fi・モバイル回線どちらでもOK）
   - ホーム画面に追加すればアプリのように使えます

### ⚠️ 知っておいてほしい制約

- 一定時間アクセスが無いとアプリは自動的にスリープします。再度開くと数十秒〜数分の
  起動待ちが発生します。
- Community Cloud上の `outputs/` フォルダは**永続的な保存先ではありません**。
  再デプロイやスリープからの復帰時に生成履歴が消えることがあります。
  長期保存したい生成物はその都度ダウンロードしてください。
- 誰でもURLを知っていればアクセスできる状態になります。他人に使われたくない場合は
  公開範囲や認証の追加を検討してください（Community Cloudには簡易的なアクセス制限機能もあります）。

### GitHubを使わずすぐ試したい場合（一時的な公開URL）

自分のPCを起動したまま、[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) の
「Quick Tunnel」を使うと、登録不要で一時的な公開URLを発行できます。

```powershell
# cloudflared をインストール後
cloudflared tunnel --url http://localhost:8501
```

表示された `https://xxxx.trycloudflare.com` のようなURLをスマホで開けます。
ただし、このURLはPCを起動している間・コマンドを実行している間だけ有効な一時的なものです。

## ディレクトリ構成

```
ai_gen_studio/
├── app.py                       # Streamlit UI 本体
├── config.py                    # .env からの設定読み込み
├── providers/
│   ├── base.py                  # ImageProvider / VideoProvider の共通インターフェース
│   ├── pollinations_provider.py # 無料・登録不要の画像生成
│   └── huggingface_provider.py  # Hugging Face経由の画像・動画生成
├── utils/
│   └── history.py                # 生成履歴の保存・一覧・削除
├── outputs/                     # 生成された画像・動画とメタデータの保存先
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 有料APIを追加したい場合

`providers/base.py` の `ImageProvider` / `VideoProvider` を継承したクラスを作り、
`generate()` を実装するだけで、`app.py` 側のエンジン選択に追加できます。
例えば OpenAI (DALL·E / Sora)、Stability AI、Runway 等を後から足す際も同じ手順です。

## 既知の制約

- Pollinations の匿名利用にはレート制限があります（約15秒に1回）。頻繁に使う場合は
  https://auth.pollinations.ai でAPIキーを取得し `.env` の `POLLINATIONS_API_KEY` に設定してください。
- Hugging Face 経由の動画生成は実験的機能であり、モデルの可用性・応答速度・無料クレジットの
  上限はHugging Face側の状況により変動します。
- 生成物は `outputs/` フォルダにローカル保存されます（クラウドには送信されません。ただし生成処理
  自体は各プロバイダーのサーバー上で行われます）。
