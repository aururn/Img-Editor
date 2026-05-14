# Img Editor

二次元キャラクター画像を対象とした **SDXL ベースの汎用画像生成・編集ローカル Web アプリ**です。
Illustrious / Pony / NoobAI 系チェックポイントと LoRA を組み合わせ、txt2img・img2img・手動 inpaint・
自動目検出 inpaint の 4 モードを Gradio UI から扱えます。


## 注意事項
- 本ツールは **二次元イラストの生成・編集** を目的としています。実在人物画像での使用は対象外です。
- ベースモデル / LoRA はユーザー側で用意してください(本リポジトリには重みは含まれません)。
- ローカル動作のみ。生成結果は端末ローカルにのみ保存されます。

## 必要環境
- Python 3.10 以上(**vast.ai 本番動作確認済み: Python 3.14.3 + PyTorch 2.11.0+cu130**)
- NVIDIA GPU(VRAM 16GB 以上推奨、本番ターゲットは **RTX 5090 (32GB)**。RTX 4090 (24GB) も可)
- CUDA 13.0 (cu130) 対応の PyTorch(`requirements.txt` では `torch`/`torchvision` を pin して
  いません。vast.ai では「PyTorch (Vast)」テンプレートに事前インストール済の torch を使うため、
  ローカルで動かす場合のみ CUDA に合った PyTorch を別途インストールしてから
  `pip install -r requirements.txt` してください)
- コンテナ容量(vast.ai): 24GB 以上(推奨 32GB)
- Python 3.13 以上では `audioop-lts` が自動で入ります(`pydub` 経由の Gradio 依存)

## セットアップ

```bash
# 1. リポジトリ取得後
cd ap

# 2. 仮想環境
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 3. PyTorch を先にインストール(CUDA バージョンに合わせる)
#   vast.ai 「PyTorch (Vast)」テンプレートを使う場合はこの手順は不要(プリインストール済)。
#   ローカル(自前環境)の場合のみ、CUDA に合わせて入れる。例(cu130):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 4. 残りの依存
pip install -r requirements.txt

# 5. モデル配置
#   models/checkpoints/  に SDXL 系の .safetensors を配置(必須)
#   models/loras/        に LoRA を配置(.safetensors)
#   models/vae/          に外部 VAE を配置(任意。HF リポジトリ ID でも可)
#   models/embeddings/   に Textual Inversion を配置(任意)
#   models/controlnet/   に ControlNet モデルを配置(任意)
#   models/ip_adapter/   に IP-Adapter モデルを配置(任意)
#   models/detection/    に YOLO 検出モデルを配置(自動目検出 inpaint 利用時のみ。
#                        face_yolov8n.pt が未配置なら起動時に Bingsu/adetailer から自動取得)

# 6. 設定確認
#   config.yaml の model.base_checkpoint を配置したファイル名に書き換える

# 7. 起動
python app.py
# -> http://127.0.0.1:7860
```

起動オプション:

```bash
python app.py --host 0.0.0.0 --port 7860 --share   # Gradio share link を出す
python app.py --config ./config.yaml               # 設定ファイルを差し替え
```

## 生成モード

| モード | 用途 |
|---|---|
| **Text to image** | プロンプトのみから新規生成 |
| **Image to image** | 入力画像全体を変換 |
| **Inpaint manual** | キャンバスにブラシでマスクを描いてインペイント |
| **Inpaint auto** | YOLOv8 顔検出で **目の領域を自動検出**してインペイント(本アプリ独自機能) |

## 機能一覧

| ID | 機能 |
|---|---|
| F-01 | 画像アップロード(PNG / JPEG / WebP, 最大 2048x2048) |
| F-02 | txt2img / img2img / 手動 inpaint / 自動目検出 inpaint の 4 モード |
| F-03 | プロンプト / ネガティブプロンプト |
| F-04 | LoRA 複数選択・強度調整・検索・トリガーワード自動挿入 |
| F-05 | LoRA / Embedding / ControlNet / IP-Adapter のアップロード対応 |
| F-06 | strength / CFG / steps / sampler(Euler a, DPM++ 2M Karras, DDIM) / seed / mask blur |
| F-07 | PNG メタデータ(プロンプト・パラメータ)付き保存 |
| F-08 | プリセット保存・読込 |
| F-09 | 履歴ギャラリー(直近 20 件、結果から img2img/inpaint へ送り返し可能) |
| F-10 | Textual Inversion Embeddings 対応 |
| F-11 | ControlNet 対応(Canny 前処理含む。txt2img/img2img/inpaint 全モード) |
| F-12 | IP-Adapter 対応 |
| F-13 | Regional Prompter(BREAK 構文、水平/垂直/グリッド/マスク、**領域別LoRA 対応**。段階インペイント方式) |
| F-14 | 起動時診断レポート(GPU / モデル / VRAM) |
| F-15 | 生成キャンセル、OOM 自動リトライ(0.75x 縮小フォールバック) |

## LoRA メタデータ

`models/loras/<name>.safetensors` の safetensors ヘッダーから推奨重み・トリガーワード・
対応ベースモデルを自動抽出します。同名の `<name>.json` を置くと情報を上書き設定できます。

```json
{
  "name": "empty_eyes",
  "file": "empty_eyes.safetensors",
  "base_model": "illustrious",
  "trigger_words": ["utsurome", "empty eyes"],
  "recommended_weight": 0.8,
  "min_weight": 0.4,
  "max_weight": 1.2
}
```

## 拡張アセット

[docs/extensions.md](docs/extensions.md) を参照してください。Embeddings / ControlNet /
IP-Adapter / Regional Prompter の詳細が記載されています。

## ディレクトリ

```
ap/
├── app.py
├── config.yaml
├── requirements.txt
├── src/
│   ├── ui/           # Gradio UI (main_view, handlers, presets, history)
│   ├── core/         # パイプライン管理・推論・LoRA・アセット・診断・Regional
│   ├── detection/    # 目検出・マスク生成(自動 inpaint 用)
│   └── utils/        # 画像 I/O・ロギング・メタデータ
├── models/
│   ├── checkpoints/  # SDXL ベースモデル
│   ├── loras/        # LoRA 群
│   ├── vae/          # 外部 VAE(任意)
│   ├── embeddings/   # Textual Inversion(任意)
│   ├── controlnet/   # ControlNet モデル(任意)
│   ├── ip_adapter/   # IP-Adapter モデル(任意)
│   └── detection/    # 顔/目検出モデル重み(自動 inpaint 利用時のみ)
├── presets/          # 保存プリセット
├── outputs/          # 生成結果(YYYYMMDD/)
├── logs/             # アプリログ
└── docs/             # 追加ドキュメント
```

## vast.ai での運用

リポジトリ直下の [process.txt](process.txt) に **vast.ai インスタンス作成 → コード転送 →
依存インストール → 起動 → SSH ポート転送 → ブラウザアクセス**までの完全手順がまとめて
あります(同梱の `Vast_EyeEditor_手順書.pdf` / `.docx` は古い版なので、`process.txt` を
優先してください)。

本番動作確認済み構成:
- テンプレート: **PyTorch (Vast)** (これ以外は使わない — torch を入れ直すと容量不足)
- GPU: **RTX 5090 (32GB)**
- Python: **3.14.3** (`/venv/main/bin/python3`)
- PyTorch: **2.11.0+cu130** (テンプレートにプリインストール済)
- 作業ディレクトリ: `/workspace/EyeEditor`

`requirements.txt` は CUDA イメージ側で torch を入れる前提のため `torch`/`torchvision`
を pin していません。Python 3.14 環境では `pip install -r requirements.txt` 後に
**`gradio<6` (5系)** を追加で入れる必要があります(`process.txt` Step 5③ 参照)。

## ライセンス・利用規約
- 本アプリケーションコード本体: MIT(LICENSE は別途必要に応じて配置)
- 取り扱う各モデル(ベース / LoRA)は配布元のライセンス・利用規約に従ってください
- 実在人物画像、権利侵害となる画像への使用は禁止します
