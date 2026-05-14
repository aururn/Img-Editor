# Img Editor — 説明書

## 目次

1. [概要](#概要)
2. [動作環境](#動作環境)
3. [起動方法](#起動方法)
4. [画面構成](#画面構成)
5. [生成モード](#生成モード)
6. [各パネルの使い方](#各パネルの使い方)
7. [モデル・アセットの配置](#モデルアセットの配置)
8. [config.yaml 設定リファレンス](#configyaml-設定リファレンス)
9. [出力ファイル](#出力ファイル)
10. [実装の概要](#実装の概要)
11. [ソースコード構成](#ソースコード構成)
12. [トラブルシュート](#トラブルシュート)

---

## 概要

**Img Editor** は、SDXL（Stable Diffusion XL）ベースのアニメ・イラスト特化型ローカル画像生成・編集ツールです。  
Gradio を使ったブラウザUIで動作し、テキストからの画像生成・画像変換・インペイント・瞳の自動補正など4つの生成モードを備えています。

### 主な特徴

| 機能 | 内容 |
|------|------|
| **4つの生成モード** | txt2img / img2img / 手動インペイント / 瞳自動検出インペイント |
| **LoRA 多重適用** | 複数LoRAを重みつきで同時使用。トリガーワード自動挿入 |
| **Regional Prompter** | 画面を複数エリアに分割し、エリアごとに別プロンプトを適用 |
| **ControlNet** | Canny エッジ抽出またはカスタム画像による構図制御 |
| **IP-Adapter** | 参照画像からスタイル・外見を誘導 |
| **Textual Inversion** | 埋め込みトークンによるスタイル固定 |
| **プリセット** | 生成設定をJSON保存・読み込み |
| **A1111互換メタデータ** | 生成画像のPNGにプロンプト・設定を自動書き込み |

---

## 動作環境

### 推奨環境（本番）

| 項目 | 値 |
|------|----|
| GPU | RTX 5090（32GB）または RTX 4090（24GB） |
| Python | 3.12 〜 3.14 |
| PyTorch | 2.x + CUDA 12.x |
| OS | Linux (vast.ai) / Windows 11 |
| コンテナ容量 | 32GB 以上推奨 |

### 最低ライン

- VRAM 16GB 以上（SDXL 1024px 生成に必要）
- SSD 空き容量 20GB 以上（モデルファイル含む）

### 主要 Python パッケージ

| パッケージ | 用途 |
|-----------|------|
| `diffusers` ≥0.31 | SDXL パイプライン |
| `transformers` ≥4.46 | CLIPテキストエンコーダ（L/G） |
| `torch` | GPU推論 |
| `peft` | LoRAローダー |
| `safetensors` | モデル読み込み |
| `gradio` <6 | Web UI |
| `Pillow` | 画像処理 |
| `opencv-python` | Cannyエッジ検出 |
| `ultralytics` | YOLOv8 瞳検出 |

---

## 起動方法

### 基本

```bash
cd /workspace/ImgEditor
python app.py
```

### オプション

```bash
python app.py --host 0.0.0.0 --port 7860 --share
```

| オプション | 説明 |
|-----------|------|
| `--host` | バインドアドレス（既定: config.yaml の値） |
| `--port` | ポート番号（既定: 7860） |
| `--share` | Gradio の公開リンクを発行 |
| `--config` | config.yaml のパス（既定: `./config.yaml`） |

### ブラウザで開く

- **ローカル**: `http://localhost:7860`
- **SSH転送 (Windows)**: 別ターミナルで以下を実行し、`http://localhost:7860` を開く

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519_vast" -p ポート番号 -N -L 7860:127.0.0.1:7860 root@IPアドレス
```

> `127.0.0.1:7860` ではなく `localhost:7860` を使うこと（Windows OpenSSHのIPv6バインド問題回避）

---

## 画面構成

```
┌──────────────────────────────────────────────────────┐
│ Topbar  Img Editor   SDXL · LoRA · Regional   [model]│
├──────────────────────────────────────────────────────┤
│ System Status ▼ (折りたたみ)                          │
├──────────────────────────────────────────────────────┤
│ [ Text to image ][ Image to image ][ Inpaint manual ][ Inpaint auto ] │
├──────────────────┬───────────────────┬───────────────┤
│ Left Rail        │ Center Panel      │ Right Panel   │
│                  │                   │               │
│ Checkpoint       │ Canvas            │ Result        │
│ LoRA             │ Prompt            │ History       │
│                  │ Negative Prompt ▼ │ Presets ▼     │
│                  │ Regional ▼        │               │
│                  │ Basic Settings    │               │
│                  │ [設定サマリー]    │               │
│                  │ [Generate][Cancel]│               │
├──────────────────┴───────────────────┴───────────────┤
│ Advanced Settings ▼                                   │
│ Strength / Mask blur / Textual Inversion              │
│ ControlNet / IP-Adapter                               │
└──────────────────────────────────────────────────────┘
```

### 左サイドバー

- **Checkpoint**: 使用するSDXLベースモデルを選択。「Reload checkpoints」で再読み込み
- **LoRA**: 複数選択可。テキスト検索・「Insert triggers」でトリガーワードをプロンプトに挿入。.safetensorsをドラッグでアップロード可能

### 中央パネル

- **Canvas（キャンバス）**: 画像アップロード + ブラシで手動マスク描画
- **Prompt**: メイン入力欄（大きく配置）
- **Negative Prompt** ▼: 折りたたみ式。不要な要素を記述
- **Regional Prompter** ▼: エリア分割生成の設定
- **Basic Settings**: CFG / Steps / Width / Height / Sampler / Seed
- **設定サマリー**: 現在の主要設定をリアルタイム表示（`CFG 7.0 · Steps 28 · 1024×1024 · DPM++ 2M Karras`）
- **Generate / Cancel**: 生成開始・中断

### 右パネル

- **Result**: 生成結果の表示。「→ img2img」「→ inpaint」で再編集へ送れる
- **History** ▼: 直近の生成画像をサムネイルグリッド表示
- **Presets** ▼: 設定の保存・読み込み

### Advanced Settings（画面下部）

- **Strength / Mask blur**: img2img・インペイントの強度、マスクぼかし
- **Textual Inversion**: 埋め込みトークン選択・アップロード
- **ControlNet**: 制御ネットワーク設定（モデル・画像・スケール・Start/End）
- **IP-Adapter**: 参照画像によるスタイル誘導設定

---

## 生成モード

### 1. Text to Image

テキストのみから新しい画像を生成します。

- 入力: プロンプト + Negative prompt
- 出力: Width × Height の新規画像
- `Strength` パラメータは無効（テキストのみ）
- 典型的な用途: キャラクターデザイン、コンセプトアート、背景生成

### 2. Image to Image

既存画像をプロンプトで変換します。

- 入力: ソース画像 + プロンプト
- `Strength`（0.0〜1.0）で元画像からの乖離度を制御
  - 0.0 = 元画像そのまま
  - 1.0 = 完全にプロンプト主導
  - 推奨値: 0.55
- 典型的な用途: スタイル変換、構図の微調整、反復的リファイン

### 3. Inpaint Manual（手動インペイント）

キャンバスにブラシで塗った領域のみを再生成します。

- 入力: ソース画像 + ブラシで描いたマスク + プロンプト
- マスク外の部分は保護され変更されない
- `Mask blur` でマスクエッジをぼかして馴染ませる
- 典型的な用途: 部分的な修正、オブジェクト置き換え、細部の調整

### 4. Inpaint Auto（瞳自動インペイント）

YOLOv8 顔検出器が瞳の位置を自動検出し、その領域のみをインペイントします。

- 入力: ソース画像（マスク不要）+ プロンプト
- 瞳が検出できない場合は手動インペイントにフォールバック
- 検出パラメータは `config.yaml` の `detection` セクションで調整
- **Img Editor の核心機能**: アニメキャラの瞳補正・瞳孔スタイル変更に特化
- 典型的な用途: 瞳の色変更、虹彩スタイル変更、瞳のリタッチ

---

## 各パネルの使い方

### Prompt の書き方

```
masterpiece, best quality, 1girl, solo, looking at viewer, empty eyes
```

- カンマ区切りのタグ形式（Danbooru タグ）
- LoRAのトリガーワードは「Insert triggers」で自動挿入
- Negative Prompt に `lowres, bad anatomy, blurry, (worst quality:1.4)` などを記述

### LoRA の使い方

1. 左サイドバーの **LoRA** パネルでチェックボックスを選択（複数可）
2. 「Insert triggers」でトリガーワードをPromptに自動挿入
3. 検索ボックスで名前・トリガーワードを絞り込み可能
4. ファイルを直接アップロードする場合は「Upload LoRA」欄にドラッグ

### Regional Prompter の使い方

複数キャラクターや異なるスタイルをエリア分割して生成する機能です。

1. **Enable regional prompt** をチェック
2. **Layout** でエリア分割方法を選択
   - `horizontal`: 横に分割（左キャラ・右キャラなど）
   - `vertical`: 縦に分割
   - `grid`: グリッド配置
   - `mask`: キャンバスで描いたマスクで分割
3. **Ratios** で各エリアの比率を指定（例: `1,2,1` で左:中:右 = 1:2:1）
4. **Base ratio**: base prompt と region prompt の混合比。空欄なら `0.2`
5. **Overlay ratio**: 隣接領域を少し重ねて境界の急な切り替わりを抑える比率
6. **Use base/common**: 互換用フラグ。Region prompts の先頭要素は消費しません
7. **Common prompt**: 全エリアに共通で付与するタグ。Region prompts 内で指定する場合は `ADDCOMM` / `ADDBASE` を使います
8. **Region prompts**: エリアごとのプロンプト（`BREAK` / `ADDROW` / `ADDCOL` / 改行で区切る）
9. **Region LoRAs**: エリアごとの LoRA（`<lora:left:0.8> BREAK <lora:right:0.8>` など）
10. **LoRA negative TE / U-Net**: 対象外エリアに他エリアのLoRAをどれだけ残すか（通常は `0`）
11. **LoRA stop step**: 指定step以降、エリア専用LoRAを止めて侵食を抑える
12. **Preview masks**: 分割/色分けマスクを生成前に確認

> **Regional 構文**: `BREAK` / `ADDROW` / `ADDCOL` / `ADDBASE` / `ADDCOMM` に対応。
> LoRAタグは `name:weight` または `name:text_weight:unet_weight` に対応します。
> `mask` layout ではキャンバスの赤・緑・青などの色ごとに別エリアとして扱います。

### ControlNet の使い方

1. **Advanced Settings** → **ControlNet** を展開
2. **Model**: 使用するControlNetモデルを選択（`models/controlnet/` 内）
3. **Control image**: 制御用画像をアップロード（なければキャンバス画像を使用）
4. **Preprocess**: `none`（画像をそのまま使用）または `canny`（エッジ抽出）
5. **Scale**: ControlNetの影響強度（0.0〜2.0、推奨: 1.0）
6. **Start / End**: 生成ステップのどの区間でControlNetを適用するか（0.0〜1.0）

### プリセットの使い方

- **保存**: 「Presets」アコーディオン → 名前を入力 → Save
- **読み込み**: ドロップダウンでプリセットを選択 → Load
- 保存内容: モード・チェックポイント・LoRA・プロンプト・全パラメータ

---

## モデル・アセットの配置

```
models/
├── checkpoints/    ← SDXLベースモデル (.safetensors)
├── loras/          ← LoRAファイル (.safetensors)
├── embeddings/     ← Textual Inversion (.safetensors / .pt / .bin)
├── controlnet/     ← ControlNetモデル (.safetensors / .bin)
├── ip_adapter/     ← IP-Adapterモデル (.safetensors / .bin)
├── vae/            ← VAEモデル（省略時はHFから自動DL）
└── detection/      ← 瞳検出モデル (.pt、省略時は自動DL)
```

### チェックポイント（必須）

SDXLベースの `.safetensors` を `models/checkpoints/` に配置し、`config.yaml` の `model.base_checkpoint` にファイル名を記載します。

```bash
# aria2c での高速ダウンロード例
aria2c -x 16 -s 16 -o illustriousXL_v01.safetensors \
  "https://civitai.com/api/download/models/MODEL_ID?...&token=${CIVITAI_TOKEN}"
```

推奨モデル系統: **Illustrious XL** / **NoobAI XL** / **Pony Diffusion XL**

### LoRA

- `models/loras/` に `.safetensors` を配置するだけで自動認識
- UIの「Upload LoRA」からドラッグアップロードも可能（上限: config.yaml の `max_file_size_mb`）
- ファイル名・メタデータからトリガーワードを自動抽出
- カスタムメタデータは同名の `.json` ファイルで上書き可能

### ControlNet・IP-Adapter

512MBを超えるファイルはUIアップロードではなく直接配置を推奨:

```bash
# ControlNet
mv your_controlnet.safetensors models/controlnet/

# IP-Adapter
mv your_ip_adapter.safetensors models/ip_adapter/
```

---

## config.yaml 設定リファレンス

```yaml
app:
  host: "0.0.0.0"        # バインドアドレス（Vast.ai では 0.0.0.0）
  port: 7860
  log_level: "INFO"

model:
  base_checkpoint: "illustriousXL_v01.safetensors"  # ファイル名（必須）
  vae: "madebyollin/sdxl-vae-fp16-fix"              # fp16 NaN 問題対策
  default_sampler: "DPM++ 2M Karras"
  precision: "fp16"          # fp16 / bf16 / fp32
  enable_vae_tiling: true    # VRAM節約（高解像度時に有効）
  enable_xformers: false     # メモリ効率化アテンション
  cpu_offload: false         # モデルをCPUにオフロード（低VRAM環境）

detection:
  eye_model: "face_yolov8n.pt"    # 瞳検出モデル（自動DL）
  confidence_threshold: 0.35     # 検出信頼度の下限
  padding_px: 8                   # 検出領域の余白（px）
  feather_px: 12                  # マスクフェザリング（px）

defaults:
  inpaint:
    mask_blur: 8
    strength: 0.55
    cfg_scale: 7.0
    steps: 28

controlnet:
  preprocess: "none"   # none / canny
  scale: 1.0

limits:
  max_file_size_mb: 512    # UIアップロードの上限
  max_image_size: 2048     # 画像の最大辺（px）

history:
  max_items: 20    # 履歴に保持する枚数

paths:
  outputs_dir: "./outputs"
  presets_dir: "./presets"
  logs_dir: "./logs"
```

---

## 出力ファイル

### 保存先

```
outputs/
└── YYYYMMDD/
    ├── HHmmSS_fff_seedXXXX.png
    └── ...
```

- `YYYYMMDD`: 生成日付
- `HHmmSS`: 時分秒
- `fff`: ミリ秒
- `seedXXXX`: 使用したシード値

同名ファイルが存在する場合は自動リネームされます。

### PNGメタデータ（A1111互換）

生成画像の PNG テキストチャンクに以下が自動埋め込みされます：

```
masterpiece, best quality, 1girl
Negative prompt: lowres, bad anatomy
Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345,
Size: 1024x1024, Model: illustriousXL_v01,
Lora hashes: "my_lora: abc123"
```

この形式は A1111（Automatic1111 Web UI）と互換性があり、他ツールでもメタデータを読み取れます。

---

## 実装の概要

### アーキテクチャ

```
app.py
  └── build_ui()          ← Gradio UI定義（src/ui/main_view.py）
       └── handlers.py    ← UIイベントハンドラ
            └── InferenceService.run()   ← 推論エンジン
                 ├── PipelineManager     ← diffusers パイプライン管理
                 ├── LoRALoader          ← LoRA 読み込み
                 ├── EyeDetector         ← YOLOv8 瞳検出
                 └── RegionalPipeline    ← エリア分割生成
```

### 生成フロー（generate_v2）

```
1. バリデーション
   └── プロンプト空チェック・モード別入力チェック

2. 画像前処理
   └── RGB変換・max_image_size でリサイズ

3. マスク生成（インペイント時）
   ├── Manual: キャンバスの描画レイヤーからアルファ抽出
   └── Auto: YOLOv8 検出 → 楕円マスク生成 → Gaussian blur フェザリング

4. GenerationRequest 構築
   └── モード・モデル・LoRA・プロンプト等をデータクラスにパッケージ

5. InferenceService.run()
   ├── Standard パイプライン（通常生成）
   │   ├── txt2img → StableDiffusionXLPipeline
   │   ├── img2img → StableDiffusionXLImg2ImgPipeline
   │   └── inpaint → StableDiffusionXLInpaintPipeline
   └── Regional パイプライン（エリア分割）
       ├── ベースパス（共通LoRA）
       └── エリアごとのLoRAスワップ + UNetブレンド

6. 出力処理
   └── PNG保存・A1111メタデータ埋め込み・履歴追加
```

### Regional Prompter

Regional Prompter の核心実装です。UNet のdenoise中に、領域ごとの positive / negative prompt 条件をマスクでブレンドします。左パネルの LoRA は全領域に共通適用され、Region LoRAs や `<lora:name:weight>` タグで指定した LoRA は領域ごとに切り替えられます。

```
base prompt / negative を通常どおりエンコード
  ↓
各 region prompt / region negative をエンコード
  ↓
UNet forward 内で latent mask ごとに noise prediction をブレンド
  ↓
txt2img / img2img / inpaint の各パイプラインが通常どおり画像化
```

### 長プロンプト対応（>77トークン）

CLIPの標準トークン上限（77トークン）を超えるプロンプトを処理するため、独自のチャンク分割エンコーディングを実装しています。

- プロンプトを75トークンずつのチャンクに分割
- CLIP-L（768次元）と CLIP-G（1280次元）の2エンコーダそれぞれでチャンクごとに hidden state を計算
- チャンクの hidden state を結合して最終的な埋め込みベクトルを生成
- 300トークンを超える長いプロンプトでも切り捨てなく処理可能

### CUDA OOM フォールバック

生成時に VRAM 不足（CudaOutOfMemoryError）が発生した場合、自動的に解像度を 0.75 倍に縮小して再試行します。

### VAE タイリング

高解像度（1536px 以上）の生成時に `enable_vae_tiling: true` を設定すると、VAEのデコード処理をタイル分割することでVRAM使用量を大幅に削減できます。

---

## ソースコード構成

```
src/
├── core/
│   ├── __init__.py
│   ├── config.py           AppConfig — config.yaml のパース
│   ├── pipeline_manager.py PipelineManager — diffusersパイプライン管理・LoRA適用
│   ├── inference.py        InferenceService — 推論エンジン本体
│   ├── lora_loader.py      LoRALoader — LoRAメタデータ抽出・管理
│   ├── assets.py           アセット探索（checkpoint・embedding・controlnet等）
│   ├── regional.py         RegionalPromptSpec — エリア分割プロンプト処理
│   └── diagnostics.py      起動時の依存関係・アセットチェック
├── ui/
│   ├── __init__.py
│   ├── main_view.py        build_ui() — Gradio Blocks レイアウト定義
│   ├── handlers.py         UIイベントハンドラ・HandlerContext
│   ├── presets.py          プリセット保存・読み込み
│   └── history.py          生成履歴管理（deque）
├── detection/
│   ├── __init__.py
│   ├── eye_detector.py     EyeDetector — YOLOv8顔検出・瞳位置推定
│   └── mask_builder.py     MaskBuilder — 検出結果からマスク画像生成
└── utils/
    ├── __init__.py
    ├── image_io.py         画像読み込み・保存・リサイズ
    ├── metadata.py         A1111互換PNGメタデータ生成
    └── logger.py           ロギング設定
```

### 各モジュールの責務

#### `src/core/pipeline_manager.py`

diffusers の3種類のパイプライン（txt2img・img2img・inpaint）を管理します。チェックポイントの切り替え、LoRAの動的ロード/アンロード、サンプラーの設定変更などを担当します。

#### `src/core/inference.py`

実際の推論処理を行う中核モジュールです。`GenerationRequest` データクラスで全パラメータを受け取り、`GenerationResult` を返します。OOM フォールバック、長プロンプトエンコーディング、Regional パイプラインのオーケストレーションなどが含まれます。

#### `src/core/lora_loader.py`

LoRAファイルの `.safetensors` メタデータを解析し、トリガーワード・推奨ウェイト・LyCORISバリアント種別などを自動抽出します。同名の `.json` ファイルでメタデータを上書きすることも可能です。

#### `src/detection/eye_detector.py`

ultralytics の YOLOv8 モデルを使用して顔を検出し、検出された顔領域から左右の瞳位置を推定します。モデルが未配置の場合は Hugging Face から自動ダウンロードします。

#### `src/detection/mask_builder.py`

検出した瞳座標から楕円形のマスク画像を生成します。`padding_px` で検出領域を拡張し、`feather_px` の Gaussian blur でエッジをぼかします。Gradio の ImageEditor コンポーネントからのユーザー描画マスクの抽出も担当します。

---

## トラブルシュート

### 起動しない

| 症状 | 対処 |
|------|------|
| `TypeError: Blocks.launch() got unexpected keyword argument 'theme'` | `gradio<6` にアップグレード |
| `ModuleNotFoundError: No module named 'audioop'` | `pip install audioop-lts`（Python 3.13+） |
| `OSError: Cannot find empty port in range: 7860-7860` | 前回のプロセスが残留。`pkill -f "python app.py"` で終了 |
| `TypeError: argument of type 'bool' is not a container` | gradio 4.x のバグ。`pip install --upgrade 'gradio<6'` |

### 生成できない

| 症状 | 対処 |
|------|------|
| `CUDA out of memory` | 解像度を下げる・LoRA枚数を減らす・`enable_vae_tiling: true` に設定 |
| 生成画像が紫色の横縞 | `config.yaml` の `vae` が `madebyollin/sdxl-vae-fp16-fix` になっているか確認 |
| チェックポイントが読み込まれない | `config.yaml` の `base_checkpoint` のファイル名が `models/checkpoints/` 内の実ファイル名と一致しているか確認 |
| LoRAが効かない | トリガーワードがプロンプトに含まれているか確認。「Insert triggers」ボタンを使用 |

### UIに関する問題

| 症状 | 対処 |
|------|------|
| `127.0.0.1:7860` が `ERR_CONNECTION_REFUSED` | `localhost:7860` を使う（Windows OpenSSH の IPv6バインド問題） |
| ファイルアップロードが弾かれる | `config.yaml` の `limits.max_file_size_mb` を増やす。大きいファイルは `scp`/`aria2c` で直接配置 |
| `Failed to create tunnel`（Vast Tunnels） | そのホストがTunnels非対応。SSHポート転送（`-N -L`）を使う |

### モデルダウンロード

Civitai からのダウンロードには API トークンが必要です：

```bash
read -s CIVITAI_TOKEN
aria2c -x 16 -s 16 -o model.safetensors \
  "https://civitai.com/api/download/models/MODEL_ID?type=Model&format=SafeTensor&token=${CIVITAI_TOKEN}"
unset CIVITAI_TOKEN && history -c
```

- URLに `&token=${CIVITAI_TOKEN}` を忘れると 401 エラー
- ダウンロード失敗後は `rm -f model.safetensors` で 0バイトファイルを削除してから再実行
