# Stable Diffusion Extension Assets

本アプリは SDXL のベースモデル/LoRA に加え、以下のオプション拡張アセットを扱えます。
ディレクトリに配置するだけで UI 側に自動列挙されます。

## 対応アセット

| ディレクトリ | 用途 | 対応拡張子 |
|---|---|---|
| `models/embeddings/` | Textual Inversion embeddings | `.safetensors`, `.pt`, `.bin` |
| `models/controlnet/` | SDXL ControlNet 重み | フォルダ または 単一ファイル |
| `models/ip_adapter/` | IP-Adapter 重み | フォルダ または 単一ファイル |
| `models/vae/` | 外部 VAE(任意。Hugging Face リポジトリ ID でも可) | `.safetensors` |
| `models/detection/` | 顔/目検出モデル(自動 inpaint 利用時のみ) | `.pt` |

## LoRA

LoRA はテーブル UI で複数選択でき、ファイルごとに重み・トリガーワード・警告メッセージを
表示します。`<name>.json` の同梱メタデータで詳細を上書き可能です(README 参照)。

UI 上の「Insert triggers」ボタンで、選択中の LoRA のトリガーワードを現在のプロンプトに
自動挿入できます。

## Textual Inversion

埋め込みファイルをアップロードまたは `models/embeddings/` に配置すると CheckboxGroup に
列挙されます。「Insert embedding tokens」で対応トークンをプロンプトに挿入します。

## ControlNet

txt2img / img2img / inpaint のすべてのモードに対応します。

- **Model**: ドロップダウンから ControlNet を選択(空の場合は無効)
- **Control image**: 制御用画像をアップロード(txt2img では必須)
- **Preprocess**: `none` または `canny`
- **Scale / Start / End**: 影響度と適用ステップ範囲

## IP-Adapter

参照画像のスタイル・特徴を反映させます。

- **Weights**: ドロップダウンから IP-Adapter 重みを選択
- **Reference image**: 参照画像をアップロード
- **Scale**: 影響度

## Regional Prompter

領域ごとに別プロンプトと **領域別 LoRA** を適用します。本家 A1111 の Attention/Latent
Couple 方式ではなく、**「下地生成 → 領域ごとに LoRA を切り替えてインペイント」方式**
(段階インペイント方式)で実装されています。

### 方式の特徴

- **キャラ LoRA の混線が起きない** — キャラA LoRA と キャラB LoRA を同時に全体適用すると
  両者の特徴が混ざってしまう問題を、LoRA を **領域ごとに逐次切り替える** ことで完全に
  回避します。
- **想定用途**: キャラA LoRA × キャラB LoRA × シチュエーション LoRA で 2 キャラ構図を生成。

### パラメータ

| 項目 | 説明 |
|---|---|
| Layout | `horizontal` / `vertical` / `grid` / `mask` |
| Ratios | `1,2,1` のように比率指定(空欄なら均等分割) |
| Common prompt | 全領域に必ず付与 |
| Base prompt | 下地生成用プロンプト(構図を決める) |
| Region prompts | 領域ごとのプロンプト(`BREAK` または改行区切り、最大 4 領域) |
| Region negatives | 領域ごとのネガティブ(任意、`BREAK` 区切り) |
| Run base pass before regions | img2img/inpaint でも下地生成パスを先に実行(**推奨 ON**) |
| **Common LoRAs** | **下地パス + 全領域**に常時適用される LoRA(シチュ/スタイル LoRA 想定) |
| **Region 1〜4 LoRAs** | **その領域のインペイント時のみ**適用される LoRA(キャラ LoRA 想定) |
| Base strength | 下地パス(img2img/inpaint)の strength |
| Region strength | 各領域のインペイント strength |

### 推奨ワークフロー (例: キャラA × キャラB × シチュC)

1. **Common LoRAs**: `situation_C @ 0.9` を選択
2. **Base prompt**: `2girls, classroom, looking at each other, ...`
3. **Region prompts**:
   ```
   1girl, [characterA_trigger], black hair
   BREAK
   1girl, [characterB_trigger], blonde hair
   ```
4. **Region 1 LoRAs**: `characterA @ 0.9`
5. **Region 2 LoRAs**: `characterB @ 0.9`
6. **Run base pass before regions**: ON
7. **Region strength**: 0.75 程度(境界が気になるなら下げる)

### 制限と注意

- **最大 4 領域**。それ以上の `BREAK` は切り捨てられます。
- **領域別 LoRA を切り替えるたびに `unload_lora_weights → load_lora_weights` が走る**ため、
  領域数だけ生成時間が増加します(2 領域なら ~2 倍 + LoRA ロード分)。
- **境界に縫い目が出やすい** ため、`mask_blur` を大きめに設定するのが安全です。
- **OOM フォールバック時は Regional Prompter は無効化** され、通常生成にフォールバックします。
- **キャンセル**は領域ループの途中でも効きます(現在処理中の領域を完了して停止 ではなく、
  即座に途中で抜けます)。

## 自動目検出 inpaint

`models/detection/face_yolov8n.pt` をベースに YOLOv8 で顔を検出し、その上半分から目領域を
ヒューリスティックに抽出してインペイントマスクを自動生成します。モデルファイルが見つから
ない場合、起動時に Bingsu/adetailer から自動ダウンロードします。
