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

領域ごとに別プロンプトを適用します。実装は hako-mikan
[sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter) の
**Latent Couple 方式**を Diffusers 向けに移植した**単一denoise内UNetブレンド方式**です。
各 UNet ステップで base CFG + 領域ごとの CFG 計算を行い、ラテント空間のマスクで
ノイズ予測をブレンドするため、ステージ分割 inpaint と異なり繋ぎ目が出ません。

LoRA は左パネルで選択したものを全領域に共通適用できます。さらに Region LoRAs 欄や
region prompt 内の `<lora:name:weight>` / `<lora:name:text_weight:unet_weight>` タグで、
領域ごとに Text Encoder 用 LoRA 重みと U-Net 用 LoRA 重みを切り替えられます。

### パラメータ

| 項目 | 説明 |
|---|---|
| Layout | `horizontal` / `vertical` / `grid` / `mask` |
| Ratios | `1,2,1` のように比率指定。`1,1;2,1` のような2D比率も可 |
| Base ratio | base prompt と region prompt の混合比。空欄なら `0.2` |
| Overlay ratio | 隣接領域を少し重ねて境界の急な切り替わりを抑える比率 |
| Use base prompt | 互換用フラグ。Region prompts の先頭要素は消費しない。base prompt は通常メイン Prompt、`ADDBASE` で明示指定 |
| Use common prompt | 互換用フラグ。Region prompts の先頭要素は消費しない。Common prompt 欄と `ADDCOMM` を共通タグとして使用 |
| Use common negative prompt | region negative の先頭要素を全領域の共通ネガティブにする |
| Common prompt | 全領域に必ず付与 |
| Base prompt | メイン Prompt。`ADDBASE` 使用時は region prompt 側の先頭要素をbase扱い |
| Region prompts | 領域ごとのプロンプト(`BREAK` / `ADDROW` / `ADDCOL` / 改行区切り、最大 4 領域) |
| Region negatives | 領域ごとのネガティブ(任意、`BREAK` 区切り) |
| Region LoRAs | 領域ごとの LoRA 指定。`BREAK` 区切りで `<lora:left:0.8>` / `name@0.6/0.9` などを指定 |
| LoRA negative TE / U-Net | 対象外領域で region LoRA をどれだけ残すか。空欄または `0` なら対象外では無効 |
| LoRA stop step | 指定step以降、region専用LoRAを止めて共通LoRAだけでUNet計算する |
| Preview masks | 「Preview masks」ボタンで現在の設定に基づく領域マスクをプレビュー表示する |

### 構文

```text
left prompt BREAK right prompt
```

```text
top-left ADDCOL top-right ADDROW bottom-left ADDCOL bottom-right
```

```text
common tags ADDCOMM base tags ADDBASE left prompt BREAK right prompt
```

`ADDCOMM` がある場合は先頭要素を common prompt に追加します。
`ADDBASE` がある場合は次の要素を base prompt として使います。

記事でよく使われる `1;3,1,1` のような比率は、上段1領域・下段3領域のような2D分割として解釈されます。

### 制限と注意

- **最大 4 領域**。それ以上の `BREAK` は切り捨てられます。
- 左パネルの LoRA は全領域に共通適用されます。Region LoRAs や inline LoRA tag は指定した領域だけに追加適用されます。
- 領域数ぶんUNet追加計算が走るため、通常生成より遅くなります。
- `mask` layout は色分けマスクに対応します。キャンバスで赤・緑・青など別色を塗ると、region prompt の順に割り当てられます。
- **OOM フォールバック時は Regional Prompter は無効化** され、通常生成にフォールバックします。

## 自動目検出 inpaint

`models/detection/face_yolov8n.pt` をベースに YOLOv8 で顔を検出し、その上半分から目領域を
ヒューリスティックに抽出してインペイントマスクを自動生成します。モデルファイルが見つから
ない場合、起動時に Bingsu/adetailer から自動ダウンロードします。
