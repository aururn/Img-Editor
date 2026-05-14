# 設計書

## Img Editor — システム/詳細設計

**版数**: 1.1
**作成日**: 2026-05-11
**更新日**: 2026-05-13
**関連文書**: requirements.md

---

## 1. システム構成

### 1.1 全体アーキテクチャ

```
+--------------------------------------------------+
|                Browser (localhost)                |
|              Gradio UI (Web Frontend)             |
+---------------------+----------------------------+
                      | HTTP (Gradio internal)
+---------------------v----------------------------+
|              Application Layer                    |
|  +---------------+  +---------------------------+|
|  |  UI Handler   |->|   Pipeline Controller     ||
|  +---------------+  +-------------+-------------+|
|                                   |              |
|  +--------------------------------v------------+ |
|  |            Inference Service               | |
|  |  - Txt2ImgPipeline                         | |
|  |  - Img2ImgPipeline / InpaintPipeline       | |
|  |  - ControlNet Pipeline                     | |
|  |  - LoRA Loader / Embedding Loader          | |
|  |  - IP-Adapter Loader                       | |
|  |  - Eye Detector / Regional Prompter        | |
|  +------------------+--------------------------+ |
+---------------------+----------------------------+
                      |
+---------------------v----------------------------+
|          GPU (CUDA) / PyTorch + diffusers        |
+--------------------------------------------------+
```

### 1.2 技術スタック

| レイヤ | 採用技術 | 用途 |
|---|---|---|
| UI | Gradio 4.x | ローカル Web UI |
| アプリ | Python 3.10+ | 全体実装 |
| 推論 | PyTorch 2.x + diffusers>=0.31 | SD 推論 |
| アダプタ | peft>=0.13 | LoRA / adapter 管理 |
| モデル取得 | huggingface_hub>=0.26 | 検出モデル自動ダウンロード |
| 高速化 | xformers / SDPA | attention 最適化 |
| 目検出 | YOLOv8(anime-face) | 目領域抽出 |
| 画像処理 | Pillow / OpenCV / NumPy | マスク処理・前後処理 |
| 設定 | PyYAML | config.yaml 読み込み |
| ロギング | Python logging | アプリログ |

### 1.3 ディレクトリ構成

```
eye-editor/
├── app.py                      # Gradio エントリポイント
├── config.yaml                 # 設定ファイル
├── requirements.txt
├── README.md
├── docs/
│   └── extensions.md           # 拡張アセット説明
│
├── src/
│   ├── __init__.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_view.py        # Gradio Blocks 定義・カスタム CSS
│   │   ├── handlers.py         # イベントハンドラ・モード定義
│   │   ├── presets.py          # プリセット保存・読込
│   │   └── history.py          # 履歴管理
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline_manager.py # パイプライン生成/キャッシュ
│   │   ├── inference.py        # 推論実行
│   │   ├── lora_loader.py      # LoRAロード/管理/メタデータ抽出
│   │   ├── assets.py           # Embeddings/ControlNet/IP-Adapter 検出
│   │   ├── diagnostics.py      # 起動時診断
│   │   ├── regional.py         # Regional Prompter
│   │   └── config.py           # 設定ロード
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── eye_detector.py     # 目検出 (YOLOv8)
│   │   └── mask_builder.py     # マスク生成
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image_io.py         # 画像入出力・リサイズ
│       ├── metadata.py         # PNG メタデータ
│       └── logger.py
│
├── models/
│   ├── checkpoints/            # ベースモデル(.safetensors)
│   ├── loras/                  # LoRA 群
│   ├── embeddings/             # Textual Inversion
│   ├── controlnet/             # ControlNet モデル
│   ├── ip_adapter/             # IP-Adapter モデル
│   ├── vae/                    # 任意の VAE
│   └── detection/              # 検出モデル重み
│
├── presets/                    # ユーザープリセット (JSON)
├── outputs/                    # 生成結果
└── logs/
```

---

## 2. モジュール設計

### 2.1 主要クラス図(概念)

```
+--------------------+
|   AppController    |  app.py のメイン制御
+------+-------------+
       | uses
       v
+--------------------+       +---------------------+
| PipelineManager    |<>-----|  LoRALoader         |
| - get_txt2img()    |       |  - load(names, ws)  |
| - get_img2img()    |       |  - unload_all()     |
| - get_inpaint()    |       +---------------------+
| - get_controlnet() |
| - apply_loras()    |       +---------------------+
| - load_embeddings()|       |  AssetDiscovery     |
| - load_ip_adapter()|       |  - embeddings()     |
+------+-------------+       |  - controlnets()    |
       | uses                |  - ip_adapters()    |
       v                     +---------------------+
+--------------------+
| InferenceService   |       +---------------------+
| - run_txt2img()    |       |  RegionalPrompter   |
| - run_img2img()    |       |  - parse_break()    |
| - run_inpaint()    |       |  - build_masks()    |
+------+-------------+       +---------------------+
       | uses
       v
+--------------------+       +---------------------+
| EyeDetector        |-------|  MaskBuilder        |
| - detect(image)    |       |  - from_boxes()     |
+--------------------+       |  - from_canvas()    |
                             +---------------------+
```

### 2.2 主要モジュール仕様

#### 2.2.1 `core/pipeline_manager.py`

**責務**: SDXL系パイプライン(txt2img / img2img / inpaint / controlnet)の生成・キャッシュ・LoRA再装着

```python
class PipelineManager:
    def __init__(self, config: AppConfig):
        self._txt2img_pipe: StableDiffusionXLPipeline | None = None
        self._img2img_pipe: StableDiffusionXLImg2ImgPipeline | None = None
        self._inpaint_pipe: StableDiffusionXLInpaintPipeline | None = None
        self._controlnet_pipes: dict[str, Any] = {}
        self._loaded_ip_adapters: list[str] = []
        self._loaded_embeddings: list[str] = []
        self._current_loras: list[LoRASpec] = []

    def get_txt2img(self) -> StableDiffusionXLPipeline: ...
    def get_img2img(self) -> StableDiffusionXLImg2ImgPipeline: ...
    def get_inpaint(self) -> StableDiffusionXLInpaintPipeline: ...
    def get_controlnet(self, model_path: str) -> Any: ...
    def apply_loras(self, specs: list[LoRASpec]) -> None: ...
    def load_embeddings(self, paths: list[str]) -> None: ...
    def load_ip_adapter(self, path: str, scale: float) -> None: ...
    def free_memory(self) -> None: ...
```

**設計上のポイント**:
- 3パイプラインは `text_encoder` / `vae` / `unet` を共有して VRAM 消費を抑える(diffusers の `from_pipe` パターン)
- LoRA 切替時は `unload_lora_weights()` → `load_lora_weights()` → `fuse_lora()` の順
- ControlNet パイプラインはモデルパスをキーにキャッシュ
- `cpu_offload` オプションで低 VRAM 環境に対応

#### 2.2.2 `core/inference.py`

**責務**: 実際の生成処理

```python
@dataclass
class GenerationRequest:
    mode: Literal["txt2img", "img2img", "inpaint"]
    image: PIL.Image.Image | None
    mask: PIL.Image.Image | None
    prompt: str
    negative_prompt: str
    strength: float
    cfg_scale: float
    steps: int
    sampler: str
    seed: int
    width: int = 1024
    height: int = 1024
    mask_blur: int = 8
    # ControlNet
    controlnet_model: str | None = None
    controlnet_image: PIL.Image.Image | None = None
    controlnet_preprocess: str = "none"
    controlnet_scale: float = 1.0
    controlnet_start: float = 0.0
    controlnet_end: float = 1.0
    # IP-Adapter
    ip_adapter: str | None = None
    ip_adapter_image: PIL.Image.Image | None = None
    ip_adapter_scale: float = 1.0
    # Embeddings
    embeddings: list[str] = field(default_factory=list)
    # Regional Prompter
    regional: RegionalSpec | None = None

@dataclass
class GenerationResult:
    image: PIL.Image.Image
    seed_used: int
    elapsed_ms: int
    metadata: dict
```

#### 2.2.3 `core/assets.py`

**責務**: 拡張アセット(Embeddings / ControlNet / IP-Adapter)の自動検出

```python
@dataclass
class EmbeddingSpec:
    name: str; path: str

@dataclass
class ControlNetSpec:
    name: str; path: str

@dataclass
class IPAdapterSpec:
    name: str; path: str

def discover_embeddings(directory: str) -> list[EmbeddingSpec]: ...
def discover_controlnets(directory: str) -> list[ControlNetSpec]: ...
def discover_ip_adapters(directory: str) -> list[IPAdapterSpec]: ...
```

#### 2.2.4 `core/regional.py`

**責務**: Regional Prompter v1 の実装

```python
@dataclass
class RegionalSpec:
    layout: Literal["horizontal", "vertical", "grid", "mask"]
    ratios: str          # 例: "1,1" (水平2分割)
    prompts: list[str]   # BREAK で分割されたプロンプト
    use_base_pass: bool

class RegionalPrompter:
    @staticmethod
    def parse_break(prompt: str) -> list[str]: ...
    @staticmethod
    def build_masks(layout: str, ratios: str, size: tuple[int,int]) -> list[PIL.Image.Image]: ...
```

#### 2.2.5 `core/diagnostics.py`

**責務**: 起動時の環境・モデル検証

```python
class Diagnostics:
    def run(self, config: AppConfig) -> str:
        """Markdown 形式の診断レポートを返す"""
    def _check_packages(self) -> list[str]: ...
    def _check_checkpoint(self, path: str) -> bool: ...
    def _check_directories(self, config: AppConfig) -> list[str]: ...
```

#### 2.2.6 `detection/eye_detector.py`

**責務**: 二次元キャラクターの目領域検出

```python
@dataclass
class EyeBox:
    x: int; y: int; w: int; h: int
    confidence: float

class EyeDetector:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = YOLO(model_path)  # 遅延ロード

    def detect(self, image: PIL.Image.Image) -> list[EyeBox]:
        """検出結果を信頼度降順で返す。顔検出のみの場合は目領域を推定分割"""
```

**補足**:
- 検出モデル未配置時は `huggingface_hub` 経由で自動ダウンロード
- 顔全体しか検出されない場合は顔領域を左右に分割して目領域を推定

---

## 3. 画面設計(Gradio)

### 3.1 モード構成

```
タブ切替(カスタム CSS によるタブバー):
  [Text to image] [Image to image] [Inpaint manual] [Inpaint auto]
```

### 3.2 主要 Gradio コンポーネント

| 領域 | コンポーネント |
|---|---|
| モード切替 | カスタム CSS タブバー |
| 画像入力 | `gr.Image`(img2img) / `gr.ImageEditor`(手動 inpaint) |
| プロンプト | `gr.Textbox(lines=3)` |
| LoRA 選択 | `gr.Dataframe`(テーブル UI、トリガーワード列含む) |
| パラメータ | `gr.Slider`, `gr.Dropdown`, `gr.Number` |
| ControlNet | `gr.Image`(制御画像) + `gr.Dropdown`(モデル・前処理) |
| IP-Adapter | `gr.Image`(参照画像) + `gr.Dropdown`(モデル) + `gr.Slider`(スケール) |
| Embeddings | `gr.CheckboxGroup` |
| Regional | `gr.Textbox`(ratios) + `gr.Dropdown`(layout) |
| 結果表示 | `gr.Image` x2(before/after) |
| 履歴 | `gr.Gallery` |
| 診断 | `gr.Accordion` + `gr.Markdown` |

### 3.3 主要イベントフロー

```
[Generate ボタン押下]
   |
   v
1. 入力バリデーション (画像有無、プロンプト空チェック)
   |
   v
2. モード判定
   +- txt2img      -> InferenceService.run_txt2img()
   +- img2img      -> InferenceService.run_img2img()
   +- inpaint(手動) -> ImageEditor からマスク抽出 -> run_inpaint()
   +- inpaint(自動) -> EyeDetector.detect() -> MaskBuilder -> run_inpaint()
   |
   v
3. LoRA / Embedding / IP-Adapter 構成変化チェック -> 必要なら再ロード
   |
   v
4. Regional Prompter 有効時: BREAK 解析 -> 領域マスク生成
   |
   v
5. ControlNet 有効時: 前処理(Canny 等) -> パイプライン選択
   |
   v
6. 推論実行
   |
   v
7. 結果表示・履歴追加・自動保存
```

---

## 4. データ設計

### 4.1 `config.yaml`(全セクション)

```yaml
app:
  host: "0.0.0.0"
  port: 7860
  log_level: "INFO"

paths:
  checkpoints_dir: "./models/checkpoints"
  loras_dir: "./models/loras"
  embeddings_dir: "./models/embeddings"
  controlnet_dir: "./models/controlnet"
  ip_adapter_dir: "./models/ip_adapter"
  vae_dir: "./models/vae"
  detection_dir: "./models/detection"
  outputs_dir: "./outputs"
  presets_dir: "./presets"
  logs_dir: "./logs"

model:
  base_checkpoint: "illustriousXL_v01.safetensors"
  vae: "madebyollin/sdxl-vae-fp16-fix"  # null なら checkpoint 内蔵 VAE
  default_sampler: "DPM++ 2M Karras"
  enable_xformers: false
  enable_vae_tiling: true
  precision: "fp16"
  cpu_offload: false

detection:
  eye_model: "face_yolov8n.pt"
  confidence_threshold: 0.35
  padding_px: 8
  feather_px: 12

defaults:
  img2img:
    strength: 0.55
    cfg_scale: 7.0
    steps: 28
  inpaint:
    strength: 0.75
    cfg_scale: 7.0
    steps: 28
    mask_blur: 8

controlnet:
  preprocess: "none"
  scale: 1.0
  start: 0.0
  end: 1.0

ip_adapter:
  scale: 1.0

regional:
  enabled: false
  layout: "horizontal"
  ratios: ""
  use_base_pass: false

limits:
  max_image_size: 2048
  max_file_size_mb: 20

history:
  max_items: 20
```

### 4.2 LoRA メタデータ(`models/loras/<name>.json`)

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

safetensors ヘッダーから自動抽出も行う。JSON が存在する場合は JSON が優先。

### 4.3 プリセット形式(`presets/<name>.json`)

```json
{
  "mode": "inpaint_auto",
  "prompt": "utsurome, empty eyes, masterpiece, best quality",
  "negative_prompt": "lowres, bad anatomy",
  "loras": [{"name": "empty_eyes", "weight": 0.8}],
  "strength": 0.75,
  "cfg_scale": 7.0,
  "steps": 28,
  "sampler": "DPM++ 2M Karras",
  "seed": -1,
  "mask_blur": 8
}
```

### 4.4 出力 PNG メタデータ
PIL の `PngInfo` を用い、`parameters` キーに以下を埋め込む(A1111互換風)。

```
<prompt>
Negative prompt: <negative>
Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 7.0,
Seed: 1234567890, Size: 1024x1024,
Model: illustriousXL_v01,
LoRAs: empty_eyes:0.8,
Mode: inpaint_auto, Mask blur: 8
```

---

## 5. 主要処理フロー詳細

### 5.1 自動目検出 inpaint シーケンス

```
User          UI Handler       EyeDetector    MaskBuilder    InferenceService
 | Generate        |               |               |               |
 |--------------->|               |               |               |
 |                | detect(image) |               |               |
 |                |-------------->|               |               |
 |                |<---- boxes ---|               |               |
 |                | from_boxes()  |               |               |
 |                |------------------------------>|               |
 |                |<------ mask ------------------|               |
 |                | run(request with mask)                        |
 |                |---------------------------------------------->|
 |                |<----------- GenerationResult -----------------|
 |<- show result -|
```

### 5.2 LoRA 切替最適化

```
新リクエストの LoRA 構成 == 現在の構成?
+- YES -> そのまま推論
+- NO  -> unload_lora_weights()
          for each (name, weight) in new_specs:
              load_lora_weights(name, adapter_name=name)
          set_adapters([name...], adapter_weights=[w...])
          (任意) fuse_lora()
```

### 5.3 VRAM 不足時のフォールバック

```
推論実行
+- try: pipe(...)
+- except torch.cuda.OutOfMemoryError:
   1. torch.cuda.empty_cache()
   2. enable_attention_slicing()
   3. enable_vae_tiling()
   4. リトライ
   5. それでも失敗 -> 解像度を 0.75 倍に下げて再試行
   6. それでも失敗 -> エラー表示
```

---

## 6. エラー処理

| エラー種別 | 検出箇所 | UI 表示 | ログ |
|---|---|---|---|
| ベースモデル未配置 | 起動時(diagnostics) | 診断レポートに警告 | ERROR |
| LoRA ベース不整合 | LoRA選択時 | 黄色警告バナー | WARN |
| 目検出 0 件 | 検出後 | 「手動モードに切り替えますか?」 | INFO |
| CUDA OOM | 推論中 | フォールバック処理、最終失敗時エラー表示 | ERROR |
| 不正な画像形式 | アップロード時 | 「対応形式: PNG/JPEG/WebP」 | WARN |
| ControlNet/IP-Adapter 未配置 | アセット検出時 | 選択肢を非表示 | INFO |

---

## 7. 起動・配布

### 7.1 `requirements.txt`(主要)

```
torch>=2.2
torchvision
diffusers>=0.31,<0.40
transformers>=4.46,<5.0
accelerate>=1.0
peft>=0.13
safetensors>=0.4.5
huggingface_hub>=0.26,<1.0
gradio>=4.40,<5.0
pillow
opencv-python
numpy
pyyaml
ultralytics
xformers         # 任意・GPU環境次第
```

---

## 8. テスト方針

| テスト種別 | 内容 |
|---|---|
| 単体 | `MaskBuilder`, `EyeDetector`, `LoRALoader`, `RegionalPrompter`, `Diagnostics` の入出力 |
| 結合 | txt2img / img2img / inpaint(手動・自動) 各モードの end-to-end |
| 拡張機能 | ControlNet / IP-Adapter / Embeddings / Regional Prompter の動作確認 |
| 性能 | VRAM 12GB / 16GB 環境での生成時間計測 |
| 異常系 | OOM、モデル未配置、目検出失敗、LoRA ベース不一致 |
| 受入 | 想定ワークフロー(画像upload -> 自動inpaint -> 保存)が 3クリック以内で完結すること |

---

## 9. 今後の拡張余地

- ControlNet(reference / lineart)追加前処理
- 顔以外の領域(口、髪)もマスク選択肢に追加
- バッチ処理モード(フォルダ単位)
- LoRA メタデータ自動取得(Civitai API 連携、ユーザー許諾下)
- macOS MPS 公式対応(性能チューニング)