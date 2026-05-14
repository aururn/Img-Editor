# 要件定義書

## Img Editor — SDXL ベース 二次元キャラクター画像生成・編集アプリケーション

**版数**: 1.2
**作成日**: 2026-05-11
**更新日**: 2026-05-13
**対象**: 二次元キャラクター画像を SDXL + LoRA で生成・編集する個人利用向けローカル Web アプリ


---

## 1. プロジェクト概要

### 1.1 目的
Stable Diffusion XL 系(Illustrious / Pony / NoobAI 等)のチェックポイントと、各種 LoRA・
拡張アセットを組み合わせて、ローカル(またはレンタル GPU)上で二次元キャラクター画像の
生成・編集を行えるアプリケーションを構築する。

### 1.2 背景
- WebUI(AUTOMATIC1111 / ComfyUI)は高機能だが学習コストが高く、特定のワークフローには機能過多
- 自動目検出インペイント・領域別 LoRA など、Civitai 系ワークフローでよくある用途を
  最短手順で実行できる UI に特化する

### 1.3 スコープ
**含む**
- 二次元キャラクター画像(イラスト)の生成・編集
  - テキストからの新規生成(txt2img)
  - 画像変換(img2img)
  - 手動マスク inpaint
  - 自動目検出 inpaint(YOLOv8 ベース)
- 拡張アセットによる条件付け: LoRA / Textual Inversion / ControlNet / IP-Adapter
- Regional Prompter による領域別生成(**領域別 LoRA 対応**、Phase 1)
- ローカル PC または vast.ai 等レンタル GPU 上での単独動作(Gradio ローカル Web UI)

**含まない**
- 実在人物写真への適用(機能的にブロック対象とはしないが、利用規約・README で明示的に対象外とする)
- クラウド推論サービスへの統合
- LoRA 自体の学習機能
- 動画への対応

### 1.4 用語定義
| 用語 | 説明 |
|---|---|
| ベースモデル | Illustrious / Pony / NoobAI 系の Stable Diffusion チェックポイント |
| LoRA | ベースモデルに追加適用する小規模な微調整重み |
| txt2img | テキストプロンプトから新規画像を生成する手法 |
| img2img | 入力画像全体をプロンプトとノイズで再生成する手法 |
| inpaint | マスクで指定した領域だけを再生成する手法 |
| トリガーワード | LoRAの効果を発動させるためのプロンプト中の特定語句 |
| ControlNet | 構図・輪郭などを画像で条件付けする拡張機構 |
| IP-Adapter | 参照画像のスタイル・特徴を条件付けする拡張機構 |
| Embedding | Textual Inversion による追加トークン重み |
| Regional Prompter | 画像領域ごとに異なるプロンプトを適用する機構 |

---

## 2. 利用者・利用環境

### 2.1 想定ユーザー
- 自作または許諾済みの二次元イラストを編集したい個人
- Stable Diffusion の基本概念(プロンプト・LoRA・strength)を理解している、または学ぶ意欲がある

### 2.2 動作環境(必須要件)
| 項目 | 要件 |
|---|---|
| OS | Windows 10/11、Ubuntu 22.04+(vast.ai 等)、macOS(MPS 動作・性能保証外) |
| GPU | NVIDIA GPU、VRAM 16GB 以上(SDXL のため) |
| CUDA | 12.x または 13.x(PyTorch ビルドに合わせる) |
| メモリ | 16GB 以上 |
| ストレージ | コンテナ容量 24GB 以上(モデル格納用) |
| Python | 3.10 以上 |

### 2.3 動作環境(本番動作確認済み、vast.ai)
- テンプレート: **PyTorch (Vast)**(これ以外は不可、torch 入れ直しで容量不足になる)
- GPU: **RTX 5090 (32GB)**
- Python: **3.14.3** (`/venv/main/bin/python3`)
- PyTorch: **2.11.0+cu130**(プリインストール済)
- コンテナ容量: 32GB 推奨

### 2.4 動作環境(推奨ローカル)
- VRAM 16GB 以上(高解像度 inpaint で快適。本番は 32GB の RTX 5090)
- SSD(モデルロード高速化)

---

## 3. 機能要件

### 3.1 機能一覧

| ID | 機能名 | 優先度 |
|---|---|---|
| F-01 | 画像アップロード | 必須 |
| F-02 | txt2img モード(テキストから新規生成) | 必須 |
| F-03 | img2img モード(画像全体変換) | 必須 |
| F-04 | 手動 inpaint モード(キャンバスにブラシ塗りでマスク作成) | 必須 |
| F-05 | 自動目検出 inpaint モード(YOLOv8 ベース、Img Editor の独自機能) | 必須 |
| F-06 | プロンプト/ネガティブプロンプト入力 | 必須 |
| F-07 | LoRA 選択・強度調整 | 必須 |
| F-08 | パラメータ設定(strength, CFG, steps, seed) | 必須 |
| F-09 | 結果プレビュー・保存 | 必須 |
| F-10 | プリセット保存・読込 | 推奨 |
| F-11 | 履歴(直近 N 件)管理 | 推奨 |
| F-12 | Textual Inversion Embeddings 対応 | 推奨 |
| F-13 | ControlNet 対応(Canny 前処理含む) | 推奨 |
| F-14 | IP-Adapter 対応 | 推奨 |
| F-15 | Regional Prompter v2(BREAK 構文 + 領域別 LoRA、段階インペイント方式) | 推奨 |
| F-16 | 起動時診断レポート | 推奨 |

### 3.2 機能詳細

#### F-01 画像アップロード
- 受付形式: PNG / JPEG / WebP
- 最大解像度: 2048×2048(超過時は自動リサイズ、ユーザー確認あり)
- 最大ファイルサイズ: 20MB

#### F-02 txt2img モード
- テキストプロンプトから新規画像を生成
- 主要パラメータ: prompt, negative prompt, width, height, CFG scale, steps, sampler, seed
- LoRA / Embeddings / ControlNet / IP-Adapter / Regional Prompter 適用可

#### F-03 img2img モード
- 入力画像全体を再生成
- 主要パラメータ: prompt, negative prompt, strength(0.0–1.0), CFG scale, steps, sampler, seed
- LoRA は複数同時適用可、それぞれ weight 設定可(0.0–1.5)

#### F-04 手動 inpaint モード
- ブラウザ上のキャンバスでブラシ塗りしてマスク作成(目に限らず任意領域)
- マスクのフェザー(ぼかし)幅指定: 0–32px
- inpaint 固有パラメータ: mask blur

#### F-05 自動目検出 inpaint モード
- 顔・目検出モデル(YOLOv8、Bingsu/adetailer の face_yolov8n.pt)で目領域を自動抽出
- モデル未配置時は Hugging Face から自動ダウンロード
- 検出失敗時はユーザーに通知し手動 inpaint モードへ誘導

#### F-06 プロンプト入力
- ポジティブ/ネガティブの2フィールド
- LoRA トリガーワードはモード切替時にテンプレ挿入
- BREAK 構文で Regional Prompter 領域ごとのプロンプト分割

#### F-07 LoRA 管理
- `loras/` ディレクトリに配置された `.safetensors` を一覧表示(テーブル UI)
- safetensors ヘッダーからトリガーワード・推奨重み・対応ベースモデルを自動抽出
- チェックボックスで複数選択、各々スライダーで強度設定
- LyCORIS 系(locon/loha/lokr 等)の自動判別・警告

#### F-08 詳細パラメータ
| パラメータ | 範囲 | 既定値 |
|---|---|---|
| strength | 0.0–1.0 | 0.55(img2img) / 0.75(inpaint) |
| CFG scale | 1.0–20.0 | 7.0 |
| steps | 10–60 | 28 |
| sampler | Euler a / DPM++ 2M Karras / DDIM | DPM++ 2M Karras |
| seed | -1(ランダム) または整数 | -1 |
| mask blur | 0–32 | 8 |

#### F-09 結果プレビュー
- before / after を並列表示
- ワンクリックで PNG 保存(メタデータに使用パラメータを埋め込み)
- 既定の保存先: `outputs/YYYYMMDD/`

#### F-10 プリセット
- 現在のパラメータ一式を名前付きで JSON 保存
- 起動時に最後のプリセットを自動復元

#### F-11 履歴
- 直近 20 件の生成結果サムネイルを保持(設定で変更可)
- クリックでパラメータ再現

#### F-12 Textual Inversion Embeddings
- `models/embeddings/` の `.safetensors` / `.pt` / `.bin` を自動検出
- プロンプト中でトークン名を使用して適用

#### F-13 ControlNet
- `models/controlnet/` のモデルを自動検出
- アップロードした制御画像を使用
- Canny エッジ前処理をアプリ内で実行
- img2img / inpaint / txt2img(制御画像必須) に対応

#### F-14 IP-Adapter
- `models/ip_adapter/` のモデルを自動検出
- 参照画像のスタイル・特徴を条件付けとして使用
- スケール調整可

#### F-15 Regional Prompter v2(段階インペイント + 領域別 LoRA)
- **方式**: A1111 本家の Attention / Latent Couple ではなく、「下地生成 →
  領域ごとに LoRA を切り替えて順次インペイント」する **段階インペイント方式**
- BREAK 構文で領域分割プロンプトを記述(最大 4 領域)
- レイアウト: 水平 / 垂直 / グリッド / 塗りマスク
- **Common LoRAs**: 下地パス + 全領域に常時適用される LoRA(シチュエーション・スタイル LoRA 想定)
- **Region 1〜4 LoRAs**: その領域のインペイント時のみ適用される LoRA(キャラクター LoRA 想定)
- LoRA を領域ごとに逐次切り替えることで、**キャラ LoRA 同士の混線**(キャラA と キャラB の特徴が
  両方の顔に混ざる現象)を原理的に防止
- Base strength / Region strength を独立に調整可
- ベースパス(下地生成)の使用可否を設定可、デフォルト ON
- OOM 時は Regional 無効化で自動フォールバック
- キャンセル要求は領域ループ途中でも即応

#### F-16 起動時診断レポート
- 必要パッケージ(torch / diffusers / transformers 等)の存在確認
- checkpoint の存在確認
- 診断結果を UI 上に表示

---

## 4. 非機能要件

### 4.1 性能
| 指標 | 目標値(RTX 5090 / vast.ai) |
|---|---|
| 起動時間(初回モデルロード含む) | 60 秒以内 |
| txt2img 1枚生成(1024×1024, 28 steps) | 10 秒以内 |
| img2img 1枚生成(1024×1024, 28 steps) | 10 秒以内 |
| inpaint 1枚生成(マスク領域のみ) | 6 秒以内 |
| 自動目検出 | 1 秒以内 |
| Regional Prompter 2 領域(下地 + 2 領域 inpaint) | 30 秒以内 |
| Regional Prompter 4 領域(下地 + 4 領域 inpaint) | 60 秒以内 |

> Regional Prompter は領域別に LoRA を切り替えながら順次インペイントするため、生成時間は
> `(1 + 領域数) × inpaint 1枚` + LoRA ロード/アンロード分のオーバーヘッドとなる。

### 4.2 信頼性
- VRAM 不足時は attention slicing / VAE tiling 有効化 + 0.75x 縮小リトライ
- Regional Prompter で OOM 発生時は Regional 無効化フォールバック
- 生成キャンセル要求は次の step で即応(`callback_on_step_end` で `pipe._interrupt`)
- モデル未配置・破損時は起動時検出し、UI 上の System Status に案内表示
- YOLO 検出モデル未配置時は Hugging Face から自動ダウンロード

### 4.3 セキュリティ・プライバシー
- 完全ローカル動作、外部送信なし
- 起動時のモデルダウンロードはユーザー明示同意後のみ
- 入力画像・生成結果は端末ローカルのみに保持

### 4.4 保守性
- モデルパス・既定値はすべて `config.yaml` で外部化
- ロギング: Python `logging` で `logs/app.log` に出力(レベル切替可)

### 4.5 利用規約・コンプライアンス
- README に「実在人物画像への使用禁止」「権利を持つ画像にのみ使用」を明記
- 起動時に初回利用同意ダイアログを表示

---

## 5. 制約事項

- LoRA 自体の配布は行わない。ユーザーが Civitai 等から取得し、所定のフォルダに配置する
- ベースモデルも同様にユーザー側で用意
- LoRA のライセンス・利用規約は LoRA 配布元のものに従う

---

## 6. リスクと対応

| リスク | 影響度 | 対応 |
|---|---|---|
| ベースモデルと LoRA のバージョン不整合 | 中 | LoRA メタデータの対応ベース種別チェック、警告表示 |
| VRAM 不足 | 高 | attention slicing / VAE tiling 自動有効化、0.75x 縮小リトライ、Regional 無効化フォールバック |
| Regional Prompter の領域境界の縫い目 | 中 | mask_blur で吸収、Phase 2 で feather/latent blend 改善予定 |
| 領域別 LoRA 切替のオーバヘッド | 中 | 現状は領域ごとに unload+reload。Phase 2 で adapter プリロード + set_adapters 切替に最適化 |
| キャラ LoRA 同士の混線(全体適用時) | 高 | Regional Prompter v2 で領域別 LoRA を逐次切替し原理的に防止 |
| 自動目検出の精度不足 | 中 | 検出失敗時は手動 inpaint へ誘導、信頼度しきい値を config で調整可 |
| 不適切利用(実在人物等) | 高 | 規約明示、README で対象外を明文化 |
| ControlNet / IP-Adapter モデル未配置 | 低 | UI から選択肢を非表示、設定なし時は無効化 |
| vast.ai テンプレートの Python バージョン更新 | 中 | `audioop-lts` / `gradio<6` 等を process.txt の手順で都度追加インストール |