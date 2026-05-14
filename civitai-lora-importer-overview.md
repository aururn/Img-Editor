# civitai-lora-importer 概要メモ

## 目的

`civitai-lora-importer` は、Civitai のモデルURLまたはモデルIDから LoRA のメタデータを取得し、ローカルの LoRA ナレッジ管理フォルダに Markdown ファイルとして保存するための Claude Code subagent。

最終的には、Civitai URL を渡すだけで、LoRA名・説明・base model・trigger words・model version・safetensorsファイル名・gallery prompt example などを取得し、カテゴリ別フォルダに整理して保存することを目指す。

保存先は以下に固定する。

```text
C:\Users\tr831\Documents\ap\.claude\lora-knowledge
```

---

## この agent が解決したいこと

手動で Civitai の LoRA ページを見て、以下のような情報を毎回 Markdown に整理するのは面倒。

- LoRA名
- URL
- base model
- trigger words
- safetensorsファイル名
- LoRA tag
- prompt example
- negative prompt
- 推奨weight
- 生成設定
- カテゴリ分類
- 保存先フォルダ選択

そのため、`civitai-lora-importer` に以下を任せる。

1. Civitai URL または model ID を受け取る
2. ローカル Python スクリプトで Civitai API から情報取得
3. gallery 画像APIから最大10件の prompt / negative prompt / settings を取得
4. ユーザーが希望した場合は、LoRA本体の `.safetensors` を `models/loras/` に保存
5. 取得できた JSON を読む
6. ユーザーが貼り付けた prompt example があれば統合
7. character / outfit / style / feature / situation などに分類
8. 対象カテゴリフォルダの `example.md` を読む
9. そのカテゴリの既存フォーマットに合わせて Markdown を作成
10. 正しいフォルダに保存する

---

## 関連 agent

### lora-researcher

既存の LoRA metadata を検索する agent。

役割:

- ローカルの LoRA metadata md を検索
- 似た LoRA があるか確認
- URL、LoRA tag、trigger words、base model、recommended weight を抽出
- `example.md` は検索結果から除外

検索対象は以下に固定済み。

```text
C:\Users\tr831\Documents\ap\.claude\lora-knowledge
```

### civitai-lora-importer

Civitai URL から新規 LoRA metadata を作成・保存する agent。

役割:

- Civitai API から基本情報取得
- ユーザー貼り付けの prompt example と統合
- カテゴリ判定
- `example.md` に準拠して md 保存
- API失敗時は空ファイルを保存しない

---

## 正しいフォルダ構成

現在想定している構成は以下。

```text
C:\Users\tr831\Documents\ap\
├─ .env
├─ scripts\
│  └─ fetch-civitai-model.py
└─ .claude\
   ├─ agents\
   │  ├─ lora-researcher.md
   │  ├─ civitai-lora-importer.md
   │  ├─ lora-compatibility-checker.md
   │  └─ sd-lora-prompt-engineer.md
   └─ lora-knowledge\
      ├─ characters\
      │  └─ example.md
      ├─ outfits\
      │  └─ example.md
      ├─ styles\
      │  └─ example.md
      ├─ features\
      │  └─ example.md
      ├─ situations\
      │  └─ example.md
      ├─ model-profiles\
      │  └─ example.md
      └─ guides\
         └─ example.md
```

---

## Python スクリプトの置き場所

Civitai API を叩く Python スクリプトはここに置く。

```text
C:\Users\tr831\Documents\ap\scripts\fetch-civitai-model.py
```

`.claude` 配下ではなく、プロジェクト直下の `scripts` に置く理由:

- Claude Code agent の定義ファイルと実行スクリプトを分けられる
- `.claude/agents` に補助スクリプトを混ぜなくて済む
- Bash から実行しやすい
- `.env` をプロジェクト直下から読みやすい

---

## API key の置き場所

Civitai API key は agent prompt に直書きしない。

プロジェクト直下の `.env` に置く。

```text
C:\Users\tr831\Documents\ap\.env
```

中身:

```env
CIVITAI_API_KEY=ここにAPIキー
```

agent は API key を直接扱わず、Python スクリプトが `.env` または環境変数から読み込む。

---

## 現在の実装状況

### 完了済み

- `lora-researcher` の設計
- `lora-researcher` の検索範囲を以下に固定

```text
C:\Users\tr831\Documents\ap\.claude\lora-knowledge
```

- `lora-researcher` が他フォルダを無駄に読まないように修正
- `example.md` を検索結果から除外する方針を追加
- `civitai-lora-importer` の設計
- `civitai-lora-importer` の保存先を以下に固定

```text
C:\Users\tr831\Documents\ap\.claude\lora-knowledge
```

- カテゴリ別フォルダの保存ルールを定義
- 各カテゴリの `example.md` を参照して md 形式を合わせる方針を定義
- API key を `.env` に置く方針を定義
- Python スクリプトを `scripts/fetch-civitai-model.py` に置く方針を定義
- API 取得失敗時に空の placeholder md を保存しないルールを追加
- Civitai gallery 画像APIから最大10件の prompt / negative prompt / settings を `gallery_prompts` として正規化する実装を追加
- `--download-lora` 指定時に `.safetensors` を `models/loras/` に保存する実装を追加

---

## 途中で発生した問題

### 1. 保存先がズレた

本来の保存先は以下。

```text
C:\Users\tr831\Documents\ap\.claude\lora-knowledge
```

しかし、writer/importer が別の `lora-knowledge` を見に行った可能性があった。

対策:

- agent の system prompt に絶対パスを明記
- 保存先・参照先・`example.md` の場所を固定
- `C:\Users\tr831\Documents\ap\lora-knowledge` には保存しないルールを追加

---

### 2. example.md を読めていない可能性

カテゴリ別に md フォーマットが違うため、各フォルダに `example.md` を置く方針にした。

agent は新規 md 作成前に、対象カテゴリフォルダの `example.md` を読む必要がある。

例:

```text
C:\Users\tr831\Documents\ap\.claude\lora-knowledge\characters\example.md
C:\Users\tr831\Documents\ap\.claude\lora-knowledge\outfits\example.md
```

対策:

- `example.md` を最優先のフォーマット参照にする
- `example.md` がない場合のみ既存 `.md` を1〜3件読む
- それもない場合だけ fallback format を使う

---

### 3. Civitai API が 403 を返した

`civitai-lora-importer` を実行した際、Civitai API が 403 を返した。

その結果、最初は情報がほぼ空の md が保存されてしまった。

対策:

- APIが失敗した場合は保存しない
- ほぼ空の placeholder md を作らない
- URLだけしか情報がない場合は保存しない
- API取得失敗時は、ユーザーにページテキスト・prompt example・trigger words・settings の貼り付けを求める

---

## 現在の期待動作

### API取得が成功した場合

ユーザーがこう依頼する。

```text
@agent-civitai-lora-importer
このCivitai URLからLoRA情報を取得して、適切なカテゴリにmd保存して。
https://civitai.com/models/xxxx/xxxxx
```

期待される動作:

1. URLから model ID を抽出
2. `scripts/fetch-civitai-model.py` を実行
3. Civitai API JSON を取得
4. gallery 画像APIから最大10件の sample prompt を `gallery_prompts` として取得
5. LoRA本体の保存が必要な依頼なら `--download-lora` で `.safetensors` を `models/loras/` に保存
6. LoRA名、説明、base model、trainedWords、files、modelVersions などを抽出
7. カテゴリ判定
8. 対象フォルダの `example.md` を読む
9. `example.md` の形式に合わせて md を作成
10. 正しいカテゴリフォルダに保存
11. 保存パスを報告

---

### API取得が失敗した場合

API が 403 などを返した場合。

期待される動作:

1. 失敗を報告
2. Markdown ファイルは保存しない
3. 空の placeholder を作らない
4. ユーザーに以下の貼り付けを求める

```text
- LoRAページのテキスト
- trigger words
- safetensors filename
- base model
- prompt examples
- negative prompt examples
- generation settings
```

---

### URL + prompt example を渡した場合

ユーザーがこう渡す。

```text
@agent-civitai-lora-importer
このURLとprompt exampleからmdを作って保存して。

URL:
https://civitai.com/models/xxxx/xxxxx

Prompt examples:
...

Negative prompt:
...

Settings:
...
```

期待される動作:

1. API取得を試す
2. APIが成功すればAPI情報と貼り付け情報を統合
3. APIが失敗しても、貼り付け情報が十分ならそれを主情報として使う
4. カテゴリ判定
5. `example.md` に合わせて md 作成
6. 保存

---

## 保存してよい最低条件

以下が揃っている場合だけ保存する。

- LoRA名またはタイトル
- Source URL
- カテゴリ、またはカテゴリを推測できる情報
- URL以外に少なくとも1つ以上の有用なメタデータ

有用なメタデータの例:

- trigger words
- base model
- safetensors filename
- LoRA tag
- description
- prompt examples
- model version metadata

保存してはいけないケース:

- APIが403で失敗した
- URLしか分からない
- ほとんどの項目が `unknown`
- category が不明
- 保存先が正しい root 配下ではない
- `example.md` を上書きしそうになっている

---

## agent 権限

`civitai-lora-importer` は Python スクリプトを実行する必要があるため、以下が必要。

```text
Read-only tools: ON
Edit tools: ON
Execution tools: ON
MCP tools: OFF
Other tools: OFF
All tools: OFF
```

agent frontmatter の tools は以下。

```yaml
tools: Glob, Grep, Read, Edit, Write, Bash
```

`WebFetch` / `WebSearch` / `NotebookEdit` は不要。

理由:

- Civitai API取得はローカル Python スクリプト経由で行う
- WebFetch / WebSearch は使わない
- notebook 編集は不要

---

## 現在の懸念

### API 403

Civitai 側が 403 を返す場合がある。

考えられる原因:

- API key が未設定
- `.env` の場所が違う
- API key が読み込まれていない
- 対象モデルが認証・年齢確認・制限付き
- Civitai API 側の制限
- `civitai.red` URL と API の相性問題

確認コマンド:

```powershell
cd C:\Users\tr831\Documents\ap
python scripts\fetch-civitai-model.py "https://civitai.com/models/xxxx/xxxxx"
```

403が返る場合は agent の問題ではなく、API取得側の問題。

### Gallery prompt の取得

スクリプトはデフォルトで以下の Civitai images API と model-version detail API を使い、対象 LoRA の gallery / version image から最大10件の prompt 情報を取得する。

```text
https://civitai.com/api/v1/images?modelId=<MODEL_ID>&limit=10
https://civitai.com/api/v1/model-versions/<MODEL_VERSION_ID>
```

取得できた prompt は JSON の `gallery_prompts` に正規化して入る。

主なフィールド:

- `image_id`
- `image_url`
- `civitai_image_url`
- `prompt`
- `negative_prompt`
- `settings`
- `resources`
- `model_version_id`
- `model_version_name`

確認コマンド:

```powershell
cd C:\Users\tr831\Documents\ap
python scripts\fetch-civitai-model.py "https://civitai.com/models/xxxx/xxxxx" --images-limit 10
```

注意:

- 最大10件まで取得する
- prompt が非公開、削除済み、meta未保存の場合は取得できない
- API key のアカウントで閲覧できる範囲は取得できる可能性が高い

### LoRA本体のダウンロード

ユーザーが LoRA 本体の保存も依頼した場合、以下のオプションを使う。

```powershell
cd C:\Users\tr831\Documents\ap
python scripts\fetch-civitai-model.py "https://civitai.com/models/xxxx/xxxxx" --download-lora
```

保存先:

```text
C:\Users\tr831\Documents\ap\models\loras
```

挙動:

- Civitai API の `modelVersions[].files[]` から `.safetensors` の model file を選ぶ
- URLに `modelVersionId=...` があればそのversionを優先する
- `modelVersionId` がなければ `.safetensors` を持つ先頭versionを使う
- Civitai側のファイル名を優先して保存する
- 同名ファイルがある場合は上書きせず、末尾に `-2`, `-3` ... を付ける
- 結果は JSON の `lora_download` に入る

事前確認だけしたい場合:

```powershell
python scripts\fetch-civitai-model.py "https://civitai.com/models/xxxx/xxxxx" --download-dry-run
```

---

## 次にやること

1. `civitai-lora-importer.md` を修正版に置き換える
2. `scripts/fetch-civitai-model.py` を作成・保存する
3. `.env` に `CIVITAI_API_KEY` を設定する
4. PowerShell で Python スクリプト単体をテストする
5. 403時に保存しないことを確認する
6. API成功時に正しいカテゴリへ保存されるか確認する
7. API失敗時は URL + prompt example 貼り付け運用で保存できるか確認する

---

## 使い方テンプレート

### APIだけで取得

```text
@agent-civitai-lora-importer
このCivitai URLからLoRA情報を取得して、適切なカテゴリにmd保存して。
https://civitai.com/models/xxxx/xxxxx
```

### URL + prompt exampleで登録

```text
@agent-civitai-lora-importer
このURLと貼り付けテキストからLoRA metadata mdを作って保存して。

URL:
https://civitai.com/models/xxxx/xxxxx

Prompt examples:
...

Negative prompt:
...

Settings:
...
```

### 保存せずに確認だけ

```text
@agent-civitai-lora-importer
このCivitai URLから情報取得だけして、保存せずに抽出結果と保存予定パスを表示して。
https://civitai.com/models/xxxx/xxxxx
```

---

## 最終的に目指す運用

1. `lora-researcher` で既存 md に同じ LoRA があるか確認
2. なければ `civitai-lora-importer` で Civitai URL から登録
3. APIで取れない場合は URL + prompt examples を貼って登録
4. 登録後、再度 `lora-researcher` で検索できるか確認
5. 複数 LoRA を使う場合は `lora-compatibility-checker` で互換性確認
6. 最後に `sd-lora-prompt-engineer` で画像生成 prompt を作る

流れ:

```text
探す
→ なければ登録
→ 互換性確認
→ prompt作成
```
