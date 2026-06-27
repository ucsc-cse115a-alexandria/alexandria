# 公開済みプロンプトと正解データを備えた精度ベンチマーク

## 概要

本ノートは、Alexandria にとって絞り込まれた一つの問いに答えるものである。すなわち、評価に用いる正確なプロンプトと正解データの双方を公開しており、圧縮の前後で同一のプロンプトを実行し、その結果を自動的に採点できる公開ベンチマークはどれか、という問いである。これは、`InstructionSet` を圧縮してもタスク精度が保たれることを検証するために必要となる。第一の推奨は **IFEval**（Instruction-Following Eval）であり、そのプロンプトはモデルへの入力そのものであり、その「正解」は決定論的でプログラムによって検証可能な制約の集合である。次点は **GSM8K** であり、各設問に単一の数値による正解と、EleutherAI の lm-evaluation-harness に記載されたプロンプトテンプレートが対応づけられている。

## 手法

### 主候補 — IFEval (Zhou et al., 2023)

- **評価対象:** プロンプトに埋め込まれた明示的かつ検証可能な指示（例えば「300語以上で書く」「JSON で応答する」「キーワード X を3回以上含める」「カンマを一切使わない」など）にモデルが従うかどうか。採点は内容ではなく *形式* を対象とするため、人手や LLM による判定を必要としない。
- **規模:** 541 件のプロンプト、単一の `train` split、25 種類の指示タイプ、英語のみ（BCP-47 `en`）。Apache 2.0 ライセンス。
- **正確なプロンプトの公開状況:** `prompt` フィールドがモデルに送られる文字列そのものであり、それを包む隠れたテンプレートは存在しない。これが IFEval が我々の A/B 検証に適する決定的な理由である。公開された成果物が、評価プロンプトそのものなのである。
- **正解データの形式:** 各行は `key`、`prompt`、`instruction_id_list`（満たすべき制約タイプ）、および `kwargs`（`num_words: 300`、`relation: "at least"`、`num_highlights: 3` といった制約ごとのパラメータ）を保持する。したがって「解答キー」は、単一の正解文字列ではなく、機械によって検証可能な制約の仕様である。
- **採点方法:** 決定論的なチェッカーが、各指示をモデル出力に対して検証する。Google Research の公式コード（`instruction_following_eval`）は4つの指標を報告する。すなわち prompt レベルと instruction レベルの精度であり、それぞれに `strict` と `loose` の変種がある。strict は厳密な遵守を要求し、loose は前置きや後置きの文言と整形上の修飾を取り除くことで偽陰性を減らす。

### 次点 — GSM8K (Cobbe et al., 2021)

- **評価対象:** 算術的推論を要する多段階の小学校レベルの数学文章題。
- **規模:** 8.5K 件の問題（7,473 件の train / 1,319 件の test）、`question` と `answer` という2つの単純な文字列フィールド。`answer` は `#### <final_number>` の行で終わる。
- **正確なプロンプトの公開状況:** 評価プロンプトそのものはデータセット自体には含まれないが、lm-evaluation-harness のタスク YAML に公開されている。すなわち `doc_to_text: "Question: {{question}}\nAnswer:"` と `doc_to_target: "{{answer}}"` である。既定のタスクは 5-shot であり、`gsm8k_cot` の変種は 8-shot である。
- **正解データの形式:** `answer` フィールドの `#### <number>` 接尾辞から抽出された単一の数値による正解。
- **採点方法:** 抽出された数値に対する `exact_match` であり、2つのフィルタを介する。すなわち `strict-match`（`#### <number>` 形式を要求する）と `flexible-extract`（生成結果中の最後の数値を取る）である。

IFEval が好ましいのは、そのデータセットの行が *そのまま* プロンプトであり、前後比較が曖昧さなく行えるためであり、また制約ごとの決定論的なチェッカーが、圧縮による細かな文言の変化に敏感なきめ細かいシグナル（instruction レベルの精度）を与えるためである。

## 検証方法

- **IFEval:** モデルには各 `prompt` がそのまま与えられ、自由形式の応答を生成する。チェッカーは `instruction_id_list` に挙げられたすべての制約を `kwargs` のパラメータを用いて評価する。プロンプトは、そのすべての制約が通った場合にのみ「従った」とみなされる（prompt レベル）。各制約も個別に採点される（instruction レベル）。精度 = （従った数）/（全体数）であり、strict と loose の基準ごとに別々に報告される。判定モデルは関与しないため、スコアは完全に再現可能である。
- **GSM8K:** EleutherAI の lm-evaluation-harness を通じて標準化されている。harness は記載されたプロンプトテンプレートを描画し、貪欲法によるデコード（`temperature 0.0`）を実行し、`strict-match` / `flexible-extract` の正規表現フィルタで最終的な数値を抽出し、それを正規化したうえで（カンマ、`$`、および `#### ` 接頭辞を無視する）、`exact_match` を用いて正解の数値と比較する。
- **共通の harness:** 両ベンチマークとも lm-evaluation-harness の第一級のタスクであるため、単一の CLI でプロンプトの描画、生成、解答の抽出、指標の集計を再現可能な形で駆動できる。

## 本プロジェクトとの関連

Alexandria は、冗長性スコアリングと貪欲なペアワイズ枝刈りによって `InstructionSet` を圧縮する。圧縮が精度を保つことを示すために、公開されたプロンプトと正解データを備えたベンチマークで A/B 検証を行う。

1. **ベースライン（圧縮前）:** 各 IFEval `prompt` を未圧縮の `InstructionSet` 入力として取り、対象モデルに送り、公式の strict/loose チェッカーを実行する。prompt レベルと instruction レベルの精度を記録する。
2. **圧縮後（after）:** 同じプロンプトを Alexandria の represent -> score -> greedy-pairwise パイプラインに通して圧縮済みプロンプトを生成し、同一のデコード設定のもとで同じモデルに送り、同じ `instruction_id_list` / `kwargs` に対して同じチェッカーを実行する。重要なのは、正解がプロンプトの文言ではなく制約に結びついている点である。そのため、プロンプトを書き換えたり短縮したりした後でも正解は有効なままであり、これはまさに、圧縮が制約を落とすかどうかを測定するために必要なことである。
3. **比較:** 精度の差分（圧縮後からベースラインを引いたもの）が忠実度を測り、トークン数の差分が削減量を測る。両者を報告し、加えて指示タイプごとの劣化を報告することで、Alexandria の枝刈りがどの制約カテゴリを失いやすいかを確認する。IFEval の決定論的で判定モデルに依存しない採点が、この差分を信頼できるものにする。
4. **推論面でのクロスチェック:** GSM8K で繰り返し、推論型のプロンプト（few-shot の例示 + 設問）の圧縮が `exact_match` のもとで最終的な数値の正解を保つことを確認する。

実務上の注記: IFEval が最初の統合に適しているのは、そのデータセットの行が正確なプロンプトそのものであり、`InstructionSet` への配線がテンプレートの再構成を伴わない直接的な対応づけになるためである。GSM8K は harness テンプレートの再現を要するが、形式の遵守を超えた、内容の正しさという第二の軸を与えてくれる。

## 関連論文

- [Zhou et al. — Instruction-Following Evaluation for Large Language Models
  (2023)] — IFEval の 541 件の検証可能な指示プロンプトと、strict/loose かつ prompt/instruction レベルの指標を定義している。arXiv:2311.07911. Dataset:
  https://huggingface.co/datasets/google/IFEval — Code:
  https://github.com/google-research/google-research/tree/master/instruction_following_eval
- [Cobbe et al. — Training Verifiers to Solve Math Word Problems (2021)] —
  `#### <answer>` という正解形式を持つ 8.5K 件の小学校レベルの数学問題からなる GSM8K を導入している。arXiv:2110.14168. Dataset: https://huggingface.co/datasets/openai/gsm8k
- [EleutherAI — lm-evaluation-harness] — IFEval と GSM8K の双方について、標準化されたタスク定義（`doc_to_text` のプロンプトテンプレート、`doc_to_target` の解答キー、指標/フィルタの設定）を提供している。Repo:
  https://github.com/EleutherAI/lm-evaluation-harness — GSM8K task:
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml
