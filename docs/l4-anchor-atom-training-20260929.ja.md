# L4・1024²でatom×site評価を減らす学習試作（2026-09-29）

## 方式

既存のCST atom、Triweight、Strip＋Torus、AdamWを維持する。64×64ブロック内の固定アンカーだけで真のCST重みを評価し、三次Lagrange基底で近似演算子を組み立てる。学習経路は全W・全dWを生成せず、アンカー行列 `S` のみを生成する。`Y=(X B_i) Sᵀ B_oᵀ` なので、逆伝播で得る `dS=(dY B_o)ᵀ(X B_i)` をアンカーのatom勾配に縮約する。基底とアンカー位置は固定し、atom振幅と中心は毎ステップ更新する。幅の勾配は現行DirectAmpWidth経路と同じく切る。

候補リストは従来の16×64 site tileで作る。アンカーkernelはその候補リストを再利用し、各候補を選択したsiteだけで評価する。候補リストは元tileの保守的な支持判定なので、アンカーの寄与を欠落させない。候補192枠を超えたtileは従来どおりbucket spanへフォールバックする。

実装は [`prototypes/anchor_atom_training.py`](../prototypes/anchor_atom_training.py)。完全ステップ比較は [`prototypes/profile_anchor_atom_training.py`](../prototypes/profile_anchor_atom_training.py)。これは実験経路であり、公開backendの選択には追加していない。

## L4測定

Colab ProのNVIDIA L4、PyTorch 2.11.0+cu128、Triton 3.6.0。1024×1024、52,429 atoms（5%）、seed 21、入力行数M=16/128。現行CST側は候補リスト・W 8×64・atom勾配16×64・512行窓・FP16全Wキャッシュ・IEEE FP32 GEMM。両方式は同じ初期atomから作り、固定ランダム入力とMSE targetでAdamWを行った。各方式2回warmup、1回Graph capture、20回Graph replay。両方式を同一プロセスに置き、順序を交互にした。表はGraph replayのwall時間中央値と、対応ペア比率の中央値。

| M | ブロック内アンカー | 評価site割合 | 現行CST | アンカーCST | 対応ペア比 | 初期出力相対L2 | 初期atom勾配相対L2 | 23更新後の出力相対L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 16×32 | 1/8 | 1.082 ms | **0.622 ms** | **0.574** | 0.750% | 0.728% | 2.424% |
| 16 | 24×48 | 28.125% | 1.087 ms | **0.769 ms** | **0.708** | 0.158% | 0.154% | 0.852% |
| 128 | 16×32 | 1/8 | 1.145 ms | **0.658 ms** | **0.576** | 0.733% | 0.733% | 4.409% |
| 128 | 24×48 | 28.125% | 1.156 ms | **0.837 ms** | **0.723** | 0.151% | 0.153% | 2.272% |

16×32の完全ステップは現行CSTの約1.74倍速く、24×48は約1.38～1.41倍速い。16×32の初期アンカー値と真のCST重みの該当siteは相対L2約2.1×10⁻⁷、最大絶対差3.73×10⁻⁹で一致した。atom勾配も同じatomを学習する。一方、上表のatom勾配誤差は**近似演算子と真のCSTとの差**であり、カーネル実装誤差ではない。

実装勾配を別途128²・seed 37で、全atom×アンカーsiteをPyTorchで直接評価・自動微分したものと照合した。アンカー値の相対L2差2.01×10⁻⁷、packed atomの振幅・中心勾配の相対L2差1.36×10⁻⁷、最大絶対差3.81×10⁻⁶。width列の実装勾配は0で、既存経路と同じdetach契約。再現コードは [`prototypes/check_anchor_atom_gradients.py`](../prototypes/check_anchor_atom_gradients.py)。

別プロセスで各方式をGraph capture・再実行したときのPyTorch最大割当は次の通り。初期化用の全Wは作らない測定であり、CUDA予約量ではない。再現コードは [`prototypes/profile_anchor_memory.py`](../prototypes/profile_anchor_memory.py)。

| M | 現行CST | 16×32 | 24×48 |
| ---: | ---: | ---: | ---: |
| 16 | 43.75 MiB | 44.20 MiB | 46.75 MiB |
| 128 | 45.94 MiB | 45.95 MiB | 48.87 MiB |

## 判断と制限

**atom×site評価の削減は、L4の完全ステップ高速化につながった。** dWを単独でなくすだけなら現行全ステップの約2～3%しか削れないが、アンカー法ではatom勾配とW生成で評価するsite数そのものを減らした。低ランク因子を直接学習する別モデルとは異なり、atomパラメータとTriweight値を毎ステップ再評価・更新する。

ただしこれは近似した演算子の学習であり、真のCSTと同一の学習軌道ではない。23回の固定ランダムMSE更新で両方式のlossは近かったが、出力差は初期より増えた。データタスクでの精度、長い学習、atom移動後の誤差、複数seedでの安定性、近似の許容閾値は未検証。16×32を即採用せず、24×48を精度寄りの候補として残す。次は同じタスク上で学習曲線と最終精度を比べ、許容できるアンカー密度を決める。

生JSONは [`data/l4-anchor-training-20260929/`](data/l4-anchor-training-20260929/)。測定ソースはコミット`3656244`のGit archive SHA256 `70fb247f447d80ea6227b50654d53fdc9f15ad7da12525dd07288ec4fcc3141a`。Colab session `torchcst-anchor-l4-20260929`は停止し、active sessionがないことを確認した。
