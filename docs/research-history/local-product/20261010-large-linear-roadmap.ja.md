# 大型 Linear の研究方針

## 追記: 規則格子 Product を主対象にする

2026-10-10 の追加方針。以降の大型化は、単一の 2 軸 Euclidean Product
Chart と Triweight profile product を主対象にする。両軸は LinePattern、
正の等間隔、当面は両軸で同じ spacing を使う。Sphere 固有の計画・数値契約を
この Product に読み替えない。下記の元計画と参照 protocol JSON は Sphere の
計画として保存し、この追記が今後の主対象と作業順序を更新する。

既存の `_candidate_span` を使い、中心・支持半径・原点・spacing から
候補区間を直接計算する。丸めを含む保守的候補に元の positive-support 判定を
適用する。入力・出力の区間は独立に求め、支持全体の norm と既存の
whole-product floor を維持する。Sphere の per-chart floor へ置き換えない。

既存 `research_profile_product_large_prepared` と
`research_profile_product_large_matrix` は N8192・atom4194304 までを宣言し、
既存の支持区間・方向別実行配置・forward 保存状態を使う。まずこの実装と
既存 Case/catalog を比較基準にする。宣言上の上限は新たな性能検証ではない。

1. dispatch の対応条件を確立する。Product + 2 Line 軸 + 正の同一 spacing
   を metadata から判定し、退化 spacing と軸間 spacing 不一致を拒否する。
   有効な宣言で候補が非対応なら自動選択は同じ意味の Torch fallback、強制 Plan は
   実行前に拒否する。Points 軸・Explicit chart は既存 profile_product の宣言検査で
   拒否される。不正な数学宣言を fallback で隠さない。
2. 支持区間を用いる既存大型ルートで、全量独立 FP64 oracle と Y/dX/全 atom
   勾配を確認する。全 FP64 runtime の完成を最適化の必須前段にはしない。
3. 支持・H/G・VJP の保存量を chunk で制限し、既存 matrix 経路と比較する。
   支持数分類は live sigma から毎 forward 更新し、dispatch key に持ち込まない。
4. N1024/2048 から 4096/8192 へ進め、構築・配置・backward・optimizer・Graph
   込みの時間と allocated/reserved peak を比較する。並べ替えは費用込みで判定する。

Chart の live start/spacing 変更は宣言 cache を更新し、exact dispatch 条件も
変える。Graph は更新後に再 capture する。公開既定 dispatcher への採用は
研究 Registry での正しさ・性能確認後に判断する。

今回の対応条件修正は CPU 1406 passed / 2644 skipped、関連回帰 89 passed /
53 skipped、Ruff、wheel/sdist build を通過した。N2048/rho3 の既存 prepared と
native-g8-p8 + dense 比較を `tools.kernel_dev prepare/check` で宣言検査した。
raw log と比較 snapshot は研究 worktree の ignored `output/regular-product-*`
に保存する。この追記では GPU の正しさ・性能・メモリを新たに測定していない。

## 元の Sphere 計画

2026-10-10。起点は main `5755a1b5`。この文書は今後の研究計画であり、
未実装方式の精度・性能・メモリ改善を示す結果ではない。

初回研究の固定条件は [参照 protocol](20261010-large-linear-reference-protocol.json)
にも記録する。設計宣言済みだが runtime/source は未凍結であり、投入可能な
benchmark catalog として扱わない。

## 目的と優先順位

主目標は、既存 CST Linear の数学的契約を保ったまま、batch 32、
N1024 → N2048 → N4096 → N8192 の完全学習 step を実行できるようにすること。
時間と CUDA Graph capture/replay 込みのメモリを同時に評価する。
規模ごとに正しさを確立してから次へ進む。

普通幅の初期 sigma=3 を性能の主条件とする。sigma=1.25 は鋭い支持、
sigma=8 は広い支持・overflow の独立した精度診断とする。
幅は各 step で既存の活動度則に従って更新し、固定しない。

直近の Sphere を主実装経路にする。Strip/Torus は別の数学的契約を持つ
比較・移植経路として維持する。Sphere の結果を Torus や radial kernel
の結果として読み替えず、Sphere の一段が成立してから同じ実行機構の
移植可能性を判定する。

研究の順序を次に固定する。

1. **全 atom・全サイトの物理 FP64 参照を確立する。**
2. **正確な支持探索で全サイト走査を置き換える。**
3. **支持情報・H・VJP 作業領域を atom chunk で制限する。**
4. **4096、8192 へ対応範囲を拡張し、完全 step を比較する。**

参照の時点から chunk を使って保存量を制限するが、その全サイト走査は
残す。次の段では探索だけを変え、最後に実行配置・融合を改善する。
一度に精度、支持集合、optimizer、演算配置を変更しない。

## 今の結果から決めること

既存 compact W は N1024/N2048、sigma=3 の有効な比較基準である。
直近の [direct recompute](20261010-sphere-direct-recompute.md) は保存量を
減らしたが、compact W より完全 step が遅く、採用条件を満たしていない。
[support-adaptive](20261010-sphere-support-adaptive.md) は鋭い条件の全量勾配
検査で停止し、[固定 0.001 の精度補正](20261010-sphere-precision-adaptive.md)
も補正対象外の atom に誤差が残っている。

したがって、新研究は 0.001 の閾値を調整する延長から始めない。
全 atom の物理計算を高精度にし、その真の費用と誤差の発生箇所を先に
測る。精度を選択的に下げる方式は、その後に必要性が示された場合のみ、
別の事前宣言した研究として扱う。既存の失敗を再試行や許容誤差変更で
成功へ読み替えない。

main の Sphere Algorithm は現在 1..2048 サイトに対応範囲を制限している。
int16 ID の上限 32768 は保存形式の条件であり、4096/8192 の対応を意味しない。
新しい対応範囲は別の研究 Algorithm/revision と検証で宣言する。

## 数学的な契約

入出力は既存の明示的 intrinsic S² chart の組。距離は実際の埋め込み座標の
chord distance、profile は Triweight、振幅・幅は既存の Polar/activity 則。
全 6 atom 成分の勾配と task-loss に対する幅の stop-gradient を保つ。

各側の raw profile と正規化を

\[
f_{a,i}=(1-\pi_a\|s_i-q_a\|^2)_+^3,\qquad
n_a=\sqrt{\sum_{i\in\text{chart}}f_{a,i}^2},\qquad
\phi_{a,i}=f_{a,i}/\max(n_a,10^{-6})
\]

とする。chart 全域で計算した norm を用い、各側で floor を適用する。
一つの Cartesian 全体の norm/floor へ変更しない。分離演算は

\[
H_{b,a}=\sum_i X_{b,i}\phi^{\mathrm{in}}_{a,i},\qquad
Y_{b,j}=\sum_a \mathrm{amp}_a\phi^{\mathrm{out}}_{a,j}H_{b,a}.
\]

atom chunk は上式の全 atom 和を分割するだけであり、サンプリング、
atom の間引き、正規化領域の切り取りを導入しない。支持容量を超えた場合も
全寄与を処理する。norm と floor が等しい点の微分は既存の `clamp_min`
規約に従う。floor 未満の正支持 singleton は非ゼロの中心勾配を持ち得るため、
その微分を保持する。等値/以上の正支持 singleton では中心勾配がゼロになる。

## 規模と費用の見積もり

主系列は \(K=\lfloor0.05N^2\rfloor\)、batch \(B=32\)、FP32 の学習状態。
一つの atom は 6 scalar なので、5% atom は dense の scalar parameter 数の
約 30% に相当する。下表は **tensor 容量の計算値**であり、実測 peak ではない。

| N | atom K | P+勾配+Adam 2 moments MiB | W+dW MiB | 全 atom の H MiB | 両側 CAP64 ID16+Phi MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 52,428 | 4.8 | 8 | 6.4 | 38.4 |
| 2048 | 209,715 | 19.2 | 32 | 25.6 | 153.6 |
| 4096 | 838,860 | 76.8 | 128 | 102.4 | 614.4 |
| 8192 | 3,355,443 | 307.2 | 512 | 409.6 | 2457.6 |

容量式はそれぞれ \(96K\)、\(8N^2\)、\(4BK\)、
\(2K\cdot64\cdot(2+4)\) bytes。optimizer の previous-P、forward 保存 P、
geometry/Jacobian、oracle、allocator/Graph pools は別途必要になる。
Phi だけを削除しても N8192 の ID 約819 MiB と H 約410 MiB は残る。
full U/V を保存する場合は同規模で約205 GiBとなるため、採用しない。

全サイト走査は両側で \(2KN\) profile-site 評価を行い、主系列では
\(O(N^3)\)。N8192 の 1 checkpoint だけで約550億評価になる。
chunk は保存量を解決するが、この計算量は解決しない。
正確な支持 index が次の構造的な変更になる。

計算量の切り分け用に K=209,715 を固定した N2048/4096/8192 も用意する。
この系列の atom 密度は5%/1.25%/0.3125%であり、主系列の代替や採用証拠には
使わない。同じ K の中で方式を比較し、サイト数と atom 数の増加を区別する。

## A: 完全な物理精度参照

最初の実装は standalone な研究 route とし、旧 FP32 準備や norm-sensitive
classifier を前段に置かない。全 atom の site 正規化、intrinsic decode、
距離、幅、profile、norm、Jacobian/VJP を FP64 で計算する。
最初は contraction/reduction も FP64 にして丸め箇所を切り分け、API 境界で
FP32 の Y/dX/dP を返す。FP32 contraction を試す場合は独立した ablation とし、
物理精度を変えず同じ全量 gate を課す。

floor・π・級数係数などの非整数定数を明示的な FP64 値として生成する。
振幅の特異原点処理では、元の FP32 parameter-domain の `tiny` 定数を維持し、
計算を FP64 にしたという理由で FP64 の `tiny` へ変更しない。

Triton の chunk 開始位置は runtime offset とし、固定 C と有効行 mask を用いる。
開始位置を constexpr にして全 chunk 数だけ specialization を生成しない。
固定 site tile T=256、FP64 norm/moment 部分和と再利用する chunk workspace を
用いる。全 N を一つの FP64 ベクトルとしてレジスタに保持せず、完全 norm の確定後に
profile/VJP を処理する。tile 化による reduction 順序も数値検証の対象にする。

fixed atom chunk C=128、full chart traversal を起点とする。forward は全体の
入力 P と live scalar/site/radius の不変 snapshot を持ち、backward で各 chunk
の profile と H を再計算する。全 K×CAP の ID/Phi、全 B×K の H を保存しない。
両側の FP64 factor tile 容量は \(16CN\) bytes。C=128、N2048 なら4 MiB、
N8192 なら16 MiBで、Jacobian/VJP scratch は別に上界を示す。

独立した物理 FP64 oracle は既存の明示的埋め込みと Torch autograd の全量
実装を用い、runtime の decoder/profile helper を共有しない。Y/dX は FP64
で全 chunk を累積し、全 6 dP を比較する。oracle も全サイト・全 atom を
処理し、巨大な factor 全体や `grads` の list+cat による不要な重複を避ける。
固定 dY に対して chunk ごとに local VJP を求め、その autograd graph を解放して
detached Y/dX を累積する。全 chunk の differentiable Y を保持してから一度だけ
backward する方式は用いない。既存 oracle の dtype 依存 `tiny` と上記 FP32
domain 定数の相違は、新 oracle revision で明示し、古い結果・helper は変更しない。
FP32 出力への cast error も診断に記録するが、誤差 gate は緩めない。

新しい固定 cohort は N1024/N2048 × sigma1.25/3/8 の参照6条件。
既存 compact W と standard dense は sigma3 の各規模で比較し、計10 worker。
既知に失敗する sharp の旧 FP32 compact は過去の負結果として扱い、新参照の
採用基準に置かない。前の full16 cohort の成功資格は転用しない。

seed41、batch32、K=floor(.05*N²)、共有 X/target/dY、初期 P/site/radius/scalar
hash を固定する。loss は MSE、fused AdamW は lr1e-4、weight-decay0.01、
betas=(0.9,0.999)、eps1e-8、amsgrad=false。FP32 IEEE 入出力、TF32/autocast
無効、Triton FP fusion 無効とする。新しい物理参照の内部計算のみ FP64 とし、
既存 compact と standard dense は内部 FP32 を保持する。外部の FP32/TF32 方針を
揃え、内部精度の違いは比較結果に明記する。共有入力の RNG stream と
model 初期化 stream は既存 fixture と同様に分ける。標準 dense の weight 初期値
は別の固定 seed41 stream を持ち、CST の初期 P と同じ値とは主張しない。

最初の GPU batch は各 worker を別 job に分割し、1 L4 の共有 queue で実行する。
各 job は outer900s / driver875s / child350s。回帰テストは実装後、投入前に
正確な collection と disjoint partition を凍結する。全条件の source、
driver/protocol、全テスト ID と原始結果の和集合を照合する。

数値検証10 worker 全てが成功してから、sigma3 の参照/compact/dense
6 worker の未計装完全 step を測り、準備・収縮・optimizer の費用は別途診断する。
CST8 worker は初期/24更新後の全 Y/dX/全6dP を検査する。dense2 worker は
独立した dense Y/dX/dW、有限 state、AdamW moments と正確な clock を検査する。
geometry update と atom dP の条件を dense に適用しない。
ここでは参照の成立と費用の確定が目的であり、FP64 化を性能向上と呼ばない。

## B: 正確な支持探索

A と同じ全 atom FP64 物理計算を維持し、探索だけを変更する。
実際の正規化済み FP64 site に対する空間 index を用い、保守的な候補範囲を
問い合わせた後、元の positive-gap predicate で最終支持を決める。
unit-sphere の距離恒等式で実座標の距離を置換しない。

cell 境界と支持半径の外向き丸めを含めた保守的な範囲を構成し、正支持 site
が必ず候補に入る根拠を記録する。丸め誤差の安全性を判断できない場合は完全走査に
落とす。site ID の順序を固定する。支持集合は完全一致を検査する。
浮動小数点 reduction の結果は独立 FP64 との numerical gate で検査する。

候補が全ての正支持サイトを含むことを、小さい domain の全列挙と
全参照条件で確認する。支持数、完全 norm、重複/欠落、各寄与の所有を検査する。
問い合わせ容量、支持容量、危険な境界、無効な index は完全走査へ落とす。
live site/radius の変更は新しい forward の index に反映し、古い backward
は保存 snapshot と index だけを読む。

index 構築、再構築、分類、fallback の全費用を完全 step に含める。
sigma から tiny 比率や高速化を推測せず、実支持数と fallback 比率を記録する。

## C: 実行と保存量の制限

B が正しさを満たした後、同じ支持集合・物理算術で atom chunk の実行を改善する。
全 K×CAP index、全 H/G、full U/V、full W/dW を必要としない経路を主候補にする。
一回の変更では chunk 処理・保存/再計算の一つの機構だけを比較する。
chunk 数に比例して autograd 保存や Graph pool が増えないことを実測で確認する。

output 所有による atomic 削減は追加機構として分ける。K×CAP の output CSR を
作って保存量を戻したり、各 output tile で全 atom を再走査したりする実装は、
その費用を明示して比較する。無条件の次の候補にはしない。

## D: 4096、8192 への拡張

各サイズの新しい宣言・index 幅・chunk tail・overflow・Graph 捕捉を検証する。
前サイズの全量初期/24更新後 gate と測定が完了した場合のみ進む。
大規模の oracle はメモリ上は chunk で全量検査できるが、時間は未確定である。
A/B の実測費用から、次サイズの全量検査と測定の job 分割・deadline を
投入前に別 protocol で固定する。失敗後の再予算化やサンプリングへの変更はしない。
実行不能なら、そのサイズは未検証/実行不能として残し、原因を次の研究へ渡す。

同サイズ・同 K・同 batch・同初期状態・同 optimizer の有効な CST control を
保持する。standard dense も同サイズ/batch/task/外部 FP32・TF32 方針/loss/AdamW 設定で
保持する。CST control が実行不能なら、CST に対する速度比は
未測定とする。dense は異なる parameterization/update 則の工学的比較であり、
同じ optimizer 軌跡や学習品質を意味しない。

## 全段階の合否と採否

- CST は initial と24回の実更新後に、全 Y、全 dX、全 atom の6 dP を物理 FP64 と照合。
  maxabs **かつ** relative-L2 が4e-4以下。近似・sampled oracle で代替しない。
- 小さい非正方形 Cartesian domain の直接列挙、空支持、singleton、両軸の
  floor 以下/等値/以上、正支持境界、重複/並べ替え site、overflow、chunk tail、
  requested gradients、regular zero amplitude を確認する。
- retained forward 後の P/site/radius/scalar 変更、可変幅、24回の Graph 更新、
  公開 optimizer と Parameter/moments/正確な clock を確認する。
- correctness、完全範囲、支持集合、clock、job 期限、原始結果 integrity の
  どれかが失敗した freeze は性能選定を停止する。欠落 worker から勝者を選ばない。
- 完全 step は forward/loss/backward/AdamW/geometry update を含み、2 warmups、
  1初回 replay、21 timed replays の計24更新。21 samples を21独立 run と呼ばない。
- capture/replay 込み peak allocated / reserved を区別する。oracle scratch と
  独立 phase 診断は主測定から除く。GPU process usage 未測定なら未測定と記載する。
- 同条件の有効 CST control に対し、時間が厳密に3%超改善、または allocated が
  5%以上減り時間悪化が3%以下なら一次候補。独立 reverse run で全条件・control を
  保持して再確認し、規模ごとに採否を決める。
- dense に対する時間比と memory 比も必ず報告する。現状は dense に対する
  速度・training peak の優位を確立していない。kernel 誤差の一致だけで
  圧縮モデルの学習品質を主張しない。

各段階を意味のある source/test/note commit にし、研究 branch、frozen source、
driver/protocol、GPU job ID、raw logs/tensors/hash、採否と残課題を保存する。
未採用 runtime は研究 branch に置く。main への統合は必要な検証後の GitHub PR、
公開 dispatcher の採用は別判断とする。

次の実装対象は A の **全 atom FP64・chunk128・H 再計算参照**の一つに固定する。
これが成立するまで、閾値 sweep、広い候補探索、4096/8192 の未検証投入は行わない。
