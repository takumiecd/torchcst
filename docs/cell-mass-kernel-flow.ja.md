# 区間確率方式：prepareからbackwardまでのkernelの流れ

## 設計の正本と、現在の実装範囲

2026-10-27。この文書を、sigmaからの階層分類・layout・Z/H/C・backwardを
つなぐ設計の正本とする。分類規約は下の
[sigmaから階層と周期を決める規約](#sigmaから階層と周期を決める規約)に置く。
他の実測ノートは、その時点の実装と測定の記録であり、この設計が全て実装済みという意味ではない。

**階層版は既存ChebyshevSquareの式とsigmaの意味を維持し、Z→H→C→Yへ接続する。**
Zの二分集約木があることと、sigmaから異なるtile/周期を選ぶことを区別する。
既存の固定bank版は比較・復元用に維持する。以前追加した「sigma=全幅、I=sigma/2」は撤回する。

| 項目 | この文書の設計 | 固定bank版 `cell_mass/`（source `7c1404c1`） |
| --- | --- | --- |
| sigmaの意味 | 既存の幅制御値。I=effective_radius(profile,sigma,P) | 同じ幅制御値・半径下限補正 |
| 階層分類 | T_minから2倍ずつ、sigma<=T_hの最小local bankを選ぶ | 未実装。軸ごとに一つのAxis設定 |
| 周期 | local bankごとにP_h=2L_h=3T_h | 軸ごとに固定P=K*T |
| sort | 入出力のbankと中心tileの辞書順 | 中心の(output owner,input owner)だけ |
| Z | bankの基底ごとに直接生成／一致modeの継承 | 固定Pの全modeを同じ基底で二分マージ |
| 小幅の直接計算／全周bankへの自動分類 | 全周bankは分類済み。小幅directは任意の高速化候補 | 未実装。宣言済みK=1の経路はあるが自動分類ではない |
| H/G・Parameter勾配 | 各threadが別atomを担当 | 実装済み、固定bank条件で検証済み |
| C→Y/dX | bankごとにCを作って展開後に加算 | owner Cの部分和と展開は実装済み。複数bankの合算は未実装 |
| 面積正規化・backward | 既存密度・面積式と解析微分を再利用。階層版backwardは実GPU検証待ち | 固定bankの独立oracle照合完了 |

同じprofile式でも選ぶPが変わるので、固定bankの時間・勾配検証を階層版の結果へ読み替えない。
実装箇所は[`CellMassAxisSpec`](../src/torchcst/kernels/cell_mass.py)、
[`executor.py`](../src/torchcst/_backends/cuda/algorithms/linear/cell_mass/executor.py)、
[`kernels.cu`](../src/torchcst/_backends/cuda/algorithms/linear/cell_mass/kernels.cu)。
現状と測定条件の入口は[実測状態](cell-mass-status.ja.md)。

### 実装の配置と、新しい階層版の段階

固定bank版の`cell_mass/`は移動・削除せず、既存テスト・結果の比較先に残す。
新しい実装は[`cell_mass_hierarchical/`](../src/torchcst/_backends/cuda/algorithms/linear/cell_mass_hierarchical/)
へ分ける。固定版executorへの暗黙のfallbackは使わず、同じprofile数学を明示的に再利用する。
同じ研究branch/Draft PRで進め、復元可能な旧sourceは`8f343c16`以前に残る。

| 新しい実装 | 処理と保存するもの | 状態 |
| --- | --- | --- |
| `plan.py` | 2倍ずつのlocal bank・全周bank・Z node位置。Tensorを持たない実行計画 | 実装済み、CPU分類検査済み |
| `executor.prepare_layout` | 分類とstable sort。元行番号、groupの(h_out,h_in,q_out,q_in)、offsetsだけをcacheへ残す | CUDA実装済み、元行・group範囲を実GPU照合済み |
| `preparation.cu::build_hierarchy` | bank別の周期のZ。直接生成と偶数harmonicの子からの継承。完成Zはglobal FP64 | CUDA実装済み、direct/mergeを独立積分参照へ照合済み |
| `contraction.cu::compute_h` | 各threadが別atomのHをFP64で完成。既存係数と面積をregisterで計算 | 実装済み、既存profileの独立参照へ照合済み |
| `compute_c` / `expand_c` | 32atom shared chunkからowner Cを作り、bank別の基底で展開後にYへ加算 | 実装済み、CとYを別々に照合済み |
| G/dX・Parameter勾配 | 同じprofile微分と選択中のlayoutを使う。各threadが別atomを担当 | 実装済み、実GPU検証待ち |

新しい前半は空の雛形でなく、`prepare_layout(plan, atoms)`と
`build_hierarchy(signal, axis, method="merge")`を実行するCUDA入口。
直接生成だけの`method="direct"`も比較に使える。
研究用の`forward_only`はprepare→Z→H→C→Yを呼び、detached入力の診断用として残す。
新しい`hierarchical_linear`は明示的なautograd bindingを持つ。公開CSTLinear/dispatcherへの登録はない。

新しいlayout cacheにatom値や全atomキーは保持しない。所有するTensorは
`rows[A]`（int64）、`groups[G,4]`（int32）、`offsets[G+1]`（int64）。
groupがない場合もoffsetsは[0]。元Parameterへの参照とversionを持ち、更新後の古いcacheの使用を拒否する。
同じatom値・planなら異なるXで再利用できる。sort/uniqueのキー・countsはprepare中だけのglobal scratch。
一時領域の再利用は可能だが、block終了で消えるsharedだけで後続kernelへ順序を渡すことはしない。
現在のsort/uniqueは動的shapeとhostへのerror確認を含むため、layoutのCUDA Graph対応は主張しない。

各Z bankはその幅のtile nodeだけを持つ。全bankそれぞれに全幅の木を重複して作らない。
local node数はN/n_min、N/(2*n_min)、...と減り、最後に全周nodeを一つ置く。
1nodeは`[batch,1+2*M]`。direct kernelのsharedは128個のdouble部分和で1,024bytes/block。
`team_size=1,2,4,8,16,32`のthread集団が一つのZ要素のsite和を分担する。
Hのatomを複数threadで分担するという意味ではない。
最初は全宣言bankのnodeを準備し、使われないbankの省略は依存関係を含めた後続の改善とする。
cacheの`tensor_bytes`やZの容量はpayloadであり、完全stepのallocatorピークと混同しない。

### 階層版backwardの実装（実GPU検証待ち）

`hierarchical_linear(plan, x, atoms)`から次の順番で呼び出す。

```text
forward: prepare_layout → build_hierarchy(X) → H → C_out → Y
backward: build_hierarchy(dY) → G
                                ├→ C_in → dX
                                └→ parameter_grad(Zx,Zy,H,G) → d_atoms
```

H/Gは振幅を含まない`[batch,sorted_slot]`のFP64配列。学習対象にatomがある場合だけ、
forwardのZx/Hをautogradの保存状態に残す。入力だけの勾配ならZx/Hを残さない。
backwardのZy/Gは同じ呼び出しのdXとparameter_gradで共用する。
C_out/C_inはbank別`[batch,node,mode]`で、展開後に解放する。
元X、密なW/dW、packed_atoms、全atom係数表、独立したdH配列は保存しない。
低メモリ用のH再計算切替は今後の実行方式であり、この初版ではZx/H保存を使う。

各threadは別atomを担当し、batch方向の和をregisterに持つ。
入力profileをv、出力profileをu、振幅をaとすると、H=Σ_i X_i v_i、G=Σ_o dY_o u_o。

```text
d_a    = sum_batch(H * G)
d_mu_i = sum_batch(a * G * dH/d_mu_i)
d_mu_o = sum_batch(a * H * dG/d_mu_o)
d_sigma= sum_batch(a * (G * dH/d_sigma + H * dG/d_sigma))
```

`backward.cu::parameter_grad`はZx/Zyから微分の縮約をregisterで計算し、元atom行へ一度だけ書く。
幅微分には既存effective_radiusの下限補正と、係数の面積正規化の商の微分を含める。
選択済みbank・ownerは固定して滑らかな枝を微分し、次のforwardで更新後の値を再分類する。
H/G作成は`contraction.cu::compute_h`の軸切替を共用し、dXは入力側Cを同じ展開kernelで評価する。

出力順layoutでは入力ownerの範囲が非連続なので、初版のC_inは非空groupのdescriptorを走査し、
該当する連続atom範囲だけを32atom shared chunkへ読み込む。atom値の転置コピーやdX atomicは使わない。
このgroup走査は高速化対象であり、完全stepの速度・メモリ達成は実測前に主張しない。
保存atomとlayout配列は`save_for_backward`で更新検査し、retained backwardに対応する。
二階微分とsortを含む全体Graph captureは未対応。

### H/C/Y接続とsigma整合の検証（現在の実装）

runtime source `d09eba32`。既存`ChebyshevSquareSpec`の式・sigma・半径下限を維持し、
分類をsigma<=T_hへ修正した。I=sigma/2の前半prototypeから意味を読み替えない。
既存固定bank版のソースとpublic APIは変更していない。

新規188件（CPU31/CUDA157）＋既存1816件をL4で実行し、**2004 passed、failed/skipped 0**。
新しいHは既存profileのFFT係数からセル積分したCPU参照へ照合し、
Cは同じ既存係数を使う独立のowner集約へ、Yは直接profile行列を掛けた参照へ別々に照合した。
さらに既存生密度を48点Gaussでセル積分した参照ともCPU上で照合した。
新しいCUDAの正の多項式を参照側へコピーして正しさを判定していない。

FP32/64、local/full bank、QがPを割り切らないchart端、最後の半セルのowner=0、
非ゼロ原点、幅境界と上限のdtype cast、振幅0、空atom/batch、batch1/3/4/5、
128/1024site、M16を含むH/C/Y、direct/merge Z、出力担当幅8/32を検査した。
検査許容差はH/CとFP64 Yがrtol2e-10/atol5e-11、FP32 Yがrtol2e-6/atol2e-7。
これは測定した最大誤差を意味しない。unit XのH=1、Yの総和も照合した。
compute-sanitizer racecheckの小さなunit-mass/cache検査は1 passed、0 hazards/errors/warnings。
Z→H→C→Yの低水準Graph replayは新しいXで照合したが、atoms/layoutは不変である。
replayではPythonのcache version検査が再実行されないため、atom更新を伴うGraph学習の検証ではない。

CPU全体2037 passed / 4721 skipped / 既存警告18件（45.09秒）、Ruff・format・差分検査・buildを通過。
wheel/sdistの336ファイルを現srcへbyte照合した。GPU snapshotの326コードと共通headerもgit/sourceへ照合。
共通headerのSHA256は`ccf5f0073d70fc16b4b44703e18cffcaac7dbcf669ebcbb50673d657d7dd7b13`。

128×128/A819/B4、FP32、M8/tail1e-3、入力T_min4/出力T_min8、Q128、spacing1、
sigma=linspace(0.2,40)、中心seed120の**forwardのみ**をfresh processで診断した。
warmup5/同期wall20標本、Graph/TF32なし、X検査・分類/sort・Z/H/C/Yを含み、
backward・Parameter勾配・AdamWを含まない。1fresh processの中央値は1.113883ms、
peak allocated105,472bytes（103KiB）/ reserved2MiB、baseline allocated15,360bytes。
独立runを複数回確認した採用値でも、完全stepのdense以下を検証した値でもない。
固定bank版の完全stepとはsigma分布・周期・処理範囲が異なるので改善率へ使わない。

| 配列 | このcaseのpayloadと場所 | 用途 |
| --- | --- | --- |
| atoms / X | global FP32、13,104 / 2,048bytes | 元Parameterと入力。atomの複製なし |
| rows / groups / offsets | global、計11,936bytes、224 groups | 元atom行への対応とグループ範囲 |
| Zx | global FP64[4,61,17]、33,184bytes | 入力側の階層moment |
| H | global FP64[4,819]、26,208bytes | slot順の完成H、振幅前 |
| C | global FP64[4,29,17]、15,776bytes | 出力bank/ownerの完成C |
| Y | global FP32[4,128]、2,048bytes | siteへ展開し、bank寄与を合計した出力 |
| atom係数 / amp*H chunk | block shared、dynamic5,376bytes＋static16bytes | 現在の32atomだけ。chunk間の同期後に再利用 |

payload和とallocatorピークは丸め・寿命・scratchが違うため同一視しない。
cuobjdumpでもC kernelのstatic shared16bytesを確認した。register/sharedとCUDA moduleは
Torch allocatorの指標外であり、全GPU processの使用量は未測定。
係数・中心位相はregisterで生成してshared chunkへ書き、atomごとの係数表をglobalへ作らない。
H/Cのglobal保存はblock間で受け渡すための基準実装。完全stepの保存／再計算・融合は今後比較する。

job `l4job-95f9b6adde8c4ccb982cf1abad4fac12`:

- source SHA256 `92450b0fc1effcd20675d1df9bd110b5a90d224558b1779ef8927d1463161cc4`
- result SHA256 `f18d87b69a993190b6ab469e470e98f1241c93150a0128f02b4196bb4b02680a`
- driver SHA256 `2622ae687e9c672889126fb62608a219afe0f2adc346a73c1dabbd2f7f5ace2f`
- binary SHA256 `2c1a30a9fbdce6f90d949013b73a35c8b13f78e1ce676706a8631381fc2a81ac`

manifest16ファイル、driver/binaryを照合。raw/source/result/spec/receiptはignored
`output/cell-mass-hc-20261027/first-run/`に保全。所有L4 runtimeの停止も確認した。
再検証は`python -m pytest tests/test_cell_mass_hierarchical.py -q`。
新規profileの数式待ちではなく、次は同じ式のG/dX/Parameter勾配を実装・検証する段階。

### 前半prototypeの検証・診断（sigma整合修正前の履歴）

runtime source `44c93543`。この節は撤回したsigma=全幅の分類での履歴であり、現在のH/C/Yの検証結果ではない。
既存`cell_mass/`からのimport・ソース変更はなかった。
新規146件（CPU分類27件＋CUDA119件）と既存のCUDA・数学1816件を一緒に実行し、
**1962 passed / failed 0 / skipped 0**。L4 / Torch2.11.0+cu130 / CUDA13.0。
FP32/64、空/端数atom、幅の境界、周期端、全周のみのplan、空/端数batch、
128/1024site、M2/4/8、Zのthread team 1/4/32を含む。
Zはセルごとの24点Gauss積分から独立に作ったCPU参照へ照合した。
親が同じharmonic番号の子を足してしまう誤りも専用caseで検出する。
Z構築だけのCUDA Graph capture/replayは新しいXで照合したが、layoutや完全stepのGraph検証ではない。
新経路のY/dX/Parameter勾配はまだ存在しないので、既存1816件をその検証として数えない。

CPU全体は2033 passed / 4683 skipped / 既存警告18件（46.18秒）。Ruff・format・差分検査を通過。
wheel/sdistをbuildし、両packageの334ファイルを現srcとbyte照合した。

診断条件はB4、atom数floor(0.05N^2)、sigma=linspace(0.2,100,A)、中心seed120、
Q128、入力T_min=4/出力T_min=8、M8。128ではspacing1とmin_tile_sites=4/8、
1024ではspacing0.125とmin_tile_sites=32/64を使う。
独立な各2fresh process、2巡目は逆順、warmup5/同期wall20標本、Graphなし。
計測範囲は**分類・sort・group作成＋Zx/Zyのみ**。H/C/Y、dX、Parameter勾配、AdamWを含まない。

| site数 | 前半のみ ms run0 / run1 | peak allocated | peak reserved | layout / Zx / Zy payload bytes |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.7623 / 0.7605 | 78,848bytes（77KiB） | 2MiB | 11,192 / 33,184 / 15,776 |
| 1024 | 3.9127 / 3.9113 | 3,422,208bytes（約3.264MiB） | 4MiB | 439,808 / 33,184 / 15,776 |

baseline allocatedは各17,408 / 871,936bytes。peakには入力atoms/X/dYと前半の一時領域を含む。
新しい完全stepのメモリがdense以下だと検証した値ではない。
これは前半の初版の診断で、速度最適化・完全step比較・Graphピーク・全GPU processメモリは未完了。
新幅定義・幅分布・処理範囲も旧測定と異なるので、旧版に対する同条件の改善率にはしない。
128のgroup数193、1024では849。各入出力階層のatom数もraw JSONへ保存した。

検証job `l4job-dc3552d6f4744d2590e234a3d6e53133`:

- source SHA256 `15dd1879f53498ff779d072abc2b4be5a2ed4d438be66956edf57978edc04181`
- result SHA256 `c9e79dfbb559b2f2c858184cf1006c01b98393423cf9635303f5f8560e3c120c`
- driver SHA256 `302bb2881d278550d4e7c50dfd9937a3f6ddd9aed3f900a1283519576816bb35`
- binary SHA256 `4f667530443b20cf271248fc792074049760b41a39cbd15398dcd8dcaf1e000c`

fresh process診断job `l4job-64d587a39ee3453c8b2c952a02eea6b8`:

- source SHA256 `bfce127c57b7f7bd0c3d6607b8ea593894f2e320e895c4738608e7b01f18e619`
- result SHA256 `b2778b0fd239741b6ebbdd80e834f63db329767e1b3707967e7b7ab0205033e3`
- driver SHA256 `39f313349f286c3bff7e251a1e33613358d0433604b3f3d4dd372bc055a90faf`
- binary SHA256 `ac74a4ab841b4ead035d72c164e69ce35ed1d4d777b05b6b6e3e454288ca1133`

両jobの325 runtimeコード、manifestの8/31ファイル、driver/binaryをhash/byte照合。
raw/source/result/spec/receiptはignored `output/cell-mass-hierarchy-20261027/verified/`と
`fresh-process/`へ保全する。検証直後の同一process診断は約8MiBの残存割当があり、
新しい前半の代表ピークには使わず、上記fresh processの値を使う。
最初のjob `l4job-b9af7d6fa60d419ca101075438e51808`はNinja不足でコンパイル前に失敗。
成功件数・時間へ含めず、archiveとverificationを同じ親directoryの`first-run/`へ保全した。
依存はjob内へ追加し、共有poolが所有runtimeを停止する。停止状態は終了後にも確認する。

再検証は`python -m pytest tests/test_cell_mass_hierarchical.py -q`。
新旧一緒の検証には`tests/test_cell_mass_coefficients.py tests/test_cell_mass_reference.py tests/test_cell_mass_cuda.py`も加える。
性能診断の再現は固定source archiveのdriverと空の`CST_JOB_OUTPUT`を使う。
共有L4ではpool submit/serve経由とし、新しいdriverを別profileの完全step測定へ流用しない。

現在のprofileは[1周期分のタイル窓](cell-mass-tile-window.ja.md)。
選んだwhole tileを共通のセル積分基底で処理し、支持境界での個別積分は行わない。
blockの担当site範囲は別の実行設定`sites_per_block`で分けられる。
固定bankの測定ではprofileのtile幅・周期を固定していた。
[担当site数の比較](cell-mass-block-sites.ja.md)は、その条件でのlaunch調整の記録。
現在の実コードには、Y/dXのatom集合を分担する`atom_splits`と、
H/G・Parameter勾配のblock幅を指定する`atom_threads_per_block`もある。
部分出力の寿命・合計kernel・係数sharedの具体的な実装は
[atom分担と位相共有](cell-mass-atom-splits.ja.md)を参照する。
実コードの投影微分を保存／再計算する分岐は
[projection cache](cell-mass-projection-cache.ja.md)、
メモリ制約の採否は[完全step比較](cell-mass-memory-frontier.ja.md)を参照。
現在の1bank runtimeでは、値検査とkey生成を融合しstable sortした行番号を保持する。
`owner_coefficient_reduction=True`は、その順序から(output owner,input owner)の
begin/endも作る。[owner C](cell-mass-owner-coefficients.ja.md)で係数を一度生成し、
[共有Z窓](cell-mass-owner-windows.ja.md)からH/Gを投影する。
最新の固定bank経路には[G/Parameter勾配とC_in生成の融合](cell-mass-cin-fusion.ja.md)、
[係数と微分のHorner評価](cell-mass-horner-and-sigma.ja.md)もある。
これらの算術・配列再利用は階層経路への移植候補であり、階層分類の代わりにはならない。

2026-10-26。[設計と数学契約](cell-integrated-profile.ja.md)に対応する、検討用の擬似コード。
以下の関数名・引数は一般化した擬似コード。明示的なChebyshevSquare密度から始めた
[初版の実コードと対応範囲](cell-mass-implementation.ja.md)は別に記録する。
全てのprofile族・複数bank分類・更新則まで実装済みとは扱わない。

最初はH/GとZx/Zyをglobalへ保存し、Parameter勾配まで再利用する構成で考える。
この保存量を基準に、HやZxの再計算・局所融合を後から比較する。
入力・出力128サイト、atom数128、batch 4から始める。thread数などは後で変更できる。

## 前準備と順序

```cpp
// forward
prepare_layout(atoms, spec, layout);
build_input_hierarchy(X, layout, Zx);
compute_h(Zx, X, atoms, layout, H);
compute_y(H, atoms, layout, Y);

// backward
build_output_hierarchy(dY, layout, Zy);
compute_g(Zy, dY, atoms, layout, G);
compute_dx(G, atoms, layout, dX);
compute_parameter_grad(Zx, Zy, X, dY, H, G, atoms, layout, grad);
```

同じstreamへ依存順に投入する。上の名前は論理段階であり、実際のlaunch数ではない。
owner方式のcompute_y/dxはC部分和を作るkernelと展開kernelに分かれる。
`fuse_g_parameter_grad=True`ではcompute_gと再計算型Parameter勾配を一つに融合し、
forward HはYで使用後に解放する。[融合経路](cell-mass-fused-pullback.ja.md)を参照。
階層構築は「signalから直接作る要素」と「子Zから継承する要素」に分ける。
前者はlevelをまたいで並列に作り、後者だけlevel順に起動して子の完成を待つ。
CPU側で毎回同期するという意味ではない。
以下ではgeometry・正規化設定などの引数を省略し、補助関数が同じspecを参照するものとする。
compute系ではソート済みの位置slotと、元のatom行番号original_idを区別する。
H/Gはslot順で保存する。atomの値は`atoms[layout.sorted_rows[slot]]`から読み、
Parameter勾配はoriginal_idの行へ書く。atom全体のソート済みglobalコピーは作らない。

## 起動するものとkernel内で呼ぶものの区別

| 名前 | 種類 | 役割 |
| --- | --- | --- |
| `prepare_layout` | host側の起動関数 | 以下の分類・ソート・配置・範囲作成を順に起動 |
| `classify_atom_keys` | CUDA kernel | 元のatom行から階層とソートキーを作る |
| `device_sort_pairs` | GPUライブラリ呼び出し | キーと元の行番号を一緒にソート |
| `mark_group_heads` | CUDA kernel | 同じキーの連続区間の先頭を検出 |
| `device_inclusive_scan` | GPUライブラリ呼び出し | 先頭フラグの累積和からグループ番号を作る |
| `write_group_begins` | CUDA kernel | 各グループのキー・beginとグループ数を書く |
| `write_group_ends` | CUDA kernel | 次のbeginから各グループのendを書く |
| `build_input_hierarchy` | host側の起動関数 | Xからの直接生成と、level順のマージを起動 |
| `build_output_hierarchy` | host側の起動関数 | dYからの直接生成と、level順のマージを起動 |
| `build_z_direct` | CUDA kernel | 全levelの直接生成要素を並列に作る |
| `merge_z_level` | CUDA kernel | 指定levelの継承要素を子Zから並列に作る |
| `compute_h` / `compute_y` | CUDA kernel | 以下の第1・第2節 |
| `compute_g` / `compute_dx` / `compute_parameter_grad` | CUDA kernel | 以下の第3〜第5節 |

各kernel節のコードブロックはkernelの本体を示す。
`assigned_atom`や`input_coefficient`などはkernel内のdevice補助関数で、追加のkernel起動ではない。
sort/scanの内部で使うkernel数はGPUライブラリに任せる。ライブラリ内部まで自作することは前提にしない。
host側でshapeから出力・scratch容量を確保し、同じstreamを全起動・ライブラリ呼び出しへ渡す。
以下の配列は明記しない限りglobalにある。

## prepare_layout：分類・並び替え・範囲の準備

目的はatomの意味を変えることではなく、実行用の順序と検索情報を作ること。
atom数A=128の例では、次の作業配列を最大A要素分用意できる。
これは一般化したsort/scanの説明用で、全配列を常設する要求ではない。
新しい初版は上記のライブラリsort/uniqueを使い、後続へ渡す順序・範囲だけを残す。
sort/scan用scratchはライブラリが要求する容量を別途数える。

| 配列 | 内容 |
| --- | --- |
| `keys[A]`, `source_rows[A]` | ソート前のキーと元のParameter行番号 |
| `sorted_keys[A]`, `sorted_rows[A]` | ソート結果 |
| `head_flags[A]`, `group_numbers[A]` | グループ先頭とその累積和 |
| `group_key[A]`, `group_begin[A]`, `group_end[A]` | 有効なグループだけを先頭から格納 |
| `group_count[1]` | 有効グループ数。GPU上で後続kernelが読む |

ここでsorted_rowsとgroup関連の出力はlayoutが参照するbufferであり、
擬似コード中の短い配列名はその別名とする。head_flagsやscan結果はprepareだけのscratch。
atomのsigma・中心・振幅は元のParameter配列にだけ置き、prepareでは複製しない。

```cpp
// host側。同じstreamへ順に起動する。
classify_atom_keys<<<...>>>(atoms, spec, keys, source_rows);
device_sort_pairs(keys, source_rows, sorted_keys, sorted_rows);
mark_group_heads<<<...>>>(sorted_keys, head_flags);
device_inclusive_scan(head_flags, group_numbers);
write_group_begins<<<...>>>(sorted_keys, head_flags, group_numbers, layout);
write_group_ends<<<...>>>(layout);
```

A=0ならhost側でprepareの空経路を選び、group_count=0をstream順に設定する。
以下のprepare kernelはA>0を対象とする。空のatom集合でも後段はY/dXへ必要な0を書く。
その場合はH/G/Parameter勾配のatom用kernelを起動せず、Y/dXの所有blockで0を保存する。
keysなどのscratchは最後の利用が済んだ後に再利用できるが、sorted_rowsと
グループ範囲などbackwardで使う状態はforward呼び出しごとに保持する。

### classify_atom_keys：各atomの所属とキーを作る

各threadが元の一つのatomを読む。ここではprofile値やHを計算しない。

```cpp
a = blockIdx.x * blockDim.x + threadIdx.x;
if (a >= atom_count) return;

level = classify_hierarchy(atoms[a], hierarchy_spec); // 入出力のbank番号の組
tile_shape = hierarchy_tile_shape(level);
q_out = center_tile_coordinates(atoms[a].output_center, tile_shape.output);
q_in  = center_tile_coordinates(atoms[a].input_center,  tile_shape.input);

keys[a] = make_lexicographic_key(level, q_out, q_in);
source_rows[a] = a;
```

キーは`(h_out[0], ..., h_out[Dout-1], h_in[0], ..., h_in[Din-1],
q_out[0], ..., q_out[Dout-1], q_in[0], ..., q_in[Din-1])`の辞書順。
全周bankには下記の専用番号を使う。tile座標はsite中心ではなくセル端を境界にする。
周期軸の中心のwrapとspacing/2の補正は、下記のqの式に従う。
多次元のキーの符号化はこの辞書順を保ち、packed整数のbit数超過を黙って切り捨てない。
同じキーのatomの順序は演算を変えない。元の行番号は対応付け用で、必須の数学的な分類軸ではない。

階層分類に使う幅・周期L・タイル寸法の関係は次節の規約を使う。
ここは従来未確定だった`classify_hierarchy`の設計。
新ディレクトリのprepareへ実装したが、H/C/Yを含むLinearへの接続は未実装。
中心の所属キーと、支持が重なる全タイルの集合は別である。

### sigmaから階層と周期を決める規約

#### 既存profileとsigmaを維持する

この実装は既存の`ChebyshevSquareSpec`を使う。profileの式や係数が未定だったわけではない。
`torch/profiles/fourier.py`の`effective_radius`、`kernel_coefficients`と、固定bank CUDAの
正の係数多項式を基準とする。profile族を選び直したり、sigmaを別の物理量へ変換したりしない。
Kaiser/DPSS等も既存の式はあるが、このCUDA初版の対象はChebyshevSquareだけである。

sigmaは従来通り**幅制御パラメータ**であり、全幅・標準偏差・支持半幅そのものではない。
先の文書／前半prototypeへ追加した`I=sigma/2`は撤回し、分類も修正する。
Iは既存式による集中半径とする。P=2L、s=P/3、epsilon=tail_bound、M=modesとして、

\[
a=\operatorname{acosh}(\epsilon^{-1/2})/M,\quad
r_0=\frac P\pi\arccos(1/\cosh a),
\]
\[
I(\sigma,P)=\left[r_0^4+\left(1-(r_0/s)^4\right)\sigma^4\right]^{1/4},
\quad\beta=\sec(\pi I/P),
\]
\[
g(t;\sigma,P)=\left[\frac{T_M(\beta\cos(\pi t/P))}{T_M(\beta)}\right]^2.
\]

半径下限を含めて既存式をそのまま使う。これは非負な有限Fourier密度そのものであり、
新しいcompact目標profileを作ってfitする処理は今回追加しない。
Iの外の裾と、選んだwhole-tile窓の外をゼロにする支持マスクは区別する。
以前のrho=sigma/(2*d)は幅制御値の比率として保持できるが、I/dと同じ量ではない。
rho<1だけで常に1siteになるとも仮定しない。

#### T_minからbankを選ぶ式

各軸はN sites、間隔d、chart周期Q=N*dを持つ。最小tileのsite数n_minとN/n_minは
2のべき乗とし、T_min=n_min*d、T_h=2^h*T_minとする。
local bankではP_h=2L_h=3T_hを固定する。
既存profileの実現可能性条件r_0<T_hの下では、

\[
I(\sigma,3T_h)\le T_h\quad\Longleftrightarrow\quad\sigma\le T_h.
\]

実際、I^4=r_0^4+(1-(r_0/T_h)^4)*sigma^4なので、I=T_hのときsigma=T_h。
したがって分類のために全atomのprofileを評価する必要はなく、

\[
h_a^{\rm raw}=\min\{h\ge0:\sigma_a\le T_h\}
=\max\left(0,\left\lceil\log_2\frac{\sigma_a}{T_{\min}}\right\rceil\right),
\quad T_a=2^{h_a}T_{\min},\quad L_a=\frac32T_a.
\]

等号は小さい階層側。h=0は0<sigma<=T_min、h>=1はT_h/2<sigma<=T_hを受け持つ。
実装はlog2丸めの代わりにsigma>Tの2倍更新で決める。sigmaをTへ丸めず、
係数生成へ元の値を渡す。T_min=4ならsigma範囲(0,4]/(4,8]/(8,16]は
h=0/1/2、T=4/8/16、L=6/12/24となる。

以前の式はbankの集中半径上限I_h_max=T_hに対して
2L_h=(ceil(2*I_h_max/T_h)+1)*T_h=3T_hとなる。同じbankでPと基底を共有する。
T_minは`sites_per_block`等のlaunch幅とは別。既存native profileはPも数学的な引数なので、
T_minを変えることは同じWの単なるメモリ配置変更ではない。異なる設定を同一条件の速度比較へ混ぜない。

#### 中心tile、全周bank、有効幅範囲

3T_h<=Qなら中心tileと左右1tileを使う。u=(mu-origin) mod Q、N_h=Q/T_hとして、

\[
q_{\rm raw}=\left\lfloor\frac{u+d/2}{T_h}\right\rfloor,\qquad
q=q_{\rm raw}\bmod N_h.
\]

窓のunwrappedなセル範囲は[(q_raw-1)*T_h-d/2,(q_raw+2)*T_h-d/2)。
I<=T_hなので中心がtile端へ寄っても集中領域を覆う。窓内は全セルを同じ基底で積分し、
Iの境界に合わせたcellごとのclipは入れない。

3T_h>Qなら切り詰めず全周bankへ移す。専用level H=log2(N/n_min)、T_full=Q、
K_full=1、P_full=2L_full=Q、q=0とする。全siteを一度だけ扱う。
この場合もprofile内部のsはP_full/3であり、tile幅Qに置き換えない。
既存profileの有効範囲に揃え、0<sigma<=min(Q_in,Q_out)/3を受け付ける。
上端の比較は既存Torch/固定CUDAと同じく入力dtypeへboundをcastしてから行う。
以前の前半prototypeが許したsigma<Qまでの範囲は撤回し、暗黙にclipしない。

```cpp
classify_axis(sigma, center, axis) {
    h = 0;
    T = axis.T_min;
    while (sigma > T) { T *= 2; h += 1; }
    if (3 * T > axis.Q)
        return full_bank(axis.H, T=axis.Q, L=axis.Q/2, q=0);
    u = positive_mod(center - axis.origin, axis.Q);
    q_raw = floor((u + axis.spacing/2) / T);
    return local_bank(h, T, L=1.5*T, q=positive_mod(q_raw, axis.Q/T));
}

// finiteと0<sigma<=min(Qin,Qout)/3を検査してから分類する。
out = classify_axis(atom.sigma, atom.output_center, output_axis);
in  = classify_axis(atom.sigma, atom.input_center, input_axis);
key = (out.h, in.h, out.q, in.q);
```

原点・spacing・T_minが異なれば入出力のhも異なってよい。多次元では各軸へ同じ規則を適用する
設計だが、現CUDAは1入力軸/1出力軸のみ。small sigmaも同じprofileの半径下限を維持する。
直接計算はその同じprofileのセル積分を評価する高速化候補で、別の狭い密度へ差し替える意味ではない。

#### 既存係数のセル積分、面積正規化、H/C

既存密度の複素Fourier係数をc_kとすると、

\[
g(t)=c_0+2\sum_{k=1}^{M}c_k\cos(\omega_k t),\quad
\omega_k=2\pi k/P,\quad S=P c_0,
\]
\[
F(t)=c_0t+2\sum_{k=1}^{M}\frac{c_k}{\omega_k}\sin(\omega_k t),\qquad
\phi_a(i)=\frac{F(b_i-\mu_a)-F(a_i-\mu_a)}{P c_0}.
\]

係数は既に定義済みであり、新しく学習・fitするものではない。通常実行では同じ係数を
正の多項式からregister内で生成し、中心位相とSによる正規化を適用する。
T_M(beta)^-2は全c_kに共通なので、c_k/(P*c_0)から消去できる。
全atomの係数表やL2計算passは作らない。

セル積分のJ_k=2*sin(omega_k*d/2)/omega_kはZと展開基底に含める。
atom側の係数ベクトルはp_0=1/P、p_cos=2*c_k/(P*c_0)*cos(omega_k*mu)、
p_sin=2*c_k/(P*c_0)*sin(omega_k*mu)。そのため、

```cpp
H[b,slot] = sum(retained_tile,mode, input_coeff(atom,mode) * Zx[b,tile,mode]);
C[b,owner,mode] = sum(atom_in_owner, amplitude[atom] * H[b,slot] * output_coeff(atom,mode));
Y[b,site] = sum(bank,covering_owner,mode, C[b,owner,mode] * integrated_basis(bank,mode,site));
```

となる。amplitude=0でもHを省略しない。C同士は異なるPのまま混ぜず、展開したsite値を加算する。
初版のHはglobal FP64[batch,A]でslot順。Cはglobal FP64[batch,output_nodes,1+2M]の小さなowner表。
C blockは入力bankごとにgroup prefixを二分探索し、対象atom区間を取得する。
32atomの係数とamplitude*Hだけをsharedに置き、使用後の同期を挟んで次chunkへ再利用する。
一つのC要素・Y要素に書き手を一つ割り当て、出力atomicやblock間spin待ちは使わない。
Z/H/Cをglobalに渡す初版は検証用の基準。H/C融合や再計算は、同じ式のまま後で比較する。

chart端では窓と中心を同じ座標像へ移す。owner=0で中心が最後の半セルにある場合はmuを-Qずらす。
Zを読むtileの像がx+jQならcos/sin momentをomega*jQだけ回転する。
Cの展開も同じ像で評価する。QがPを割り切るとは仮定せず、同じsiteを二重計上しない。

#### 周期が変わるZの継承と、残る検証

local bankでP_h=2P_(h-1)なので、共通原点・spacingでは

\[
\omega_{h,2j}=\omega_{h-1,j},\qquad
Z_{h,t,b,2j}^{\cos/\sin}
=Z_{h-1,2t,b,j}^{\cos/\sin}+Z_{h-1,2t+1,b,j}^{\cos/\sin}.
\]

親の偶数harmonic（j=1,...,floor(M/2)）と定数modeを継承し、奇数harmonicは直接生成する。
全周bankはP=Qの別基底なので初版は直接生成する。atomがいないbankの子要素も親が使うなら準備する。

- 実装: 既存profileとsigmaの分類、group offsets、bank別Z、H、C、Y。検証範囲は冒頭の表と記録で追跡する。
- 未実装: 階層版G/dX/Parameter勾配、公開CSTLinear/dispatcherへの統合。
- 継続検証: branch境界の誤差、完全step時間とメモリ、直接計算/融合等の速度比較。

整数の階層・owner選択は微分せず、選択中のbankのprofileを微分する既存方針を維持する。
同じ式でもPを変えるとprofile値は変わり、階層遷移で関数が連続とは限らない。
forwardのlayoutをbackwardへ引き継ぎ、optimizer後の次forwardで再分類する。
新しいprofile式を待つことをH/C実装の条件にせず、既存密度の独立oracleで各段階を検証する。
完全stepでのY/dX/全atom勾配・更新と、dense以下のallocated/reserved、速度目標を同条件で確認する。

### device_sort_pairs：キーと元の行番号を一緒に並べ替える

```cpp
// GPU sortライブラリへ渡す意味。内部kernelはライブラリに任せる。
(sorted_keys, sorted_rows) = sort_by_key(keys, source_rows);
```

atomのParameter本体やoptimizerのmomentsを並べ替える操作ではない。
多wordキーを使う場合も辞書順を維持できるsort経路を選ぶ。

### mark_group_heads：同じキーの区間の先頭に印を付ける

```cpp
s = blockIdx.x * blockDim.x + threadIdx.x;
if (s >= atom_count) return;

if (s == 0)
    head_flags[s] = 1;
else
    head_flags[s] = (sorted_keys[s] != sorted_keys[s - 1]);
```

その後にinclusive scanを行う。例えば`head_flags=[1,0,1,0]`なら
`group_numbers=[1,1,2,2]`となり、0始まりのgroup番号は`group_numbers[s]-1`。

```cpp
group_numbers = inclusive_sum(head_flags);  // GPU scanライブラリ呼び出し。
```

### write_group_begins：groupごとのbeginを記録する

```cpp
s = blockIdx.x * blockDim.x + threadIdx.x;
if (s >= atom_count) return;

if (head_flags[s]) {
    g = group_numbers[s] - 1;
    layout.group_key[g] = sorted_keys[s];
    layout.group_begin[g] = s;
}
if (s == atom_count - 1)
    layout.group_count = group_numbers[s];
```

beginとgroup_countの書き手はそれぞれ一意。全beginが完成するのをkernel境界で待つ。

### write_group_ends：次のgroupのbeginからendを決める

```cpp
g = blockIdx.x * blockDim.x + threadIdx.x;
if (g >= layout.group_count) return;

if (g + 1 < layout.group_count)
    layout.group_end[g] = layout.group_begin[g + 1];
else
    layout.group_end[g] = atom_count;
```

`[begin,end)`は同じキーのatomの連続区間。全区間を合わせるとソート済みatomを一度ずつ覆う。
gridは最大Aグループ分で起動でき、group_countをCPUへコピーして起動数を決める必要はない。

### blockが使うrangeはどう取得するか

出力タイルや入力タイルに寄与するatomが、一つの連続区間に全て収まるとは限らない。
各blockは必要なgroupの`[begin,end)`を順に読む。この範囲はsorted_rows上の位置であり、
元のatomパラメータがglobalで連続していることは要求しない。
最初の128atomの説明形では、groupを走査するdevice補助関数から始められる。

```cpp
for (g = 0; g < layout.group_count; ++g) {
    if (!group_can_contribute(layout.group_key[g], target_tile, side))
        continue;
    consume_range(layout.group_begin[g], layout.group_end[g]);
}
```

これは追加のkernelではない。`group_can_contribute`は階層ごとの最大支持幅を用いる
保守的な候補判定で、中心が対象タイルの外にあるgroupも必要なら含める。
支持幅の上限を保証できないgroupは、この判定で除外しない。
支持が周期境界をまたいでも、同じgroupの区間を二重に消費しない。
候補に含まれるが実際の寄与が0のatomを含めて処理してよい。範囲を狭めるためのatom別判定は必須にしない。
group検索の索引は後で改善できるが、タイルごとのatomパラメータ全体の複製は前提にしない。

### load_atom_chunk_together：必要なatomだけsharedへ集める

追加のkernelではなく、Y/dXなどのblock内で呼ぶdevice補助関数。
同じatomを複数threadで再利用するときに、そのblockが必要な小さなchunkだけを集める。

```cpp
// rangeとchunkのループはblock内で共通。全threadが参加する。
for (base = range.begin; base < range.end; base += ATOM_CHUNK) {
    count = min(ATOM_CHUNK, range.end - base);
    for (j = threadIdx.x; j < count; j += blockDim.x) {
        original_id = layout.sorted_rows[base + j];
        shared_atoms[j] = atoms[original_id];
    }
    __syncthreads();

    consume_atom_chunk(shared_atoms, count, base, H_or_G);
    __syncthreads();  // 全員が使い終わってから次のchunkで上書き。
}
```

shared_atoms[j]はソート順に並び、対応するH/Gの行は`H_or_G[b,base+j]`。
実装では必要なparameter成分だけをロードしてよい。shared全量はATOM_CHUNKと成分数で制限する。
Cの担当がないthreadもchunkロードと全barrierへ参加する。
Cを保持するshared領域とは、同時に必要な間は別の領域を使い、容量を両方数える。
globalからの最初のgatherは飛び飛びになり得る。sharedでの再利用とロード・同期の費用を測る。
block間でsharedを共有することはできず、別blockはそれぞれ必要なchunkをロードする。

一つのthreadが一つのatomだけを使うH/G・Parameter勾配では、sorted_rowsを介して元のatomを
直接registerへ読む。共有する相手がいなければsharedを経由する必要はない。
いずれの場合も、Fourier係数やその微分はsigma・中心・bankから利用時に計算する。
全atom分の係数表や係数微分表をglobalへ用意しない。

## build_input_hierarchy：XからZxを作る起動関数

以前の`build_z_level`には、signalを読む処理と子Zを読む処理が一つに入っていた。
ここを分ける。二分マージを出発点にする。
nodesはタイルのsite範囲・bank・mode・子ノードを記述する実行設定で、atom値を含むZとは別。
最初は宣言済みのbankとlevelの全ノードを作る単純な構成とし、必要ノードだけへの限定は後で考える。

### 並列な部分と、依存で順序が必要な部分

| 処理 | 並列にできる範囲 | 待つ相手 |
| --- | --- | --- |
| signalからZを直接生成 | 異なるlevel・タイル・batch・mode | 入力XまたはdYが利用可能になることだけ |
| 一つのZへのsiteの加算 | siteを複数threadに分担し、部分和を作る | 部分和を合わせるときの担当thread |
| 子Zから親Zを継承 | 同じlevelの異なる親・batch・mode | その要素で使う子Zの完成 |

**親が子を使う依存は残るが、全levelの全処理が直列という意味ではない。**
直接生成するmodeは、子Zがなくても作れる。最初に全level分をまとめて起動できる。
マージのlevel単位の完了待ちは、必要な子ごとの待機より粗いが、安全で単純な基準になる。

基底設定から、各`(node,mode)`を直接生成か継承のどちらかへ一度だけ割り当てる。
`direct_jobs`は直接生成するmodeをnodeごとに束ねた記述、`merge_jobs[level]`は
継承する要素と子modeの対応を記述する。GPUで空きthreadを監視するjob管理ではなく、固定の担当表である。
同じZ要素を両方のkernelから書かない。これはatomの支持分類とは別の、基底間の対応である。

```cpp
// 並列部分：直接生成する要素は全level分をまとめて処理。
if (has_input_direct_jobs)
    build_z_direct<<<input_direct_grid, threads>>>(X, input_direct_jobs, Zx);

// 依存部分：子の完成後に次のlevelを起動。各起動の内部は並列。
for (level = 1; level < input_level_count; ++level) {
    if (has_input_merge_jobs(level))
        merge_z_level<<<input_merge_grid(level), threads>>>(
            input_merge_jobs[level], Zx);
}
```

forはCPU側から起動順序を記述するもの。タイルをCPUの一つのthreadで計算しているわけではない。
各起動を同じstreamへ投入し、CPU側で毎levelのGPU完了を待つ呼び出しは入れない。
例えば128サイト、最小タイル幅16なら、二分マージのノード数は8→4→2→1。
マージでは4親を並列→2親を並列→1親へ進む。どのmodeを直接生成するかは基底の継承条件による。
これは構造の例で、最小幅16や各levelの周波数・mode数を固定する決定ではない。

## build_output_hierarchy：dYからZyを作る起動関数

同じ処理にdYと出力側のタイル・基底設定を渡す。
入力側と出力側で次元、spacing、タイル寸法、bankが異なっていてよい。

```cpp
if (has_output_direct_jobs)
    build_z_direct<<<output_direct_grid, threads>>>(dY, output_direct_jobs, Zy);

for (level = 1; level < output_level_count; ++level) {
    if (has_output_merge_jobs(level))
        merge_z_level<<<output_merge_grid(level), threads>>>(
            output_merge_jobs[level], Zy);
}
```

Zyはbackwardで新しく作る。Zxを入力にしてZyを作るという意味ではない。

## build_z_direct：site方向も複数threadで並列に足す

以前のコードも、タイル・batch・modeは並列だった。ただし、各Z要素のsite方向の加算は
一つのthreadに任せていた。ここでは一つのZをP個のthreadで分担する案を示す。
これはatomを複数threadで処理する話ではなく、複数atomが再利用するZの生成である。

一つのblockは一つの直接生成nodeとbatch範囲を担当する。
Pは調整可能な2のべき乗で、blockDim.xを割り切る値とする。warp単位に固定しない。
例として128thread、P=4なら、32個のZ要素を同時に処理できる。

| thread | 担当 | 分担するsiteの番号 |
| --- | --- | --- |
| 0〜3 | 一つ目の`(batch,mode)` | 0,4,8,… / 1,5,9,… / 2,6,10,… / 3,7,11,… |
| 4〜7 | 二つ目の`(batch,mode)` | 同じ分割を次のZ要素に適用 |
| … | … | … |
| 124〜127 | 32個目の`(batch,mode)` | 同じ分割を次のZ要素に適用 |

batch 4・直接生成mode 8なら32個のZなので、この例では全128threadに担当がある。
要素が32個より多い場合は、次の波でまた32個を並列に処理する。
下のteamはP個のthreadをまとめるソフトウェア上の番号であり、新しいCUDAの実行単位ではない。

```cpp
job = direct_jobs[blockIdx.x];  // 異なるlevelのnodeも同じgridに含む。
team = threadIdx.x / P;
lane = threadIdx.x % P;
team_count = blockDim.x / P;
work_count = batch_size * job.direct_mode_count;
__shared__ Scalar partials[THREADS];

// 全threadが同じ回数繰り返す。各wave内の全teamは並列。
for (wave = 0; wave < ceil_div(work_count, team_count); ++wave) {
    q = wave * team_count + team;
    (b, mode) = decode_direct_job(q, job, blockIdx.y);
    valid = (q < work_count && b < total_batch_size);

    partial = 0;  // 各threadのregister。
    if (valid) {
        for (i = lane; i < job.site_count; i += P) {
            site = job.site_at(i);
            partial += signal[b, site] * integrated_basis(job.node, mode, site);
        }
    }
    partials[threadIdx.x] = partial;
    __syncthreads();

    // teamごとの部分和を並列reduction。P=4なら2段。
    for (offset = P / 2; offset > 0; offset /= 2) {
        if (lane < offset)
            partials[threadIdx.x] += partials[threadIdx.x + offset];
        __syncthreads();
    }
    if (valid && lane == 0)
        Z[job.node, b, mode] = partials[threadIdx.x];
    __syncthreads();  // 次のwaveでsharedを再利用する前の同期。
}
```

`decode_direct_job`は範囲外qでも不正な配列を読まず、valid=falseとなる番兵を返す。
担当のないthreadもpartial=0を書き、全barrierへ参加する。途中return/continueはしない。
siteのループはthreadごとにsite数/P程度の加算をするものであり、全siteを一人で走査しない。
足し算の依存は各thread内の部分和と、log2(P)段のreductionに残る。

`signal`は入力側ならX、出力側ならdY。signalはglobal、部分和はregister/shared、完成Zはglobal。
shared容量はこの例でTHREADS個のScalar分。128thread・FP32なら512 byteであり、
タイル全体や全levelのZをsharedに置くという意味ではない。
同じsignalをmode間で共有ロードする追加案もあるが、この擬似コードではまだ導入しない。
P=1なら以前と同様に一人で一つのZを作る割り当てになる。短いタイルやmode数が多い場合は
Pを増やさない方がよい可能性もあり、reductionと同期の費用を含めて比較する。

## merge_z_level：子Zの完成後、親の各要素を並列に作る

ここではsiteを走査しない。二つの子の完成値を足すだけなので、一つの親Z要素を
一つのthreadへ割り当てる。他の親・batch・modeは別threadで並列に進む。

```cpp
q = blockIdx.x * blockDim.x + threadIdx.x;
if (q >= merge_jobs.count * total_batch_size) return;
(job, b) = decode_merge_job(q, merge_jobs);

left = Z[job.left_child, b, job.left_mode];
right = job.has_right_child ? Z[job.right_child, b, job.right_mode] : 0;
Z[job.parent, b, job.parent_mode] = left + right;
```

子Zは先に起動したkernelで完成済みなので、ここにblock内barrierは不要。
親要素の書き手も一つ。次のlevelはこのkernelの完了後に読む。
rootが一つになってもbatch/modeは並列にできるが、上位ほど仕事が少なくなる点は残る。
この縮小した並列度や起動費用は、後でlevel融合や複数nodeの同一block処理と比較する。

## セル積分基底と継承条件

`integrated_basis`はsiteの担当セル全体を積分した基底値。例えば共通原点のcos基底なら、

```cpp
if (omega == 0) return spacing;
return 2 * sin(omega * spacing / 2) / omega
       * cos(omega * (site_position - common_origin));
```

sin基底も同じ積分係数にsinを掛ける。多次元の積基底なら軸ごとのセル積分を掛ける。
ここにはatomの中心・幅・振幅を混ぜない。選択する領域はwhole tileのみなので、セル境界の補正は不要。

`can_inherit_same_basis`は、子タイルのsite集合が親を重複なく覆い、該当modeの周波数、
sin/cosの種類、座標原点、spacing、セル積分係数が一致するときだけtrueにする。
葉ノードや一方の子で該当modeがない場合はfalse。原点が異なる場合の位相変換は、この単純な継承には含めない。
親を二つの子から作ることだけでは、基底まで一致するとは限らない。
一致しないmodeはdirect_jobsへ、一致するmodeだけmerge_jobsへ入れる。
全levelの直接生成を先に行っても、継承できる要素までsignalから作り直すわけではない。
必要なZの総要素数も変えない。signalの再読込は残り、処理順によるcacheの効き方や
同時に扱う作業範囲は変わり得る。速さは未測定であり、元のlevel単位の混合kernelも比較候補として残す。

## 1. compute_h：入力側の全寄与をatomごとにまとめる

blockはatom群とbatch範囲を担当する。各threadが別々のatomを担当し、
そのatomに必要な全入力タイルを足す。最初の例では128thread、各threadに1atom、batch_size=4。
Hは振幅を掛ける前の値。Zxはglobalから読み、Hの累積値はregisterへ置く。

```cpp
slot = assigned_atom(blockIdx.x, threadIdx.x);
if (slot >= atom_count) return;
original_id = layout.sorted_rows[slot];
atom = atoms[original_id];  // 必要なparameterをregisterへ読む。
h[batch_size] = {0};

for (tile : full_input_tiles(atom)) {
    for (mode : tile.modes) {
        c = input_coefficient(atom, tile, mode);
        for (b : assigned_batches(blockIdx.y, batch_size))
            h[local_batch(b)] += c * Zx[b, tile, mode];
    }
}
for (b : assigned_batches(blockIdx.y, batch_size)) {
    H[b, slot] = h[local_batch(b)];
}
```

`full_input_tiles`は選択した1周期分の窓を重複なく覆うwhole tile集合。親とその子を二重に含めない。
modeのループは`sum_k input_coefficient(atom,tile,k) * Zx[b,tile,k]`を計算する。
係数には中心・幅と面積による正規化を含め、必要なmodeごとにregisterで評価し、batch間で再利用する。
窓の外はタイル単位でゼロ扱いする。窓内は共通の積分基底を使い、端セルの直接積分は行わない。

この案では異なるthreadのHを足す必要はない。thread自身が必要な入力寄与を全て足す。
sharedへの共同ロードやblock内同期はまだ入れず、完成したHを一度globalへ書く。
後に共同ロードを導入する場合は、担当atomがないthreadも必要な同期へ参加させる。

## 2. compute_y：担当出力タイルのCを作り、Yまで完成させる

blockは出力タイルとbatch範囲を所有する。Yを0から累積し、最後に一度globalへ書く。
一つのY要素に対する全グループの寄与をこのblockが集める。

```cpp
output_tile = assigned_output_tile(blockIdx.x);
init_output_accumulator(y);  // 各threadの担当site/batchは重ならない。

for (bank : contributing_output_banks(output_tile)) {
    build_output_c_together(bank, output_tile, H, atoms, layout, shared_c);
    __syncthreads();
    expand_output_c_into_y(bank, output_tile, shared_c, y);
    __syncthreads();  // 全員が読み終えてからshared_cを再利用する。
}
store_output_tile(output_tile, y, Y);
```

内部部分でのCは`C[b,k] = sum_a alpha[a] * H[b,a] * output_coefficient(a,tile,k)`。
`build_output_c_together`はCの各batch/modeに書き手を一つ決め、対応atomを走査して完成値を
sharedへ書く方式を出発点にする。その走査中、必要なatomを上記のchunkロードでsharedへ集める。
係数を担当するthreadはchunk内のatomから係数をその場で算出する。
atomごとの部分Cを並列生成してreductionする案は後で比較する。
係数を作るthread配置と、Yのsiteを展開するthread配置は異なってよい。

bankごとに基底が異なるため、C同士を混ぜず、展開したsite値をyへ足す。
窓内の全寄与をCで表すので、境界セルの追加処理はない。
bankのループと二つのbarrierには全threadが参加し、担当要素がないthreadも途中returnしない。
globalのHは読み取り専用、Cはshared、yは各threadのregisterを基本とする。
各threadが持つyの個数は担当タイル寸法とbatch範囲で決まり、容量に応じて変更する。

## 3. compute_g：dY側でHと対称の集約を行う

atom群とbatch範囲を担当し、各threadが別々のatomを処理する。
Zyを読み、振幅を掛ける前のGを完成させてglobalへ書く。

```cpp
slot = assigned_atom(blockIdx.x, threadIdx.x);
if (slot >= atom_count) return;
original_id = layout.sorted_rows[slot];
atom = atoms[original_id];
g[batch_size] = {0};

for (tile : full_output_tiles(atom)) {
    for (mode : tile.modes) {
        c = output_coefficient(atom, tile, mode);
        for (b : assigned_batches(blockIdx.y, batch_size))
            g[local_batch(b)] += c * Zy[b, tile, mode];
    }
}
for (b : assigned_batches(blockIdx.y, batch_size)) {
    G[b, slot] = g[local_batch(b)];
}
```

`compute_h`と入出力の役割が逆になる。選ぶタイル窓と係数の定義は、
forwardで使う同じprofileに対応させる。G全体のほかにdH全体を保存することは前提にしない。

## 4. compute_dx：担当入力タイルの係数を作り、dXまで完成させる

blockは入力タイルとbatch範囲を所有する。`compute_y`の入力側版として構成する。

```cpp
input_tile = assigned_input_tile(blockIdx.x);
init_input_gradient_accumulator(dx);

for (bank : contributing_input_banks(input_tile)) {
    build_input_c_together(bank, input_tile, G, atoms, layout, shared_c);
    __syncthreads();
    expand_input_c_into_dx(bank, input_tile, shared_c, dx);
    __syncthreads();
}
store_input_tile(input_tile, dx, dX);
```

内部部分の係数は`C_in[b,k] = sum_a alpha[a] * G[b,a] * input_coefficient(a,tile,k)`。
振幅は一度だけ掛ける。bankごとに展開して担当dXを完成させる。境界セル補正は不要。
Gはglobal、集約したC_inはshared、dxはregisterを基本とする。barrierの規則は`compute_y`と同じ。
atomのロードもYと同様に、元配列から必要なchunkだけをsharedへ集める。
Y/dXの各要素を一つのblockが所有するため、block間の出力加算atomicは不要にできる。

## 5. compute_parameter_grad：H/Gと微分を組み合わせる

各threadが一つのatomの全batchを担当し、そのatomのParameter勾配を完成させる基準案。
batch分割で複数blockが同じ勾配を書く方式は、追加reductionを伴う別候補にする。

```cpp
slot = assigned_atom(blockIdx.x, threadIdx.x);
if (slot >= atom_count) return;
original_id = layout.sorted_rows[slot];
atom = atoms[original_id];

grad_amp = 0;
for (b : all_batches) {
    grad_amp += H[b, slot] * G[b, slot];
}
store_original_amp_grad(original_id, grad_amp, grad);

for (parameter : profile_parameters(atom)) {
    grad_parameter = 0;
    for (b : all_batches) {
        dh = input_parameter_derivative(atom, parameter, b, Zx, X);
        dg = output_parameter_derivative(atom, parameter, b, Zy, dY);
        grad_parameter += atom.amplitude *
            (G[b, slot] * dh + H[b, slot] * dg);
    }
    store_original_profile_grad(original_id, parameter, grad_parameter, grad);
}
```

ここで`dh/dg`はそれぞれ`partial H / partial parameter`と`partial G / partial parameter`。
逆伝播の中間勾配`dH = alpha * G`とは別である。
入力側だけに効くparameterならdg=0、出力側だけならdh=0、両側に効くなら両方を含める。

微分関数は、内部の`sum partial_coefficient * Z`と残りのセルの正確な微分を合計する。
面積正規化の微分も含める。H/Gだけからこの情報を復元できるとはしない。
Zx/Zyは読み取り専用のglobal、gradの累積値はregisterへ置く。
元のParameter行への対応はlayoutで戻し、optimizerによる更新はここでは行わない。

このコードはparameterごとにZを読む単純な説明形である。同じZを読んだときに複数parameterの
微分を同時に累積すれば読み取りを共有できるが、register使用量との交換になる。
幅のtask VJP・更新則、支持境界・階層遷移の微分は数学契約を確定してから実装する。

## 確認すること

各段階の保存値、配列のshape、書き手、寿命を先に確定し、独立oracleで
Y/dX/全atom勾配を検証してから速度を比較する。上記は実機での正しさ・性能の結果ではない。
H/Gの保存を省く案では、同じforward状態での再計算と、全利用先の終了を明示する。
