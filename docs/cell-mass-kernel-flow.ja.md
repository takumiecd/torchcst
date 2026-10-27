# 区間確率方式：prepareからbackwardまでのkernelの流れ

## 設計の正本と、現在の実装範囲

2026-10-27。この文書を、sigmaからの階層分類・layout・Z/H/C・backwardを
つなぐ設計の正本とする。分類規約は下の
[sigmaから階層と周期を決める規約](#sigmaから階層と周期を決める規約)に置く。
他の実測ノートは、その時点の実装と測定の記録であり、この設計が全て実装済みという意味ではない。

数式は[sigmaと周期の規約](#sigmaから階層と周期を決める規約)、処理順は
[起動とkernel内関数の区別](#起動するものとkernel内で呼ぶものの区別)、
実測は[最新の再確認](#最新の再確認完全stepと1024のadamw状態)、
設定は[現在使う研究設定](#第7段までの独立再確認と現在使う研究設定)を参照する。

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
| 面積正規化・backward | 既存密度・面積式と解析微分を再利用。階層版backwardは独立参照へ照合済み | 固定bankの独立oracle照合完了 |

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
| `executor.prepare_layout` | 分類とsortまたはcounting。元行番号、groupの(h_out,h_in,q_out,q_in)、offsets、任意の逆slot indexをcacheへ残す | CUDA実装済み、元行・group範囲を実GPU照合済み |
| `preparation.cu::build_hierarchy` | bank別の周期のZ。直接生成と偶数harmonicの子からの継承。完成Zはglobal FP64 | CUDA実装済み、direct/mergeを独立積分参照へ照合済み |
| `contraction.cu::compute_h` | 各threadが別atomのHをFP64計算し、指定dtypeで保存。係数と面積はregister | 実装済み、既存profileの独立参照へ照合済み |
| `compute_c` / `expand_c` | 32atom shared chunkからowner Cを作り、bank別の基底で展開後にYへ加算 | 実装済み、CとYを別々に照合済み |
| G/dX・Parameter勾配 | 同じprofile微分と選択中のlayoutを使う。各threadが別atomを担当 | 実装済み、全atom勾配とAdamWを実GPU照合済み |

新しい前半は空の雛形でなく、`prepare_layout(plan, atoms)`と
`build_hierarchy(signal, axis, method="merge")`を実行するCUDA入口。
直接生成だけの`method="direct"`も比較に使える。
研究用の`forward_only`はprepare→Z→H→C→Yを呼び、detached入力の診断用として残す。
新しい`hierarchical_linear`は明示的なautograd bindingを持つ。公開CSTLinear/dispatcherへの登録はない。

新しいlayout cacheにatom値や全atomキーは保持しない。基本のTensorは
`rows[A]`（int64）、`groups[G,4]`（int32）、`offsets[G+1]`（int64）。
countingでは任意で`reverse_slots[A]`（int32）、`reverse_offsets[input_nodes+1]`（int64）を加える。
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

### 階層版backwardの実装（検証済みの初版）

`hierarchical_linear(plan, x, atoms)`から次の順番で呼び出す。

```text
forward: prepare_layout → build_hierarchy(X) → H → C_out → Y
backward: build_hierarchy(dY) → G
                                ├→ C_in → dX
                                └→ parameter_grad(Zx,Zy,H,G) → d_atoms
```

H/Gは振幅を含まない`[batch,sorted_slot]`配列。autograd経路では入力dtypeで保存し、
H/G縮約・微分・Z/Cの計算はFP64を維持する。低水準診断の既定保存dtypeはFP64。
学習対象にatomがある場合だけ、
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
選択中のbankでは周期Pとsite基底が固定されるため、Z自身の幅微分は不要である。
atom側の係数だけを微分する。raw Fourier係数をr_kと書くと、

\[
A_k=\frac{2r_k(\beta)}{P r_0(\beta)},\qquad
\partial_\sigma A_k=
\frac{2\partial_\sigma\beta}{P r_0}
\left(\partial_\beta r_k-\frac{r_k}{r_0}\partial_\beta r_0\right).
\]

cos/sinの中心微分はそれぞれ-omega*sin、omega*cos。
定数係数1/Pの幅・中心微分は0。共通密度係数T_M(beta)^-2は商から消えている。
H/G作成は`contraction.cu::compute_h`の軸切替を共用し、dXは入力側Cを同じ展開kernelで評価する。

出力順layoutでは入力ownerの範囲が非連続なので、初版のC_inは非空groupのdescriptorを走査し、
該当する連続atom範囲だけを32atom shared chunkへ読み込む。atom値の転置コピーやdX atomicは使わない。
このgroup走査は高速化対象であり、完全stepの速度・メモリ達成は実測前に主張しない。
保存atomとlayout配列は`save_for_backward`で更新検査し、retained backwardに対応する。
二階微分とsortを含む全体Graph captureは未対応。

### 最適化の着手：元の版の処理別実測

ユーザーの指定により階層・sigmaの意味・セル積分と面積正規化・H/Gの意味を維持し、
完全stepの速度をdenseの約2倍（時間は約半分）へ近づける。T_min/Pを変えて別のWを測ったり、密Wを作って置換したりしない。
allocatedとreservedの両方をdense以下に保ち、Y/dX/全atom勾配の検査許容差も変えない。
GPT-6.1 Terraの3担当でC集約・atom数学・pipelineを調査し、GPUは単一の共有poolで測定する。

最初にsource `48d2d4a0`（runtime `1ec0f66b`と同じ）を凍結して診断した。
同一fixtureの各低水準stageをCUDA Graphへ一度captureし、21 replayのevent中央値を測った。
prepareはGraph未対応なので同期wall中央値。完全stepの採用時間へ足したり差し引いたりしない。
別のprofile traceでも1024のC kernel2回が約16.17msを占め、局所診断と対応した。

| 処理 | 128診断ms | 1024診断ms |
| --- | ---: | ---: |
| Zx | 0.237568 | 1.795872 |
| H | 0.088064 | 0.501536 |
| C_out | 0.156672 | 7.999072 |
| Y展開 | 0.197632 | 0.198432 |
| Zy | 0.227328 | 1.746048 |
| G | 0.080896 | 0.535552 |
| C_in | 0.242688 | 8.179552 |
| dX展開 | 0.256000 | 0.256640 |
| parameter_grad | 0.191488 | 1.250592 |
| prepare（同期wall） | 0.294586 | 0.513347 |

1024は849 group、1 ownerの最大atom数は入力・出力とも10,539。
約20%が全周bankの単一ownerへ集まり、従来は1warpが全chunkを順に集約していた。
C_outとC_inが主対象であり、次にZ生成の三角関数評価を削る。

job `l4job-67913d6841dc437ba6f17f3727c3676f`、L4/Torch2.11.0+cu130/CUDA13.0、driver239.16秒。
source SHA256 `7c51f473225632601327ee4ce7887c870ba9bf65a4e2ed528afa816e6918596d`、
result SHA256 `ea32824b0a88647ebe18f822512c9aee680dd6b36be1a5edd177e6a772c8b2c3`。
327 runtimeコードと結果11ファイルを照合し、ignored `output/cell-mass-opt-20261027/baseline-profile/`へ保全。
共通headerの別途byte照合も行う。profileは診断専用で、検証済み完全step/ピークの代替にしない。

候補source `8f129556`は次の変更を実装し、下記のCUDA検証を通過した。

- owner×固定split数のFP64部分Cを作って縮約。容量はatom数でなくowner数とsplit数で制限する。
  逆側の非一致groupにはblock同期を置かず、warp ballotで対応rangeを選ぶ。
- 既存の正の多項式をHorner評価し、値と解析微分をまとめる。全atom係数表は作らない。
  basisだけの定数表は444 doubles、3,552bytes/CUDA translation unit。
- Zのsite位相と出力のharmonic位相はFP64漸化式で再利用し、同じセル積分の基底を評価する。
  chart端の3tile周期回転は整数mod3のroots of unityで表し、全周bankは回転0。
- `fuse_g_grad=True`で、同じthreadがGとatom勾配を作る。Gを保存dtypeへ丸めてから勾配へ使う。
  `c_splits`と融合の有無は独立に比較し、採用前の既定は1/Falseのままとする。

CPU上の数式回帰や相互レビューはGPU検証・独立oracle・速度改善の証拠には数えない。
候補の全gateと完全stepピークが通るまで、下の検証済み版の結果と混在させない。

### 最適化第1段：C分割と位相再利用の実測

runtime `8f129556`、L4/Torch2.11.0+cu130/CUDA13.0。
階層版270件＋既存1816件で**2086 passed、failed/skipped 0**。
独立oracle/AdamW検査では、追加引数なしのautograd呼び出しもテストdriver内で
小さいatom数はC2分割、大きいatom数は16分割・G/勾配融合へ切り替え、候補を通した。
低水準stageの独立照合、分割数2/7/16・片側M16・空/端数batch・retained backwardも検査した。
M16の1024rootのZをteam1/32で独立Gauss積分へ照合した。
racecheckはsplit/fusedの8件で0 hazards/errors/warnings。

128/819atomと1024/52,428atomを、C分割1/2/8/16と融合の6設定で
既存FFTセル積分の独立CPU参照へ照合し、全Y/dX/全atom4列がPASS。
1024の最大絶対誤差は全設定でY1.7711e-8、dX1.99152e-8、atom勾配2.7011e-8。
profile式、sigma、T_min/P、許容差とH/G保存精度を変更していない。

同じ完全step条件で、各設定1 fresh process、warmup5/同期wall21標本。
以下は分割・融合を選んだ候補であり、公開dispatcherへの採用ではない。

| 設定 | 完全step中央値 | allocated | reserved |
| --- | ---: | ---: | ---: |
| 128、C1・非融合 | 1.376251ms | 166.5KiB | 2MiB |
| 128、C2・融合 | 1.182218ms | 242.5KiB | 2MiB |
| 同job dense128 | 0.666325ms | 264.5KiB | 2MiB |
| 1024、C1・非融合 | 14.312060ms | 5.332MiB | 8MiB |
| 1024、C8・非融合 | 4.925133ms | 5.332MiB | 8MiB |
| 1024、C16・非融合 | 4.416706ms | 5.332MiB | 8MiB |
| 1024、C16・融合 | 4.100332ms | 5.854MiB | 8MiB |
| 同job dense1024 | 0.664340ms | 16.063MiB | 22MiB |

128のC2融合はdenseの約1.77倍の時間で、中間目安とメモリ条件内。最終速度目標は未達。
128のC8/C16は約1.2msでもallocatedがdenseを超えたため選ばない。
1024は約6.17倍で未達だが、元の23.21msから大きく短縮した。
別jobとの大きな差は初版改善の方向を示すが、採用性能は選択設定の独立再測定で確認する。

候補の1024診断（Graph event、完全stepへの加減算は禁止）は、
Zx/Zy=0.203/0.189ms、H/G=0.412/0.445ms、C_out/C_in=0.770/0.633ms、
Y/dX展開=0.055/0.070ms、非融合勾配0.956ms、G/勾配融合1.136ms。
次はatom側の基本位相・累乗・正規化逆数を再利用し、Zの大きなnodeを小区間のblockへ分割する。
sortを置換するカウント配置は同じgroup順序でindexだけを扱い、別途検証する。

job `l4job-41b59f34aa8c4c5aa02455316d9f5e79`（driver498.89秒）。

- source SHA256 `88faf220099f145a358ae4fe810bcdf64838d7ddc443a9fe9a1cab596eec810a`
- result SHA256 `6a63e8e37228bb6bcafbfa5850448010fbbfd05403471ad4d85750eb7d6e5cfa`
- driver SHA256 `a9dee371e0980c87d27da1ee304ab9b9e47a4cfbab9a5b935a0b5908f2dafdcd`
- binary SHA256 `bc0d8c91c0d942b6aec50aa447549708ad7c6e9f40e8937e2209c596d03702bf`

327 runtimeコードとmanifest58ファイルを照合し、ignored
`output/cell-mass-opt-20261027/first-candidate/`へsource/result/spec/receiptを保存。
queued状態でdriverのreport hook修正のため取り消したjob `l4job-c503f15621bb405e8d903f3aaee10117`は
GPU成功や実測へ数えない。検証前の候補・失敗・不採用設定も削除しない。

### 最適化第2段：harmonic stream・counting配置・Zの分割

runtime `0c4fc64d`、L4/Torch2.11.0+cu130/CUDA13.0。
階層版304件＋既存1816件の**2120 passed、failed/skipped 0**。
CPU全体2037 passed / 4837 skipped / 既存警告18件（38.76秒）、Ruff/format/build通過。
wheel/sdistの337ファイルをsourceへbyte照合した。

1 thread内で基本位相、betaの累乗、正規化逆数を再利用し、harmonic順に係数を生成する。
C_outとC_inでは、同じownerのatom列を均等なS区間へ分ける。
元の係数とprofile式は変えず、

\[
C_{h,q,b,k}=\sum_{s=0}^{S-1} C^{\mathrm{partial}}_{h,q,s,b,k}
\]

と最後に集約する。各atomはそのowner内で一つのsplit・一つのthreadへ割り当てる。
係数表を全atom分globalへ保存する方式ではない。

Zは大きなtileも32site以下の小区間へ分け、同じbank周期で基底を積分してから足す。

\[
Z_{h,t,b,k}=\sum_{j\,\subset\,t} Z^{\mathrm{partial}}_{h,j,b,k}.
\]

この小区間は実行上の分担であり、数学のT_hやP_hを変更しない。
1024/B4/M8のpartial Zは入力87,040bytes・出力69,632bytes、呼び出し中だけ保持する。
完成した実tileのZから、従来通り親の偶数harmonicを継承する。

counting配置は固定された辞書順のCartesian groupを使う。1024の現planでは1769 groupで、
空groupもoffsetsに残す。atom値のコピーは行わず、元row番号だけを並べる。
カウントとrow番号の予約にwarp集約atomicAddを使うが、Y/dX/atom勾配へatomicで加算しない。
一時keys/count/cursorはprepare内だけ、保存cacheはrows/groups/offsetsだけである。
4096 groupを超えるplanはcounting経路を拒否し、従来sort経路は残す。

独立FFT参照、空group/重複offset、空atom/batch、片側M16、32siteのZ分割、
全team・retained backward・幅/有限値拒否を検査した。128/1024の全atom勾配も全候補でPASS。
許容差は以前と同じ。racecheckとsource/result検証の詳細は保存artifactへ置く。

| 完全step設定 | 中央値 | allocated | reserved |
| --- | ---: | ---: | ---: |
| 128、C2/融合/sort/Z非分割 | 1.132184ms | 242.5KiB | 2MiB |
| 同job dense128 | 0.668032ms | 264.5KiB | 2MiB |
| 1024、C16/融合/sort/Z非分割 | 3.273565ms | 5.854MiB | 8MiB |
| 1024、C16/融合/sort/Z32分割 | 3.003025ms | 5.854MiB | 8MiB |
| 1024、C16/融合/counting/Z非分割 | 3.090996ms | 5.875MiB | 8MiB |
| 1024、C16/融合/counting/Z32分割 | 2.795126ms | 5.875MiB | 8MiB |
| 同job dense1024 | 0.667932ms | 16.063MiB | 22MiB |

1 fresh process/方式、warmup5・wall21標本、その他の条件は第1段と同じ。
128は約1.69倍、1024は約4.18倍。メモリ条件は両方で満たすが、1024の時間目標は未達。
1024の診断値はZx/Zy=0.050/0.048ms、H/G=0.313/0.313ms、C_out/C_in=0.467/0.403ms、
Y/dX展開=0.055/0.070ms、G/勾配融合=0.929ms。今後は係数の定数比とgradのregister保持が対象。

job `l4job-af3d11b157de415ca147825b7eb73a0d`（driver499.96秒）。

- source SHA256 `ef46a7db0977d21f02054fa29444d0945fada72c584f9146262aa3973cb4e536`
- result SHA256 `85b3b9c5092fa655a01182f788e1425b7068bff6503e0ea28ff30a9c74c4b09f`
- driver SHA256 `452575e1fbeb4f7c0d30c5ebc7924a374845c0db0b8899d48e6825f7b38bdc32`
- binary SHA256 `c093812a93223aa4dc7dd1a9764d87124f5ba056d2295110c8315109886fb701`

327 runtimeコードと結果manifest40ファイルを照合し、ignored
`output/cell-mass-opt-20261027/stream-candidate/`へsource/result/spec/receiptを保存した。

### 最適化第3段：定数比・partialページ・atom blockサイズ

runtime `8dc468cb`。basisだけの前係数72個を定数表へ加え、毎harmonicの定数比を
除算し直す処理を省いた。多項式444個と合わせ4,128bytes/CUDA translation unit。
β累乗の微分も逐次更新する。profileの近似次数やprecisionは変更しない。

C32/64分割は、単一partialが1MiBをわずかに超えて大きなallocator予約へ移ることを避けるため、
各partial Tensorが1MiB以下になるようsplit方向をページへ分ける。総partial容量は減らさず、
同じFP64値を同じ順番で縮約する。pointerはkernel引数として渡し、GPU pointer表は作らない。
元C一枚自体が1MiB超ならページを小さくできないため、そこで条件達成とは主張しない。

atom担当のblockを32/64/128/256 threadsへ変えても、1 threadが別atomを処理する。
同じlayoutを使った比較ではY/dX/各atom勾配がbit単位で一致することを検査した。
counting配置での3stepのAdamWと階層遷移も追加し、**2165 passed、failed/skipped 0**。
全52,428atomの独立Y/dX/4勾配は全候補でPASS。

| 1024、counting/Z32/融合 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| C16、128threads | 2.586894ms | 5.875MiB | 8MiB |
| C16、64threads | 2.554607ms | 5.875MiB | 8MiB |
| C32、128threads | 2.600685ms | 6.382MiB | 8MiB |
| C32、64threads | 2.561422ms | 6.382MiB | 8MiB |
| C64、64threads | 2.501205ms | 7.396MiB | 10MiB |
| 同job dense | 0.678842ms | 16.063MiB | 22MiB |

128のC2/128threadsは1.125077ms（dense0.686617ms）、allocated242.5KiB/dense264.5KiB、
reserved両方2MiB。両サイズでメモリ条件内、1024は約3.68倍で時間目標は未達。
単一fresh processずつ、wall21標本の診断であり、この小差を未再現の普遍的勝者にはしない。

job `l4job-3752c02b51b046988a9396aa7f5464fa`（driver502.43秒）。

- source SHA256 `94edf7e0793c90c284103d28b3f2135495adf9f177491706a4847808895b7b7e`
- result SHA256 `07d210687f848fe6ee54d1846c3a4cee18ae49ddd82e061eeea078c752c4d4cb`
- driver SHA256 `01db43bf9f3b475fd848fc64a3a8ab317062034f05abdd8e1b63a78eb3e44bff`
- binary SHA256 `ea6609d53591b1afaa6660b609fc79e7b9ce4c9853c2af75408330bc446677c9`

327 runtimeコードと結果43ファイルを照合し、ignored
`output/cell-mass-opt-20261027/paged-candidate/`へsource/result/spec/receiptを保存した。

次の第4段では、Cartesian groupの直接参照と加重微分を検証した。


### 最適化第4段：加重微分とgroup参照の簡略化

source `b0cb69a5`。Gとparameter勾配の融合で、batchごとの微分配列を作らず、
`sum_b amplitude * H_b * Z_b` と `sum_b amplitude * G_b * Z_b` を先に縮約する。
Gの保存dtypeへの丸めと振幅微分で使うGは従来通り。FP64加算順の変更は独立参照で検査した。
counting layoutでは既知のCartesian group番号を直接参照し、C_inで全groupを走査しない。
また、cacheを作らないcounting経路ではXとatomの有限値検査を同じerror wordへまとめる。
cache再利用時の新しいXの検査は残す。

同一の128/1024、B4、M8、初期sigma0.2..40、seed120、T_min/P不変。
各設定はfresh process、warm5、同期wall21回の中央値。prepare、全forward/backward、
fused AdamWと一時配列の作成・解放を含む。Graphなし、TF32無効、cuBLAS workspaceなし。

| 条件 | 完全step | peak allocated | peak reserved |
| --- | ---: | ---: | ---: |
| 128、C2/融合/sort/Z0/128threads | 1.099659ms | 242.5KiB | 2MiB |
| 128、matched dense | 0.679848ms | 264.5KiB | 2MiB |
| 1024、C16/融合/counting/Z32/64threads | 2.386650ms | 5.875MiB | 8MiB |
| 1024、C32/同上 | 2.391108ms | 6.382MiB | 8MiB |
| 1024、C64/同上 | 2.277464ms | 7.396MiB | 10MiB |
| 1024、matched dense | 0.654805ms | 16.063MiB | 22MiB |

1024初版23.211msから約10.2倍短縮したが、dense比3.48倍の時間を要する。
128もdense比1.62倍である。**ユーザー指定の最終目標はdenseの2倍の速度、時間は半分**であり、
両方とも未達。過去の「時間をdenseの2倍以内」は中間目安としてのみ読む。
この1jobの結果を全幅分布・全GPUへ一般化しない。

低水準Graph-event診断は、1024 H/G各0.289792ms、融合G/勾配0.823296ms、
C_out0.409600ms、C_in0.325632ms、Zx0.050176ms、Zy0.048128ms、
Y展開0.055296ms、dX展開0.069632ms。診断値を加算して完全stepとしない。
独立Y/dX/全52,428atomの4列勾配、およびGPU・数学2171件が通過。
CPU全体2037 passed / 4888 skipped / 既存警告18件。
128のAdamW Parameter/moments/step照合はsort/counting両方を含む。
1024はこのjobで全勾配を照合し、optimizer状態の独立照合まで行ったとは主張しない。

job `l4job-78ac2e52bb064b3ca575953af131dde7`。
source archive SHA256 `425469c004887b648c1d17d537ae7b417092416a1386f0b61659218df28716d1`、
result archive `3be67c2a2f0e8a36722230b2556425c5b5fdb02048d7569ccec097900bf78f8c`、
frozen driver `b9f5998a340f65ff80b0e6c1063031e16b6b3c2acb0989e6142181de99ae10b7`、
loaded binary `5075323aa51a7599dffc9f8450394e98f5c82c688f621181db0d619d28dadc87`。
ignored `output/cell-mass-opt-20261027/weighted-candidate/`にsource/results/spec/receiptと照合記録を保存。



### 最適化第5段：bankの定数と互換階層範囲

source `2a507b1b`。bankのinverse_periodとomegaをhost計画で一度計算する。
make_profileは同じsigma/scaleを使い、半径を経由する式を無次元のrootへ整理した。
Fourierセル積分のk=1..16の逆数を定数表へ追加し、表全体は4,264bytes/CUDA translation unit。
Cartesian Cの探索では、sigmaの分類区間に共通部分がないbank組を省く。
全周bankの上限はatom dtypeへ丸めた共通sigma上限であり、FP32端点を落とさない。

GPU・数学2171件、128/1024の独立Y/dX/全4列勾配が通過。
128 C2/融合/sort/Z0/128threadsは1.106592ms（dense0.666637ms）、
1024 C64/融合/counting/Z32/64threadsは2.201958ms（dense0.661160ms）。
allocated/reservedは第4段と同じでメモリ条件内。単一fresh processずつの診断値であり、
数%の差を普遍的な改善率としない。速度目標は引き続き未達。
1024の段階診断はH/G各0.278528ms、融合G/勾配0.741376ms、C_out0.408576ms、C_in0.343040ms。

job `l4job-0b0b8d63e3b044ed86a5b324e1ad0f90`、driver474.24秒。
source SHA256 `c865dc5e36ccfaa594487dd1f5e1e9953be123d98229fb84e0e17117999319dd`、
result `405bc3499a649efe7ecd94b920819d38880955dbf7fb3c51ba5ce48d7f731208`、
driver `b9f5998a340f65ff80b0e6c1063031e16b6b3c2acb0989e6142181de99ae10b7`、
binary `766c6ab807d14cf4dced308f244be47bc578f69b0ed6d835abaded5f7dbced2c`。
ignored `output/cell-mass-opt-20261027/bank-candidate/`へ327 runtimeと結果34ファイルの照合を保存。

### 最適化第6段：shared配置、M8専用化、owner window

- source `7c610fcc`：Cの論理32atomは維持し、shared物理strideを33にする。
  M8/B4ではsharedは5,376から5,544bytes/block（+168bytes）。
  G/勾配のbatch和を先に縮約し、正規化のk=0値・微分も一度に計算する。
- source `468c858d`：M8のH/C/融合勾配を固定mode数で展開する。他のeven Mは汎用経路を残す。
  registerやcode sizeの増加で逆に遅くなる可能性を含め、同条件で比較する。
- source `760d4933`：全raw Z階層の完成後、ownerごとに3タイルの位相を補正して合算する。
  そのwindow ZだけをH/G/全勾配が一度読む。full-chartはraw値と同じ。
  `window_moments=True`の研究flagで切り替え、元のraw Z/低水準APIは維持する。

windowを元の階層Zと区別してRと書くと、各bank/owner qで

\[
R^{c,s}_{b,q,k}=\sum_{\delta=-1}^{1}
 R_{\mathrm{image}(q+\delta),k}\,Z^{c,s}_{b,\operatorname{wrap}(q+\delta),k}.
\]

**階層のマージは2つずつのまま。3枚の和は、同じ階層でH/Gが使う窓を先に作る別の処理である。**

| 値 | 担当する幅 | 作り方 |
| --- | --- | --- |
| raw Zの親node | 2T_h | 周波数が一致する親の偶数harmonicを左右2子から継承。奇数は直接生成 |
| R_h,q | 3T_h=P_h（local bank） | 同じ階層のZ_h,q-1・Z_h,q・Z_h,q+1を、chart image補正後に合計 |
| 全周bankのR | Q | 全周bankのraw Zと同じ |

Rを作ってもh、T_h、P_hやsigma分類は変えない。三分木へ変更する処理ではない。
この和はatomではなくownerにだけ依存するので、同じwindowを使うatom間で共有できる。
DCは単純和。中心の最後の半セルに対するmake_profileのimage補正は別の処理として残す。
windowはmergeへ戻さず、mergeが全て完了してから別bufferで作る。
rawとwindowは同shapeなので低水準APIには`windowed=True`を必ず渡す。
C/展開は変えない。入力windowは33,184bytes、出力は15,776bytes（1024/B4/M8）。
変換時のraw/window同時生存を含めた完全stepピークと、FP64再結合後の全勾配を検査する。

これらの単一jobのGPU検証結果は次の通り。全てsame physical T_min/P、同じ初期profileからのdense、
TF32なし、Graphなし、workspaceなし、warm5/同期wall21、fresh processずつ。

| runtime / 設定 | 128 CST / dense | 1024 CST / dense | GPU・数学 |
| --- | ---: | ---: | ---: |
| `7c610fcc` shared stride・加重回転 | 1.108148 / 0.673994ms | 2.169511 / 0.669591ms | 2171 passed |
| `468c858d` M8専用化 | 1.090994 / 0.660951ms | 2.094633 / 0.683576ms | 2171 passed |
| `760d4933` owner window R | 1.141979 / 0.663074ms | 1.840612 / 0.669264ms | 2217 passed |
| `833e68dc` Axisコピー回避・shape共用 | 1.123588 / 0.677137ms | 1.661862 / 0.666132ms | 2217 passed |

128はC2/sort/Z0/128threads、1024はC64/counting/Z32/64threads、G/勾配融合あり。
Rは後半2行で有効。allocated/reservedは全行、128=242.5KiB/2MiB、
1024=7.396MiB/10MiBで、各dense264.5KiB/2MiB、16.063MiB/22MiB以下。
全52,428atomの4列勾配、Y/dXを各jobの独立FFT参照へ照合した。
CPU回帰はwindow段階2037 passed / 4934 skipped / 既存警告18件。

Rは1024で有効だった一方、128では追加起動の費用を補えない結果だった。
全サイズ一律の有効化はせず、最終設定は別fresh processで再確認する。
source833の1024はdenseの約2.49倍の時間で、最終目標の時間半分は未達。
初版23.211msから約14倍短縮したが、固定bank版の別周期・別幅分布とは改善率を比較しない。

source833はAxisのmode値を書き換える代わりに、係数生成へ固定mode数をtemplateで渡す。
入力・出力でperiod/inverse_period/omega/modes/radius_ratio4が全て一致するときのみ、
同一sigmaのbeta・正規化形状と微分を共用する。中心・owner/imageは軸ごとに計算する。
このfixtureでは52,428中47,422atom（90.45%）が同じ形状を共有できる。
mode固定前後のresource出力は、H STACK648→40、C split688→80、融合勾配1256→40bytes/thread。
融合勾配のREGは248→244であり、register圧迫まで解消したとはしない。

1024の段階診断（Cは32分割）は、source833でH/G=0.152/0.151ms、
C_out/C_in=0.359/0.316ms、融合G/勾配0.360ms、Zx/R_x0.049ms、Zy/R_y0.047ms。
展開Y/dXは0.045/0.056ms。これらを加算して完全stepとはしない。



検証archive（各327 runtime＋共通header・検査ソースを照合、結果manifestも検証）:

- `7c610fcc` / `l4job-e08e148f295644ca9b7104dee4673bac`：source `211763d0560f20d0963f561bf0f7dd3438cba503222eeba4981234e4585bfebf`、result `8dd26e3cb25ba020cbc746ebe42ed1e38e24bb959bb171930c4b31fe4ccddfad`、driver `b9f5998a340f65ff80b0e6c1063031e16b6b3c2acb0989e6142181de99ae10b7`、binary `1563457bb3882a602b00fda15b124928f55ba2cbec2111fa45df2f6b590da2b5`。
- `468c858d` / `l4job-cfbc5d6ff6594bedb152c4ab9185be77`：source `1166fbc96ac950c30b0920e5009fcad8c1684ae0ff8caf3e91cceb3e7162baf1`、result `682e0dac7b3120187504a347b5cfc93e0215a2792f2d18e9a1a0ed01188eec76`、driver `b9f5998a340f65ff80b0e6c1063031e16b6b3c2acb0989e6142181de99ae10b7`、binary `bb640378186bbada9463cbb71e4080fc76b1d0c8e7f1d0037ca8cc3020b07fb8`。
- `760d4933` / `l4job-5cf94d223d37477d9a4caba176f5b706`：source `9571fed484351a86e5981da72c773782772a47c20e80ca47057abe3b892b0f0a`、result `d97011ad766039359779125bec3d253ca038cd6bc93a024a34ea0bab7a12317a`、driver `b95b332a67d11e49c0bdc2281e806da7fceabdb7dde353438c46656aa3e85bda`、binary `6b06002db885a8bbe68cb7a7160fd02af35b4067d42cce0607b6cdbc9db7668d`。
- `833e68dc` / `l4job-763872192be04cea9a0013015dcdc7bb`：source `2fa471732daa025db55a8417233393e5f822b0ab02967272faec894125e82973`、result `dc8ea5468866d76593b2f2076972b3c86e8c3e9683ea41f537b5bf20cd883c6d`、driver `b95b332a67d11e49c0bdc2281e806da7fceabdb7dde353438c46656aa3e85bda`、binary `f48edac099ee855bf583c75e2e192bbc16648b6fc55c2a221317611fc7029423`。

ignored `output/cell-mass-opt-20261027/` の各candidateにsource/results/spec/receiptを保全。

### 最適化第7段：G・Parameter勾配・C_in融合の負の結果

source `dcf48273`。研究flag `fuse_backward_cin=True`を追加した。
両方の勾配が必要で1<=batch<=4のとき、input owner/splitの各laneが別atomについて
G→保存dtypeへ丸め→全4勾配→input係数のshared保存を行い、C_inを縮約する。
Hの保存値と丸めたGを使う規約は維持する。C累積はsharedへ置き、重い勾配計算中のregister保持を減らす。
M8/B4のsharedは6,088bytes/block。Gはglobalへ保存し、全atom係数表は追加しない。
B0/B>4/片側だけの勾配は既存経路へ戻る。新header `projection.cuh`はJIT digestにも含める。
CPU全体2037 passed / 4995 skipped / 既存警告18件、Ruff/format/diffは通過。
GPUでは通常経路2278 passed、tuned経路2278 passed、failed/skipped 0。
全52,428atomの独立Y/dX/4列勾配、AdamW、fallback分岐のspy、racecheckも通過した。
しかし1024完全stepはC16=2.259520、C32=2.176199、C64=2.276359ms、dense0.671774ms。
128/C2は1.172671ms、dense0.674572ms。allocated/reservedは従来と同じでdense以下だが、
source833分離版1.661862msより遅いため**速度面では不採用、既定Falseを維持**する。
別sourceの小差を一般化せず、現在のsourceでflag False/Trueをfresh processずつ再確認する。

1024/C32のGPU段階診断は融合1.153024ms、分離G/勾配0.360448ms、C_in0.319488ms。
融合REG194、STACK40、LOCAL0であり、register増大が主因と断定できる証拠ではない。
同じfixtureのgroup/splitをCPUで列挙すると、逆方向S64の32lane枠の実atom比は30.97%、
S32は43.10%、S16は56.69%。S64は5291 chunkで、連続atomだけなら必要な1639 warpの約3.23倍。
これはGPU stall計測でなく、group区間ごとの末尾paddingの構造的な集計である。
重い勾配計算をその断片化したchunkへ移した影響は、今後の候補を設計する根拠になる。

次の未実装候補は、atom値をコピーせず`reverse_slots[A]`にforward sorted slotを並べ、
`reverse_offsets[input_nodes+1]`で入力ownerの連続範囲を渡す方法。
正規rowsは出力→入力順を維持し、H/Gも従来slot、勾配は元Parameter行に対応させる。
counting prepareでgroup内の既存scatter順位を再利用すれば、追加atomicなしにreverse配置を作れる。
int32 slotなら1024の恒常増分は209,712bytes＋offsets496bytes。
owner別の有効split数、追加prepare時間、group復元、保存状態の変更検出、全勾配とピークを検証するまで、
改善や採用は主張しない。現在はこの逆indexを実装していない。

job `l4job-6efb835e93b74ae48c03f8dbba502198`、source `23958a971ff86d7c328d24e3426f12c0795fa492e7e3c0fac04e7f17f249d72f`、result `161ab9561fc8af50862268ffbcf47fceacb3c7858f426bed7fa5833264932aec`、driver `ae0664bfc469ad720788d08f68341790f2f6118de254070cff79d63eace6075b`、binary `99cc6d5d0202d892753c71a584f619ee23795f82994e2854ddcdd0bd5f9cc14c`。
327 runtimeとcommon/projectionの両header、検査ソース、結果35ファイルを照合。
ignored `output/cell-mass-opt-20261027/fused-candidate/`へsource/results/spec/receiptを保存。
wheel/sdistの338 runtimeファイルもsourceへ照合した。



### 最適化第8段：逆方向indexとowner別の有効split数

source `a9995b07`。counting prepareに任意のreverse indexを追加した。
正規rows/groups/offsetsは出力→入力順のまま、atom値や係数はコピーしない。

```text
reverse position r → reverse_slots[r] = forward sorted slot s
                   → rows[s] = original atom a
H/G[:,s]、atoms[a,:]、gradient[a,:]を使用
```

`reverse_offsets[input_node:input_node+2]`がその入力ownerの連続範囲。
prepareの既存group内scatter順位を再利用して、追加atomicなしに逆slotを書く。
増分はint32 reverse_slots[A]とint64 reverse_offsets[input_nodes+1]のみで、
1024 fixtureは209,712+496bytes。group開始位置は一時int32 scratchで保存しない。
`save_for_backward`へ両配列を加え、in-place変更検出とretained backwardを維持する。

逆方向のowner範囲について、grain=atoms_per_splitが正なら
`active=min(max_splits,max(1,ceil(owner_atom_count/grain)))`とする。
active外blockは部分和をゼロにして全thread一斉に戻る。active内は区間を重複なく分担。
grain0は固定split数を維持する。Zの二分マージ・Rの3枚窓・sigmaの分類は変更しない。

同source、B4/FP32/M8/Q128/A52428/sigma初期.2..40、counting/Z32/64threads、C64、R有効。
通常/tuned各2369 passed / failed0 / skip12（unsupported sort+reverse combinations）。
全52,428atomのY/dX/4列勾配は全設定で独立FFT参照を通過した。

| 逆index / grain / Cin融合 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| なし / 0 / 分離 | 1.665969ms | 7.396MiB | 10MiB |
| あり / 128 / 分離 | 1.454741ms | 7.596MiB | 10MiB |
| あり / 0 / 融合 | 2.125936ms | 7.596MiB | 10MiB |
| あり / 64 / 融合 | 1.605266ms | 7.596MiB | 10MiB |
| あり / 128 / 融合 | 1.477567ms | 7.596MiB | 10MiB |
| あり / 256 / 融合 | 1.737101ms | 7.596MiB | 10MiB |
| matched dense | 0.688162ms | 16.063MiB | 22MiB |

各1 fresh process、warm5/wall21、全prepare/forward/backward/AdamWを含み、Graph/TF32なし。
逆index＋owner適応で分離版を約13%短縮したが、dense2倍の速度目標は未達。
128はreverse無し/C2/sort/Z0/128threadsで1.224288ms、dense0.674702ms、
allocated242.5KiB/264.5KiB、reserved両方2MiB。
段階診断はC32/grain128でCin0.218112ms、融合G/勾配/Cin0.743424ms、
Cout0.359424ms、G/勾配0.360448ms。stageを足して完全stepとはしない。
出力側Cが次の改善対象であり、grainをforwardへも適用する候補を検証する。

job `l4job-4e4975d749a54a6488085e238b3649e8`、source `95ebc10b5e9d3cba0a0c9db5882a626f9147cccfbcb25af0a9a23e511284c829`、result `b7048b1361741252c5f6eaeef39b72407c728dcec284c78395b584ec0d06d269`、driver `85a0cf3229f8f2e7d95106376a3a2fe719fa0817b6542b90d928e3a2c5639546`、binary `dfa70f7493160bebbf95f56fffddd24ae5ca4ff1cc846c3e47074e5c89534923`。
327 runtimeと両header・検査ソース、結果46ファイルを照合し、ignored
`output/cell-mass-opt-20261027/reverse-candidate/`へsource/results/spec/receiptを保存した。

出力Cにも適応splitを加えたsource `69a7806b`を別jobで検証した。
通常/tuned各2385 passed / failed0 / skip12、racecheckの対象caseも通過。
全52,428atomのY/dX/4列勾配は6設定全てで独立FFT参照を通過した。
最大絶対誤差はY 1.78e-8、dX 2.00e-8、全atom勾配2.71e-8未満。

| 条件 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 1024、逆indexなし・固定C64・Cin分離 | 1.686256ms | 7.396MiB | 10MiB |
| 1024、逆indexあり・grain128・Cin分離 | 1.494954ms | 7.596MiB | 10MiB |
| 1024、逆indexあり・grain128・Cin融合 | 1.499301ms | 7.596MiB | 10MiB |
| matched dense1024 | 0.697796ms | 16.063MiB | 22MiB |
| 128、従来のC2/sort/128threads・Rなし | 1.211821ms | 242.5KiB | 2MiB |
| matched dense128 | 0.703213ms | 264.5KiB | 2MiB |

診断C32/grain128のCoutは0.348160ms、Cinは0.214016ms、G/勾配融合は0.360448ms。
前jobのCout0.359424msとの差は小さく、出力側への適応split追加だけが有意に速いとは
主張しない。逆indexを含む分離版の改善は再度観測したが、dense比は時間約2.14倍であり、
中間目安の時間2倍以内も最終目標の速度2倍も未達。
準備段階の`c5796712`は旧input wrapperの引数渡し漏れでsuiteが154件失敗した。
これは性能採用対象外として保存し、修正後に上記の全suite/gateを再実行した。
数学/精度条件を緩める修正ではない。

job `l4job-70e3ce933e2949e1b6566b9506bf5901`、source `411b2095515fa2a67018dc27af719bb954570105aa28793cd9bec88261bfe9d0`、result `d0515dc338715206b8c7bb7f916c0fd9ccc5ab691ad652c38339869eb42af156`、driver `08170a732474b821da80bc17282ad52afbf12bafe8c8906dedeec0ecfa1adc39`、binary `a6e6baf0ab66ff7409386a21d3972e8fcbffc95484241dbb1fcdf90acecbffbc`。
327 runtimeと両header・検査ソース、結果46ファイルを照合し、ignored
`output/cell-mass-opt-20261027/adaptive-both/`へsource/results/spec/receiptを保存した。

### 最適化第9段：補償されたM8の値評価

source `0f08e84e`はM8多項式だけのdouble-single補償演算を実験用templateとして追加した。
既定Falseを維持し、現在の完全stepは全てDouble評価。定数表増分は352bytes、合計4616bytes/TU。
β・位相・正規化・Z/R/Cの縮約をDoubleのままにし、単独micro診断で価値を測った。
M8/A52428のGraph-event中央値はvalue 0.088064→0.078848ms、derivative
0.145408→0.116736ms。Doubleに対するchecksum差は最大1.46e-14だった。
registerはvalue 36→40、derivative 46→56へ増えた。
これは低registerの診断kernelであり、完全stepや独立全勾配の改善証拠ではない。
次に値計算H/Cだけの明示的opt-inで、完全step・正規化・全勾配を検証する。
勾配kernelの演算はDoubleのままにする。

micro job `l4job-6e5a58f5d60c4d42aa6107dc74c5eb8b`、source `9e2dded124f5b249547012920a4a5e4033e69dcd54ad05db2125c061b7acceab`、result `9ed48ae43fd9473dee83a449a187d51e34c74ca5af9ed403d5b1bdd2340a97cc`、driver `a22b607976e584a7dbdf0d54a0482ad9e9fc2a8ef2d3d3a50f58916c8fb351c4`、binary `63192d9779c5cd5f0906f9cfe68e5b07f3c524d7c69e20b3184171b5161e8cbf`。
このdriverはmain extensionを呼ばず、common.cuhを使う独立した診断kernelのみを実行した。
前述のc579 wrapper不具合を通るmain計算の合格とは扱わない。
327 runtime、両header、検査ソース、結果7ファイルを照合し、ignored
`output/cell-mass-opt-20261027/ds-micro/`へ保存した。

source `290e38b1`は`compensated_values=True`を研究入口へ接続した。
対象はFP32保存、M8/tail1e-3の両軸のみで、既定Falseを維持する。
H・Cout・単独G・分離Cinだけが補償演算を選べる。G/勾配融合とG/勾配/Cin融合は
従来のDoubleのまま。位相・正規化・縮約・Z/R/Cの保存もDoubleを維持する。
値評価の近似とDouble解析導関数の組合せを、独立oracleと同じ精度基準で検査する。
通常のCPU suiteは2037 passed / 5153 skipped / 既存警告18件。
wheel/sdistの338runtime byte照合とRuff/formatは通過した。
GPU job `l4job-2806b8176cb64be48be73b60f93f4ce2`で2424 passed / failed0 / skip12。
独立FFTのY/dX/全4列、unit mass、retained backward、128の3step AdamWも通過した。
1024は全52,428atomを各設定で照合し、最大誤差は従来の保存dtype丸めと同じ範囲だった。

| 条件 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 1024、逆index/grain128、G/勾配融合、Double値評価 | 1.477870ms | 7.596MiB | 10MiB |
| 同上、補償値評価 | 1.428999ms | 7.596MiB | 10MiB |
| 補償値評価、G/勾配を分離 | 1.676313ms | 6.796MiB | 8MiB |
| matched dense1024 | 0.676794ms | 16.063MiB | 22MiB |
| 128、従来設定、Double / 補償値評価 | 1.158747 / 1.163540ms | 242.5KiB | 2MiB |
| matched dense128 | 0.693166ms | 264.5KiB | 2MiB |

1024の補償値評価はこの単一job内で約3.3%短縮したが、独立再確認前の条件付き候補とする。
128は改善せず、既定や128推奨を切り替えない。G/勾配を分けるとメモリは減る一方で遅くなる。
診断はH 0.150528→0.137216ms、Cout 0.348160→0.321536ms、
Cin 0.225280→0.202752ms。完全stepへの加減算はしない。
速度の最終目標も、中間目安の時間dense2倍以内も1024では未達。
source `1d7a2c016e1b0d7d348d56588ced1d789d9f8df69658d0e0a6263c9405c31367`、result `34b5fa0c418ccdbba12f4dbffbdaec0e3f6736128885df2fd0349fbed03184ed`、driver `ebbffe58c47d4801601cf478a01b02dbd2b75011b38cfa0987fe101fa2c87857`、binary `b8ce9ffe5e0ba685796b4190e821d64c5f7f04d0d59210a7753e74c8c2fff142`。
327runtime、両header・検査ソース、結果39ファイルを照合し、ignored
`output/cell-mass-opt-20261027/compensated-values/`へsource/results/spec/receiptを保全した。

### 最適化第10段：HとCの融合と負の結果

source `e1d84c33`はHとCoutを同じatom担当threadで作る`fuse_h_c=True`を追加した。
両軸M8のみを対象とし、他modeは分離経路へ戻す。既定Falseである。
このfixtureのCPU分類では47,422/52,428 atom（90.4517%）で両軸のprofile形状が一致し、
その場合は一つの多項式streamから、中心差の位相回転で出力係数を生成する。
形状が異なるatomでは別の係数streamを使う。中心/imageの補正は各軸で独立に保つ。

```cpp
// owner・batch tile・splitごとのblock。各threadは異なるatom。
H_in_register, output_coefficients_in_shared = project_and_make_coefficients(atom, Zx);
H_saved = cast_to_atom_dtype(H_in_register);
amp_H_in_shared = amplitude * H_saved;  // 丸め前のHはCに使わない
sync_block();
C_partial = reduce_atom_chunk(amp_H_in_shared, output_coefficients_in_shared);
// 全chunkの後、従来の別kernelでsplit部分和をCへ合計する。
```

Hはbackward用に引き続き保存し、globalからの再読込と係数の重複生成を減らす。
Zの二分階層・Rの3枚窓・sigma分類は変えず、Y/dXやParameter勾配の担当も変えない。
M8/B4のsharedは6,088bytes/block。追加の全atom係数表やatom値コピーは作らない。
位相の再利用で丸め順序は変わるため、独立参照・AdamW・racecheckを同じ基準で検証する。
GPU job `l4job-5a1444db6d614fd2bea9289dbdf75369`は2438 passed / failed0 / skip12、
融合stageのracecheck、独立H/C/Y/dX/全4列、128の3step AdamWを通過した。
1024の全52,428atomも4設定全てで独立参照を通過したが、融合は完全stepで悪化した。

| 方式 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 1024、Double値評価・H/C分離 | 1.479868ms | 7.596MiB | 10MiB |
| 1024、Double値評価・H/C融合 | 1.529752ms | 7.596MiB | 10MiB |
| 1024、補償値評価・H/C分離 | 1.427589ms | 7.596MiB | 10MiB |
| 1024、補償値評価・H/C融合 | 1.509354ms | 7.596MiB | 10MiB |
| matched dense1024 | 0.692989ms | 16.063MiB | 22MiB |
| 128、従来設定・H/C分離 | 1.183312ms | 242.5KiB | 2MiB |
| 128、Double値評価・H/C融合 | 1.161017ms | 242.5KiB | 2MiB |
| 128、補償値評価・H/C融合 | 1.135810ms | 242.5KiB | 2MiB |
| matched dense128 | 0.713953ms | 264.5KiB | 2MiB |

H/C融合単体の診断はDouble0.709632ms、補償0.679936ms。完全stepへの加減算はしない。
融合のregisterはFP32 atomで106/123（Double/補償）、分離Hは56、分離C splitは76。
融合はgroup参照・HとCの縮約を同じ32thread blockのchunk内にまとめるため、
係数共有だけで速くなるとは限らない。占有率低下などの因果を個別に実証したとはしない。
1024では不採用、既定Falseを維持する。128の小差は独立再確認前の候補に留める。
補償値評価の分離版は前jobと同じ約1.43msを再観測したが、最終速度目標は未達。

source `147ea8fac350307dbdb033d5c4a9351afbe04968d49296328c1d02a179673550`、result `f651f5f3e51da60c87447bc13376e3cf33c398d5eb21659d30c17091e095854d`、driver `080f4da97d4d76fac68c705d1c29447e3525f090b8d7b6b9dc2fbeb38e70e8e5`、binary `27d669e31aa81bf649c8428c213df52c3a85604c42a1b80f1adf1cb5c0ce65a9`。
327runtime、両header・検査ソース、結果45ファイルを照合し、ignored
`output/cell-mass-opt-20261027/fused-forward/`へsource/results/spec/receiptを保全した。

### 最適化第11段：Gと勾配の専用化と汎用化

source `7fd9132b`は、既存のG/Parameter勾配融合kernelのdispatchだけを切り替える
`specialize_g_grad=False`を追加した。axisのMは変えず、M8を既存の汎用kernelへ渡す。
低水準の同名操作では`specialize_modes=False`を指定する。
source `69a7806b`の資源診断ではM8専用版は244register/thread、汎用版は128だった。
演算の展開と占有率の関係を調べる候補であり、register数だけを速度改善の証拠にはしない。
既定Trueを維持し、G/勾配/Cin融合やParameter勾配単独には作用しない。
CPU2037 passed / 5176 skipped / 既存警告18件、Ruff/format/buildを通過。
GPU job `l4job-cfe3e961427946f38f40609358a7e549`で階層版suiteは631 passed /
failed0 / skip12、融合stage racecheckを通過した。このjobは階層版だけを再実行し、
前jobの固定bank/coefficients/referenceを含む2438件を再実行したとは扱わない。
128の全819atom、1024の全52,428atomについて、全設定のY/dX/4列勾配を独立参照へ照合した。

| 1024の条件 | 完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 補償値評価・H/C分離・M8専用G/勾配 | 1.432525ms | 7.596MiB | 10MiB |
| 同上、汎用G/勾配 | 1.481324ms | 7.596MiB | 10MiB |
| 補償値評価・H/C融合・M8専用G/勾配 | 1.504512ms | 7.596MiB | 10MiB |
| 同上、汎用G/勾配 | 1.546615ms | 7.596MiB | 10MiB |
| matched dense | 0.683191ms | 16.063MiB | 22MiB |

G/勾配単体はM8専用0.360448ms、汎用0.404480ms。register数の削減だけでは改善せず、
`specialize_g_grad=True`を維持する。H/C分離・補償値評価の約1.43msを3job目でも再観測。
128も専用/汎用は分離1.153474/1.177473ms、融合1.139130/1.190129msで汎用が遅かった。
dense128は0.682831ms、allocated242.5/264.5KiB、reserved両方2MiB。
選んだ設定の別L4・2 fresh process再測定と、1024全atomの3step AdamW状態は
次の最新再確認節で検証した。

source `320a041d3801405afd673a93921e491a24333cbd2c814643db3ea6fdf5c10875`、result `678f035b3c2785c23401f81c26939449493d39d9988645d6dc5ff72f2f896f5a`、driver `494c38a190f82e01a88fad7a0d925a81ad964694a041a9ca9f4e5ea8285a16bf`、binary `ef0cdb0d72b3b55443f4f2667a6f19c76676a7ee1085cbe2bf92ab157bf26fc0`。
327runtime、両header・検査ソース、結果54ファイルを照合し、ignored
`output/cell-mass-opt-20261027/generic-grad/`へsource/results/spec/receiptを保全した。
wheel/sdistの338runtimeファイルもsourceへbyte照合した。

### 最新の再確認：完全stepと1024のAdamW状態

runtime `7fd9132b`。job `l4job-3d9f12cdfca6486fa532f515d7b04cda`で別の所有L4へ切り替え、
各設定2 fresh process、2巡目は順番を逆転した。warm5/同期wall21、全検査/prepare/
forward/backward/fused AdamWを含む。TF32/Graph/追加workspaceなし。
前jobとatom/X/dY/初期dense Wのbit一致を確認した。

| 設定 | 完全step（2 process） | allocated | reserved |
| --- | ---: | ---: | ---: |
| 128、保守的なH/C分離・Double値評価 | 1.182075 / 1.149119ms | 242.5KiB | 2MiB |
| 128、H/C融合・Double値評価 | 1.149945 / 1.146039ms | 242.5KiB | 2MiB |
| 128、H/C融合・補償値評価 | 1.164949 / 1.127448ms | 242.5KiB | 2MiB |
| matched dense128 | 0.690988 / 0.686519ms | 264.5KiB | 2MiB |
| 1024、逆index/grain128・H/C分離・補償値評価 | 1.434035 / 1.431054ms | 7.596MiB | 10MiB |
| 同上、Double値評価 | 1.476370 / 1.481064ms | 7.596MiB | 10MiB |
| matched dense1024 | 0.669389 / 0.688117ms | 16.063MiB | 22MiB |

全CST設定で、同repeatのdense以下のpeak allocated/reservedをassertした。
128の融合・補償の差は順序やjobで揺れがあるため、保守的な分離設定を維持する。
1024の補償値評価は独立再確認でも約3%短縮し、メモリ増加はなかった。
最終目標の速度dense2倍には未達。1024は中間目安の時間dense2倍以内にもまだ届かない。

このjobでは1024の全52,428atomを3step更新し、各更新後の参照Parameterから
独立CPU FFT/autogradでY/dX/全4列勾配を再計算した。
Y/dX/勾配はrtol3e-6/atol2e-7、Parameterはrtol2e-6/atol2e-7、
AdamW momentsとstepはrtol4e-6/atol2e-8を維持して全step通過した。
最大勾配絶対誤差は3.16e-8未満、最大Parameter絶対差は1.53e-5未満で、
Parameterの相対・絶対誤差の複合基準を通過した。bit一致とは主張しない。
この正確性検査用の参照Tensorは計測に持ち込まず、時間/ピークは別fresh processで測った。
GPU suiteの再実行はこの確認jobでは行わず、直前の2438件/階層版631件を根拠とする。

source `0dc5210d24e864ce8dbde9abd32db44444899af3c62c2e5469382a6584c2c106`、result `9623c160c9fc06bd9ab7c8f770533324cae1d7475bcb620319fc810ca5440a26`、driver `71fc5970a7484a11bae226c954a4edee1d4e60fbec873e7b8b15efd07d09d67d`、binary `bf1bb7eec946cca2ccf7a246a1093bd94f4d37ec4379fb112801033518630c4a`。
327runtime、両header・検査ソース、結果54ファイルを照合し、ignored
`output/cell-mass-opt-20261027/final-confirmation/`へsource/results/spec/receiptを保全した。
CPU Actionsはsnapshot `9c972124`（runtimeは同じ）でsuccessを確認した。

### 第7段までの独立再確認と、現在使う研究設定

runtime `dcf48273` / snapshot `46b069e2`（差は文書のみ）。別の所有L4で各設定2 fresh process、
2巡目は測定順を逆転した。warm5/同期wall21、同じatom/X/dY/初期dense Wは前jobとbit一致。
検査/prepare/全forward/backward/fused AdamWを含む。JIT/oracle/初期W構築は除外。
TF32なし・Graphなし・cuBLAS workspaceなし。前jobの通常/tuned各2278 gateを根拠とし、
このjobで正確性suiteを再実行したとは扱わない。

| 条件 | 2 fresh processの完全step | allocated | reserved |
| --- | ---: | ---: | ---: |
| 128、C2/sort/Z0/128threads、Rなし、Cin分離 | 1.205635 / 1.121764ms | 242.5KiB | 2MiB |
| 128、同上Rあり、Cin分離 | 1.148473 / 1.144093ms | 242.5KiB | 2MiB |
| 128、同上Rあり、Cin融合 | 1.121268 / 1.120112ms | 242.5KiB | 2MiB |
| dense128 | 0.699004 / 0.687199ms | 264.5KiB | 2MiB |
| 1024、C16/counting/Z32/64threads、Rあり、Cin分離 | 1.704220 / 1.723238ms | 5.875MiB | 8MiB |
| 1024、C64/同上、Cin分離 | 1.657808 / 1.655620ms | 7.396MiB | 10MiB |
| 1024、C16/同上、Cin融合 | 2.238962 / 2.236008ms | 5.875MiB | 8MiB |
| 1024、C64/同上、Cin融合 | 2.302701 / 2.263328ms | 7.396MiB | 10MiB |
| dense1024 | 0.668562 / 0.683754ms | 16.063MiB | 22MiB |

1024のCin融合は同sourceの比較でも悪化した。128のR/Cin融合はこのjobで小幅に速いが、
過去jobとのゆらぎが小差を上回るため、全サイズ一律の自動採用はしない。
通常のprivate API既定値は比較用に残し、公開dispatcherの既定は変更しない。
現在の保守的な128設定と、確認した1024設定は次のkwargsで明示する。
128でR/Cin融合を選ぶ場合も、同じ数学/メモリ契約の研究選択肢であり、普遍的勝者とはしない。

```python
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.executor import hierarchical_linear

settings128 = dict(
    c_splits=2, fuse_g_grad=True, layout_method="sort",
    z_chunk_sites=0, atom_threads=128,
    window_moments=False, fuse_backward_cin=False,
)
settings1024 = dict(
    c_splits=64, fuse_g_grad=True, layout_method="counting",
    z_chunk_sites=32, atom_threads=64,
    window_moments=True, fuse_backward_cin=False,
    reverse_layout=True, atoms_per_split=128,
    compensated_values=True,
)
y = hierarchical_linear(plan, x, atoms, **settings1024)
```

上の1024 kwargsは第8段の逆index/grain128も含む。第7段の再確認表とは区別し、
第8段の逆indexで1.45〜1.49ms、補償値評価を加えると2jobで約1.43msを確認した。
H/C融合は1024で悪化したため、上の設定には含めない。

この設定はB4/FP32/M8/tail1e-3/Q128、A=floor(.05*N*N)、sigma初期linspace(.2,40)、
物理T_min入力4/出力8のfixtureで確認したもの。別の幅分布・多軸・他GPUの勝者へ一般化しない。
1024は初版23.211msから約14倍短縮したが、dense比約2.42〜2.48倍の時間であり、
ユーザーの最終目標である**denseの2倍の速度＝時間半分**は未達。

job `l4job-b3fdda00495f458d8706f9e0b74a4e13`、source `e87dda784b5b41dcf87505be5319242b002df6e8273659abdca9451d1a4eb872`、result `13aa1b7f83e6420a6948d79a564a596f0e56b5eab3561f0f2243645efdb38853`、driver `665b8c3f78fa1d0abcb383227891b188622f921f2cd70d122013182e0e8a7adf`、binary `b960d6fded186f6b49ddb96842bc02c25d88bd362616b1b75fb2318e04eb2219`。
327 runtime＋両header・検査ソース、結果69ファイルとfixture bit一致を照合し、
ignored `output/cell-mass-opt-20261027/confirmation/`へsource/result/spec/receiptを保全。
所有L4は共有poolで停止を確認した。source `46b069e2`のCPU validationもsuccess。


### 現在：入力dtypeでH/Gを保存するbackward

runtime source `1ec0f66b`。FP32入力ではH/GをFP32で完成保存し、途中の縮約・正規化係数・
微分・Z/Cは引き続きFP64で計算する。FP64入力は保存もFP64。
FP64の全Hを作ってからcastするのではなく、最終値を指定dtypeへ直接書く。
低水準`compute_h/compute_g`の診断既定はFP64のままで、`storage_dtype`で保存dtypeを指定できる。
元Parameter、profileの式、sigmaの意味、tile分類、許容誤差は変更していない。

階層版218件（CPU31/CUDA187）＋既存1816件の**2034 passed、failed/skipped 0**。
前段の全勾配・retained backward・mutation検査・gradcheck・3stepのAdamW比較を再実行し、
保存Hが入力dtype、保存ZがFP64であることも検査した。
racecheckはretained backwardの3件で0 hazards/errors/warnings。
CPU全体2037 passed / 4751 skipped / 既存警告18件（44.17秒）。
Ruff・format・差分検査・wheel/sdist buildを通過し、packageの337ファイルをsourceへbyte照合した。

1024/B4/A52,428でも全atom4列を512atomずつ独立FFTセル積分参照へ再照合し、PASS。
最大絶対誤差はY=1.7711e-8、dX=1.9916e-8、全atom勾配=2.7011e-8。
許容差は前段と同じFP32 rtol3e-6/atol2e-7である。1024のoptimizer状態の独立比較は未実施、
128の全atom/AdamW状態比較は再通過した。

完全stepの条件・処理範囲は前段と同じ。sigma0.2..40、Q128、M8、batch4、
物理T_minは入力4/出力8（128では4/8sites、1024では32/64sites）。
各方式1 fresh process、warmup5/同期wall20標本、入力/dY固定、TF32なし、cuBLAS workspace 0、Graphなし。
元atom profileからdense Wを作る時間とoracle/JITは計測から除外する。

| 完全layer step | CST中央値 | dense中央値 | CST allocated / dense | CST reserved / dense |
| --- | ---: | ---: | ---: | ---: |
| 128、819atom | 2.249732ms | 0.703702ms | 166.5 / 264.5KiB | 2 / 2MiB |
| 1024、52,428atom | 23.211011ms | 0.674475ms | 5.332 / 16.063MiB | 8 / 22MiB |

現在の両fixtureでallocated/reservedともdense以下。1024のCST allocatedは5,591,040bytes、
H/Gは各838,848bytes。FP64保存初版のreserved26MiBを8MiBへ減らした。
時間は128で約3.20倍、1024で約34.41倍で、**約2倍の速度目標は未達**。
保存量削減で速度も改善したとは判断しない。プロセス全体のGPUメモリ、全体Graph学習、
複数独立runでの性能再現は未検証。C_inのgroup走査とregister保持量は次の分解計測候補であり、
現時点でどちらが最大のボトルネックかを断定しない。

job `l4job-84892b903dae42c68640c7705652b8d2`、L4 / Torch2.11.0+cu130 / CUDA13.0。

- source SHA256 `882983d42d4985f106526bc9741027c3e74fad8e8df57d5255909b9410a5e1ac`
- result SHA256 `34a29566dfbdc89808b0408a2f70b2a8010eb6bad50be6e12de0c1caba71b614`
- driver SHA256 `fe8f555d2bff1cc648456b943c87e53635dbb5e15b8adffa8b3c2002b0eb59e3`
- binary SHA256 `4aaedbe3f72282d8cc514eb1198e917414c0fd9aeb1d7a8423250827ddbce3fb`

327 runtimeソース・共通header・検査ソース・埋込1024 driverを照合し、manifest42ファイルのhashを確認。
source/result/spec/receiptをignored `output/cell-mass-backward-20261027/storage-run/`へ保全した。
再検証は`python -m pytest -q tests/test_cell_mass_hierarchical.py`と既存3つのcell-mass検査。
同じGPU実行は保存済みdriverを共有poolのsubmit/serveへ渡す。
旧FP64保存型の結果と負のreserved結果も下に残し、現版の値と混在させない。

### 前段：FP64保存型backwardの検証と128の完全step

runtime source `33d7f9dd`、L4 / Torch2.11.0+cu130 / CUDA13.0。
階層版216件＋既存1816件の**2032 passed、failed/skipped 0**。
独立参照には既存profileのFFT係数から作る微分可能なセル積分を使い、
Y/dX/全atomの4列をFP32/64、direct/merge Z、local/full bank、chart端、
非ゼロ原点、異なる入出力サイズ、M16、batch1/3/4/5で照合した。
128/B4/A819は全atomを検査。1024siteの小さな35atom検査も含む。
幅・ownerの整数選択を固定した枝の解析微分であり、階層遷移の連続性の証明ではない。

FP32はrtol3e-6/atol2e-7、FP64はrtol3e-9/atol2e-10の許容差。
これは実測最大誤差の値ではない。数値gradcheck、空batch/atom、必要な勾配だけの呼び出し、
retained backward、非連続dY、保存atom/layoutのin-place更新検知も検査した。
128/819atomの3stepでは幅の階層とtorus ownerを移し、毎forwardの再分類を行い、
AdamWのParameter・exp_avg・exp_avg_sq・stepを独立勾配で更新した同じAdamWへ照合した。
これは物理atomへのAdamWであり、公開CSTOptimizer/Parameterizationとの統合検証ではない。
compute-sanitizer racecheckはretained backwardの3件で0 hazards/errors/warnings。
二階微分とsortを含む全体Graph学習は未対応で、backwardのGraph replayも未検証。

完全stepは128×128/A819/B4、FP32、M8/tail1e-3、Q128、入力T_min4・出力T_min8、
sigma=linspace(0.2,40)、seed120。入力/dY固定、grad初期化・検査・sort・全forward/backward・
fused AdamWを含む。JIT・oracle・初期dense Wの生成は含めない。
同じ初期profileからdense Wを作り、両者ともX勾配と重み側勾配を計算する。
TF32なし、cuBLAS workspace 0、Graphなし。各方式1 fresh process、warmup5/同期wall20標本。

| 完全layer step | 中央値 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: |
| 階層版CST | 2.239738ms | 197,120bytes（192.5KiB） | 2MiB |
| dense、workspaceなし | 0.678630ms | 270,848bytes（264.5KiB） | 2MiB |

時間比約3.30倍で約2倍目標は未達。allocatedはdenseの約72.8%、reservedは同じ。
単一runの初版診断であり、採用性能として複数独立runの再確認はしていない。
全GPU process使用量は未測定。固定bank版の過去結果とは周期・幅分布が異なるため改善率にしない。

job `l4job-b59c3c9f88d64b239e13952d2bdc4cae`、driver実行411.20秒（JIT・検証込み、step時間ではない）。

- source SHA256 `d24ba011d7160a99eebc702b039ea342d4f26ea89343ce211c70c1954be33288`
- result SHA256 `75987af45be3f960e7e8febab64b659102ca7aed8394d1fddef082eaca7d4102`
- driver SHA256 `c0e9967b444aef36d57cc72d2b8b972e78d6b6bfeaf2b61c2ab926ed10ca2b44`
- binary SHA256 `5cad5b6ece85b5292f00680ac1e9559256dae15a6d9e4c2b141871617f2e69f5`

327 runtimeコード・共通header・検査ソースをcommitへbyte照合し、結果manifest22ファイルのhashを確認。
source/result/spec/receiptをignored `output/cell-mass-backward-20261027/first-run/`へ保全した。
CPU全体2037 passed / 4749 skipped / 既存警告18件（43.50秒）。Ruff・format・差分検査・build通過、
wheel/sdistの337ファイルを現sourceへbyte照合した。GPU/DBのskipは検証完了へ数えない。

### 前段：1024全atom検証とFP64保存の負の結果

同じruntime `33d7f9dd`を1024/B4/A52,428、Q128、spacing0.125、入力最小tile32sites・
出力64sites（物理T_min4/8）で検査した。sigmaは0.2..40の等間隔値、seed120。
512atomずつ既存FFTセル積分でCPU参照を作り、全atom4列の勾配を省略せずに照合した。
最大絶対誤差はY=1.4745e-8、dX=1.4850e-8、全atom勾配=2.7011e-8、全てPASS。
1024のoptimizer状態の独立照合はこのjobでは実施していない。

| 完全layer step、FP64 H/G保存 | 中央値 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: |
| 階層版CST | 23.130443ms | 7,268,352bytes（6.932MiB） | 26MiB |
| dense、workspaceなし | 0.669730ms | 16,843,264bytes（16.063MiB） | 22MiB |

同じ初期profileからdense Wを作った。比較方法は128と同じ、方式ごとに1 fresh process。
約34.54倍で速度目標未達。allocatedはdense以下だが、**reservedはdenseを超える**。
このFP64保存型をメモリ条件を満たす採用値にはしない。各H/Gは1,677,696bytesであり、
FP32入力では入力dtype保存へ変更して再度全勾配・stepメモリを検査する。
allocatorの予約はTensor容量と異なるため、保存量だけで達成を判断しない。

job `l4job-558ff9aa21ec4ca7ac83e1d61db4eb46`（driver250.01秒、性能値とは区別）。

- source SHA256 `9ea59b717276f3ae5f0c18c573e8d91de2c9d8df16b8e1fa7455a731bf575785`
- result SHA256 `f281ec92603f5b88c02fcd5d5f3d3b1eab0b105076bc55d784eab46c46a72dd7`
- driver SHA256 `7ba7882c19f0acf568c4543c1fcc381c3c32f7a9619e2042692af8a1a06329e3`
- binary SHA256 `42634e664408a62a176ed0e28f548bf7fc3a4b702ee1a53457ccefd43cc20624`

327 runtimeソースと結果manifest20ファイルを照合し、
ignored `output/cell-mass-backward-20261027/run1024/`へsource/result/spec/receiptを保全した。
GPU process全体のメモリは測定していない。旧固定bank版とprofile周期・幅分布が異なり、
既存の1.354msからの退行率にはしない。

### 前段：H/C/Y接続とsigma整合の検証

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
新規profileの数式待ちではない。同じ式のG/dX/Parameter勾配を実装し、検証記録を冒頭に追記する。

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
compute_h(Zx, atoms, layout, H);
compute_y(H, atoms, layout, Y);

// backward
build_output_hierarchy(dY, layout, Zy);
compute_g(Zy, atoms, layout, G);
compute_dx(G, atoms, layout, dX);
compute_parameter_grad(Zx, Zy, H, G, atoms, layout, grad);
```

同じstreamへ依存順に投入する。上の名前は論理段階であり、実際のlaunch数ではない。
owner方式のcompute_y/dxはC部分和を作るkernelと展開kernelに分かれる。
階層版の`fuse_g_grad=True`はGとParameter勾配を融合するが、forward Hはbackwardのため保存する。
旧固定bank版のH再計算型[融合経路](cell-mass-fused-pullback.ja.md)とは保存寿命が異なる。
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
新ディレクトリのprepareとH/C/Yへ接続済み。backwardも同じforwardの分類結果を使う。
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

- 実装: 既存profileとsigmaの分類、group offsets、bank別Z、H/G、C、Y/dX、Parameter勾配。検証範囲は冒頭の表と記録で追跡する。
- 未実装: 公開CSTLinear/dispatcherへの統合、階層版の保存／再計算切替。
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
        dh = input_parameter_derivative(atom, parameter, b, Zx);
        dg = output_parameter_derivative(atom, parameter, b, Zy);
        grad_parameter += atom.amplitude *
            (G[b, slot] * dh + H[b, slot] * dg);
    }
    store_original_profile_grad(original_id, parameter, grad_parameter, grad);
}
```

ここで`dh/dg`はそれぞれ`partial H / partial parameter`と`partial G / partial parameter`。
逆伝播の中間勾配`dH = alpha * G`とは別である。
入力側だけに効くparameterならdg=0、出力側だけならdh=0、両側に効くなら両方を含める。

微分関数は窓内の全セルについて`sum partial_coefficient * Z`を計算する。
支持境界をさらう補正やraw X/dYの追加読込みはない。
面積正規化の微分も含める。H/Gだけからこの情報を復元できるとはしない。
Zx/Zyは読み取り専用のglobal、gradの累積値はregisterへ置く。
元のParameter行への対応はlayoutで戻し、optimizerによる更新はここでは行わない。

このコードはparameterごとにZを読む単純な説明形である。同じZを読んだときに複数parameterの
微分を同時に累積すれば読み取りを共有できるが、register使用量との交換になる。
physical sigmaの解析微分は実装済み。bank/ownerの離散的な選択は固定した枝として微分し、
次のforwardで再分類する。階層遷移が滑らかだと保証するものではない。

## 確認すること

各段階の保存値、配列のshape、書き手、寿命を先に確定し、独立oracleで
Y/dX/全atom勾配を検証してから速度を比較する。上記は実機での正しさ・性能の結果ではない。
H/Gの保存を省く案では、同じforward状態での再計算と、全利用先の終了を明示する。
