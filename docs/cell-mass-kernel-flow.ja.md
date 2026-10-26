# 区間確率方式の5つの計算kernel

2026-10-26。[設計と数学契約](cell-integrated-profile.ja.md)に対応する、検討用の擬似コード。
関数名・引数は仮名であり、CUDA実装ではない。profile族、係数とその微分、支持境界の
数式が未確定なので、以下の補助関数を実装済みとは扱わない。

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

同じstreamへ依存順に投入する。階層構築はこの5つの計算kernelに含めず、
親子の依存を安全に満たす起動単位を別途決める。CPU側で毎回同期するという意味ではない。
以下ではgeometry・正規化設定などの引数を省略し、補助関数が同じspecを参照するものとする。

## 1. compute_h：入力側の全寄与をatomごとにまとめる

blockはatom群とbatch範囲を担当する。各threadが別々のatomを担当し、
そのatomに必要な全入力タイルを足す。最初の例では128thread、各threadに1atom、batch_size=4。
Hは振幅を掛ける前の値。Zx/Xはglobalから読み、Hの累積値はregisterへ置く。

```cpp
atom = assigned_atom(blockIdx.x, threadIdx.x);
if (atom >= atom_count) return;

for (b : assigned_batches(blockIdx.y, batch_size)) {
    h = 0;
    for (tile : full_input_tiles(atom)) {
        h += contract_input_tile(atom, tile, b, Zx);
    }
    h += input_boundary_contribution(atom, b, X);
    H[b, atom] = h;
}
```

`full_input_tiles`は支持内部を重複なく覆うタイル集合。親とその子を二重に含めない。
`contract_input_tile`は`sum_k input_coefficient(atom,tile,k) * Zx[b,tile,k]`。
係数には中心・幅と面積による正規化を含め、必要なmodeごとに評価する。
`input_boundary_contribution`は内部タイルに含めなかった残りのセルを、正確な積分で計算する。
この分割の具体式は未確定だが、二つを合わせて全profileの寄与を一度ずつ数える契約にする。
内部タイルがなくても境界側で必要な寄与を処理し、空支持なら0を書く。

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
    build_output_c_together(bank, output_tile, H, shared_c);
    __syncthreads();
    expand_output_c_into_y(bank, output_tile, shared_c, y);
    __syncthreads();  // 全員が読み終えてからshared_cを再利用する。
}
add_output_boundary_contributions(output_tile, H, y);
store_output_tile(output_tile, y, Y);
```

内部部分でのCは`C[b,k] = sum_a alpha[a] * H[b,a] * output_coefficient(a,tile,k)`。
`build_output_c_together`はCの各batch/modeに書き手を一つ決め、対応atomを走査して完成値を
sharedへ書く方式を出発点にする。atomごとの部分Cを並列生成してreductionする案は後で比較する。
係数を作るthread配置と、Yのsiteを展開するthread配置は異なってよい。

bankごとに基底が異なるため、C同士を混ぜず、展開したsite値をyへ足す。
境界補正は内部のCで表した寄与と重複しない残りを加える。
bankのループと二つのbarrierには全threadが参加し、担当要素がないthreadも途中returnしない。
globalのHは読み取り専用、Cはshared、yは各threadのregisterを基本とする。
各threadが持つyの個数は担当タイル寸法とbatch範囲で決まり、容量に応じて変更する。

## 3. compute_g：dY側でHと対称の集約を行う

atom群とbatch範囲を担当し、各threadが別々のatomを処理する。
Zy/dYを読み、振幅を掛ける前のGを完成させてglobalへ書く。

```cpp
atom = assigned_atom(blockIdx.x, threadIdx.x);
if (atom >= atom_count) return;

for (b : assigned_batches(blockIdx.y, batch_size)) {
    g = 0;
    for (tile : full_output_tiles(atom)) {
        g += contract_output_tile(atom, tile, b, Zy);
    }
    g += output_boundary_contribution(atom, b, dY);
    G[b, atom] = g;
}
```

`compute_h`と入出力の役割が逆になる。内部タイルと残りのセルの分割・係数の定義は、
forwardで使う同じprofileに対応させる。G全体のほかにdH全体を保存することは前提にしない。

## 4. compute_dx：担当入力タイルの係数を作り、dXまで完成させる

blockは入力タイルとbatch範囲を所有する。`compute_y`の入力側版として構成する。

```cpp
input_tile = assigned_input_tile(blockIdx.x);
init_input_gradient_accumulator(dx);

for (bank : contributing_input_banks(input_tile)) {
    build_input_c_together(bank, input_tile, G, shared_c);
    __syncthreads();
    expand_input_c_into_dx(bank, input_tile, shared_c, dx);
    __syncthreads();
}
add_input_boundary_contributions(input_tile, G, dx);
store_input_tile(input_tile, dx, dX);
```

内部部分の係数は`C_in[b,k] = sum_a alpha[a] * G[b,a] * input_coefficient(a,tile,k)`。
振幅は一度だけ掛ける。bankごとに展開し、境界補正を加え、担当dXを完成させる。
Gはglobal、係数はshared、dxはregisterを基本とする。barrierの規則は`compute_y`と同じ。
Y/dXの各要素を一つのblockが所有するため、block間の出力加算atomicは不要にできる。

## 5. compute_parameter_grad：H/Gと微分を組み合わせる

各threadが一つのatomの全batchを担当し、そのatomのParameter勾配を完成させる基準案。
batch分割で複数blockが同じ勾配を書く方式は、追加reductionを伴う別候補にする。

```cpp
atom = assigned_atom(blockIdx.x, threadIdx.x);
if (atom >= atom_count) return;

grad_amp = 0;
for (b : all_batches) {
    grad_amp += H[b, atom] * G[b, atom];
}
store_original_amp_grad(atom, layout, grad_amp, grad);

for (parameter : profile_parameters(atom)) {
    grad_parameter = 0;
    for (b : all_batches) {
        dh = input_parameter_derivative(atom, parameter, b, Zx, X);
        dg = output_parameter_derivative(atom, parameter, b, Zy, dY);
        grad_parameter += alpha[atom] *
            (G[b, atom] * dh + H[b, atom] * dg);
    }
    store_original_profile_grad(atom, parameter, layout, grad_parameter, grad);
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
