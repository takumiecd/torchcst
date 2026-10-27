# 区間確率方式：prepareからbackwardまでのkernelの流れ

## 設計の正本と、現在の実装範囲

2026-10-27。この文書を、sigmaからの階層分類・layout・Z/H/C・backwardを
つなぐ設計の正本とする。分類規約は下の
[sigmaから階層と周期を決める規約](#sigmaから階層と周期を決める規約)に置く。
他の実測ノートは、その時点の実装と測定の記録であり、この設計が全て実装済みという意味ではない。

**現在のCUDA runtimeは1軸1bankの固定tile/周期であり、sigma別階層分類は未実装。**
Zの二分集約木があることと、sigmaから異なるtile/周期を選ぶことを区別する。
この追記は設計の確定と実装差分の明示であり、ParameterやCUDAの計算は変更していない。

| 項目 | この文書の設計 | 現CUDA runtime（source `7c1404c1`） |
| --- | --- | --- |
| sigmaの意味 | 新規約は集中領域の全幅、I=sigma/2 | 旧幅制御値。Pに依存する半径下限補正あり |
| 階層分類 | T_minから2倍ずつ、Iを収める最小bankを選ぶ | 未実装。軸ごとに一つのAxis設定 |
| 周期 | local bankごとにP_h=2L_h=3T_h | 軸ごとに固定P=K*T |
| sort | 入出力のbankと中心tileの辞書順 | 中心の(output owner,input owner)だけ |
| Z | bankの基底ごとに直接生成／一致modeの継承 | 固定Pの全modeを同じ基底で二分マージ |
| 小幅の直接計算／全周bankへの自動分類 | 下記の実行計画に含む | 未実装。宣言済みK=1の経路はあるが自動分類ではない |
| H/G・Parameter勾配 | 各threadが別atomを担当 | 実装済み、固定bank条件で検証済み |
| C→Y/dX | bankごとにCを作って展開後に加算 | owner Cの部分和と展開は実装済み。複数bankの合算は未実装 |
| 面積正規化・backward | 分類後のprofile契約で再検証する | 旧幅定義／固定bankの独立oracle照合のみ完了 |

新しい幅規約と階層経路へ、既存の時間・勾配検証結果を読み替えない。
実装箇所は[`CellMassAxisSpec`](../src/torchcst/kernels/cell_mass.py)、
[`executor.py`](../src/torchcst/_backends/cuda/algorithms/linear/cell_mass/executor.py)、
[`kernels.cu`](../src/torchcst/_backends/cuda/algorithms/linear/cell_mass/kernels.cu)。
現状と測定条件の入口は[実測状態](cell-mass-status.ja.md)。

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
ここは従来未確定だった`classify_hierarchy`の設計であり、runtimeへの接続は未実装。
中心の所属キーと、支持が重なる全タイルの集合は別である。

### sigmaから階層と周期を決める規約

#### 幅の意味と、旧実装との境界

新しい階層設計では、sigma>0を意図する集中領域の**全幅**として扱い、半幅を

\[
I_a=\sigma_a/2,\qquad \rho_a=I_a/d=\sigma_a/(2d)
\]

と定義する。sigmaは標準偏差ではない。Iは集中領域の半幅、LはFourier半周期、
実際に評価する窓の幅は2Lであり、これらを同一視しない。
この幅規約だけで密度族の形状まで確定したとは扱わない。近似の比較先となる目標profileは
bankに依存しない数学定義にし、同じ目標に対して各bankの近似誤差を検証する。
Fourierの裾を近似として残す／ゼロ扱いすることと、厳密なcompact支持も区別する。
rho<1だけで「1siteにしか寄与しない」とは判断しない。セルとの位置関係も必要になる。

これは本節で定める新しい幅規約である。既存runtimeのsigmaは
`effective_radius(sigma,P)`により集中半径が変わる旧幅制御値であり、同じ数値を
新規約のsigmaとして渡しても同じprofileにはならない。旧測定のsigma=0.2..12を
新幅の測定として扱わず、旧checkpoint/Planの暗黙の読み替えもしない。
以下では1入力軸・1出力軸で同じ物理sigmaを共有する。軸別幅を持つ将来拡張でも
同じ式を各軸のsigmaに適用する。

#### T_min、二分階層、3タイル周期

各軸についてsite数N、間隔d、chart周期Q=N*dを持つ。
最小tileのsite数n_minはNを割り切る2のべき乗とし、

\[
T_{\min}=n_{\min}d,\qquad T_h=2^hT_{\min}.
\]

n_minは実行計画で宣言する。特定GPU向けに選んだ`sites_per_block`とは別の値である。
T_minとPは近似式・誤差とbank切り替えに影響するので、単なるlaunch調整とは扱わない。
実行計画を変えたときも、同じ目標profileの許容誤差を守ることを検査する。

atomのIを収める最小の候補階層を選ぶ。

\[
h_a^{\rm raw}=\min\{h\ge0:I_a\le T_h\}
=\max\left(0,\left\lceil\log_2\frac{\sigma_a}{2T_{\min}}\right\rceil\right).
\]

local bankではK=3を固定し、

\[
P_h=2L_h=3T_h,\qquad L_h=\frac32T_h.
\]

h=0は0<sigma<=2T_min、h>=1は2^h*T_min<sigma<=2^(h+1)*T_minを受け持つ。
境界の等号は小さい階層側へ入れる。実装ではlog2の丸めで分類を変えず、
2倍更新とI>Tの比較、または同じ境界を保証する整数化した方法を使う。
**bankを選んでもsigma自体は丸めない。** 係数と微分は元のsigmaから生成する。

以前の関係式は、bankの半幅上限I_h_max=T_hへ適用すると

\[
2L_h=\left(\left\lceil\frac{2I_h^{\max}}{T_h}\right\rceil+1\right)T_h=3T_h.
\]

各atomの小さなIに対して毎回最短の周期を作るのでなく、同じbankで周期と基底を共有する。
例としてT_min=4なら、sigma範囲(0,8]/(8,16]/(16,32]は
それぞれh=0/1/2、T=4/8/16、L=6/12/24となる（local窓がchartに収まる場合）。

#### 中心のtileと、chart全周への切り替え

3T_h<=Qなら中心tileと左右1tileを評価する。中心の原点相対座標を
u=(mu-origin) mod Q、N_h=Q/T_hとすると、

\[
q_{\rm raw}=\left\lfloor\frac{u+d/2}{T_h}\right\rfloor,\qquad
q=q_{\rm raw}\bmod N_h.
\]

unwrappedな窓は、セル端を基準に

\[
[(q_{\rm raw}-1)T_h-d/2,\ (q_{\rm raw}+2)T_h-d/2)
\]

とする。中心から窓の端まで最低T_hあるため、I<=T_hの集中領域を覆う。
qはソート用にcanonical化するが、演算の座標像・位相も同じwrapに揃える。
QがP_hの整数倍でなくてもよい。chartの継ぎ目で同じsiteを二重に足さない。

3T_h>Qならlocal窓を切り詰めず、全周bankへ移す。
その専用番号はH=log2(N/n_min)、T_full=Q、K_full=1、P_full=2L_full=Q、q=0とする。
これは二分木の全周rootを使う例外で、途中の収まらない3tile bankは作らない。
全siteを一度だけ扱う。新幅の初期対応範囲は0<sigma<min(Q_in,Q_out)とし、
それ以上の幅や端点sigma=Qのprofile/安定評価は未対応として明示的に拒否する設計にする。
全周bankの自動分類・新幅profile・勾配はまだ実装／検証していない。

```cpp
classify_axis(sigma, center, axis) {
    I = sigma / 2;
    h = 0;
    T = axis.T_min;
    while (I > T) { T *= 2; h += 1; }

    if (3 * T > axis.Q)
        return full_bank(axis.H, T=axis.Q, L=axis.Q/2, q=0);

    u = positive_mod(center - axis.origin, axis.Q);
    q_raw = floor((u + axis.spacing/2) / T);
    q = positive_mod(q_raw, axis.Q / T);
    return local_bank(h, T, L=1.5*T, q);
}

// 事前にfinite・sigma範囲・gridの整数条件を検査する。
out = classify_axis(atom.sigma, atom.output_center, output_axis);
in  = classify_axis(atom.sigma, atom.input_center,  input_axis);
key = (out.h, in.h, out.q, in.q);
```

多次元では各軸へ適用し、上記の辞書順キーへ座標を追加する。
T_minやspacingが異なれば、同じsigmaでも入力と出力のhは異なってよい。
小幅の直接計算は各軸で独立に選べるため、入力direct・出力Fourierなどの組も許す。
最小local bankをdirectとする方式は初期候補であり、速度による切り替え閾値の確定ではない。
直接用profileのセル積分・正規化とFourier側の近似誤差の接続は、実装前に検証する残件とする。

#### 周期が変わるZと、profile係数に必要な変更

local bankの基底はomega_h,k=2*pi*k/P_h。hが一つ上がるとPも2倍になるので、
**同じ番号kのZを子からそのまま足しても、親の同じkのZにはならない。**
同じmode数・共通原点・spacingの場合、親の偶数harmonicは

\[
\omega_{h,2j}=\omega_{h-1,j},\qquad
Z_{h,t,b,2j}^{\cos/\sin}
=Z_{h-1,2t,b,j}^{\cos/\sin}+Z_{h-1,2t+1,b,j}^{\cos/\sin}
\]

で継承できる（j=1,...,floor(M/2)）。定数modeも継承でき、親の奇数harmonicは直接生成する。
式の2j/jはharmonic番号であり、配列内のcos/sin interleave位置とは区別する。
全周bankはP=Qの別基底なので、この偶数対応を自動適用しない。初期実装は直接生成する。
後の`can_inherit_same_basis`は、この周波数対応を担当表へ落とす条件になる。
atomがいないbankでも、親Zの生成に必要な子要素は準備する。

既存の半径下限補正をそのまま使ってI=sigma/2と呼び替えてはいけない。
新幅のChebyshev候補ならbeta=sec(pi*I/P)とする写像と微分を別に検証する必要がある。
M8・tail_bound=1e-3の集中半径下限は約0.158067P=0.474201T。
最小以外のlocal bankではI/P>1/6なのでこの下限を満たすが、最小bankの小幅は保証されない。
このため小幅を直接処理する経路が必要になる。単に幅を下限へ膨らませて条件を満たさない。
beta式は係数側の移植候補であり、新profileの誤差・極小幅・遷移の検証完了を意味しない。

階層・tile所属の整数選択は微分せず、選択中のbankのsigma/中心の微分を計算する。
階層遷移で関数が連続になる保証はまだない。遷移両側のY・全勾配・optimizer更新を検証し、
許容誤差に収まらなければbank間の整合や遷移方式を修正する。forwardで選んだlayoutを
backwardでも使い、optimizer更新後の次forwardで再分類する。

#### 実装へ進む前の対応項目

- 設計確定: sigma/Iの新幅規約、二分候補階層、3tile周期、全周の例外、入力出力bankのキー。
- 未実装: 新分類とbegin/end、bank別Axis・係数写像、対応modeでのZ継承、複数bankのC展開合算。
- 未確定／未検証: bankに依存しない新目標profileの形状、小幅directとの接続、全周bank、
  新旧幅の明示的移行、階層遷移の誤差と微分。これらを新幅の実装完了として扱わない。
- 採用条件: 新幅の独立oracleでY/dX/全atom勾配と更新を照合し、sort込みの完全step時間と
  allocated/reservedを同条件のdenseと比較する。メモリはdense以下を守り、
  速度はdense約2倍以内のstep時間という目標との距離を別に記録する。

atomのコピーや全atom係数表は引き続き作らない。必要なbank/nodeと継承依存だけを用意し、
bank数分のdense行列・全site×atom表へ展開しない。T_minや直接処理の閾値は、
この数学契約を明示した別caseで時間・メモリを測って選ぶ。

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
