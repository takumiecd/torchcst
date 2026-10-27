# 同じownerのZ窓を共有し、位相を漸化式で作る

同じtile ownerに分類されたatomは、参照するwhole-tile窓も同じになる。
`owner_window_moments=True`では、その窓のZをatomごとに集計する代わりに、
ownerごとに一度集計して共有する。atomの中心・幅・係数を固定する方式ではない。
[面積正規化とタイル窓](cell-mass-tile-window.ja.md)の数式を変えず、保存と再利用を変える。

## Xから共有するものを作る

セルiの共通積分基底を \(b_{ik}\)、owner qのKタイル窓を \(J_q\) とする。

\[
Z^{\rm win}_{bqk}=\sum_{i\in J_q}X_{bi}b_{ik},\qquad
H_{ba}=\sum_k v_{ak}Z^{\rm win}_{b,q(a),k}.
\]

\(v_{ak}\) は幅と中心からcompute中に作る入力側の正規化係数。
Hの幅・中心微分も、同じ窓ベクトルに係数の微分を掛ける短い内積で得られる。
出力側もXをdYへ置き換えて同じ方法でGを作る。

窓がchart端をまたぐと、基底の位相をunwrappedな位置へ合わせる。
chart周期Q、profile周期P、\(\theta_k=2\pi k\,image\,Q/P\) として

\[
\begin{pmatrix}Z_c'\\Z_s'\end{pmatrix}
=\begin{pmatrix}\cos\theta_k&-\sin\theta_k\\
\sin\theta_k&\cos\theta_k\end{pmatrix}
\begin{pmatrix}Z_c\\Z_s\end{pmatrix}.
\]

これは[owner Cの回転](cell-mass-owner-coefficients.ja.md)と逆符号。
PがQを割り切るとは仮定しない。端のtileを捨てたり、cell境界をclipしたりしない。

```cpp
// 同じstream。level間だけ依存し、各levelの要素は並列。
z_tree = build_z(signal);                 // 既存direct/mergeを選択
parallel_for(batch, owner, mode) {
    total = 0;
    for (node : cover_owner_window(owner))
        total += rotate_chart_image(z_tree[node], mode);
    z_window[batch][owner][mode] = total;  // 各要素に一意writer
}
// kernel境界で公開。z_treeの一時領域はここで不要になる。
parallel_for(atom_slot) {
    original = sorted_rows[atom_slot];
    profile = make_profile(atoms[original]);  // register
    H[batch][atom_slot] = dot(profile, z_window[batch][owner]);
}
```

`cover_owner_window`の小ループは各threadの集約。全ownerやmodeを直列に処理する意味ではない。
Z階層の一時領域と窓配列は集約中だけ共存する。
N1024/B4/M8では、各側138,720bytesだった全階層の保存が、
入力窓4,352bytes、出力窓2,176bytesになる。階層構築そのものは省略していない。

## 一度のsin/cosからharmonicを作る

`coefficient_method="polynomial_recurrence"`は、
[mode定数表](cell-mass-polynomial-constants.ja.md)に加えて位相も共有する。
\(\alpha=2\pi\mu/P\) のdouble sin/cosを一度計算して

\[
c_0=1,\ s_0=0,\quad
c_{k+1}=c_k\cos\alpha-s_k\sin\alpha,\quad
s_{k+1}=s_k\cos\alpha+c_k\sin\alpha
\]

で最大16harmonicを生成する。係数・微分・位相の内部精度はdouble。
全atomの係数表は保存せず、H/Gはthreadのregister、Cは現在の32atom chunkだけsharedへ置く。
scalar polynomial、定数表のみの方式、DFTも比較用に残す。

## source 48d317dbでの測定

L4 / Torch2.11.0+cu130 / CUDA13.0、N1024/A52428/B4、FP32、
double Z、G/Parameter勾配融合、owner C、幅256/split128。
Q128、入力tile128/出力tile256、M8、sigma初期0.2..12、seed120。
固定X/dYのforward/backward/fused AdamW、Graphなし、TF32無効。
2fresh process、逆順、warmup10/同期wall30標本。

| 方式 | step ms run0 / run1 | allocated MiB | reserved MiB |
|---|---:|---:|---:|
| table・全階層・64thread | 2.724 / 2.719 | 5.261 | 6 |
| table・共有窓・64thread | 2.276 / 2.278 | 5.003 | 6 |
| recurrence・共有窓・32thread | 1.935 / 1.930 | 5.003 | 6 |
| recurrence・共有窓・64thread | 1.932 / 1.965 | 5.003 | 6 |
| recurrence・共有窓・128thread | 1.973 / 1.974 | 5.003 | 6 |
| dense通常 | 0.439 / 0.439 | 32.313 | 42 |
| dense workspaceなし | 0.632 / 0.622 | 16.063 | 22 |

32threadの時間比は通常denseの4.41/4.40倍。約2倍の目標には未達。
メモリはworkspaceなしdense以下。時間とメモリの基準は別のdense設定であり、
allocatorの値にCUDA moduleの定数表9864bytesや全process使用量は含まない。

別のevent診断ではtableのG/Parameter融合が全階層1.15→共有窓0.78ms、
recurrenceで0.67ms。recurrenceのowner Yは0.34、dXは0.30ms。
準備は約0.26msで、窓生成は各側約0.073ms。phase時間の和をstepの代用にしない。

同じsourceの128/A819/B4・微分保存・幅16/split8/16threadでは、
recurrence0.722/0.776ms、table0.750/0.736ms、通常dense0.497/0.442ms。
CST200KiB / reserved2MiB、workspaceなしdense264.5KiB / 2MiB。
この小さいcaseでは約2倍以内のstep時間を満たす。

## 検証と保全

CUDA・数学1,277 passed（37.43秒）、failed/skippedなし。
全52,428atomのY/dX/全4列勾配と、同一GPU fused AdamW更新・moments・stepも
CPU float64 oracleへ照合。許容差は緩めていない。
CPU2006 passed / 4025 skipped / 既存警告18件（42.44秒）。

job `l4job-059a4324b021479783b7d345d54e83ef`:

- source SHA256 `0787ef25a9208b91c2972f78b30c7a5ed854d6cb7b3eb3afdb850d5d322cf4f9`
- result SHA256 `a4bbe036d9340411c1b8fe85d649508c88f9508fa3c66c9c00a87955d5fd6f6d`
- driver SHA256 `178a30935a4353b12a73b930a883594031b069d1fa9f9b36eabe00fb0bcb47da`
- binary SHA256 `2ea0bd72529b0d18ab31de3c3ed2a1dbb746d483c81d108da4ddad14a2386ee8`

189manifest files、321runtime source、driver、128helper、binaryを照合。
ignored `output/cell-mass-1024-20261027/window-recurrence/`へarchiveと検証記録を保持する。
再実行はarchiveの`__pool_driver__.py`にsrc/rootのPYTHONPATHと空の`CST_JOB_OUTPUT`を渡す。
公開dispatcher・CUDA Graph・複数bank/多軸は未接続。

直前のjob `l4job-54107ba3735c465bbe0f70733d375d31` はC++のbuild_z宣言の引数不足で
build失敗した。48d317dbで修正。性能結果には数えず、`window-build-failed/`に保持する。
