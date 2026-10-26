# GとParameter勾配の再投影をまとめる

`fuse_g_parameter_grad=True`は微分再計算経路で使う実行設定。
forwardのHはYで使い終えたら解放し、backwardは保存したZxからHを再投影する。
その同じthreadがZyからGとその微分も計算するので、GをdX用に書き出す。
別途Gだけをもう一度計算するkernelを省ける。

この設定と`cache_projection_derivatives=True`の同時指定は明示的に拒否する。
保存案との比較用に、既定は両方Falseを維持する。

## 実際の呼び出し順

```cpp
// forward
rows, offsets = prepare_inputs_and_layout(with_groups=true);
Zx = build_z(X);
H = compute_h(X, Zx, atoms, rows);
Y = compute_y_from_owners(H, atoms, rows, offsets);
save_for_backward(X, atoms, rows, Zx, offsets);
// Hは以後のctxに保持しない。

// backward
Zy = build_z(dY);
G, grad_atoms = compute_g_parameter_grad(X, dY, atoms, rows, Zx, Zy);
dX = compute_dx_from_owners(G, atoms, rows, offsets);
return dX, grad_atoms;
```

owner Cの設定を使わない場合は、Y/dXの部分だけ従来の出力部分和経路を使う。
Zの保存精度は別設定`double_intermediates`で選ぶ。
全52,428atomの極小勾配の更新検証ではTrueを必要とした。

## 1threadの処理

各threadは一つのatomの全batchを担当する。
4batchごとに投影値と幅・中心微分をregisterへ得る。

```cpp
for (batch_tile : batches_by_4) {
    input  = project_input_value_and_derivatives(Zx, atom);
    output = project_output_value_and_derivatives(Zy, atom);

    for (batch : batch_tile) {
        H = input.value[batch];
        G = output.value[batch];
        accumulate_amplitude_width_and_center_gradients(H, G, input, output);
        global_G[batch, sorted_slot] = G;
    }
}
global_grad_atoms[original_id] = accumulated_gradients;
```

数学的に必要なHはこの再投影のregister値として存在する。
元X/atomsはautogradのversion検査のためにも保持し、更新後の古いforwardを使わない。
retained backwardは新しいG/勾配tensorを生成し、前回の出力を上書きしない。
Gとatom勾配が同じstreamで完成した後、dXのconsumerを起動する。

## 同じGPU・同じ条件での測定

source `5da4dc43`、N1024 / A52428 / B4、FP32、doubleのZ、owner C、
幅256 / split128 / atom64thread。固定X/dYのforward/backward/fused AdamW。
比較元も同じsourceの`fuse_g_parameter_grad=False`。各2fresh process、逆順、warmup10/30標本。

| 経路 | step ms run0 / run1 | allocated MiB | reserved MiB |
|---|---:|---:|---:|
| owner C・G別生成 | 3.784 / 3.745 | 5.530 | 8 |
| owner C・G/勾配を融合 | 3.273 / 3.268 | 5.261 | 6 |
| dense通常 | 0.445 / 0.443 | 32.313 | 42 |
| dense workspaceなし | 0.636 / 0.617 | 16.063 | 22 |

融合で約13〜14%短縮。メモリも減り、workspaceなしdense以下。
時間は通常denseの7.35/7.38倍で、目標の約2倍には未達。
128threadの融合版は3.294/3.288msで、今回の64threadより少し遅かった。

別測定では64threadのG単独が0.486ms、再計算Parameter勾配が1.428ms、
融合した両方が1.430ms。G単独の重複を除いたことと整合する。
16threadでは融合2.779msに対し128threadは1.433msだった。
128の小さいatom集合で速いthread幅を、52,428atomへそのまま移す判断はしない。

## 検証と証拠

CPU2006 passed / 3325 skipped / 既存警告18件（42.51秒）。
CUDA・数学577 passed、failed/skippedは0（L4、Torch2.11.0+cu130、CUDA13.0）。
FP32/64、両Z方式、両中間精度、owner C有無、batch/atomの0・端数、
retained backward、極小中心勾配のAdamW更新を含む。
全52,428atomは独立CPU oracleとY/dX/全4列の勾配、同一GPU融合optimizerの更新・moments・stepを照合した。

job `l4job-596f6278a65a43ab910285c3a95837ba`:

- source SHA256: `e0e6dfb89c38db72cabb63aa7717620979a31d6b226e8dcc6bcb6469f94ea0cd`
- result SHA256: `eff98b16daaf63c6d105597c60dcefcdaaae06a372cfc6ce2ecd5fb413ae9312`
- driver SHA256: `a84af8c3cd1565badbc59306f5074fd2d298df1dc743fc204de80f6a77573b8c`
- binary SHA256: `09b1df51cc17057e1c1d732d90c343b4eceb7a3cb27a613a781f7274f33461da`

111 manifest files、321 runtime sources、driver・128 helper・binaryを照合した。
wheel/sdistも321 source filesをcommitの内容と比較済み。
rawはignored `output/cell-mass-1024-20261027/fused/`。
再実行はsource archive内の`__pool_driver__.py`へsrcとrootのPYTHONPATH、
空directoryの`CST_JOB_OUTPUT`を渡す。公開dispatcherへの統合・Graphは未実施。
