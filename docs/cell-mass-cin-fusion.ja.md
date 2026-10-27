# C生成・site展開の分離とbackward C_in融合

1024/A52428/B4、面積正規化とwhole-tile窓を保ったまま、Cの生成とsite展開を分けて測る。
source/result archiveを回収して照合した値だけを採用し、段階別event時間の和で
完全step時間を置き換えない。現在の実測の入口は[状態一覧](cell-mass-status.ja.md)。

## 分けた境界

```cpp
C_partial = compute_owner_partials(H_or_G, atoms, sorted_rows, group_offsets);
output = expand_owner_partials(C_partial);
```

前者はowner/splitごとに32atom chunkの係数をsharedへ生成し、C部分和を作る。
後者はsplitと隣接ownerのCを合計し、共通の積分基底からY/dXを展開する。
後者にatomの係数生成は含まない。旧compute_y/dx_from_ownersも同じ二つの処理を呼ぶ。

C部分和は[split,batch,owner,mode]のFP64。
入力owner8 / 出力owner4、split128/B4/M8では、入力544KiB / 出力272KiB。
呼び出し内だけのscratchであり、atom・Jacobianの全表は作らない。

## 分離した実測

source `835dc92a`、NVIDIA L4 / Torch2.11.0+cu130 / CUDA13.0。
N1024/A52428/B4、FP32 H/G、FP64 Z/C、M8、Q128、入力tile128/出力tile256、
split128、C thread128、atom thread32。係数はpolynomial_recurrence。
2fresh process、2巡目は段階の測定順を逆転し、warmup10/31event標本。

| 段階 | median ms run0 / run1 |
|---|---:|
| C_out生成 | 0.23757 / 0.23757 |
| C_outの合計・Y展開 | 0.11571 / 0.11571 |
| C_in生成 | 0.22221 / 0.22118 |
| C_inの合計・dX展開 | 0.08602 / 0.08294 |
| 従来のY一括呼び出し | 0.34509 / 0.34406 |
| 従来のdX一括呼び出し | 0.29594 / 0.29389 |
| G/Parameter勾配 | 0.67584 / 0.67379 |
| 従来のG/Parameter勾配→dXをまとめた呼び出し | 0.96154 / 0.97382 |

各native呼び出しはその段階のTensor確保を含む。独立して測ったため、分離時間を
加算して一括時間を再現するものではない。生成側が重い一方、site展開も無視できない。
G/Parameter投影の入力係数をC_in生成へ渡す融合を、次の実装候補に選んだ。

追加した分離の8件のCUDAテストが通過し、分離/一括のY/dXはbit一致。
全52,428atomのY/dX/全4列勾配・AdamW更新/moments/stepも独立CPU oracleへ照合した。

job `l4job-1ccb699a62f84a24be5a3665e621d278`:

- source SHA256 `635c8d3b8d445352c425413154f0feff32327679da50b71d1ec25d5273daa4c0`
- result SHA256 `53c87e970667f9eea6d4f8011ab6fc0a9975d034630f7b077439a28e9aa0a337`
- driver SHA256 `978dc664888053c7b744e1bd11b2809caacc50a4efb64465c1d0f5843e20dd12`
- binary SHA256 `14c90f4a0619023c2064951807d8197b5f753eb9a09456770f49a54e2e6febbe`

24manifest files・321runtime sources・driver・binaryを照合。
rawはignored `output/cell-mass-cin-fusion-20261027/split-short/`。

初回 `l4job-ae617edcb3314490b2890a71dab38aef` はColab接続が切れ、
完了応答が990秒でtimeoutした。結果archiveを回収できていないため正しさ/速度に数えない。
source SHA256 `1f9e8a637b7a2ece0bb1ec54f77ffaf575c193c71cac4dec5304d2c940b38232`、
321runtime sourceを照合し、要求・source archive・通信logを`unretrieved-split/`へ保存。
所有VMの停止とpool recoverを確認してから短い計測を再提出した。

## 融合版の所有と順序

batch4以下の試作では、blockが入力ownerとatom chunk範囲を担当する。
同じ入力ownerは、出力ownerごとに連続したsorted_rowsの範囲を順に読む。
全範囲をchunk数で分割し、各atomを一つのblockへ一度だけ割り当てる。

```cpp
for (atom_chunk : assigned_owner_ranges) {
    // 各threadが異なるatom。入力係数をH/dHとC_inで共有する。
    H_and_dH, input_coeff = project_input(Zx);
    G_and_dG = project_output(Zy);
    store_original_order_parameter_grad(H_and_dH, G_and_dG);
    publish_chunk_G_and_input_coeff_in_shared();
    sync();
    accumulate_C_in_in_shared();
    sync();
}
store_partial_C_in();
// 次kernelでsplitと隣接ownerを合計してdXを展開する。
```

Parameter勾配は全batch分を一つのthreadで計算するため、同じatomへのatomicは不要。
C_inの部分和は入力owner/splitの一意writer。dXも既存の一意writerを維持する。
4を超えるbatchでは別の勾配集約設計が必要なので、試作flagで明示的に拒否する。

従来dXが読むGの入力dtypeへの丸めをshared上でも保つ。
Parameter勾配では丸める前のdouble H/Gを使う。
係数・その微分・正規化・phaseの内部精度はdoubleのまま。

## 実装の入口

研究入口の`physical_linear(..., fuse_backward_cin=True)`を選ぶ。
`owner_coefficient_reduction=True`、`owner_window_moments=True`、微分再計算が必要。
`fuse_g_parameter_grad`の指定にかかわらず、融合時はforward Hを保存しない。
既定のFalseと従来経路は維持する。4を超えるbatchにはこのflagを使わない。

`kernels.cu`の`project_atom<true,true>`が、Hとその微分を作りながら、同じ入力係数を
現在のchunkのsharedへ書く。`backward_owner_partials_kernel`はその係数とGからC_inを作る。
Cの蓄積もsharedに置き、両軸の投影中にC配列のregisterを保持し続けない。
M8では32atom係数4,352bytes、G chunk1,024bytes、振幅256bytes、C544bytes、合計6,176bytes。
グローバルに残るのはC_in部分和と元順序のatom勾配。G全表を作らない。

`compute_g_parameter_grad_cin`がそのproducerを起動し、`expand_owner_partials`が既存の
split/隣接owner合計とsite展開を行う。`executor.py`のbackwardがこの二つを順に呼ぶ。
同じstreamのkernel境界で依存をつなぎ、block間のspin待ちは入れない。

## 融合版の最初の比較

source `660cb0ea`（runtimeは`907132b3`とbyte一致）、同じL4/条件で、
固定X/dYのforward/backward/fused AdamWを2fresh process・逆順で測定。
warmup10/同期wall30標本、TF32無効、Graphなし、検査/sortを含む。
JIT・oracle・初期dense W構築は測定外。batch4のlayer stepであり全networkではない。

| 経路 | step ms run0 / run1 | allocated MiB | reserved MiB |
|---|---:|---:|---:|
| 従来・C128・幅256・split128 | 1.847 / 1.852 | 5.003 | 8 |
| 従来・C32・幅256・split128 | 1.832 / 1.832 | 5.003 | 8 |
| C_in融合・C32・幅256・split128 | 1.755 / 1.748 | 4.261 | 6 |
| C_in融合・C32・幅256・split64 | 1.888 / 1.893 | 4.261 | 6 |
| C_in融合・C64・幅256・split128 | 1.780 / 1.783 | 4.261 | 6 |
| dense通常 | 0.435 / 0.437 | 32.313 | 42 |
| dense workspaceなし | 0.639 / 0.627 | 16.063 | 22 |

thread数を揃えた従来C32に対し約4〜5%短縮。
G表838,848bytesを作らなくなり、完全stepピークは約0.743MiB減った。
除いたbuffer容量とピーク差は同じではない。全配列が同時に生きるわけではないため、
単純な容量差でピークを代用しない。S64はC_in scratchが半分でもピークが下がらず遅かった。

別event診断のbackward pair（G/Parameter→dX）は従来0.954/0.953ms、
融合C32/S128で0.889/0.891ms。融合producer単独は0.823/0.815ms。
融合が投影の全計算を消すわけではなく、まだ両軸の係数・微分計算を行う。

site展開の幅32は、Y0.0686/0.0686ms、dX0.0696/0.0686msで、
幅256のY0.1157、dX0.0819msより短かった。
この段階別結果をもとに、完全stepも幅32で別jobへ確認する。

GPU・数学1468 passed（42.48秒）、failed/skippedなし。
全52,428atomのY/dX/全4列勾配・AdamW更新/moments/stepを、従来と融合32/64threadで照合。
CSTの最大差はY/dX各1.49e-8、atom勾配2.91e-11、更新delta/stepは0。
exp_avg3.64e-12、exp_avg_sq2.78e-17。許容差を緩めていない。
CPU2006 passed / 4216 skipped / 既存警告18件（44.20秒）。
Ruff・clang-format・wheel/sdistと321runtime sourceのbyte照合も通過。

job `l4job-e1ae5c45798e4f0aaa6cfc97f6f6bc84`:

- source SHA256 `89dd562eec28ab5ddf01bacbf224835c2e247648acbe4b7fdae6b6b28759a92e`
- result SHA256 `07fccc46b1b44ef5cf3f7fd9ff7fb79aee8759c420d40231bd02eebe1be1ce99`
- driver SHA256 `d076b13e5b2d13a70cb72ed2606aabb48dd2f8610e5cebdd72573f7bb7344325`
- binary SHA256 `9ab7c2bf98c681f9d2fcf859fdfaa52e833fa170e63d21ec16eb1e238d06043c`

73manifest files、321runtime sources、driver、binaryを照合。
rawはignored `output/cell-mass-cin-fusion-20261027/fusion/`。

直前のjob `l4job-cccd85c746b24f1f91a05923065b9c14` は1436 passed / 32 failed。
追加した1サイトfixtureにK3を指定し、宣言検査で拒否された32件だった。
source `907132b3` のCUDA本体を変えず、`660cb0ea`でfixtureをK1へ修正した。
この失敗jobには性能結果がない。source/result SHA256は
`20fb7d1e9079bd5c212be5a7bf9bc2e928df3c5efcf7d924fdf11a158307a803` /
`3b6f078b9f794dee63a67638ab8050e5a0b8bb883349cec3c9240c77aa2295ee`。
回収済み11manifest filesと321runtime source、binaryを照合し`invalid-single-site-test/`へ保持。
回収後のremote cleanup応答も通信timeoutしたため、VM停止・pool recover後に再開した。

## 別jobでの確認と採用設定

同じruntime source `660cb0ea`、独立したL4で再確認した。
同じfixture/精度/optimizer、2fresh process、逆順、warmup10/同期wall30標本。

| 経路 | step ms run0 / run1 | allocated MiB | reserved MiB |
|---|---:|---:|---:|
| 従来C128・幅256・split128 | 1.854 / 1.885 | 5.003 | 8 |
| 従来C32・幅32・split128 | 1.778 / 1.778 | 5.003 | 8 |
| C_in融合C32・幅256・split128 | 1.763 / 1.772 | 4.261 | 6 |
| C_in融合C32・幅32・split128 | 1.706 / 1.710 | 4.261 | 6 |
| dense通常 | 0.450 / 0.441 | 32.313 | 42 |
| dense workspaceなし | 0.727 / 0.611 | 16.063 | 22 |

同じC thread32・幅32で融合だけを比べると約4%短縮。
前の推奨C128/幅256からは、thread/幅の再調整も含め8〜9%短縮。
時間の基準は通常dense、メモリの基準はworkspaceなしdenseという別の比較点。
最短設定でも通常denseの3.79/3.88倍で、約2倍の時間目標には未達。
割当・reservedは両方ともworkspaceなしdense以下。

owner C方式ではC生成とsite展開が元から別kernelなので、site幅を32へ下げても
atom係数を生成する回数は増えない。今回の分離計測で、展開側の幅を選び直せた。

```python
# N1024 / A52428 / B4、Q128、spacing0.125、input tile128 / output tile256
settings = dict(
    z_method="merge", coefficient_method="polynomial_recurrence",
    sites_per_block=32, atom_splits=128, atom_threads_per_block=32,
    cache_projection_derivatives=False, double_intermediates=True,
    owner_window_moments=True, owner_coefficient_reduction=True,
    fuse_g_parameter_grad=True, c_threads_per_block=32, fuse_backward_cin=True,
)
```

全52,428atomの独立oracle・AdamW更新/moments/stepを、幅32の従来/融合で再照合。
CSTのY/dX最大差は各1.49e-8、勾配2.91e-11、更新delta/step差0。
先の1468件と同じruntimeであるため、再確認jobでは全suiteを繰り返さず、
採用設定の全shape検証と未計装step測定を行った。

job `l4job-5d615c06ba4f41cebe411b9ef4fcb1b2`:

- source SHA256 `4996013701ea92ba33753997a39a22f56efca3eab73dce99c6b91ffcd5b1fe47`
- result SHA256 `fd5e1dec1c903b86f79b22ac1e181541e889bfc49cc85c0fa6031a9d7bd34246`
- driver SHA256 `0e24f099f1aa5b683bc3c845cb7d32b800450af46bbdf14d922ae3537bf1cba1`
- binary SHA256 `7b9ea7b53c13d4ad78a9fdd3365292c1ac4cab6be25fd5ca84321cfab079f475`

55manifest files、321runtime sources、driver、binaryを照合。
rawはignored `output/cell-mass-cin-fusion-20261027/confirmation/`。
所有runtimeの停止を確認し、未回収/失敗のarchiveも削除せず保持する。

## 採用範囲と残るコスト

1024/B4の推奨候補は融合C32・site幅32・split128。
研究入口の既定値False、公開dispatcher未接続、batch<=4の制約は維持する。
他のbatch・幅分布・多軸・複数bank・他GPUへ一般化しない。
全GPU process使用量とCUDA Graphのcapture/replayピークは未測定。
module定数表9864bytesはallocator外として別に数える。

融合producer約0.82msが引き続き大きい。両軸の係数・微分生成と投影は残っており、
kernel数を減らすだけで目標へ届いたという結果ではない。
次はharmonicごとの係数生成にある重複演算を減らす候補を検討する。
現時点でその効果は未測定で、数学や精度を変えた結果はこの速度値に含めない。
