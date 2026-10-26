# profileタイル内でCを共用する

2026-10-26。[タイル窓の定義](cell-mass-tile-window.ja.md)を固定し、
Y/dXのCを8site leafごとに重複生成していた部分だけを変更する。
比較元はcommit `97babc4c`。周期・profile・近似誤差・block配置は同じ。

## なぜ共有できるか

profileタイルtに対する係数は

\[
C_{btk}=\sum_{a:\,t\in\mathcal T_a}
  \alpha_a H_{ba}\,u_{atk},\qquad
Y_{bi}=\sum_k C_{b,t(i),k} B_k(i).
\]

タイルの所属条件とunwrappedな位相はprofileタイル内で共通。
siteに依存する部分はB_k(i)だけなので、同じprofileタイルを構成するleafごとに
Cを作り直す必要はない。dXもHをG、出力側を入力側へ置き換えた同じ構造。
Zのleaf幅は入力階層の構築単位であり、Cを分ける根拠にはしない。

128site、batch4、mode8（17基底）では、旧方式は両側とも
16leaf×4batch×17=1,088個のCスカラーを生成していた。
入力profile tile16/出力tile32の固定caseで、Yは4tile×4×17=272個、
dXは8tile×4×17=544個になる。

## コードで変える部分

blockが所有するsite範囲から、そこに含まれる異なるprofileタイル番号を求める。
threadは(profile tile,batch,mode)に対応するCをregisterへ蓄積し、
全atom chunkの後にsharedへ渡す。各siteは自分のprofileタイルのCを読み、
自分の積分基底を掛けて出力する。

```cpp
first_tile = block_first / profile_tile_sites;
tile_count = (block_end - 1) / profile_tile_sites - first_tile + 1;
C = accumulate_by_profile_tile(first_tile, tile_count);
sync();
for (site : owned_sites) {
    tile = site / profile_tile_sites;
    Y[site] = dot(C[tile], integrated_basis(site));
}
```

対象caseのY/dXは各4blockのまま。atom chunk、batch幅、dtype、
係数計算方式、sin/cos、Z、H/G、Parameter勾配の計算は変更しない。
sharedの確保上限も今回は同じにし、有効なCの個数だけを減らす。
全Cのglobal bufferは追加しない。

profileタイルがblockの担当幅より大きい場合、Cは各block内で一度ずつ作る。
blockをまたいで一度だけ計算する方式は、global共有やlaunch変更を伴う別の比較にする。
今回のtile16/32・block32のcaseではprofileタイルがblockをまたがない。

比較では旧CUDA sourceをbyte単位で保存し、同じexecutorと入力から両extensionを呼ぶ。
独立参照でY/dX/全atom勾配を確認した後、検証とは別のfresh processで
forward+backward時間とallocated/reservedピークを測る。

## 検証と同条件の実測

比較元`97babc4c`のCUDA source SHA256は
`2904984f6d35d73cc99e8ef7754fae3efcab7ddc3b2cd1a32f89847840852621`。
旧sourceを保存し、両extensionへ同じexecutorから同じ入力を渡した。
候補source commitは`778a8911`。CPU全体は2006 passed / 2827 skipped / 既存JIT警告18件。
L4対象79件（CUDA51件・CPU/math28件）はpassed、failed/skippedは0。
追加したtile16/32とchart全体のtile128は、tile境界の両側とbatch末尾も含む。

819atomのFP32/FP64両方で、変更前と変更後のY・dX・全atom勾配の最大絶対差は全て0。
独立したTorch参照に対する最大絶対誤差も同じだった。

| dtype | Y | dX | 全Parameter勾配 |
| --- | ---: | ---: | ---: |
| FP32 | 5.96e-8 | 3.73e-8 | 5.96e-8 |
| FP64 | 1.80e-15 | 2.33e-15 | 1.50e-15 |

近似のprofile定義、正規化、支持タイルの選択、係数計算と微分は変えていない。
従来の実装誤差基準も緩和していない。

NVIDIA L4、Torch2.11.0+cu130、CUDA13.0、128×128・819atom・batch4・FP32・mode8。
入力T=16/出力T=32、K=3、局所周期48/96、spacing1、sigma0.2〜12、seed120。
merge Z・1要素4thread、係数は直接多項式、Y/dXのgridは各4blockで固定。
検証とは別のfresh processで、旧/新を交互に2processずつ測定した。
各processはwarmup5回の後11サンプル。build/JIT時間は含まない。

| 方式 | process1中央値 | process2中央値 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| leafごとのC | 8.019 ms | 7.912 ms | 85,504 byte | 2 MiB |
| profileタイルごとのC | 5.215 ms | 5.310 ms | 85,504 byte | 2 MiB |

forward+backwardの改善は1.54倍/1.49倍（時間は約35.0%/32.9%減）。
prepare・入力検査を含むphysical entryのwall時間で、optimizer・Graph・公開CSTLinearは含まない。
同じ数学の変更として比較できるが、denseに対する優位や別shapeの改善は未検証。

別の段階別GPU event診断の中央値は以下。和を全体時間として使わない。

| phase | leafごと | profileタイルごと |
| --- | ---: | ---: |
| prepare | 0.407 ms | 0.387 ms |
| Zx | 0.0379 ms | 0.0369 ms |
| Zy | 0.0358 ms | 0.0358 ms |
| H | 0.196 ms | 0.197 ms |
| Y | 3.522 ms | 1.619 ms |
| G | 0.214 ms | 0.215 ms |
| dX | 2.454 ms | 1.749 ms |
| Parameter勾配 | 0.620 ms | 0.621 ms |

Yは約2.18倍、dXは約1.40倍。Cの有効要素数を4分の1/2分の1にしても、
atomの準備・H/Gの読込み・基底展開が残るので全体が同じ比で速くなるわけではない。
主に変更したY/dXが改善し、変更していないH/G/Parameter勾配はほぼ同じ時間だった。

Tensor確保ピークは両方式83.5KiB、shared確保上限も9,984byte/blockで同じ。
Y/dXは両方式register90/thread・stack40・表示上LOCAL0。
ここでの値はコンパイル時情報であり、warp占有率・spill trafficの実測ではない。

gridは各4blockのままなので、次の比較ではprofileの周期・タイル幅を固定して
実行上のblock担当範囲を調整できる。今回の変更には含めない。

## 証拠と再現

job `l4job-1616c235461d426f8b2c7c6a1ebb161f`。
source archive SHA256
`d7e99da4051ca6d37888b5d6b075b2582c93cc31f9577fe451abf7d5459d04f2`、
result archive SHA256
`d8d8718884e6c95c76960e3e58c29901c71506dcdaee673f8ac135e59079e604`。
両hash、保存した旧sourceのhash、候補321 runtime source filesのsnapshotとのbyte一致を照合した。
wheel/sdist buildと両packageの321 runtime source filesのbyte照合も通過。

raw evidence・旧source・比較driver・build/CPU log・package照合は
ignored `output/cell-mass-c-reuse-20261026/`へ保持する。
source/result archive原本はhost共通poolのjob directory。job成功と所有slotの停止を確認済み。

再実行: CPUは`python -m tools.kernel_dev test --suite cpu`。
CUDA/nvcc/Ninja環境では
`python -m pytest tests/test_cell_mass_coefficients.py tests/test_cell_mass_reference.py tests/test_cell_mass_cuda.py -q`。
速度比較のdriverは上記ignored directoryとjob snapshotに保持する。
2process/方式は1つのGPU job内で実行したもので、11サンプルを11独立runとは数えない。

続く[block担当site数の比較](cell-mass-block-sites.ja.md)では、同じprofileのまま
32/16/8siteの実行上の分担を切り替える。上記のC共有比較の測定条件は変更しない。
