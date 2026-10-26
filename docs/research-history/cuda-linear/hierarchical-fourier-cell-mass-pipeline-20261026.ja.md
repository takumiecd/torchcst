# 区間確率128版：処理をどう分けるか

2026-10-26。現行block128 CUDAから新しい正規化への処理分割案。
新しい関数名は設計上の仮名で、実行可能なCUDAへの追加ではない。
現在動くのは[ソート付きL2版](hierarchical-fourier-block128-reading-20261026.ja.md)。
数学契約は[区間確率profile](../../cell-integrated-profile.ja.md)を正本とする。
新方式のforwardはprepareとcomputeの2回のkernel起動を基本構成とする。
独立したnorm/mass準備kernelは置かない。これは新方式の設計で、現行L2版の起動数ではない。

## 全体と、一つのblockの流れ

入力128・出力128・128 atomsで、現在の4バッチ×出力32 sitesを一つのblockが
担当する構成を出発点にする。これらの値を一般化後の最適値として固定しない。

```cpp
// forward全体。prepareとcomputeだけを起動する。
prepare<<<prepare_grid, prepare_threads>>>(atoms, spec, layout);
compute<<<compute_grid, compute_threads, shared_bytes>>>(X, layout, spec, Y);
```

```cpp
// computeの一つのblock。
init_block();
load_block_inputs();
sync_block();

build_z_hierarchy();       // 子Z → 親Z。依存する段階の間で同期する。
compute_atom_h();          // 別々のthreadが別々のatomを担当する。
publish_h();
sync_block();              // 全員がZを使い終え、Hを公開した境界。

build_output_c();          // 一つのC要素の書き手を一つにする。
sync_block();
evaluate_and_store_y();    // Cの展開＋直接成分＋境界部分。
```

prepareはソートkeyの作成、並べ替え、実行用pack、groupとbegin/endの作成を担当する。
現在のatom値から毎回準備し、新profileの支持則とセル幅も候補範囲へ反映する。
全siteの走査や面積正規化の別kernelを、prepareとcomputeの間へ追加しない。

computeは係数の面積正規化とZ/H/C/出力を担当する。
正規化用のmake_atom_profileはcompute内部のdevice関数とし、kernelを起動しない。
必要なatomの生係数と面積Sから、その場で正規化済み係数を作る。
H担当threadのregisterを、別のC担当threadが直接読めるとは仮定しない。
初版のC担当threadは必要な出力係数を同じdevice関数で評価する案とし、
小さな係数chunkをsharedで共有する最適化は、容量・同期・再評価コストを見て選ぶ。

block内の関数は同じkernelのdevice関数で、関数分割とkernel分割は別である。
prepareとcomputeはまず同じstreamで順に起動する。
一部threadだけが入る分岐の中に、block全員の同期を入れない。

これは最初に読む処理の分割案であり、準備担当と計算担当の重ね合わせ、
担当割合の調整、固定担当表の将来案を取り消さない。
必要なZの完成後にHを始められる依存関係を保ち、担当の調整と数式を分離する。
この整理のためだけに動的job管理や待機whileを追加しない。

## 関数の役割と現行コード

| 仮名 | 読むもの → 作るもの | 現行CUDA |
| --- | --- | --- |
| prepare | 中心・幅・支持則 → 順列、group、blockのbegin/end | prepare_atom_layout、get_range |
| make_atom_profile（compute内部） | atom・宣言 → 面積から正規化した係数と必要な微分 | prepare_norms/alpha等の役割をcomputeへ融合する部分 |
| init_block | block/thread番号 → 小バッチ、担当出力範囲、atom担当、sharedのview | block128先頭 |
| load_block_inputs | globalのX・実行用atom情報 → shared | block128のinit |
| build_z_hierarchy | X・共通基底・完成した子Z → Z | build_z_tileとnodeループ |
| compute_atom_h | Z・入力の端・atom情報 → thread内の完全なH | compute_atom_responses、contract_component |
| publish_h | thread内のH → sharedのH | valuesへの書き込み |
| build_output_c | H・正規化係数・担当範囲 → sharedのC | build_c |
| evaluate_and_store_y | C・セル積分基底・直接/境界成分 → 担当Y | store_output |

## 数式を差し替える場所

新しい実行契約はNormalizationSpecのcontinuous_mass/profile_supportを要求する。
現行FourierLinearSpec/Algorithmを暗黙に新しい意味へ読み替えない。

1. 全siteの二乗norm計算を、支持全体の面積Sの解析評価へ置き換える。
   正規化係数と必要な微分を用意し、混合成分の正規化もSpecの意味に合わせる。
2. 生係数をSで割り、中心による位相をatom側に持たせる。同じ階層・bankでは
   周波数を共有する。sigmaから係数への写像はまだ確定していない。
3. site側はセル区間で積分した基底を使う。積分係数Jをこの基底へ含める案とし、
   atom係数に二重に掛けない。入力のZだけでなく、出力のCの展開にも反映する。
4. 候補範囲はセルと支持が正の長さで重なる範囲にする。中心だけを判定する現行の
   bounds/get_rangeを流用して境界セルを落とさず、支持半幅にspacing/2を加えた外側まで考える。
5. 完全なセル・タイルは共有基底で、支持で切れたセルは正確な区間積分で扱う。
   Z/Cへ含める集合と直接部分を分け、取りこぼし・二重加算を防ぐ。
6. 親基底を子基底から同じ式で表せるときだけ子Zを継承する。周波数、原点、spacingと
   積分係数を確認し、現行の偶数mode則を検証せず流用しない。

現行は有限格子をclipするため、面積1でも観測site上の合計1とは限らない。
非負性、最小幅、支持、周期化/有限境界を決めてから新方式を実行へ接続する。
現行fit表には非負密度の保証がなく、新方式の係数として未検証のまま使わない。

## メモリをどう持つか

| 場所 | 最初の整理案 |
| --- | --- |
| global：正本 | 元のatom Parameter、X/Y。並べ替えは実行用コピーで行う |
| global：実行用配置 | sorted parameters、元の行対応、group、begin/end |
| global：配置に必要な情報 | 支持/bank情報は必要な分をlayoutへ含める。面積専用bufferは初版の必須配列にしない |
| register | 面積の逆数、必要な正規化係数、H/Gと微分。compute内で利用時に評価する |
| shared | 小バッチの入力、Z、公開するH/G、担当範囲のC。同期後に領域を再利用する |

全atom分の係数をglobalへ展開することは前提にしない。
例として128 atoms×2軸×4成分×17基底×FP64は139,264 bytesとなり、
128×128 FP32の密な重み65,536 bytesを超える。これはpayload比較で、GPUピークではない。

将来scale[atom][axis][value, d_mu, d_width]をglobalへ保存するなら、
128×2×3 FP64で6,144 bytesとなるが、初版の必須配列ではない。
保存する場合も2起動を維持するにはprepareへの融合等を検討する必要がある。
まずcompute内で評価し、不要な微分・軸間共有・再評価・register spillを後で測る。

H/Cはまずblock内で作って使う。大きいsigmaのHをblock間で共有するglobal経路と、
backward用のH保存は、重複計算と保存量を比較する選択肢として残す。
係数・面積・支持は現在のatom値から再準備し、layout_versionだけで再利用を判断しない。

## backwardと整理の順序

2回のkernel起動という今回の決定はforwardが対象である。
backwardのdP部分和の合算、元の行への戻し方まで2回に固定したという意味ではない。

```cpp
// dXの経路。同期はforwardと同じ公開・再利用の境界に置く。
init_block();
load_block_inputs(dY);
sync_block();
build_z_hierarchy(dY);
compute_atom_g();
publish_g();
sync_block();
build_input_c();
sync_block();
evaluate_and_store_dx();
```

atom勾配にはHも必要で、保存・再計算・一部保存の選択肢を残す。
H/Gを同じthreadで作る場合と別の集団で作る場合を切り替えられる境界で分ける。
別の担当が作るなら両方の完成・公開後にatom勾配を計算する。
現行128版の小バッチ群のtile0が全atomのdP部分和を作る所有関係を出発点にできる。
L2の微分補正を面積正規化の商の微分へ置き換え、最後に元のParameter行へ戻す。

実装は、(1)現行の処理境界を読む、(2)新profileの式とCPU直接参照、
(3)新宣言とcompute内の面積正規化・セル評価、(4)同じ参照に対するZ/H/CとY/dX/全atom勾配、
(5)threadの割合・必要Z・融合・保存/再計算の実測、の順に進める。

本書と現行CUDAへの段階コメントの追加では、実行文・演算順序・配列配置を変更しない。
新しい確率profile kernelの実装・GPU検証が済んだことは意味しない。
コメントと空白を除いたCUDA token列が変更前と一致すること、文書の相対リンク14件、
code fence、例示したpayloadの計算、git diff --checkを確認した。GPU実行は行っていない。
