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
