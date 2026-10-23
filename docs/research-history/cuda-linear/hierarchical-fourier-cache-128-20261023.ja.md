# N128: profile保存容量と再計算の比較

2026-10-23。`kernel/hierarchical-fourier-tiles` の研究候補。
直前の[packed実験](hierarchical-fourier-packed-128-20261023.ja.md)から、
cache容量だけを分離して比較する。公開dispatcherへは未採用。

## 配置と寿命

ここでいうNative/profile cacheは、準備kernelがglobal memoryへ明示的に保存する配列。
ハードウェアのL1/L2、block内shared memoryとは別物である。
この比較でL2 hit率・実DRAM転送量・SM同時実行は測らない。

| 値 | 保持場所・寿命 | 今回の扱い |
| --- | --- | --- |
| Meta/係数/Norm | global、forwardまたはbackward呼び出し内 | 全atomの正規化・支持を維持 |
| Native | global、準備から局所出力/縮約/残差出力の最終利用まで | 軸ごとの支持が容量以下のときだけ保存 |
| 局所H | local output kernelの作業値、出力へ直ちに消費 | globalへのstoreなし、既存kernelを維持 |
| 非局所H | global `[B,A,3]`、そのforward内で消費 | 容量は据え置き |
| Z/C | global、階層/複数blockで利用 | 必要nodeと既存の階層依存を維持 |
| saved X/P/routes/順列表 | backwardまで | 同じforward状態で再計算し、元atom IDへ勾配を戻す |

Nativeの形状は`[A,2,3,S]`、FP64。3 fieldは値・中心微分・幅微分である。
S=128/64/32で、A128なら768/384/192 KiBとなる。矩形容量を確保するため、
実際に保存しないatomのslotも残る。atomごとの可変長圧縮やsharedへの移動は今回含めない。

Normはすべての観測siteについて混合後profileの離散L2とその微分を計算する。
Nativeを保存できない軸はBoundsに`-1`を記録し、既存の完全な係数評価へ戻る。
共有bankの完全タイルはZから、direct成分と境界はprofileから計算する。
これはcacheの保存上限であり、支持を32/64 siteで打ち切る変更ではない。
保存できない残差cacheの生成も省く。幅・中心が更新されたら毎回判定し直す。

旧Packed/Local/Fourier Recipeのschemaを変えず、独立Recipe/Algorithmとして登録する。
同じP/routesの並べ替え、32 group worker、8 local worker、stream分岐位置を保つ。
Hのstride変更、forward微分cache省略、normの係数化、FP32 cache化は混ぜない。

## 検証

新たにA7/B3で空支持、ゼロ振幅、domain端、floor付近、cache容量を超える広い支持、
幅を交換して保存対象が入れ替わる場合のY/dX/全P勾配、retained backwardを検査する。
既存独立FP64 quadrature/KKT oracleと従来cache128経路の双方を対照とする。
既存runnerでN128/B32/A128/M8 mixed seed41、direct/cache128/cache64/cache32/Denseを
同じ完全step条件で比較し、Graph replayによる動的な支持変更も確認する。

宣言検査はPASS。GPUの正しさ・完全step時間・capture/replay込みallocated/reserved
ピークはまだ未取得。保存量の減少から速度改善を推定しない。
