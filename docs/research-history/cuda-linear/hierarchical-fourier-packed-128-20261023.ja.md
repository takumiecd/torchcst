# N128: live atom packing とグループ一括縮約

2026-10-23。`kernel/hierarchical-fourier-tiles` の研究候補。
公開dispatcherへの採用ではない。

## 実装と比較範囲

支持の階層集合・使用bank集合・local/direct/shared経路で分類し、入力支持の
開始位置、出力支持の開始位置、元atom IDの順に並べる。同分類の最大4 atomを
1グループとする。N128/M8/A<=128専用の1 CTAで分類、sort、P/routesの物理配置、
group開始表を作る。係数・正規化・cacheは並べたPから一度だけ生成する。
group数はdeviceに残し、固定worker数の1 launchで全groupのHを縮約する。
出力側のlocal tile参照表は既存経路を使う。Z生成は既存の階層依存順を保つ。

Parameterとoptimizerの並びは変えない。packed row→元atom IDを保存し、
backwardで全4 fieldの勾配を元の順序へ戻す。元X/Pのversion検査も維持する。
学習による支持・階層変更は毎回GPUで再分類し、Graph replayでも更新する。

小さいlocal atomのHは既存kernel内で消費する。一方、今回は比較を切り分けるため
非局所Hの割当 `[B,A,3]` とNative cache `[A,2,3,128]` を据え置く。
N128/A128では後者だけで768 KiB、dense W FP32の64 KiBを超える。
したがって今回の候補だけでは構造的なメモリ目標を満たさない。
追加のpacked P/routes、逆順表、group表も完全stepピークの比較に含める。
微小なH scratchを強制して多段のwaveに分けることはせず、必要量を測って決める。

## 検証計画と現在の状態

- CPU全suite: 2260 passed / 5744 skipped。CUDAのskipはGPU検証に数えない。
- live分類、A7/B5端数、retained backward、元入力のversion検査を追加。
- 既存local経路の回帰と新候補をL4で検証する。
- 既存Linear runnerのN128/B32/A128/M8 mixed seed41で、direct、従来local、
  packingのみ、grouped serial/overlapを同じ独立oracle・完全step条件で比較する。
- Y/dX/全atom勾配、動的支持、capture/replay込みallocated/reservedピークを確認する。

GPU結果は未取得。初期配置の39 groupはCPU設計からの予測であり、実機で検証する。
profile cache削減は別変更として、この比較後に扱う。
