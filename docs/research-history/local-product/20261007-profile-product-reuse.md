# 新しいprofile productへの小型最適化の移植

ユーザーが以前の小型コアに比べて遅いことを指摘したため、PR59を再検討する。
初回profile productのorderedは収縮kernelを再利用したが、最新のlayout準備を採用していなかった。
小型コアの最適化を十分に移植した結果として初回測定を扱ったのは不適切だった。

旧integrationのL4 N128 rho3 UCは67.28us、初回profile product orderedは152.12us。
別source/数学契約の過去値なので直接のspeedup比較とはせず、実装を見直す根拠とする。

## コードで確認した不足

- orderedの位置順sortと32-bit compact keyが無効だった。
- owner rangesの並列準備・tight histogramと融合copyが無効だった。
- forward/dXのatom chunkは16、最新の小型候補は32だった。
- parameter VJPはatom16を維持してforward/dXとは別に調整する必要がある。

新route `reuse` は既存のhisttightinline/paramatom16/param4/param2のlayoutとlaunchを移植する。
forward/dXのatom chunk32、VJP atom16/warps4/batch partitions2、owner partitions4。
正規化はPR58のfull product L2とglobal floorを保持し、canonical [polar_x,polar_y,c_o,c_i]
のsource勾配を維持する。全site正規化のpreparationは初回版と同じで、順序cacheと
support-bounded preparationはこの候補では変更しない。

旧routeを変更せず、新routeを追加して初回saved/orderedと同じfixture/更新で比較する。
GPU検証後、N64/N128・rho3/8を既存完全step runnerで測り、rho3は実行順反転で再測定する。
数学的に異なる旧fixtureを速度baselineには使わない。
新routeのempty/singleton/tiny/global-floor、旧source/scalar変更後のVJP、
20 captured Torch/fused optimizer updates、長方形/spacing0.5のFP64 oracleを検査する。
