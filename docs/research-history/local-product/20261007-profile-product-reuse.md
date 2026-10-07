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

## 検証と初回screen

source `634cb32b2a37122f16b8a6ca6d06d6bc9e665482`。
GPU check `l4job-a5fcb9f368814eeba9daf3dc61762fbe`: reuse経路12 actual GPU +
3 metadata/CPU checks PASS。CPU全suite968 passed/1172 skipped、Ruff/diff、
wheel/sdist build、wheel metadata-only import、同sourceのGitHub CPU CIもPASS。

N128 job `l4job-74a4db46c34642ef9df57049d5bda89e` の1独立execution screen。
同じ新定義、同じ初期p/input/targetとfused AdamW/Polar。完全step median us:

|初期rho|saved|初回ordered|reuse|dense|
|---|---:|---:|---:|---:|
|3|244.28|152.99|77.90|45.19|
|8|243.92|140.49|79.88|45.58|

全3CST経路のfull-site FP64 Y/dX/全atom/非zero中心微分と更新oracleがPASS。
reuse/orderedのallocated peakは316416 bytes、saved293888、CST reserved6291456。
準備も含む完全stepで速くなり、メモリはorderedと同じ。rho3の差は約49%。
launch/layoutのまとまった移植の効果であり、個々の変更の因果寄与は分離していない。
旧契約の過去67usへは未到達だが、その差の原因は現段階では未計測。

N64 job `l4job-92f861eb03824416a500b46e3eb9d41a` もPASS。

|初期rho|saved|初回ordered|reuse|dense|
|---|---:|---:|---:|---:|
|3|64.32|71.92|53.27|38.90|
|8|64.23|61.64|52.91|38.72|

N64 reuse/ordered allocated122368、saved102400、CST reserved6291456 bytes。
rho3で最速の従来profile-product候補savedから約17%短縮した。
メモリ優先ならsaved、測定したrho3/8の速度優先ならreuseが有力。
まだnarrow/mixedや任意の幅軌跡を一般化していない。
初回64はsaved/128はorderedという選択は、今回の候補追加前の結果として扱う。

## 同一sourceの実行順反転

N64 `l4job-98ba9b2dc699474fb39fa53f2ccff68b`、N128
`l4job-1d9d7510abc5447a9a259b1978f6773d`。sourceは両初回とも同じ634cb32bの完全SHA。
候補順をreuse→ordered→savedへ反転し、rho3だけを別executionで再測定した。

|N|saved 初回/反転|ordered 初回/反転|reuse 初回/反転|
|---|---:|---:|---:|
|64|64.32 / 64.38|71.92 / 71.75|53.27 / 53.61|
|128|244.28 / 243.51|152.99 / 152.51|77.90 / 77.51|

rho3の速度優位性が両runで一致し、allocated/reserved peakも一致した。
全6完成artifactはsubmission形式/adapter4検査、within-run初期p/input/target一致、
source archiveを当該commitの全tracked filesと照合した検査がPASS。
21 timing samplesを独立run数には数えず、per-run mediansをそのまま記録する。
[機械可読要約](20261007-profile-product-reuse-summary.json)に各値とsource/result/driver hashesを保存する。
raw logs/source/result archivesはignored evidenceと共有プールjobディレクトリに保持する。

既存の小型計算方式はprofile productにも使える。初回結果は新定義の速度限界ではなく、
layout/launchの移植不足が大きかった。過去の小型契約との差の分解は未実施。
全site正規化準備、snapshot生成、順序cacheなどの残る改善余地は今後の候補として区別する。
今回の条件ではreuseを大型化用の速度候補にできるが、他のrho/mixedや大型Stripは未検証。
公開既定選択器、以前の小型routeの実装、数学定義は変更していない。

再現: `python -m tools.kernel_dev test --suite profile-product`。
測定は既存`benchmarks.cuda.linear.run`へplans-profile-product.jsonと各caseを渡し、
`--polar-update fused --source-commit <verified SHA>`を指定する。
今回のペアはcaseのplansをsaved/ordered/reuse、baselineをsavedに絞って実行した。
