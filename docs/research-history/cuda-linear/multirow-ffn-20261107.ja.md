# 多入力FFNの計算共有pilot

[English](multirow-ffn-20261107.md) | **日本語**

## 契約と候補

既存の256行・256→1024／1024→256 FFN、各map A4096、
Polar modes8/base1/merge2/spacing1を対象とする。
既存active-nativeのFP64 moment経路を対照に、別登録の研究Algorithm
`cuda_radius_cell_mass_row8/v1`で出力展開・Parameter勾配の幾何計算を
4行から8行へ共有する。明示的な`ROW8_NATIVE` Planは既存active-native recipeを使う。
production catalog/defaultには登録しない。

CPU宣言とnative checked入口の検査を維持する。
source hashとbuild suffixで候補binaryを区別し、default buildは4行のままとする。
候補はbaselineの専用execution programを使わず、generic dispatchを用いる。
Tensor/function変換時は既存Python入口へfallbackする。
今回は演算共有だけのpilotで、FP64 Z/H/G、保存Tensorの所有・version検査、支持、
radiusのtask微分のdetach、正規化、Atom数・形状、初期化、更新則を維持する。
8行の端数maskで全入力行を保持し、Atom勾配の最終batch縮約は元の昇順を保つ。
shared-memory配置を共有行数に合わせて計算する。資源負担増で遅くなる可能性もある。

## GPU計時前に固定するgate

1. CPUのmetadata・support・直列化・経路選択と既存nativeの回帰確認。
2. A100 MIG3g.40gbで実Torch/CUDAを記録する。非連続な2×128 token入力で、
   FFN両方向のY・dX・全Atom勾配をchunk化したCPU FP64 Fourier/autograd参照と比較。
   既存atol3e-5/rtol3e-4を維持する。共有Fourier数式は独立polynomial求積で補う。
   周期境界、狭い・広い支持、ゼロ振幅、端数行、retained backward/version検査を確認し、
   Graph/captureは未対応として拒否する。
3. 既存CUDAと出力・勾配、optimizerのParameter・moment・clockを比較し、
   実4block FFN taskでresumeを確認する。完全一致と許容差合格を区別し、許容差は緩めない。
4. 正しさ合格後のみ、既存cst-research FFN learnerで旧経路・row8・Denseをfresh process比較。
   pre-LN4block、幅256、FFN展開4、head4、c8、q0、minimum-001 envelope、
   native finite_chord更新、Atom LR16倍、baseLR3e-4、AdamW(.9,.95)、decay0、
   clip1、TF32off、seed17。batch2/context128/accumulation4で1回256行、1更新1024token。
   data/LR horizon20M token固定、100実更新（102400token）、developmentのみ。
   計時内にprobeを入れず、転送・全microstep・更新を含む。
   順序Aは旧経路→Dense→row8、順序Bはrow8→Dense→旧経路。
   同一allocation内の2つのfresh-process観測で、6台の独立GPU再現ではない。
5. 同期した完全更新sample、別枠の準備、model・optimizer常駐とeager一時領域を含む
   peak allocated/reservedを報告する。Graph/captureやprocess全使用量は主張しない。
   CST間は全modelの初期Parameterと分散合わせreportを比較する。
   Denseは通常の分散互換初期化であり、CSTと同じ行列ではない。

資格検証失敗・OOM・nonfinite・時間予算で固定試行を停止する。
負の結果と実source/result hashを保持する。
正しさ合格だけで採用せず、メモリと完全step時間で判断し、残るDense差も報告する。
短期性能pilotであり、長期学習精度・最小Atom数・他GPU性能は示さない。
4090は使用しない。raw evidenceはignored outputに保全する。

本書は準備の記録であり、GPU検証完了や高速化の結果ではない。
