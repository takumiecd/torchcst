# #135後の設計探索とGraph構造検査の補強

## 基準と到達点

ユーザーが選んだgoalは「有望設計を絞り、採用可否まで検証する」。改善が成立しない場合も、
反証・検討範囲・残る不確実性を保全して終了できる。基準runtimeは#132
`fba2e72d733325fbd7be5aac334a0a7463d96b27`、N1024/A52428/真B4、FP32 body＋FP64 geometry/Polar、
TF32off、MNIST全モデル、CUDA Graph。数式・全atom勾配/dX・optimizer・hooks/live settingsの契約を維持する。

#133 phase-delta、#134 known-bank、#135 8係数保存の負結果は変更しない。#135は限定診断成立後、
正式比較の最初の有効pairで0.578078→0.617372 ms/更新となり、事前6/6条件で停止した。
新しい探索で負けを捨てた再測定や、その案の別layoutによる救済は行わない。

## 独立探索で絞った三候補（まだGPU未実施）

| 候補 | 根拠 | 未確認・増える費用 | 最初の診断候補 |
| --- | --- | --- | --- |
| G: flatなwhole-step guard | 毎stepの再帰module/parameter/buffer走査とmetadata構築。保存profileの呼出し数もsourceと整合 | profileで未計装費用やCPU律速は断定できない。weakrefs、ordered edges、Tensor metadata、optimizer list/値、hooks/TLS、native guardsを全て維持する必要 | 同じL4で非計装eligible-only/signature-onlyとvalidated real-step referenceを分けて測る |
| K: period専用の正しく丸めるFP32除算 | 現scopeのPはK*2^n、K=2/3。normal fastpathの整数モデルをCPU全mantissaで検算 | 整数演算・分岐・register増が相殺し得る。CUDA bit一致も速度も未確認 | whole CSTへ接続する前に、同じGPUのRN除算とscalar/chainをbyte比較・生成コード確認 |
| O: 既存FusedAdamWのCTA再分割 | 保存traceのordinary16 CTA/atom4 CTAは公式chunk65536のsourceと整合 | disjoint aligned views、unique clockを一回だけ更新、version/alias/lifetime/private APIの検証が必要 | 同じP/m/v/clockから1更新を比較し、条件を通ればGraphへ接続 |

G/Oは共通の実行・optimizer基盤改善であり、denseにも同等方式を与える。CST数式kernelの改善と
混同しない。Kと対象費用が重なる100-byte bank-only shared tableは補欠とし、同時実装しない。
Tensor Core/大batch/精度変更/密行列化は、現在条件の第一候補から外した。

CPU periodモデルではnormal exponent一つの全2^23 mantissa×2符号＝16,777,216 case、さらに
22実period各524,300 random/edge caseで不一致0だった。GPUの正しさ・速度の証拠ではない。
normal binary32の絶対値をm*2^(E-127-23)として、P=3*2^nでは

\[
s=\begin{cases}1&m\ge3\,2^{22}\\2&\text{otherwise},\end{cases}\quad
m_{out}=\left\lfloor\frac{2^s m+1}{3}\right\rfloor,\quad E_{out}=E-n-s.
\]

非normal入力/出力・非対象Pは元のGPU除算を残す設計。`raw/zero`等の非period除算や加算順は変えない。
静的PTXのdivision個数を動的回数や回収時間へ換算しない。

## 先に分離して直す構造signatureの検出漏れ

旧`PreparedStep._signature()`は`model.modules()`でunique nodeだけを記録していた。
一方、Sequentialは登録順の全edgeを実行するため、末尾へ既存ReLUのaliasを追加すると、
CPU toyのeager logitsは[-1,1]→[0,1]へ変化しても、旧signatureは同一になった。
None edge追加も旧node走査に現れず、eager側のTypeErrorをsignatureが捉えなかった。

Proレビュー後、canonical root Sequentialのraw `_modules`について、登録数・順序付きkey・
各childへのweak参照を記録する最小案へ絞った。live childを`ref() is child`で比較し、aliasを
含むedge変更をeligibleの先頭で拒否する。外されたModuleやTensorの寿命を延ばさない。
既存のsignature全体は#132へ戻し、全moduleのedge追加走査は採用しない。
これはflat certificateの実装ではなく、速度最適化前の共通baseline補強である。
既存CST/Polarのguard、hooks/TLS、optimizer・Tensor metadataの検査は残す。

ローカルCPUでは修正前に対象2件がfailした。最小weak root案ではalias/None edge、削除・順序変更・
再binding・rename、weak lifetime、学習Parameter値の更新を検証し、memory判定器と合わせ
30 passed/2 GPU skipped。Torch2.13/Python3.11でのCPU検証であり、
対象GPUのTorch2.11/Python3.13でのeligible/replay検査は未実施。
GPU用には、alias拒否時の状態・grad・metrics・version不変性と、Graph launchせず変更後の
eagerとbyte一致するfallbackを分けた回帰検査を用意した。

このCPU反例から、過去の固定5要素・構造不変の数値/性能結果を無効とはしない。
また旧`eligible()`全体が実CUDA上で受理することまで確認済みとは主張しない。
今後、補強referenceの追加費用を取り戻した分だけを元#132からの改善と見せないよう、
旧固定model reference・補強reference・将来candidateのscopeを分ける。

独立red-teamは19のCPU反例も確認した。登録dictだけではstorage/grad差替え、定数buffer version、
optimizer nested list、traversal code変更、hooks/TLSを代替できない。flat化はこれらの検査と
weak lifetime/capture後再準備を保つ場合だけ検討する。

## 現状と証拠

新CUDA/optimizer variant、Plan、GPU測定はまだない。Proレビューは完了し、最初はGのguard費用診断を
選んだ。flat certificateは未実装。別のprospective screening記録に固定手順と停止基準を残す。採用guide（両状態5%以上/各6勝、byte/全状態、capture込みmemory-v2.1）は緩めない。

元CPU検算・反例・source/profile hashはこのworktreeのignored `output/design-search-20261104/`に保持。
公開source確認には以下の一次資料を用いた。

- [Torch2.11 Module traversal](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/modules/module.py)
- [Torch2.11 Sequential](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/modules/container.py)
- [Torch2.11 MultiTensorApply](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/MultiTensorApply.cuh)
- [Torch2.11 Adam functor](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/fused_adam_utils.cuh)
- [NVIDIA floating point](https://docs.nvidia.com/cuda/floating-point/index.html)

研究worktreeはmain・#132基準・#135最新負結果・本探索の4つをactiveにし、旧12treeのsourceと
ignored証拠は内容を削除せず中央snapshotへ移設した。全HEAD pinとverified Git bundle、
全payloadのSHA/inode照合、旧path互換link、復元手順をローカルに保全した。
