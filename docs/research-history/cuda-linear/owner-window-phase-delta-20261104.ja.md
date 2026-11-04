# FP32局所変位共有の研究（#132基準）

## 事前固定した契約と判定

基準は#132 Flat C（fba2e72d、runtimeは34b44a0cと同じ）。
候補は`GLOBAL_OWNER_WINDOW_FP32_PHASE_DELTA`のみ。
分類kernelが現在のParameterからFP64 geometryとint64 owner keysに加え、
FP32 `[2,A]`の局所変位を生成する。

\[
\delta_{i/o,a}=\mathrm{float32}\left(
\operatorname{wrap}_{i/o}(\mu_{i/o,a})
-t^{\mathrm{signed}}_{i/o,a} T_{i/o,a}\Delta_{i/o}\right).
\]

引き算までは既存FP64、既存cast直後の値を共有する。signed first tileは
ownerの周期moduloより前の値を使う。逆方向は保存したfull keyからsource bankを
復元し、同じ局所変位を使う。位相乗算、Fourier/profile、cell mass、正規化、
C縮約順、全atom勾配、Polar pullbackとoptimizerは維持する。
追加payloadはA52428で419,424 bytes（約0.40MiB）。追加launch/barrierはない。
通常のSavedVariableとしてforward単位に保持・検証し、replayごとに上書きする。
公開dispatcherとGraphのguard/hook/live settings契約は変えない。

N1024/A52428/真B4/seed41、L4、FP32 body・FP64 geometry、TF32 off、
native CSTAdamW、MNIST全モデルGraphで比較する。モデルは重み付き3層、CSTは中央1層。
基準の8実更新後と#132自身の45,000更新checkpointを凍結する。
初期値・最終checkpoint・入力gzipのSHA256をdriverに固定する。

短期は各状態256更新×6 balanced-order fresh-process pairs。
同等条件のdense Graphを含める。全12 CST cellの42 state、5 gradient、
metrics/data orderをbyte一致、fallback 0とし、過去#132 numeric controlとも照合する。
両状態でpaired geometric reduction >=5%、6/6勝利を要する。
captureを含むpeak増加allocated<=2MiB、reserved<=8MiB。
memory-v2.1の8 cycle×16実更新（128実更新/120replay）でphase・epoch peak・
cycles3–8のallocated/reserved plateau、全状態・勾配・metricsも確認する。

すべて通過した場合のみ、基準→候補と候補→基準の2順序、各45,000更新を実行する。
準備・capture/restore・再準備・旧Graph解放GC・最終retirementを含む時間で各順序>=5%、
各3 epochの47 Tensor byte一致と長期peak予算を要求する。
3 epochだけを8 cycle lifecycleの証拠とはしない。
独立FP64 oracleは既存`3e-5 + 3e-4 abs(ref)`を維持し、byte不一致の救済に使わない。
別processの16更新profileは診断専用。短期償却推定を長期実測と混同しない。

## ローカル事前検証

CPU suite: 1742 passed / 2529 skipped。host C++ syntax、Ruff、Plan/Case宣言検査pass。
Linux CPU CIは1741 passed / 2530 skipped、wheel/sdistを含めpass。
[測定ソースCI](https://github.com/takumiecd/torchcst/actions/runs/41784234271)。

## 結論：短期速度基準を不通過、45,000更新へ進めない

実装・数値・メモリの検証は通ったが、この候補は全モデルで遅くなった。
基準は#132のままとし、この候補を採用しない。短期の結果を見て閾値を緩めたり、
類似候補を追加探索したりしていない。PRは研究Draftで保全し、mainへ統合しない。

| 凍結した開始状態 | #132平均ms/step | 局所変位共有平均ms/step | paired geometric時間増加 | 候補の勝利 |
| --- | ---: | ---: | ---: | ---: |
| #132の8実更新後 | 0.578893 | 0.597644 | 3.2391% | 0/6 |
| #132自身の45,000更新checkpoint | 0.587289 | 0.599543 | 2.0867% | 0/6 |

各状態6対、各cell256更新。6対すべてが同じ遅延方向で、事前の>=5%短縮・6/6勝利を
満たさない。今回の候補を45,000更新した結果は存在しない。
同一条件dense Graphは初期0.209728、学習後0.207469ms/step。
このCST/dense比は基準2.76/2.83、候補2.85/2.89。未計装Graph stepはgather、loss、
全層backward、dX、全atom勾配、実optimizer、guard/index/version通知を含む。
短期の45000償却推定も初期約2.79–3.07%、学習後約1.82–1.91%の遅延だが、
これは長期実測の代用ではない。

## 数値とメモリ

- 全12 CST cell/状態で開始・終了42 state、全5 gradient、metrics、data/orderがbyte一致。
  過去#132の保存済みnumeric controlとも一致。dense controlsも一致。finite、fallback 0。
- 8 cycle×16更新の両reportはmemory-v2.1をpass。記録されたphase、summary/epoch peak、
  全state/gradient/metrics/order、128実更新/120replayを照合した。
  endpoint allocatedは両者245.407227MiBで全8回一定。
  reservedは基準442/444/444/444/444/444/444/444MiB、候補444MiB×8。
  両者cycles3–8で一定、capture-inclusive peak差はallocated/reservedとも0。
- 短期全CST cellのcapture-inclusive peakはallocated251.534180MiB、reserved442MiBで同じ。
  追加payloadが存在しても、この測定の総peak増加は0だった。
  allocator計測であり、GPU process全体の使用量や転送量を示さない。
- L4環境の契約検証622 passed。B0–4、rank6/12/17、owner chunk境界、空/部分勾配、
  saved geometry/phaseのdtype・shape・stride・device拒否、offload・複数forward・retained、
  current P・Graph変更検知、既存独立FP64 fixture oracleを含む。
  signed first tileと周期境界でkey・FP64 geometry・FP32 displacementの従来式一致も確認。
- 回収したcheckpoint本体もローカルで8×47 Tensor（lifecycle各epoch）と2×42 Tensor
  （短期2状態）を直接byte比較し、全460比較が一致。各短期rowの5勾配hashも一致。
  候補自身の45k全atom独立oracle・学習後Adam診断は長期へ進まなかったため未実施。
  過去のFP32 cold-Adam負結果を再分類しない。

## 別processのprofile：contract自体も遅い

| 16更新診断の平均us/step | 初期#132 | 初期候補 | 学習後#132 | 学習後候補 |
| --- | ---: | ---: | ---: | ---: |
| forward/reverse contract合計 | 185.784 | 199.308 | 191.554 | 201.160 |
| classifier | 110.566 | 112.152 | 110.724 | 112.328 |
| Radix sort | 61.180 | 61.026 | 61.096 | 60.910 |

kernel-category eventsは両者96/step。contract合計が初期+13.524us、学習後+9.606us、
classifierが約+1.6us。削除したFP64 wrap/classifyの仕事だけを見た期待とは異なり、
contract自身が遅くなっている。未計装全step増加18.751/12.255usと方向は整合する。
profilerのstage合計をthroughputの代用にはしない。

source上で候補はfull int64 keyのdivision/moduloとbank探索、phase payloadのreadを
追加する。register配置・spill・occupancy・loadや整数除算の費用が利益を上回った可能性は
あるが、因果関係は未確定。nsys/ncuはこのVMに存在したが追加計装はしていない。
SASS・counter・register/spillは未測定。CPU律速や帯域律速とも断定しない。

## 来歴と保全

- PR: [#133](https://github.com/takumiecd/torchcst/pull/133)、base #132 `fba2e72d`。
- 測定source `ff4ae86a7d62b930de9de2d0f9b58322a4829075`。後続変更は研究文書とprocessed JSONのみ。
- job `l4job-ea6a050a795c464e9bade1297daa0a07`、L4 UUID
  `d753f024-2f47-16da-dc18-ce77f63f2a3a`、driver580.82.07、Torch2.11.0+cu130、
  CUDA13、Triton3.6、Python3.13.16、Xeon2.20GHz/12 logical cores、math threads1。
- source archive SHA256 `94d6f48bc633a8dd1edd285b4aa7e86b9e5a0fe60cb569145651f491316cfd5b`。
- result archive SHA256 `96fd757c84c3459980fb9143856dff2944b2418a4217f9b09e6d61d3c1405ae7`。
- archiveと1024 manifest filesを手元で再hash。22個の元PTをlossless CASから復元し、
  元ファイルのbyte/SHA256をすべて照合。raw結果のpool copyは変更していない。
- raw logs、trace、driver、固定source、checkpoint、input manifestはignored
  `output/phase-delta/`と共有pool job directoryに保全。driverの入力checkpointは
  #132の実ファイルSHA256を固定した。過去のcheckpoint/output/worktreeは削除しない。
- driver returncode0/timed_out=false、結果回収/remote cleanup ACK、所有VM停止を確認。
  supervisorとcaffeinateは正常終了。1回のL4割当・1 seed/モデル/shape/真B4に限定した負結果。

再現driverは`output/phase-delta/driver.py`（入力同梱、hashはprocessed JSON）、
同一host-wide poolでsubmitする。各短期cellの主要引数は
`full_step_graph_comparison --bootstrap-fp32 --flat-c-baseline --mode graph --steps 256`、
precisionは`fp32_geometry_sort32_flat_c`対`fp32_phase_delta`。life probeは
`full_step_graph_learning --flat-c --mode graph --epochs 8 --steps-per-epoch 16`に
候補のみ`--phase-delta`を加える。全command/順序はdriverとphase logsに残した。

[processed結果・判定・hash](owner-window-phase-delta-results-20261104.json)に保存した。
