# RegularGrid: 支持幅生成と dH/dG 即時消費の組み合わせ

2026-10-11。比較完了。組み合わせは保存量を削減したが、支持幅単独より約3%遅く、
現行より29–45%遅い。採用せず現行BM16/G16/H32を維持する。

## 比較の目的

支持幅に合わせた16-site panelの生成と、dH/dGを生成したkernel内で
パラメータ勾配へ消費する方式を組み合わせる。
現行 `input-site-reciprocal-bm16-g16-h32`、固定幅 `tile-dot32-bm16`、
支持幅のみ `tile-window16-bm16`、即時消費のみ `tile-derivative-bm16`、
組み合わせ `tile-combined-bm16` を独立したplanとして比較する。
既存単独版の演算・保存方式は変えない。

組み合わせでは各BM16 chunkで H → G/dG → H/dH → dX の順に進める。
Hは支持幅panelで先に完成させ、G/dGの生成中にdA/dc_outを完成させる。
続く入力側の生成ではdHをGと縮約してdc_inへ消費する。
Hの値だけを作る追加passは残る。H/Gの二つのslabを使い、dH/dGは保存しない。
全支持範囲を縮約してからパラメータ勾配を作り、tile間の交差項を保持する。
全operator-siteに対する正規化、floor、singleton、source VJP、広支持fallback、
32-site ownerと集約方式は既存版に揃える。

## 固定プロトコル

RTX 6000 Ada、N=2048、B=32、A=209715、rho初期値3/8、seed41。
IEEE FP32、TF32無効。24回の実AdamW/Polar更新、21個の完全step計測。
各rhoで独立2cohortを実行し、2回目はplan順を反転する。
各planはfresh process。準備・routing sort・生成・集約・backward・optimizerを含む。
capture/replayを含むpeak allocated/reservedをdenseと比較する。
全224件のGPU qualification後、初期とpost24で独立FP64の
全軸・全atom・全4パラメータ勾配を4e-4の既存基準で検証する。
失敗した比較を再試行して選び直さない。

matrix/factor/onchipおよびdense controlも同じcohortで測る。
9 workers × 4 cohorts = 36 workers、756 samples、64 full FP64 gates。
詳細phaseは別の固定post24 snapshotで計測し、primary stepへ足し合わせない。
結果が確定するまでは研究用planに留める。

## 正しさと完全step

計測source `e5757dada359399d947cf5eb24b6b4163cfb35d2`。Torch2.9.1+cu128/CUDA12.8/Triton3.5.1。
GPU qualification 224 PASS、0 failures/errors/skips（17.48秒）。
36 fresh workers/756 primary samples、64 initial/post24 full FP64 gates PASS。
各候補と現行について独立2cohortのpost24 Parameter/Y/dX/all_dPはbitwise一致。
8つの別Graph診断もprimaryおよび未計装版とbitwise一致、Parameter不変。
全operator-site正規化、floor/singleton/empty、seam/stride/tail、gradient mask、
retained snapshot、広支持fallback、Graph/public optimizer更新を検証した。
実際のbackward割当要求で、組み合わせもdH/dG slabを作らないことを確認。
CPU1790 PASS/4219 GPU・DB skips/18既存warnings、35.15秒。
catalog47 plans、Ruff/diff PASS。
[計測sourceのCPU/build CI](https://github.com/takumiecd/torchcst/actions/runs/38132623562) PASS。

前回campaignの値を混ぜず、今回の同cohort対照で比較する。
準備・routing・全生成・集約・勾配・optimizerを含む未計装中央値（ms）。

| cohort | 現行 | 固定tile | 支持幅のみ | 即時消費のみ | 組み合わせ | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rho3-run1 | 1.415538 | 3.194810 | 1.986797 | 3.544445 | 2.050862 | 0.329215 |
| rho8-run1 | 1.815305 | 3.897053 | 2.346861 | 4.343487 | 2.420664 | 0.329250 |
| rho3-run2 | 1.435783 | 3.206241 | 1.995910 | 3.578037 | 2.055364 | 0.329205 |
| rho8-run2 | 1.875340 | 3.926245 | 2.359188 | 4.314809 | 2.426472 | 0.329088 |

| 全cohortで同じcapture/replay peak (MiB) | allocated | reserved |
| --- | ---: | ---: |
| 現行 | 57.430664 | 104 |
| 固定tile・支持幅のみ | 93.717285 | 150 |
| 即時消費のみ・組み合わせ | 68.116699 | 122 |
| dense | 81.752441 | 106 |

組み合わせは即時消費単独から42.14–44.27%短縮した。しかし支持幅単独から
2.85–3.22%遅く、現行から29.39–44.88%遅い。denseの6.23–7.37倍。
支持幅単独に対しallocatedは25.600586MiB減（27.32%）、reservedは28MiB減。
それでも現行より両peakが大きい。denseに対しallocatedは小さいがreservedは16MiB大きく、
両peak条件を満たさない。全cohortで最速の条件適合CSTは現行。公開default変更なし。

| control | rho3-run1 | rho3-run2 | rho8-run1 | rho8-run2 | allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| matrix-torch | 0.706570 | 0.713183 | 1.363230 | 1.370357 | 96.163086 / 142 |
| factor | 1.951565 | 1.974771 | 4.294607 | 4.339323 | 83.114258 / 130 |
| onchip-h | 1.809589 | 1.812140 | 3.969243 | 3.894527 | 42.866211 / 114 |

## 固定post24の細部診断

各snapshotで5 replay、各stageのchunkを合計した中央値。未更新の別Graphであり、
primary stepへ足したり差し引いたりしない。狭支持とguarded fallbackの両launchを含む。
profile/norm準備にはsource/amp cloneを含まない（primaryには含む）。

| stage (ms) | rho3 支持幅 | rho3 即時消費 | rho3 組合せ | rho8 支持幅 | rho8 即時消費 | rho8 組合せ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prepare_support_norm | 0.100672 | 0.099968 | 0.100096 | 0.118880 | 0.116512 | 0.116416 |
| support_membership_routing | 0.110208 | 0.110496 | 0.110432 | 0.110176 | 0.110496 | 0.110208 |
| forward_H | 0.233600 | 0.574880 | 0.233760 | 0.289600 | 0.716896 | 0.290272 |
| forward_Y_local_products | 0.221312 | 0.222112 | 0.221664 | 0.273568 | 0.274368 | 0.273632 |
| backward_H_value_recompute | 融合 / なし | 0.575296 | 0.234720 | 融合 / なし | 0.716544 | 0.291744 |
| backward_G_dG | 0.425216 | 0.925152 | 0.434496 | 0.531904 | 1.153088 | 0.541376 |
| backward_H_dH | 0.423904 | 0.743712 | 0.273312 | 0.530144 | 0.927712 | 0.336896 |
| backward_dX_local_products | 0.221664 | 0.222656 | 0.222400 | 0.274496 | 0.275968 | 0.274144 |
| physical_gradients | 0.028000 | 融合 / なし | 融合 / なし | 0.028192 | 融合 / なし | 融合 / なし |
| source_vjp | 0.010208 | 0.010240 | 0.010240 | 0.010208 | 0.010240 | 0.010240 |

組み合わせの入力側dH生成・即時消費は、支持幅単独のH/dH生成より軽い。
一方、H valueを先行して作るpassが約0.235/0.292ms残り、G/dG側の融合勾配も
必要で、完全stepでは約3%の費用になる。準備・sort・Y/dX ownerはほぼ変わらない。
各owner partial reductionは約0.0064ms、source VJPは約0.0102ms。

組み合わせの狭支持producerはH value80、G/dG142、dH消費111 registers、
全て0 spills/shared4096bytes。支持幅単独のdHは142 registers。
両ownerは153 registers/0 spills/shared9728bytes。全8診断でwide flagはfalse。
広支持fallbackには255 registersとspillがあるが、今回の狭支持計算ではguardを通らない。
コンパイルmetadataから実DRAM traffic/cache hit/達成occupancy/stall原因を断定しない。
入力value accはsource上にあるが、消費2ではHの保存はなく、実行命令の除去量は未測定。

## 証拠・復旧

- source archive SHA256 `3d008557a03bdd6cebb6c6a147e4e4ab67f02e3f8a1a1c80abcd3a53e3a51145`。
- compact results SHA256 `25442ded89b5183aa15e1dae77c06a150cb4ebeb19312d3c63307adc08f4ed9d`。
- audit SHA256 `4ef1f39ead04c75ace6ec2a8d3c28f4a82a4e5d4e3f1864334df1ab06c6eb2fe`。
- compactの151 entriesと1665 pinned Git blobsをローカルでhash照合済み。
- ローカルignored evidence: `output/regular-grid-h/tile-combined-*`。
- remote: `/home/jovyan/runs/cst-tile-combined-ada-e5757dad-20261011`。
  全64個のraw FP64 gate snapshots（.pt）はremoteに保存。compactはJSON/log/scriptsを保持。
- controller PID226005、followup PID228256。controller完了後に固定診断/auditを起動。
  exact commandとpostprocess PIDはcompactの各pid.json、運用メモに保存。
- 源泉・入力・runtime・cohort順・24更新・21samples・全gate・snapshot hash・
  両peak・固定診断bitwise一致をread-only auditで検証。retryなし。
- Hubはユーザー所有の起動済みsession。作業完了後もreadyのまま残す。

## 可変タイル数の次の検討

ユーザーは支持範囲がまたぐ隣接tile数を2に限定せず、任意数へ一般化する案を提示した。
これは次の設計議論であり、このcampaignには実装・計測を混ぜない。
今回rho3/8は全て狭支持経路のため、上限を外すだけで今回の速度差が改善する根拠はない。
一般化では全支持の正規化、周期siteの重複回避、H/Gを完成してから微分を縮約する契約を保つ。
atom ownerが支持tileを順に読む方式と、tile ownerがatomを束ねる方式について、
所属indexの保持量・X/dY共有・partial集約の費用を比較する必要がある。
