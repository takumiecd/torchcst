# Quadratic B-spline: backwardの分解とforward Hの保存

2026-10-19。前の[batch連続配置比較](quadratic-bspline-h-coalesced-20261019.ja.md)のB32混合σ・約1.99ms、そのうち広いσ側backward融合区間約.86msを基準にする。研究用revision v14。公開dispatcherや数学的KernelSpecを変更しない。

## 保持する数学と精度

生の入力・出力profileをそれぞれ `fi, fo` として、

\[
D_a=\max\bigl(\sqrt{Q_{i,a}Q_{o,a}},\epsilon\bigr),\quad
H_{ab}=\sum_i f_{i,a}(i)X_{bi},\quad
G_{ab}=\sum_j f_{o,a}(j)dY_{bj}.
\]

`Q`は全観測siteにおけるprofile二乗和。既存のdegree-4離散momentによる正確なsampled L2、積の後で一度だけのfloor、その中心微分を維持する。幅はPolar状態からatomごと・stepごとにdecodeし、task VJPの幅stop-gradientという既存契約を維持する。幅の直接微分を追加した実験ではない。

入力側中心微分を `Hμ`、出力側を `Gμ` とし、

\[
S_a=\sum_b H_{ab}G_{ab},\quad
T_{i,a}=\sum_b H_{\mu_i,ab}G_{ab},\quad
T_{o,a}=\sum_b H_{ab}G_{\mu_o,ab},
\]
\[
dA_a=S_a/D_a,\qquad
d\mu_{d,a}=(A_a/D_a)\left(T_{d,a}-S_a\partial_{\mu_d}\log D_a\right),
\]
\[
dX_{bi}=\sum_a (A_a/D_a) f_{i,a}(i)G_{ab}.
\]

振幅勾配は既存Polarの2成分へ戻す。floor枝では分母の中心微分は0。振幅0でも振幅勾配は残す。
Hだけを保存しても入力中心勾配のためHμは必要になる。G/Gμはbackwardで初めて得るdYに依存するため、forwardには保存できない。

準備・係数・累積moment・縮約は従来どおりFP64、H/Hμ/G/Gμは既存queryが行っていたFP32丸めを保つ。保存版はそのFP32値を保存し、backwardでFP64へ戻して使う。shape、支持、微分式、IEEE設定、誤差閾値を変更しない。

## 保存と実行の候補

- `save_h=none`: forward Hを保存せず、backwardで入力bankとH/Hμを再計算。融合版は測定済みv13と同じCUDA forward/backward kernelを呼ぶ対照。
- `save_h=h`: forwardの未スケールHだけを保存。backwardの入力bankは必要だが、Hμだけを2個のmomentと微分係数から再計算し、Hの係数積と3番目のmoment読み取りを省く。
- `save_h=hp`: forwardのHとHμを保存。backwardの入力bank再構築とH/Hμ queryを省く。forwardではHのmoment読み取りをHμにも共有するが、微分係数・FP64演算・保存writeは追加される。

保存はautogradの各forward専用Tensorで、同じforwardのmetadata/coefficients/count/orderと結び付ける。更新後のσ・中心や別のXへ使い回さない。次のforwardは別の保存領域を作り、retained backwardの旧状態を上書きしない。

保存配列は `[A,B]` FP32。Graph内でnative数が変わってもshapeは固定し、native slotは未使用。Hのみは`4AB` byte、H/Hμは`8AB` byte。A=52,428ならB32で6.4/12.8MiB、B64で12.8/25.6MiBの容量となる。これは保存Tensorの容量であり、capture/replay込みの測定peakとは別。
dPを要求しないforwardでは保存しない。Xはnative側のatom勾配にも必要なため、H/Hμ保存版でもforward状態として保持する。

`backward_mode=fused`はG/Gμ生成・H再計算またはload・勾配縮約・dX差分注入を融合する。
`backward_mode=staged`は診断対照として、H/Hμ再計算、G/Gμ再計算、broad atom勾配縮約、dX差分注入を別kernelに分ける。native側VJP、bank構築、dXのscan/復元も別に記録する。

stagedはH/Hμ/G/Gμの一時Tensorと追加launchを伴う。各stage時間の和を融合版.86msの厳密な内訳とは呼ばない。融合を解いた場合に各仕事が要する時間と、融合を維持した完全stepの両方を示す。融合モードの個別stage markerは空であり、0msの仕事を意味しない。

## 検証と測定

L4一台を共有poolから排他的に使う。1024²、A=52,428、spacing .1、初期σ .1/.4/1.6/6.4/12各20%、可変幅を含む既存fixture、B32を主対照としB64・切替付近・小σだけも別に比較する。
AdamWと既存fused Polar更新、構築・routing・sampled L2・全勾配を含む完全step、capture/replay込みのallocated/reserved peakを測る。診断は主測定後の別Graphで、未計装の完全stepを置き換えない。

Y/dX/全atom4成分の独立FP64 gateはmaxabsとrelative L2をともに4e-4未満とする。live幅・中心、毎stepの新しいX、Graph内の幅切替とX変更、retained backward、ゼロ振幅、勾配の部分要求、空atom、B64も検査する。

以下は回収archive、全manifest、source byte照合、各jobのcleanup応答を確認した結果。主比較は同じ初期P・X・target、TF32 off、warmup 5回、21 replayのmedianを2個の独立processで測る。各stepでAdamW + fused Polarが更新し、全52,428 atomのσ変化を実測した。性能fixtureのX/targetは固定し、Xも変わる場合の正しさは別のeager/Graphテストで検証した。

## 初回B32混合σの比較

job `l4job-56eef9fd777e4c438ade972b2c316632`、source `304c924e`。新規対象61件（60実CUDA + 1宣言）が通り、全shapeの独立FP64 Y/dX/全atom gateも通過した。以下は1 processの探索結果。2 processの最終比較は別に記録する。

| 方法 | 完全step ms | allocated / reserved MiB | broad backward融合区間 ms |
|---|---:|---:|---:|
| v13再計算対照 | 1.9879 | 68.488 / 134 | .8582 |
| v14再計算 | 1.9823 | 68.488 / 134 | .8704 |
| Hのみ保存 | 1.8312 | 74.888 / 174 | .7096 |
| H/Hμ保存 | 1.7082 | 81.288 / 172 | .5929 |
| W range reuse | 3.2259 | 78.590 / 154 | — |
| Dense | .0823 | 49.002 / 106 | — |

H/Hμ保存時は入力bank再構築も省ける（再計算版は別区間で約.032ms）。forward融合区間は.453～.456msで、この条件ではHμ計算と保存writeによる大きな遅延は観測されなかった。backward register/threadは126→56（H保存）→48（H/Hμ保存）、すべてspill 0。これはcompiler報告で、occupancyやcache hit率の実測ではない。

### 融合を解いた診断対照

| 保存方法 | H/Hμ再計算 ms | G/Gμ再計算 ms | broad atom縮約 ms | dX差分注入 ms | 完全step ms |
|---|---:|---:|---:|---:|---:|
| なし | .2888 | .2888 | .1475 | .4413 | 2.3203 |
| Hのみ | .1208（Hμのみ） | .2888 | .1475 | .4413 | 2.1494 |
| H/Hμ | skip | .2908 | .1495 | .5192 | 2.0073 |

native側のatom VJPは別区間で約.110ms。dXのscan/多項式復元、native側GEMM、bank構築も完全stepに含まれるが、上の4つのkernel区間の外で記録する。stagedにはmaterializedなH/Hμ/G/Gμのwrite/readとlaunchが増え、保存なしで94.643/214MiBのpeakになった。この対照の時間を足して融合版.86msを再現したものとは扱わない。

保存により入力H/Hμの仕事を減らせたが、G/GμはdYから生成する必要があり、dX差分注入も残る。staged対照ではdX注入が単独で最も長い区間だった。

## 回帰検証

- CUDA対象suite: **546 passed**（539実CUDA + 7宣言/metadata）、642.00s。job `l4job-4b93b9af98c3489c9b3b3e8bfc64bf1e`、source `4d8ba585`。
- 新規対象: **61 passed**（60実CUDA + 1宣言）、71.17s。
- CPU全体: **1974 passed / 3315 skipped**、42.68s、18件の既存warning。GPU/DBのskipは検証完了を意味しない。
- wheel/sdistの345 runtimeファイルを現checkoutとbyte照合。

新規テストはlive σ/中心・毎stepの別X、Graph replayでのcutoff切替とX変更、retained forwardのsnapshot、振幅0、dXのみ/dPのみ、空atom、B64を含む。誤差は独立FP64 oracleに対するmaxabsとrelative L2の両方を4e-4未満に維持した。full suiteのAccumulateGrad stream warningは既存経路の1件で、全件完了した。

## 係数とnative Wの保持量を減らす追加対照

job `l4job-0156aeede91d44cea70d78262a1b9ddd`、source `4d8ba585`、B32混合、各候補1 process。

| 方法 | 完全step ms | allocated / reserved MiB |
|---|---:|---:|
| 再計算対照v13 | 1.9914 | 68.488 / 134 |
| H保存 | 1.8271 | 74.888 / 174 |
| H/Hμ保存 | 1.7092 | 81.288 / 172 |
| H/Hμ保存 + compact係数 | 1.7793 | 75.688 / 158 |
| H/Hμ保存 + native W再構築 | 1.7604 | 77.288 / 172 |
| H/Hμ保存 + compact係数 + native W再構築 | 1.8295 | 71.688 / 140 |
| H保存 + native W再構築 | 1.8742 | 70.888 / 174 |
| Dense | .0834 | 49.002 / 106 |

compactは係数5→3 planeで中心微分係数を既存の代数式から復元。native W再構築は同じforward metadataを使い、dXの後はその作業領域をdWへ再利用する。新しいL2近似や低精度保存ではない。全候補の全shape FP64 gateを通過した。

ここではH/Hμ保存とcompact+W再構築が、再計算対照よりallocated/reservedそれぞれ3.2/6MiBの増加で約8%短縮する候補になった。以下の主比較でも確認した。Denseの両peak以下という目標は未達。

## 最終比較: 混合σ、独立2 process

source `eebe1bdc`、job `l4job-500dcf9673814538b1e6577aeb51b41c` / `l4job-717f368112de4c07abcb64c4fb5db629`。時間はprocessごとのmedianの範囲。peakは両processで同じだった。完全stepには毎forwardのrouting検証・必要時の再配置、metadataと係数構築、正確なsampled L2、forward/backward、AdamWとfused Polarを含む。

| 方法 | B32 完全step ms | B32 allocated / reserved MiB | B64 完全step ms | B64 allocated / reserved MiB |
|---|---:|---:|---:|---:|
| v13 H再計算 | 1.9825–1.9844 | 68.488 / 134 | 3.2912–3.2929 | 72.693 / 132 |
| H保存 | 1.8287–1.8292 | 74.888 / 174 | 2.9396–2.9579 | 85.493 / 160 |
| H/Hμ保存 | **1.7024–1.7094** | 81.288 / 172 | **2.7779–2.8071** | 96.791 / 188 |
| H/Hμ保存 + compact + W再構築 | 1.8263–1.8288 | 71.688 / 140 | 2.9204–2.9217 | 87.190 / 196 |
| 既存W range reuse | 3.2203–3.2230 | 78.590 / 154 | 3.2497–3.2508 | 78.840 / 156 |
| Dense | .0825–.0831 | 49.002 / 106 | .1040–.1047 | 49.502 / 106 |

H/Hμ保存は再計算版に対しB32で約14%、B64で約15%短縮した。Dense比はB32約20.6倍、B64約26.5–27.0倍で、Dense並みには届かない。
B32でメモリ増を抑えるならcompact + W再構築はallocated +3.2MiB、reserved +6MiBで約8%短縮する。一方B64ではそのreservedが196MiBとなり、通常H/Hμ保存の188MiBより大きい。allocatedとreservedの両方で常に優位という選択肢ではない。

主比較全体の独立FP64最大誤差はY `1.218e-6`、dX `1.539e-6`、全atom勾配 `3.405e-5`。relative L2はすべて `1.402e-7` 以下。3状態を各候補で検証し、maxabs/relative L2の両方が既存閾値 `4e-4` を通過した。

## 切替付近と小さいσだけ

job `l4job-fe1e81ff3a724b3da825f8d70e89e2ce`、source `eebe1bdc`、B32、各1 process。
切替fixtureの初期σは `.1/.199999/.200001/.20001/1.6` 各20%。小σfixtureは `.1/.14/.18` 各約1/3。いずれも全52,428 atomのσが更新された。

| 方法 | 切替付近 完全step ms | 小σだけ 完全step ms | allocated / reserved MiB |
|---|---:|---:|---:|
| v13 H再計算 | 1.8396 | 1.4822 | 68.488 / 134 |
| H保存 | 1.8046 | 1.4813 | 74.888 / 174 |
| H/Hμ保存 | 1.7649 | 1.4544 | 81.288 / 172 |
| H/Hμ保存 + compact + W再構築 | 2.1006 | 1.7823 | 71.688 / 140 |
| 既存小支持点グループ版 | — | **1.2075** | **50.231 / 114** |
| Dense | .0830 | .0834 | 49.002 / 106 |

切替fixtureはσ診断で20,971 atomが経路変更し、最終device分類はnative 41,942 / H 10,486。配置検証52呼び出しのうち再構築3回、再利用49回だった（warmup/eager/captured replayを含む回数で、replayだけの率ではない）。Graph replayの動的切替とsnapshotの対応は独立テストでも検証した。

小σだけは全atomがnativeであり、H保存slotは未使用なのに固定容量分のメモリを確保する。小σだけで観測した約.028ms差はH再利用の成果とは扱わず、全nativeで不要なbankを省く実行差も含む。Hのみはほぼ同じ速度で、compact + W再構築は悪化した。小σだけにこの保存版を適用する根拠はなく、既存グループ版が速く、省メモリだった。

この2条件を合わせた最大誤差はY `2.399e-6`、dX `2.497e-6`、全atom勾配 `1.761e-5`、relative L2は `1.726e-7` 以下。全gateを通過した。ここでの結果を新しい自動dispatcher閾値へ昇格させてはいない。

## 判断と残る仕事

forwardのHだけではHμ再計算が残るため、H/Hμ両方の保存が最速だった。保存とbackwardの対応をforward単位で固定することで、stepごとのσ変更やretained backwardでも同じ数式を保てた。毎stepで値の変わるHを次stepへ流用する方式ではない。

保存版のbroad backward融合区間は約.86→.59ms。staged診断では、入力H/Hμの再計算約.29msを取り除けた一方、G/Gμ約.29ms、broad atom縮約約.15ms、dX差分注入約.44–.52msが残る。native atom VJP約.11msも別に残る。次に削減効果が大きい候補はG生成とdX集約であり、係数構築だけを速めてもDenseとの差全体は埋まらない。この測定は時間の所在を示すもので、帯域上限やatomic競合率をhardware counterで確定したものではない。

## 証拠と再現

L4 / driver `580.82.07` / Python `3.13.15` / Torch `2.11.0+cu130` / CUDA `13.0` / Triton `3.6.0`。
今回の6 jobはすべて正常終了し、結果archive・manifest・sourceのhash照合とjobごとのcleanup応答を確認した。runtime 345ファイルと対象test/runnerは全jobの凍結sourceとbyte一致。Plan catalogは後から候補を追加したが、凍結済みPlanの内容は変更していない。

| job | source archive SHA256 | result archive SHA256 |
|---|---|---|
| `56eef9fd777e4c438ade972b2c316632` | `8a8ef633a264227e8658a46b4ea3addd266f78efadb042cd0cf0a7047a042c94` | `4d17f3a7b512e7797ed16a5afb9ddcc5997da7020d67c1c2a64b301997fc1013` |
| `0156aeede91d44cea70d78262a1b9ddd` | `ac97f5ebf1da2497ee50dde6827fad7d07366e7e3d9c6e9299b11bc7d375ded4` | `59c95103d3485aecd44b8cacc4c034f0ece197ee62b0aeaf3f36a42474a093f3` |
| `4b93b9af98c3489c9b3b3e8bfc64bf1e` | `ea1e6000b66365b1f3ba2bb1479e0bd99dd1e2ff13192414d068e3ababff84f8` | `63ddb2a29ff49ae95a83a9be19dc5b657a456aeb02b180684de59e425e74becf` |
| `500dcf9673814538b1e6577aeb51b41c` | `d3a2025c158ad3274900f7defcb6775612ca6c0ad8f7f626f4143d66d86ccb26` | `6a3a2d3df1a82836bfa39753715664f7e6c4bc9407af9c60108c3ee5a14b5f27` |
| `717f368112de4c07abcb64c4fb5db629` | `aeb66c6156cf1c3e2f19ed445dc7d378695a9b16621ab7c5c29ca1d241dec5a7` | `4b5ad8089e48f71b4173bea47c4dac2bdf103f0ebc6dc7e1973ac40592558837` |
| `fe1e81ff3a724b3da825f8d70e89e2ce` | `aeb66c6156cf1c3e2f19ed445dc7d378695a9b16621ab7c5c29ca1d241dec5a7` | `7c61ff11dac7e74ce285e1e860532f3b501bd5efd5fd730594ff8ad401561f89` |

表のjob IDは共通の`l4job-` prefixを省略。ignored evidenceはworktreeの `output/bspline-h-save-20261019/` に保持し、回収archiveを `verified-jobs/` に複製した。`audit-primary.json` / `audit-aux.json` に全phase・routing・精度・peak、`source-byte-proof.json` に凍結source照合、`package/proof.json` にwheel/sdist照合を残す。

GPUホストでの主比較再現（新規結果先を使う）:

```bash
PYTHONPATH=src python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-h-save-final.json \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --source-commit "$(git rev-parse HEAD)" --output output/h-save-reproduction.json
python -m benchmarks.submissions check output/h-save-reproduction.json
```

`b64-mixed` / `switch` / `small` の対応するCaseでも同じ手順を使う。所有者L4では共有poolへ隔離snapshotを提出し、GPU測定後に回収・照合・停止を行う。
