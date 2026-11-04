# #132を基準にした既知bank再利用の独立研究

## 研究の境界

ユーザーは候補や診断の停止を研究全体の終了とせず、Proへの結果相談と次の実装・実験を
継続することを依頼した。#133の負結果・部分診断・raw判定は保持し、過去の条件を救済しない。
本件は#132 `fba2e72d`を基準とする新しい前向き比較で、#133の採用再審査ではない。
Proは#132から独立した既知bank再利用の短期研究を支持した。事前に固定した基準で比較した。

## 一つだけ試す変更

`GLOBAL_OWNER_WINDOW_FP32_KNOWN_BANK`は#132のgeometry共有・sort32・Flat Cに、
CTAが既に持つbank index `bi`の再利用だけを加える。両軸の同一live settingsを検証し、
現在Pから毎forwardに生成したkey/rangesを使うこの限定経路では、同じRに対する
source/destinationのbankが一致する。中心はsigned first tileだけに影響する。

元の二つの`classify(R, mu, axis)`を、既知bankの`tile_sites`と`count`を使う次の式へ置き換える。

\[
t=\left\lfloor\frac{\operatorname{wrap}(\mu)+\Delta/2}{T\Delta}
-\frac{K-1}{2}\right\rfloor.
\]

wrap・floor・source ownerへのmoduloの位置、FP64局所減算→FP32 cast→位相乗算の順は
#132のまま。Kはrankから推定せずbank.countを使う。追加payload、keyの除算によるbank復元、
追加launch/barrier、process-global numeric cacheはない。H/G/dP/Flat C、chunk/atom順、
`emit_x=false`、Polar更新、保存Tensorの数・寿命、hooks/live settings/guardは維持する。
classifier/sortは変更しない。公開dispatcherも変更しない。

旧診断の通常build SASSで各contractにF2I.F64.CEILが2個残っており、削減対象がないとは
言えない。ただし費用や速度倍率の予測には使わない。旧診断のclock差やallocation差を
今回の時間比較へ持ち込まず、独立した同一L4 allocationで未計装の完全stepを比較する。

## 正しさ

直接FP64 Rを渡すprobeで、旧分類と新helperのbank・signed first tile・source owner・
既存cast後のFP32局所変位を照合する。R=.5と直後、1/2/4/8/16/32の直前・一致・直後、
50の直前・一致、周期/first-tile境界、負の中心、異なる入出力owner、同一非ゼロoriginを含める。
この直接Rの検査と、実際のFP32 ParameterからdecodeしたRの検査を区別する。

rank6/12/17、B0–4、owner chunk境界、部分勾配、offload/multiple/retained/current-Pと
既存Graph契約のテストを再利用する。rank3を追加し、rank2の境界検査だけは両軸の
radius min/birth/max/upper_floorを合法な.5へ揃える。これはテストfixture限定であり、
通常学習や性能測定の半径は固定しない。不一致live axesをprobeがkernel起動前に拒否する。

Y/dX/全dPと実optimizer後のP/m/v/clockは#132とのbyte一致を要求する。
独立FP64参照の許容値`3e-5+3e-4 abs(ref)`は維持し、byte不一致の救済に使わない。

## prospective採用guide

N1024/A52428/真B4/seed41、FP32 body/FP64 geometryとPolar、TF32 off、native CSTAdamW。
MNISTの重み付き3層モデル全体で、CSTは中央1層。両Planを#132でbootstrapし、初期8実更新後と
#132自身の45k checkpointを凍結する。新しい同一L4 allocationで各状態256更新×6 balanced
fresh-process pairsと同等条件dense Graphを測る。gather/loss/全backward/dX/全atom勾配/
実optimizer/metrics/guard/index/version通知を含む未計装の区間境界同期とする。

- 両状態でpaired geometric短縮率>=5%、各6/6勝利。
- 開始・終了42状態＋全5勾配、metrics/data/orderがbyte一致、finite、fallback0。
- capture込みpeak増加allocated<=2MiB、reserved<=8MiB。
- prospective memory-v2.1のちょうど8cycle×16実更新をpass。
  report自身のendpoint phase、summary peak>=全epoch peak>=endpoint、cycles3–8 plateau、
  128実更新/120replay、全state/gradient/metrics/orderを確認する。
- setup/recapture/restore/retirement/GCは同じscopeで記録し、内訳を包括値へ二重加算しない。
  短期45k外挿は長期実測と区別する。

すべて通過した場合のみ、2順序の各45,000実更新（44992replay）へ進む。初期8更新は
学習時間へ一度だけ計上し、準備・再capture・Graph退役GC込みの実測で各順序>=5%、
各3epochの47 Tensor/metrics/order一致、長期peak予算を要求する。3epochを8cycleの証拠にしない。
短期失敗なら候補を保全して見送り、その結果をProへ相談して次の研究方針を選ぶ。

## ローカル事前検証

CPU suite: 1746 passed / 2535 skipped。対象fixture/probe構築テストは8 passed / 120 skipped。
host C++ syntax、Ruff、Plan/Case宣言、git diff --checkはpass。
GPU検証・短期比較の結果は以下。条件付き長期は短期速度基準未達のため実行しなかった。


## 独立したL4 allocationでの結果

測定sourceは`4261aad3eaa6beb346def04b7a1c5edb986b1877`。
GPUはNVIDIA L4、UUID `9d5c0c4b-b4e9-3248-fb29-2e54ea0624fd`、driver580.82.07。
Torch2.11.0+cu130/CUDA13/Triton3.6、Python3.13.16、CPU math threads1。
同一allocationで全検証・36短期cell・4計装診断を完了し、GPU環境の632件が通過した。

| 凍結状態 | #132 baseline ms/update | known-bank ms/update | paired geometric短縮率 | 勝利数 | 同等dense Graph ms/update |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初期8更新後 | 0.580366850 | 0.565393696 | 2.580545% | 6/6 | 0.204510816 |
| #132自身の45,000更新後 | 0.589550924 | 0.575424524 | 2.396177% | 6/6 | 0.210274626 |

ms/updateは各256実更新の未計装区間境界同期を6 fresh processesで測った算術平均。
短縮率は各対応pairの比の幾何平均であり、上の平均値の比とは定義が異なる。
全モデルは重み付き3層中、中央の1層だけがCST。別モデル・seed・shapeの結果ではない。
学習後状態は#132の保存checkpointであり、候補自身の45k学習結果ではない。
短期の45k償却推定は初期1.52–2.58%、学習後2.06–2.25%で、長期実測と区別する。

両状態12 CST cellsずつの開始・終了42状態＋5勾配、metrics/data/orderはbyte一致し、
既存#132 numeric controlとも一致。finite、unexpected fallback0。
短期capture込みpeak allocated251.5341797MiB/reserved442MiBで各pairの増分0。
別途8cycle/128実更新/120replayのprospective memory-v2.1も全項目pass。
両Planのlifecycle peakは251.9921875MiB/444MiB、cycles3–8のendpointは
allocated245.4072266MiB/reserved444MiBでplateauとなり、増分0だった。

**両状態で5%以上という事前速度基準を満たさなかった。既知bank再利用は単独では採用せず、
45,000更新を実行しない。#132を研究baselineとして維持する。**
2%以上の短期改善は今回の限定条件での観測として保全し、基準を事後に緩めない。
候補の見送りを研究全体の終了とはせず、この結果をProへ相談して次の単独候補を選ぶ。

## 計装診断と機構の限界

別processのtorch.profiler16更新/条件では96 kernel events/updateで変化なし。
forward contractは初期88.951563→80.983625us、学習後90.395625→83.609688us、
reverse contractは初期98.263625→89.751438us、学習後103.561688→96.983625us。
classifier、sort、raw moments等は今回の変更対象ではない。
contract短縮と未計装の全モデル改善は整合するが、演算・register・dependency等の因果分解は
未実施。計装時間を採用判定へ混ぜず、旧#133のNCU結果をこのbinaryの証拠として扱わない。

## 証拠の同定と再確認

- pool job: `l4job-129d2f19c6594832a2c1933ff4a22b3f`
- source archive SHA256: `6e5f489b5e196bd2be98f4fe8aad9587f8a59b4f100100439059cf959d67af98`
- driver SHA256: `baf97143e9e4a0c1ee2466f88701170f602e19fe6632ee26b0c79fd4a475c5af`
- result archive SHA256: `788c6b94c96c3633163b5c6ba0fcbd8d5a58925e2319dd88da80b00fa088ff37`
- driver returncode0/timed_out=false、source照合、archiveとraw manifest1024ファイルを再hash。
- lossless CASから22 PT原本を元SHA一致で復元。8cycleの各47 Tensorと短期2状態の各42 Tensor、
  合計460件を直接loadしてbyte一致を確認。短期の5勾配は全cellのraw hashも比較した。

[処理済み結果](owner-window-known-bank-results-20261104.json)に全pair、memory判定、
各診断と証拠同定を残した。raw log/trace/checkpoint/driverはignored
`output/bank-reuse/results/`と共有pool job directoryに保管する。
実行は共有L4 poolの`submit --source <this-checkout> --script output/bank-reuse/driver.py --timeout 2400`
と同一キューの`serve --workers 1 --idle-seconds 0`。driverの7入力原本とhashはinput manifestで追跡する。
GPUは回収後停止を確認し、source CIは[41799959845](https://github.com/takumiecd/torchcst/actions/runs/41799959845)でpass。
main/dispatcher/mergeは変更しない。
