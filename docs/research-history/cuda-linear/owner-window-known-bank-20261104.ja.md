# #132を基準にした既知bank再利用の独立研究

## 研究の境界

ユーザーは候補や診断の停止を研究全体の終了とせず、Proへの結果相談と次の実装・実験を
継続することを依頼した。#133の負結果・部分診断・raw判定は保持し、過去の条件を救済しない。
本件は#132 `fba2e72d`を基準とする新しい前向き比較で、#133の採用再審査ではない。
Proに部分診断の結果と次の研究方針を送信済み。回答確認まではCPU検証の準備を進める。

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
GPU correctness・未計装短期・条件付き長期は未実施。
