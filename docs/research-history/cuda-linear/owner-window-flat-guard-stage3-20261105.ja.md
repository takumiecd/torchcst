# G1 Stage3: 正式な256更新pairの事前仕様

Stage1、Stage2の通過を受け、以下を結果取得前に固定する。最初のpairそのものが正式比較であり、
追加のtrial256は作らない。G1の四つのguard/core、数値body、`src/`はStage1/2から変えない。

## 固定範囲と順序

- L4同時1台、1 job、hard timeout900秒、再試行0。最初のCST processはcompile込み最大500秒、他は各150秒、全体残り時間以内。
- 最大48 fresh process＝2状態（initial8/自身45k）×6 pair×CST/dense×qualified/flat、各256実更新。
- `full_step_graph_comparison`の固定flagsは`--mode graph --precision fp32_geometry_sort32_flat_c --bootstrap-fp32 --sort32-baseline --steps 256`。
  bootstrapは既存Sort32 FP32 Plan、8更新後はFlatC。
  この既存routingを変えず、両Planを分けて記録する。自身45kは`fp32_final.pt`/`dense_final.pt`を使う。
- pair index0–5。偶数pairはinitial→final、奇数はfinal→initial。state indexは常にinitial=0/final=1。
  CSTは`(pair index+state index)`が偶数ならqualified→flat、奇数なら逆。denseは同じstateのCSTと逆順。
- 各stateでCST二セルを先に完結させ、通過したときだけ同stateのdense二セルへ進む。
  最初の有効なCST pairでflat時間≥qualified時間なら、同値も勝利とはせず即停止し、後続セルを追加しない。

## 正しさ・memory・来歴

各cellは実Plan/recipe/entry・実loaded extension path/SHA、backend/flat有効性、入力・checkpoint・order SHA、clockを時計外に保存する。
全cellはStage1/2と同じnative backend、FP32 FLAT_C、TF32off、math threads1、同じ7凍結入力を使う。
clockは8→264または45000→45256。通常fallback0、finite、capture復元を必須にする。

全repeatへ`.evidence.pt`を保存し、開始/終端の全状態・5勾配・totalsを直接byte照合する。
既存timed loop、memory reset、retirementは変更せず、追加hash/receipt/evidence出力は時計外とする。
同pairだけでなく同kind/stateの全repeatも同一の数値結果を要求する。
CSTは既存の外部凍結start/end/grad/totals参照とも照合する。dense initialも旧Gpilotの凍結256参照を使う。
dense finalの開始は保存済Stage1の20状態を参照し、終端は**この正式runの最初のqualified cell**を事前指定した参照とする。
後者を外部検証済の終端とは呼ばず、flat/後続repeatに同じ参照を適用する。参照作成用の追加GPU試行はしない。

capture込みpeak allocated/reservedの増分は、各pairで≤2MiB/≤8MiBを要求する。
Stage2のCST公式v2.1/dense手動継続条件は既に通過した証拠を継承し、このshort runを再admissionとは呼ばない。
code/input/clock/backend/binary/数値/finite/fallback/memoryの違反は停止し、性能負けとは区別して記録する。

## 性能採用の条件

CSTは両状態それぞれ6/6勝利、かつ

\[
1-\exp\left(\frac16\sum_{i=1}^{6}\log\frac{t_{flat,i}}{t_{qualified,i}}\right)\ge0.05
\]

を要求する。倍率1.05を短縮率5%の代用にはしない。6勝でも5%未達は不採用。
最初の有効CST負け、または正誤/memory違反で止め、有利な反復だけを選別しない。
denseにも同じguard変更・実行条件を与えて全時間を記録するが、CSTの未達を救済しない。
denseの正誤/memoryは同じく必須で、速度差は共通の実行基盤効果を分離する対照として報告する。

主時計は区間境界同期の未計装全model step（gather/loss/backward/全optimizer/metrics/guard/index/version通知）。
kernel時間やeligible費用をmodel時間へ換算しない。資格補強済G0からの改善を、元#132より速い証拠とも呼ばない。
全条件を通過した場合だけ、既定の二順序45,000更新へ進む。Stage3単体で採用/mergeはしない。
