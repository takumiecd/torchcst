# HIGGS: CSTとDSTの学習曲線を比べる参照pilot

## 目的と範囲

2026-11-05、所有者からDSTとの学習曲線比較が追加で要求された。
実行中の低密度CST確認実験(source 97580582ed28c4e494bf5c2656bbe94a85c544c3)
の算術・選択・確認データの扱いは変更しない。別sourceでdevelopmentだけを使うpilot。

今回のDSTはdense Tensorにmaskを持つ参照実装であり、実際の疎行列実装の
学習速度・メモリ効率を評価するものではない。CSTのL0的な正則化という解釈も、
曲線に整合する仮説であって因果の証明ではない。

## 固定する条件

- HIGGS: 同じ100k train、公式validationの先頭50k development。
  後半50k confirmationとofficial testは開かない。
- データSHA256: 62d32a1789ed737a5997ca6a0630548e6211af86c9cd6a0a605cd057d8bd32ac。
- 28→1024→1024→2、ReLU×2、中間bias無し、全層学習。
- CST nominal alpha0.001のA=floor(1024²/1000)=1048に対応。
  4A=4192個のactive weight slotsをDSTの中間層に用意する。
  全体の学習scalar budgetはCST/DSTとも35938。これはmask/indexの保存量ではない。
- Staticとrow-balanced RigL-styleを比較する。同じ初期mask/値を使う。
  各行4または5slotを持ち、再配線後も各行のslot数を維持する。
  active slotは新生時のゼロ値も含み、数学的な非ゼロ数とは区別する。
- 入出力層の初期化、seedごとのデータ順はCST/denseと一致させる。
  sparse重みは各行のfan-inに基づくU(-1/sqrt(k),1/sqrt(k))で初期化する。
- seed17、lr0.0003、AdamW betas(0.9,0.999)、eps1e-8、WD0.01。
  true B4、FP32、TF32 off、CPU math1 thread、L4×1。
- 両方式とも10epoch、250000 minibatches/1000000 example presentations。
  checkpointは最小development CE、tieは早いepochで選ぶ。
  lr0.0003は現時点のdense/CST選択値を用いるpilotであり、DSTの調整済み最良値ではない。

## 接続更新

RigL-styleはminibatch step tについて、0<t<0.75Tかつt mod250=0で更新する。
drop fractionは0.3からcutoffまでcosineで0へ下げる。
このバッチでは全層のAdamW更新を行わず、勾配による接続探索を行う。
Staticでは全250000バッチ、RigL-styleでは249251回のoptimizer更新になる。

小さい絶対値の重みをglobal thresholdで落とし、空になる行は最大重みを残す。
各行の不足slotはdense-map勾配の絶対値が大きい場所へ増やし、tieは小さい列を優先。
保持された重みとmomentsは維持し、prune/growされた場所は値・momentsをゼロにする。
直後に再生した同じ座標もリセットし、global Adam clockは維持する。

原論文の更新置換タイミングを採用するが、行budget維持、cadence250、
prune直後の同座標再生時のmoment resetを明示したvariantである。
旧cst repoのoptimizer更新後に別バッチでrewireする方式とは混ぜない。

- [RigL原論文 Algorithm1](https://proceedings.mlr.press/v119/evci20a/evci20a.pdf)
- [固定した公式実装](https://github.com/google-research/rigl/blob/d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9/rigl/sparse_optimizers_base.py#L451-L482)

## 検証と記録

実データ学習前にCPU FP64 masked referenceとのY/dX/dW比較と、両幅・両端密度・
両方式のeager/Graph再配線軌跡一致を必須にする。状態、勾配、全dense growth勾配、
topology hash、moment reset、minibatch数と実optimizer更新数を保持する。

図の横軸は共通のminibatch数または処理example数を主とし、optimizer更新数も記録する。
全epochの訓練損失、development CE/accuracy/AUCをdense/CSTと重ねる。
時間には参照実装のdense処理・接続探索・監査が含まれ、疎実装の優劣とは解釈しない。

source/results archive、checkpoint、curvesはignored outputに保全する。
この文書は事前条件であり、GPU実施済みや性能優位を意味しない。
