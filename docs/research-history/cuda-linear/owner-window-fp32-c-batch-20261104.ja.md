# #131基準のFP32 C縮約batch×feature配置

## 事前に固定する変更

基準は#131のint32 sort版。算術・実行コードは `8ecb3d7f` と同じで、
公開メモリ判定v2を追加した `486fc48c` から分岐する。
候補は `GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C` の明示選択だけで使う。

現C縮約は128 threadsのCTA内で `threadIdx.x < rank` のthreadがfeatureを担当し、
thread内でbatchを順次処理する。候補は `thread=b*rank+f` とし、各出力に1 threadを割り当てる。
rank12/B4では12→48 threadsとなるが、速度倍率をこの比から推定しない。
各出力のchunk順・atom順・float積和式、partials形式、保存payload、geometry/classifier/sort、
H/DH/G/dP、Polar pullback、optimizerを維持する。追加global buffer・launch・barrierは0。
forward Cとreverse dXのCの双方に適用し、dP-onlyの `emit_x=false` はC処理をしない。
threadからbatch/featureへの整数演算とwarp/sharedアクセス、register割当は変化し得る。

以前の#126ではFP64/eager条件であり、今回のFP32/Graph/geometry共有/sort32条件の
単独変更の代わりにはしない。追加候補やTensor Core、fast math、精度変更を併用しない。

## 事前の検証・採否基準

1. #131とY/dX/全dPのbyte一致を要求する。B0–4、rank6/12/17、空、部分勾配、
   集中ownerのatom数127/128/129/255/256/257を含める。
2. saved hooks/offload/metadata、retained backward、最新P、version保護、
   hooks/settings/storage/callback/finite-checkとGraph境界の契約を維持する。
3. 独立CPU FP64参照の既存 `3e-5+3e-4 abs(ref)` も維持する。
   #131とのbyte不一致を許容誤差で救済しない。
4. N1024/A52428/真B4、seed41、L4、TF32 off、CPU math threads1、native CSTAdamW。
   MNISTの3重み付き層（中央のみCST）の完全Graph stepを比較する。
5. 開始状態は#131 FP32で8実更新後と、FP32自身の45,000更新後。
   同じ256 minibatch、6組/stateのbalanced order、fresh process、同一L4 allocation。
   基準の数値状態は#131既存256更新の保存hashにも照合する。
6. 未計装・区間境界同期でgather/loss/全backward/dX/全atom勾配/実optimizer/metricsと
   guard/index供給/version通知を含める。denseも同等のGraph/FP32/TF32条件。
7. 両状態でpaired幾何平均5%以上短縮、各6/6勝、開始/終了42状態・全5勾配・集計・順序一致、
   fallback0の場合だけ継続する。準備・退役費用は別記し、45,000更新への償却推計と実測を区別する。
8. メモリは `graph-lifecycle-memory-v2` をprospectiveに使う。
   8 cycles×16実更新、評価→train復帰→同期後・checkpoint保存/旧Graph退役前のendpointで、
   cycle3–8のallocated/reservedが各byte一定。全cycle状態hash/勾配/集計/順序一致、fallback0。
   初回から最終退役までのpeak候補−基準がallocated≤2MiB、reserved≤8MiB。
   8回未満は判定不能、8回超から平らなsuffixを選び直すことはしない。
9. 数値・短期速度・メモリ/lifecycleがすべて合格した場合だけ、順序反転2対の各45,000実更新へ進む。
   長期は全3epochの状態・勾配・集計一致、準備/初期capture/復元/再準備/recapture/
   旧Graph解放GC/最終退役込み5%以上短縮、peak予算を要求する。
   長期3epochを8cycle判定へ偽って入力せず、合格した8cycle runをevidence_scopeで参照する。
10. 未達なら負結果を保存して候補を終了し、#131を基準に残す。類似候補の追加探索はしない。

別processの16更新profileで両contract、他kernel群、kernel数を記録する。
必要な場合に限りNsightのSASS/register/shared/warp情報で原因を調べるが、計装時間は採否に使わない。
初期・学習後の両状態を固定して比較し、kernel単体の改善だけで採用しない。

## 結果

未実測。source/result hash、判定version、run ID、完全step・メモリ・状態・採否を取得後に追記する。
