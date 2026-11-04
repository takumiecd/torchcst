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

**短期・prospective lifecycle v2を通過し、条件付き45,000更新でも準備・退役込み10.61–12.17%短縮した。**
全数値状態・勾配はbyte一致し、両peakの増分は0。全事前gateが合格した。
研究用の明示opt-in候補として保全し、公開dispatcherと依存PRのmergeは別の判断とする。

### 短期の未計装完全step

| 開始状態 | #131平均ms/step | Flat C平均ms/step | paired幾何平均短縮 | 勝数 | dense Graph平均ms/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| #131で8実更新後 | 0.654075619 | 0.582469812 | 10.9476015% | 6/6 | 0.207086212 |
| FP32自身の45,000更新後 | 0.666998834 | 0.591725081 | 11.2853379% | 6/6 | 0.207343999 |

各状態のCST12cellで開始/終了42状態・全5勾配・集計・data/input/order hashが一致。
基準の数値は以前の#131の256更新時の保存hashとも一致した。dense6cell/stateも数値一致。
全CST cellのcapture込みpeak allocated251.5341797MiB、reserved442MiBで同じ。fallback0。
短期の準備・退役込み45,000更新償却推計も記録したが、下の長期実測とは区別する。

### Prospective memory v2

`graph-lifecycle-memory-v2`、`evidence_scope=prospective` を測定前から使用した。
両者とも8 cycles×16実更新、全cycleの42状態/5勾配・集計・順序一致、fallback0。
両者のepoch末allocatedは245.4072266MiBで8回一定、reservedは両者とも
442/444/444/444/444/444/444/444MiBだった。cycle3–8のbyte一定とpeak増分0を満たしてpass。
記録位相は評価→train復帰→同期後、checkpoint保存・旧Graph退役前。
最終退役後の全Graph pool解放や、PyTorch外のprocess全体GPU割当を証明する値ではない。

raw判定内のrun名はこのjob archive内の相対識別子であり、processed JSONの
`lifecycle_evidence_context` にjob IDと完全修飾したrun識別子を別添した。
今回、#131の旧false判定は再利用・上書きせず、新候補のprospective証拠だけで継続を判断した。

### 条件付き45,000実更新

短期の全gate通過後だけ、全train60,000/test10,000、3epoch/45,000実更新を両順序で実行した。
各cellは初期8実更新を計時・学習集計へ含み、44,992 Graph replay。
capture/recaptureは8/15,000/30,000更新後。各captureで42状態と計数済みmetricsの復元が一致した。

| 実行順序 | #131学習ms/step | Flat C学習ms/step | #131準備・退役込みms/step | Flat C準備・退役込みms/step | 準備・退役込み短縮 |
| --- | ---: | ---: | ---: | ---: | ---: |
| #131 → Flat C | 0.701822276 | 0.610419773 | 0.749144739 | 0.657983380 | 12.1687244% |
| Flat C → #131 | 0.693479724 | 0.615229516 | 0.741526604 | 0.662848001 | 10.6103548% |

全4cell・全3epochで42数値状態/全5Parameter勾配・loss/正解数・順序hashが一致。
回収した実checkpointも全epochでbyte一致し、旧#131最終checkpointの47 Tensorとも一致した。
最終test精度97.3699987%、loss0.08367239683866501、全saved値finite、fallback0。

学習時間はgather/loss/全層backward/dX/全atom勾配/実optimizer/metricsと
guard/index供給/version通知を含む。準備・退役費用は各cell2.1295–2.1621秒で、
うち旧Graph解放GCと最終退役は0.3292–0.3451秒。初期optimizer生成、capture/復元、
境界の再検証/再準備/recapture、旧Graph退役・GC・最終退役を対称に加算した。
評価・checkpoint保存・data/model配置・epoch順序準備/転送・転送用梱包は別扱いで、
process全体wall timeではない。順序対ごとの実測差は平均へ隠さず両方残す。

長期のcapture込みpeak allocatedは全cell251.9921875MiB、reserved444MiBで、候補増分0。
epoch末allocated245.4072266MiB一定、reserved442/444/444MiBも両者・両順序で同じ。
長期はpeak予算と状態・速度を検査し、lifecycleは合格済みの8cycle runを明示参照する。
3epochを8cycle判定へ渡して合格扱いにはしていない。`long_gates_pass=true`。

同条件dense Graphは学習0.201733438ms/step、準備・退役込み0.229459773ms/step、
test精度97.87%/loss0.070017516613。Flat C/dense時間比は学習3.0259–3.0497、
準備・退役込み2.8675–2.8887。denseとの所要時間差はまだ残っている。

### 別process profileと説明できる範囲

16更新の独立したtorch.profiler CPU/CUDA + cProfile runを各状態/各Planで実行した。

| 開始状態 | #131両contract µs/step | Flat C両contract µs/step | #131分類 µs/step | Flat C分類 µs/step |
| --- | ---: | ---: | ---: | ---: |
| 初期 | 255.349 | 188.628 | 111.268 | 111.458 |
| 学習後 | 264.116 | 196.644 | 111.448 | 111.566 |

kernel-category eventは両者とも96/step。Radixは両者約61.2–61.4µs/step。
両contractの約67µs減は、未計装短期完全stepの約72–75µs減と概ね整合する。
ただしcontractには係数/H/G/dPも融合されている。register割当やspill、shared memory競合、
eligible warp、SASSを計測していないため、削減分をCループだけの時間とは断定しない。
Nsightは今回実行しておらず、CPU/GPU律速や物理的原因は未確定。profileを採否の時間へ使わない。
owner分布は各profileの16診断更新後の値で、計時開始時点そのものではない。

### 正しさと独立診断

GPU環境478検証が通過。新候補のB0–4、rank6/12/17、集中owner/chunk端、空、
部分勾配と既存payload/offload/retained/current P、Graphのhooks/settings/storage/callback/
finite-check・capture境界を含む。既存#131の整数sort検証も同じbuildで通過した。
ローカルCPU1731 pass/2396 skip、Linux CI1730 pass/2397 skip、宣言・wheel/sdist buildが通過。

候補自身の最終checkpointで固定train/test minibatch各1個の実X/dYを用い、全52,428 atomの
独立CPU FP64 Fourier Y/dX/dPと、stratified8 atomの48点quadratureが既存基準で通過した。
最大絶対差はY8.4077e-7、dX1.3420e-10、dP8.1639e-10。
実m/v/clock45,000を用いたAdam方向差は最大1.1921e-7、次のnative更新のP差は
振幅列3.7253e-9以下/他列0、clock45,001。同じ勾配のcontrolはPとoptimizer状態がbyte一致。
以前のFP32 cold-Adam gateの負結果を合格へ変更したものではない。

### Provenance・復元・限界

実測source `c3b77eb811fcc852eaef35d4c9d7ed9b70e05dc4`、
job `l4job-df8a9fce519b4027bc3a226fc92ae5ae`、L4 UUID
`d508b2c1-146b-a75e-4970-6951276aba48`、driver580.82.07、Torch2.11+cu130/CUDA13/Triton3.6、
Python3.13.16。CPUはXeon2.20GHz/12logical、math threads1。
一つのL4 allocationで全fresh-process比較を順次実行した。1seed/1モデル/1shape/真B4であり、
他モデル・batch・hardware・eagerの速度や汎用公開Graph対応へ外挿しない。

source archive SHA256 `2311de1d4e850513e4fb29a4ae03d07d75331d9720031294e98afd63f26937bd`、
private driver SHA256 `fd1f2a70f4d445c9bc755e912c74e70e77ebd8e07124513c302a812af9361265`、
result archive SHA256 `083c4fc133c2808c9238317554963ff6991eb3b0f1097cb5f05ed77f36a18292`。
archiveと1,431 manifest fileをローカルで再検証し、39元PTファイルを全byte/SHA256一致で復元した。
直接Tensor比較は18比較×47 Tensor（lifecycle/長期/旧#131最終）と2比較×42 Tensor（短期）で全一致。

raw source/spec/driver/archive/manifest/trace/checkpointはpoolの当該jobとworktreeの
`output/fp32-c-batch/results` に保全。受信archiveは元の梱包形式で残し、復元先copyへ元PTを展開した。
短期の個別sample・推計、各versionの判定、状態hash、profile、checkpoint hashは
[`owner-window-fp32-c-batch-results-20261104.json`](owner-window-fp32-c-batch-results-20261104.json)
へ保存。GPU停止をfresh pool statusで確認し、supervisorと一時的なsleep抑制も正常終了した。

今回の実装・事前判定整備・短期・条件付き45,000更新は完了。この候補へ別の最適化は追加しない。
