# 2026-10-07: exact ordering-key cache

Branch `kernel/order-key-cache`, based on PR #35 checkpoint `08c87c19`。
PR #36のtile/VJP実験とは独立したbranch。朝08:00 JSTまでの改善目標を継続する。

## 省略する処理と維持する処理

現在のmetadataからdirectionごとにcanonical composite keyを計算する。
bucketは実際のsingleton/active状態とrho帯、positionはclipped support開始位置。
keyは`((bucket * (max(K,N)+1) + position) * atoms + canonical_id)`。
key全体が前回と一致すると、同じ安定順序とbucket offsetsを再利用できる。
変化があればそのdirection全体をsortし、cacheを更新する。

省略するのはsortとhistogram/scanだけ。毎forward、live Parametersから全coefficients・
normalizers・幅・位置・支持を準備し、physical値とownerの正確なmin/max範囲を再構築する。
支持の終端だけ伸びるときはkeyが同じでもowner範囲を更新する。
位置勾配は既存のnormalized-factor VJPで計算し、keyへの微分で置き換えない。
Hはglobal tensorへ保存しない。

cacheはmodel所有の`OrderKeyCache(nn.Module)`。canonical keys、現在のID permutation、
bucket offsetsとcounterだけを保持する。各forwardのviews/orders/offsets/rangesは新規snapshotで、
autogradへmutable cacheを保存しない。更新はmodelのstream上で直列化する。
固定atom数・domain・recipeを検査し、Algorithm instanceへ実行状態を持たせない。
public dispatcherは変更しない。範囲確認用のstateless builderはoracle comparisonで使うが、
実行Planのcached routeはmodel-owned cacheがなければ拒否する。

int32キーが安全なshapeではint32、既存と同じsentinel境界を超える場合はint64を使う。
N128/A819の追加resident buffersはkeys 6552、order 6552、offsets104、counter48 bytes。
この13256 bytesはtensor予算であり、実測のcapture/replayピークとは区別する。
global key buffersの物理L2常駐やDRAM削減は実測なしに主張しない。

## 候補とprotocol

| suffix | sort warps | payloadコピー | owner範囲 |
| --- | ---: | --- | --- |
| ordered_cache_copy4 | 4 | owner CTAで分担 | direction × owner |
| ordered_cache_copy8 | 8 | owner CTAで分担 | direction × owner |
| ordered_cache_range4 | 4 | sort CTA内 | direction × owner |
| ordered_cache_range8 | 8 | sort CTA内 | direction × owner |

全候補はconsumerの16 batch行 / 16出力次元 / 32atom records / 4分割を維持する。
baselineはcopy8、controlsはparallel4と元のsplit4/int32。
N64/N128 B32 A204/A819、seed41、FP32 IEEE、TF32 off、21 samples、productionの
可変幅fused AdamW/Polar更新とdense。性能fixtureのdecoded初期rho>1をassertする。
rho1.25/3/8/mixedを層別し、rebuild/reuseの実数counterと完全step時間・allocated/reservedピークを測る。
counterの書き込み費用もprimary graph時間へ含まれる。

## 検証 / 02:37 JST checkpoint

source `df02c89f57330640f4b4090be8eedf469cb790ac`。
ruff、宣言往復、8ケースprepare/check成功。
CPU:861 passed / 597 skipped (22.02s)、wheel/sdist build成功。
GPU用に41 testsを追加。実際のslice gradients、20 Graph更新とmoments/step、
旧forward後の再構築を既存oracle testsで確認する。
key再利用のtopology unit testは、payload変更・支持endだけの変更・方向片方の開始変更・
band/singleton/inactive変更をcontrolled metadataで分離して、fresh rebuildと全snapshotを比較する。
これらのmetadata unit checksと、実Parametersの独立FP64勾配oracleを区別する。

初回長いscreen `l4job-3c7f62b45f124d5c8e458839fee849c0` はqueuedだったため短いjobsへ
分割する予定。raw evidence: ignored `benchmarks/cuda/linear/evidence/owner-cache-20261007/`。
driver/analyzer/DB scriptsを保存した。runtime/result/source archive hashとraw Parameter/inputs hashを
比較してから採用判断する。cache速度・メモリ結果はまだない。

PR #35 full repeat1 `l4job-3989854491c041178dfd79a847f148e3`はColab CLI transport timeoutで
interrupted。結果未回収なので性能・test成功へ使わない。source/spec/transportログを保持。
pool supervisor22376は異常終了し、owned L4 session停止とserver no-activeを確認した。
後続未実行jobsをcancelし、検証とサイズ別の短いmeasure jobsへ置き換える。
回収済みの3 screen jobs / 12 artifactsは独立にhash検証・DB保存済み。

## 03:08 JST: 再提出

長い旧jobは実行前にcancel済み。source `652b5fb9f47771a942d4683c5d66ac9945cdbb5d`
(runtimeは`df02c89f57330640f4b4090be8eedf469cb790ac`と同一)を凍結して、
one L4 queueへ短い3 jobsをsubmitした。

- `l4job-96b73b763ed84eb98bacad150b7c8c9f`: 新cache41 tests。
- `l4job-3e124d65f6694434a974c113e65b53fd`: N64 rho1.25/3/8/mixed、4候補と3controls。
- `l4job-6a64ce010f344510b8799ad4d079a2fc`: N128、同じ比較。

検証とscreenは未完了。測定の各候補はfull-shape FP64 oracle比較を先に行う。
独立反復やG4比較はL4結果から有望な候補を選んで実施する。
