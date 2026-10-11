# RegularGrid tile16: Graph poolとbuffer寿命

2026-10-11。組み合わせ版 `tile-owner-all-bm16` のreserved124 MiBを調査した。
正規化・floor・支持・Y/dX/全4種類のatom勾配を維持し、計算kernelとbuffer寿命を分けて比較する。
最終版は **dP遅延 + AdamW momentsの事前初期化 + capture前のwarmup勾配解放**。
完全stepのallocated/reservedは **62.774/88 MiB**、同条件denseは **81.752/104 MiB**。
通常版に対する完全step差は−0.255〜+0.239%。大きな速度低下を伴わず36 MiB削減した。
GPU process全体のメモリは未測定。公開dispatcherの既定は変更せず、研究runnerの明示policyと研究planで実行する。

## 124 MiBの内訳

Torch2.9.1+cu128のnative allocatorで `torch.cuda.memory._snapshot()` の
`segment_pool_id`、segment/blockとallocation/free履歴を記録した。
メモリ診断は未計装の完全stepと別のfresh workerで行う。

| replay後の領域 | reserved MiB | 生存tensorのallocated MiB | inactive MiB |
| --- | ---: | ---: | ---: |
| 通常pool | 60 | 13.562 | 46.438 |
| Graph専用pool | 64 | 3.701 | 60.299 |
| 合計 | 124 | 17.263 | 106.737 |

inactiveのGraph領域にもreplayで読み書きするアドレスがあり、Graphが生きている間は
単純に解放できない。inactiveを全部「不要なtensor」とは解釈しない。
allocatedの完全stepピークは65.975 MiBで、replay終了時の17.263 MiBとは異なる。

通常poolのsegmentは20+20+14+2+2+2 MiB。20 MiBの一つはcapture後には
完全inactiveだが、warmupのdPを保持した状態でcaptureへ入り、通常poolと
private poolが重なる間にreservedピークを作る。
もう一つの20 MiBにはParameter3.200 MiBと2つのint64 ID対応表計3.200 MiB、
14 MiBにはAdamW moments計6.400 MiBが残る。scalar step/input/target/dy/Graph補助状態を
含む小さいsegmentも残る。したがって通常poolの46.438 MiBには、解放できる全inactive
segmentと、生存bufferが一部を占めるため解放できない分割segmentの両方が含まれる。

Graph専用poolは20+20+14+2×5=64 MiB。H/G、packed、source snapshot、sortの
一時領域、owner partial、physical勾配、dP、Polar更新などの確保・解放を経て保持される。
ここで独立のGraphを複数残したことによるpool数の増加は観測されていない。

## buffer寿命

N2048/B32/A209715、BM16での論理サイズ。これはallocatorピークの代用ではない。

| buffer | MiB程度 | 必要な生存区間 |
| --- | ---: | --- |
| source snapshot | 3.2 | forwardからsource VJPまで。retained backward用にforwardが所有 |
| packed 13 fields | 10.4 | 正規化準備からbackwardのH/G生成・dX集約まで |
| routing + 全owner metadata | 5.06 | forwardのH/YとbackwardのH/G/dXで再利用 |
| H | 12.8 | forwardはbatch chunk内。backwardでは再計算し、G/dG生成で消費 |
| G | 12.8 | backward chunk内。dH生成とdX集約で消費 |
| physical勾配 | 2.4 | backward chunk間で累積し、最後のsource VJPで消費 |
| owner partial | 1.0 | 各chunkのownerからpartial reductionまで |
| dP | 3.2 | source VJPからoptimizer更新まで |
| sort keys/values/order64/workspace | 可変 | 各軸のrouting準備中。workspaceの上界は未規定 |

H/Gの全支持を縮約してからパラメータ勾配を作る契約を維持する。
dH/dGを全量保存せず、既存の即時消費を維持する。

H共用版はforwardごとのH/scratchを `save_for_backward` で保持し、backwardで上書きする。
別のforwardへ共有しない。retained backwardでもHを毎回再計算する。
arena版は512-byte境界の独立領域を一つのallocationから切り出し、source/packed/metadata/
H/G/physical/partialを配置する。sourceの既存cloneをarenaへのcopyに置き換え、追加の全量コピーはしない。
physicalの初期化はTritonで行い、同じstorageを持つ保存viewのversion counterを変更しない。

## 寿命変更の完全step比較

source `a9508e3d7b04efbc2f333561204fcd771c958824`、通常native allocator。
2独立cohort/rho、2回目はworker順を反転。各cohort10 fresh workers、24更新/21 samples。

| Plan | rho3 ms | rho8 ms | allocated / reserved MiB |
| --- | ---: | ---: | ---: |
| 組み合わせ版 | 1.375–1.381 | 1.720–1.722 | 65.975 / 124 |
| H/partial共用 | 1.375–1.380 | 1.719–1.722 | 65.975 / 130 |
| dP確保を後ろへ | 1.376–1.379 | 1.716–1.718 | 62.774 / 124 |
| 共用 + dP遅延 | 1.376–1.379 | 1.719–1.729 | 62.774 / 130 |
| 単一arena | 1.380–1.382 | 1.720–1.720 | 71.490 / 122 |
| dense | 約0.329 | 約0.329 | 81.752 / 106 |

H/partial共用は通常poolを60→66 MiBに増やし、private pool64 MiBは変わらない。
dP遅延は生存ピークを3.200 MiB減らすが、segment数/サイズを変えずreservedは減らない。
arenaは通常poolを60→44 MiBに減らす一方、private poolを64→78 MiBへ増やす。
G/physicalをforwardから確保するため、sort等との生存期間の重なりも増え、allocatedが5.515 MiB増える。
同じallocationにまとめることと、短い寿命の領域を再利用できることは別の条件である。

40 workers/840 samples、72 initial/post24 full-shape FP64 gates PASS。
541 GPU qualification PASS、candidateと組み合わせ版のpost24 P/Y/dX/dP、各独立repeatはbitwise一致。
前段source `6661e2de93f65349be2853e9cee44a25680e83c8` の共用/dP遅延比較も
36 workers/756 samples、64 full-shape gates、509 GPU qualificationを通過し、同じ負のメモリ結果だった。

## allocatorとcapture境界

[PyTorch2.9 CUDA allocator資料](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-memory-management)
に従い、native/expandable/topology reuseを別processの設定として試した。
`PYTORCH_CUDA_ALLOC_CONF` の実設定をsnapshotの `allocator_settings` と
`is_expandable` で照合する。設定だけを文字列で記録して有効とは判定しない。
初回は新しい変数名 `PYTORCH_ALLOC_CONF` がこのTorchでは無視された。
その24診断は設定変更の証拠から除外し、原本を保存した上で旧名の新規jobを使った。

以下は24更新の別workerでのメモリ診断。各値はrho3/8で一致。
experimental設定はメモリで候補を絞るための診断で、完全shape oracle付き性能結果とは扱わない。

| 設定とcapture準備 | 組み合わせ版reserved | arena reserved | 同条件dense reserved |
| --- | ---: | ---: | ---: |
| native、従来境界 | 124 | 122 | 106 |
| native、warmup勾配解放 | 102 | 122 | 90 |
| expandable、従来境界 | 134 | 112 | 126 |
| expandable、warmup勾配解放 | 112 | 112 | 126 |

`graph_capture_record_stream_reuse=True` はこの単一capture-streamのfixtureでは
native/expandableそれぞれの上の値を変えなかった。別streamや長いGraph一般の効果は未測定。
expandableは同設定dense126 MiBより小さいarena/解放版112 MiBを作れるが、
従来dense106 MiBを上回り、dense自身のメモリも悪化するため本課題の改善とは採用しない。

`release-warmup` は2 eager warmupsの後、stream同期・model.zero_grad(set_to_none=True)・
x.grad=None・gc.collect・empty_cacheを一度だけ行い、その後captureする。
optimizerのmoments/clockやParameterは保存する。peak counterはwarmup前から最後まで
リセットせず、2 warmups + initial replay + 21 timed replays =24実更新を保つ。
毎stepのforward/metadata準備/aggregation/backward/optimizerにはcache clearを追加しない。
公開runtimeや既定のcapture policyを変更せず、研究runnerの明示optionとして比較する。

nativeの組み合わせ版は通常pool60→38 MiB、private pool64 MiBのままで計102 MiB。
denseも通常pool88→72 MiB、private pool18 MiBのままで計90 MiB。
従来dense106 MiBだけを参照して「dense以下」とは判定しない。
同条件dense90 MiBへ届くには、通常pool38 MiBを前提にprivate pool64→52 MiB以下が必要になる。

Graphオブジェクトと出力を削除後もdP/dXが残るため、private poolの一部が残る。
勾配tensorを解放しempty_cacheした時点まで診断し、Graph由来tensorの所有とpool寿命を確認した。
終了時のreservedの低下を、capture/replay込みのpeak低下として報告しない。

## 同sourceのcapture境界だけの完全step比較

source `b74f8bfd31001e00e453ec0927ff8d064375bf07`。
通常と解放のそれぞれ2独立cohort/rhoで比較した。optimizer stateは従来通りlazy初期化。
64 fresh workers/1344 samples、112 initial/post24 full-shape FP64 gates、543資格検証PASS。
各routeのP/Y/dX/dPは2つのpolicy間でも各独立repeat間でもbitwise一致。

| policy | 組み合わせrho3 ms | 組み合わせrho8 ms | 組み合わせallocated / reserved | 同policy dense reserved |
| --- | ---: | ---: | ---: | ---: |
| standard | 1.368–1.377 | 1.712–1.719 | 65.975 / 124 MiB | 106 MiB |
| optimized | 1.378–1.380 | 1.719–1.724 | 65.975 / 102 MiB | 90 MiB |

小さな時間差を無劣化の証明とはしない。memory gateは解放版102 > dense90で未達。
次の切り分けとして、AdamW momentsを作るタイミングを移す。

## AdamW state初期化とdP寿命の組み合わせ

source `d5405af19ff446c79ce3d8eb34e81933ff13efa2`。
`--optimizer-state-policy preallocate` は、勾配が無いことを確認して一時的な空gradを与え、
Torch AdamW自身の `_init_group` にstep0とmomentsを確保させる。optimizer.stepや
geometry updateは行わず、直後にgradを解放する。これはTorch2.9の研究用初期化境界で、
公開optimizer実装へ依存を追加しない。CPU/CUDAで未更新のParameter/step0/zero momentsと、
3更新後のlazy初期化との完全一致を検証した。

初期化はprimaryのpeak resetより**後**に行い、一時gradとmomentsの確保も計測に含める。
別processのoptimizerや外部poolへ状態を逃がさない。全primaryはnative allocatorを使う。
source snapshot/packed/全量H/G/gradientの追加copyは作らない。

| policy・plan | rho3 ms | rho8 ms | allocated / reserved MiB | 同policy dense reserved |
| --- | ---: | ---: | ---: | ---: |
| 通常lazy・組み合わせ | 1.377419–1.380539 | 1.717309–1.720608 | 65.975 / 124 | 106 |
| moments先行+解放・組み合わせ | 1.380582–1.382292 | 1.720569–1.723288 | 65.975 / 100 | 104 |
| moments先行+解放・dP遅延 | 1.380659–1.380717 | 1.716219–1.718220 | 62.774 / 88 | 104 |
| moments先行+解放・arena | 1.384093–1.386038 | 1.721904–1.724300 | 71.490 / 102 | 104 |

事前momentsだけはreserved108 MiB。capture前解放だけは102 MiB。
両方の組み合わせでは元のowner planが100 MiBまで下がり、dP遅延を加えると88 MiBになる。
従来境界でdP遅延がreserved124 MiBのままだった負の結果と合わせて、効果を分離した。
H共用やarenaを採用する必要は無い。

最終dP遅延版のreplay時の通常poolは22 MiB、Graph poolは66 MiB。
元の通常60/Graph64 MiBから、通常poolを38 MiB減らし、Graph poolは2 MiB増えて計36 MiB減った。
単にGraph pool自体を小さくした結果ではない。
最終版のwarmup peakは86 MiB、capture/replay peakが88 MiB。
元のdP遅延版はwarmup100 MiB、capture/replay124 MiBだった。
原owner版をmoments先行+解放で実行した場合、replay時点は88 MiBだがwarmup peak100 MiBが残る。
この100を88と置き換えて比較しない。

同policyのdense peak104 MiBに対して16 MiB少ない。
denseにとって最小だったlazy+解放の90 MiBよりも2 MiB少なく、dense側の悪化だけで
成立する比較ではない。公開CSTLinearの一般shape/別GPU/別optimizerでの保証とは扱わない。

2独立cohort/rho/policy、逆順2回目、8 fresh workers/cohort。
64 workers/1344 samples、112 initial/post24 full-shape FP64 gatesが不変の4e-4でPASS。
545資格検証もFAIL/error/skip0。sourceと入力/Plan/Case/runtimeをread-only auditで照合した。
同routeの通常/最適化policy間、候補同士、独立repeatのP/Y/dX/全dPはbitwise一致。
別の4 fresh workersでは24更新後のP/dP/dX/Y/loss/step/exp_avg/exp_avg_sqを保存し、
通常ownerと最適化dP遅延版が全fieldでbitwise一致することを確認した。primaryのParameter hashにも一致する。
初期化による学習時計・geometry更新回数の変更は無い。

速度差はrho3 run1 +0.239%、run2 +0.009%、rho8 run1 +0.053%、run2 −0.255%。
厳密に0の差とは主張しない。大きな速度低下は観測せず、正規化や勾配の近似で速くしていない。
capture前のgc/empty_cacheとmoments初期化は一度の準備で、21 replayのstep時間には入らない。
その一度の準備の壁時計時間・compile/capture latencyは今回の速度指標として未測定。
各stepのsupport/routing準備・H/G生成・Y/dX集約・backward・AdamW/Polar更新は全部含める。

## 再現・保全

最終測定sourceは上のd5405af1。条件はRTX6000 Ada48GB、Torch2.9.1+cu128/CUDA12.8/
Triton3.5.1、D2 full-period RegularGrid/FlatTorus、N2048/B32/A209715、rho_initial3/8、seed41、
MSE、IEEE FP32/TF32 off、24実AdamW/Polar更新。source/inputs/Plan/Caseは同一cohort内で一致。

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:native PYTHONPATH=src:. python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/regular-grid-h-2048-rho3.json \
  --plans matrix-torch factor onchip-h input-site-reciprocal-bm16-g16-h32 \
    tile-owner-all-bm16 tile-owner-late-bm16 tile-owner-arena-bm16 dense \
  --isolated-oracle --phases --optimizer-state-policy preallocate \
  --graph-pool-policy release-warmup --output output/new-memory-cohort/comparison.json
```

rho8はcaseを差し替え、独立2回目は新しい出力先と `--reverse` を使う。
対照は両policyを `lazy` / `standard` にする。sourceを移した場合も各tracked fileとarchive hashを照合する。
57-plan declaration/Ruff/diff PASS。CPU1817 PASS/4574 GPU・DB skips/18既存warnings。
[測定sourceのCPU/build CI](https://github.com/takumiecd/torchcst/actions/runs/38141808610) PASS。

raw oracle/state `.pt` はremoteに保存し、compact evidenceとallocation snapshotsはlocalに検証して取得した。
前段の負の完全step比較、設定名が無効だった試行、validなexperimental memory triage、
native-controllerのdirectory copy失敗（GPU qualification/timing前）も別分類で保存した。
失敗・不適格な試行を性能の独立runへ数えず、性能再試行・oracle閾値の変更はしていない。

| evidence | value |
| --- | --- |
| final remote | `/home/jovyan/runs/cst-tile-state-ada-d5405af1-20261011` |
| controller / finish PID | 290257 / 300171 |
| source archive SHA256 | `2e367c7221b6a8f59ea07794b6d0d5ee8627f6dc9153269b7ccd16ba8aefc1a5` |
| compact archive SHA256 | `288765a0d6e3d9085abc7910dbfa487771b7471b1d9427a5018e8c613ec45817` |
| audit SHA256 | `66e652e3f1d53d06c5d7fc94151a918d47f9f391a61cafefe9f7ef6d13c6d2d7` |
| verified compact entries | 306 |
| verified pinned tracked files | 1676 |
| forensic archive SHA256 | `5f02b30bd548339368772dccc544a190dbc8b2cbaa32fd866b85498566c1a480` |
| verified forensic entries | 264 |
| local evidence | `output/regular-grid-h/tile-state-*`, `tile-capture-native-v2-*`, `tile-arena-*`, `tile-pool-*` |

前段source a9508e3dのcompact/audit hashは
`ed08161b0b42780cb88fc3a0a027a6e50b54e3bea2c1d5b656a8745640dc5fe5` /
`a4cb8cb34858e3d875fda71205ec21028ade458e85c5e2370a15703a1a156c03`（163 entries）。
b74f8bfdのcapture比較は
`5dcb1e190c6267452ff2dd26613a9a265b032920c0499d6991cc15bc00457697` /
`c48dbf3ad813045c105f52b955e94739d59a73e4dad3ea0cde46b7b52cf90d0a`（253 entries）。
前段を最終64 workersの統計へpoolしない。Hubはuserが開いたready状態を維持し、実験終了後のidleを確認する。
