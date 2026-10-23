# 階層Fourierタイルの再開用引継ぎ

読み合わせ用：[output_localのCUDA C++例](examples/output_local_walkthrough.cu)。
ユーザーがCUDA C++で設計・実装を学べるよう、4 warp×32 threadsの担当と
profileのshared配置を明示した。数学的な処理を対応させる例であり、Tritonの
実thread/shared配置の再現ではない。未compile・GPU未検証・runtime未接続。

2026-10-23最新：[profile cache容量32/64/128の比較](hierarchical-fourier-cache-128-20261023.ja.md)。
`656c1c7f`でGPU追加3 testsとrunner全9 workerがPASS。
cache32はNative768→192 KiB、全step allocated1.529→0.940 MBへ減ったが、
Graph時間1.037→1.606 msへ悪化。cache64も遅い。容量上限だけを下げる案は不採用。
Norm/Meta/atom順序は旧cache128とbitwise一致。初期23 atomの境界cacheが全て
失われるため、次は再利用の多い広いprofileも残し、未使用slot/穴/保持fieldを
詰める候補を検討する。L2 hit率や物理転送量は未測定。job回収・停止確認済み。

2026-10-23先行：[N128 live packing・グループ一括縮約の実装と実測](hierarchical-fourier-packed-128-20261023.ja.md)。
`3327f8b3`でGPU 13 testsと既存runner全13 workerを通過。初期39 groupをGPU生成し、
P/routesを物理配置、勾配は元IDへ戻す。Graph完全stepは従来local比8%短縮したが
direct/denseに未達、Eager改善も未達。Native 768 KiBとH容量は据え置きで、
構造的な省メモリ目標は未達。job回収・hash照合・VM停止確認済み。
ユーザーは適量のメモリ使用を許容。極小H scratchのための多wave化は強制せず、
次はcache容量・H保持・準備の分離を独立に比較する。公開dispatcherへは未採用。

2026-10-23設計追記：[全128 atomの39グループとHの生成・最終利用の担当案](hierarchical-fourier-128-groups-20261023.ja.md)。
保存済みGPU geometryからproducer/consumer表を作成し、CPUで所属と数値和を検査した。
この設計時点ではgrouped CUDA未実装。上の最新記録でpacking/縮約を実装・実測した。
説明図のHscratch領域再利用とC/残差consumer表の刷新は未実装。

2026-10-23先行実験：[128の実機実行・保持と受け渡しの確認結果](hierarchical-fourier-128-execution-20261023.ja.md)。
Y/dX/全P勾配の検査を通過。局所Y→残差→C展開で同じY storageを使うことを確認。
Hの全容量は残り、128では時間的な重なりは観測されなかった。job回収・停止確認済み。

2026-10-23追加：[メモリ保持の監査と128からの試作](hierarchical-fourier-memory-and-128-20261023.ja.md)。
ユーザーの方針で試作をN128/A128へ縮小した準備段階の記録。
局所Hのstore省略と確保容量の削減を区別する。GPU実行結果は上の最新記録を参照。

2026-10-23追記：ユーザーと多段の支持分類・配置・block分担を設計し、
局所出力とZ/Hの重ね合わせを限定prototypeで検証した。
1024条件の実装・2 jobの検証結果・停止状態は
[局所block数と重ね合わせの記録](hierarchical-fourier-local-overlap-20261023.ja.md)を参照。
時間的な重なりは確認したが、今回のcaseでは既存directを上回らず採用していない。
以下の停止状態・checkpointは2026-10-21時点の記録。

ユーザーの依頼で2026-10-21に探索を一旦終了した。
**階層タイルはユーザーが最も有望と考えている方針として保持する。**
この文書は次sessionの入口。自動再開、他chatへの送信、GPUの予約はしていない。

## 保存場所と停止状態

- branch: `kernel/hierarchical-fourier-tiles`
- worktree: `/Users/ware10sai/.codex/worktrees/quadratic-bspline-reference/torchcst`
- 数値実装checkpoint: `ecb6987cc224de79b6d96e27b3a9cbad636dede2`
- [数式・測定・source/result hash](hierarchical-fourier-dispatch-20261021.ja.md)
- [前sessionの通常密度比較](hierarchical-fourier-reuse-20261021.ja.md)
- raw: worktree内 `output/hierarchical-fourier-dispatch-20261021/`
- pool raw: `~/.local/state/colab-l4-pool/jobs/<job>/`
- 所有する待機jobを取消済み。pool全slot stopped、serveプロセスなし。Kyutech未使用。
- mainを変更せず、push/PR/mergeもしていない。worktreeとignored証拠は削除しない。

## 保持する数学と実行設計

研究専用profile-product linear。公開CSTLinear/Polar更新への採用ではない。
Pの列は振幅・出力中心・入力中心・I logit。
I=.1+9.9 sigmoid(logit) は有効半幅で、Gaussian標準偏差ではない。
中心と幅をatomごとに独立・連続に学習する。固定アンカー・座標一致・幅量子化を仮定しない。

既存Fourier係数、quintic周期/階層blend、中心±Lの支持、混合後の離散積L2とfloorを維持する。
FP64 profile/geometry/norm、FP32共有縮約。同じ値を高速に計算する研究である。

Xから必要なタイルのZを生成し、各atomの係数と縮約してHを作る。
出力側はatomをタイル係数へ集約しsiteへ展開する。境界はL2計算で作った残差Rと微分を再利用する。
Hはbackwardまで保存可能。入力依存のZ/Hは更新されたXに対して毎step再構築する。
親の偶数周波数は子から継承できるが、奇数周波数は新しい読込を必要とする。

実装は `src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/`。
recipeに `minimum_sharing`、`boundary_cache`、`coefficient_norm`、
`coefficient_norm_warps`、`sort_atoms` を追加済み。
sortは一時的なpacked atomと逆順序書戻しを使い、optimizerのParameter/momentsは元の順序を保持する。

現在の共有数gateは**成分が必要とする両軸の全完全タイル**で閾値を満たす場合だけ共有する。
候補共有数で選ぶため、選択後の共有数が閾値以上になる固定点保証はない。
atlas確保容量も最大形状のまま。細かいnode単位の振り分けや省メモリは未解決。

## 検証済みと未確認

修正後critical GPU gate 30件成功。各回収済みbenchmark caseでも独立FP64による
Y/dX/全atom勾配、retained backward、変更後Graph replayを照合した。
成功gateのsnapshotに対し、現checkoutの関連runtime/benchmark/test計1128ファイルのhash一致を確認済み。

境界値再利用は初期A1024大Iで旧tile経路比30〜40%短縮したが、同条件directより遅い。
幅sortは小atom数で悪化し、一部A8192で改善した。独立repeat未実施。
一般有限和による離散係数L2は正しいが、測定したM8中Iでは大きく悪化した。
この有限和の実装結果だけから階層タイル全体の不適合を結論しない。

新候補の通常密度A52428、大Iの残り、cluster比較、全82 GPU regressionは待機中に取消した。
一般的な共有数別損益分岐と最終dispatcherは未確定。Dense2倍以内は未達成。
通信timeoutで回収不能となった長いmatrix jobは全結果を除外している。

## 再開時の候補（未実装・未測定）

1. 正確な離散Parsevalを適用できる一周期・非aliasの場合だけO(M)の係数L2を使い、
   それ以外は既存sampled FP64 normへ戻す候補を独立kernel/recipeで試す。
   現実装はParseval分岐があっても一般FF×FF有限和コードを同じkernelに含む。
   一般行列経路をcompileから除外した効果はまだ測っていない。これを最初の小さい比較候補とする。
2. 現在の成分全体gateに代えてnodeごとに共有/残差へ振り分ける案。
   残差cacheで穴を表せる可能性があるが、atomの利用maskと階層依存maskを分け、
   二重加算を防ぐ証明と独立照合が必要。未実装。
3. まず短い単一caseでdirect・境界cache・新候補・Denseを比較し、
   満たした候補だけを通常密度へ進める。長い全matrix一括jobは再利用しない。

再開はユーザーの新しい依頼を受けてから行う。GPUはColabのみ、Kyutechは学習用なので使わない。
Colab skillとpool運用を読み直し、所有権を確認する。raw logを成功扱いせず、
archive回収・hash照合・全worker PASS・独立oracleを確認する。
精度閾値atol=rtol=4e-4を緩めず、振り分け・sort・係数・L2・cache・AdamW込みの
完全stepとGraph capture/replay込みallocated/reservedピークを測る。

## ローカルで状態を確認する入口

```bash
cd /Users/ware10sai/.codex/worktrees/quadratic-bspline-reference/torchcst
git status --short
git log -3 --oneline
python3 /Users/ware10sai/.codex/skills/colab-l4-pool/scripts/pool.py status
python3 output/hierarchical-fourier-dispatch-20261021/collect.py
```

collectorは51測定行・11成功jobを復元する。`short-jobs.json`は投入時のstatusが古いため、
停止時の状態は `stop-status.json` と `final-pool-status.json` を参照する。
短い再現driverは `one-case.py`、精度gateは `critical-gates.py`。
ベンチの実際の選択planは各jobのdriverとartifacts/case JSONを正本にする。
local Pythonはprimary checkoutの `.venv/bin/python`、`PYTHONPATH=src:.` を指定する。
