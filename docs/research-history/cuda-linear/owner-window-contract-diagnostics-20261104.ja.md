# #132/#133 contractの限定診断と停止記録

## 結論

Proの追加回答を約3分間隔で確認し、完成後にユーザーの依頼どおり限定診断を自動開始した。
4 scientific cellsの取得・状態一致は確認できたが、集計器の不備に伴うGPU割当の分離と
SM frequency counterの差が残り、同一GPU・clock条件を揃えた4セル比較としては未成立。
部分診断として保全して終了し、known-bank候補の実装・短期採用比較・45,000更新には
進めなかった。基準は#132、#133の不採用判断は変わらない。追加counterや別replay方式を
探索していない。main/defaultの変更・mergeはしていない。

## 固定した仕様と実行した範囲

測定時の`ff4ae86a7d62b930de9de2d0f9b58322a4829075`をgit archiveで別に固定し、
そのbuild内の#132 Flat C Planと#133 phase-delta Planを比較した。
これは#133の実測比較を説明する診断であり、pristine #132 binaryとの機械語一致を
新たに主張するものではない。通常の`-O2 --ftz=false`を維持し、`-G`、lineinfo、
fast math、register上限の変更はない。

既存の短期runnerを呼ぶPython診断adapterで、bootstrap/capture/restore後の
16 replayだけをProfiler Start/Stopで囲った。`torch.profiler`との併用はしない。
application replayの各PIDに、開始・終了42 state/5 gradients、metrics/order、
clock、実際にloadしたextensionの絶対path/SHA256/build.ninjaを別々に保存した。
初期clockは8→24、学習後は45000→45016。入力checkpointは#132自身の45kであり、
候補の45k学習実験ではない。

NCUはnode graph profiling、application replay、grid matching/strict、
cache-control none、clock-control base、application-only、LaunchStatsと固定16 metrics。
classifier/forward contract/reverse contractだけを選び、launch-countで切らず、
結果が各16件、合計48件であることを検査した。metric/CLIを実機queryで確認してから使った。
`--import-sass yes`でreportにcodeを保持し、実際の.soからcuobjdumpのresource/SASS/ELFを取得。
sm_89 native cubinを確認した。

仕様に使用した公式資料：
[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)、
[CUDA 13.0 Binary Utilities](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-binary-utilities/index.html)。
実際のversion/help/query、全commandと出力はraw evidenceに保存した。

## 集計器の不備と継続の境界

最初のjobは初期#132の48件を正しく収集したが、NCUのdemangled名がboolを`false/true`
ではなく`0/1`と表すため、driverがforward16件をreverseへ誤分類し、予定の後続3セルを
実行せず停止した。元のraw判定（classifier16/reverse32、diagnostic_established=false）は
変更していない。手元で元CSVを再検査し、実際にはclassifier/forward/reverse各16件で、
4 application passesの状態・勾配・metrics/orderも一致していたことを確認した。

boolの両表記を認識するよう集計だけを修正し、未実行の3セルだけ別jobで継続した。
最初のNCUセルは繰り返さず、counterセット・replay方式・clock要求も変更していない。
ただし最初の所有VMは既に停止していたため、全体は2 allocationとなった。
各jobで別途行った1回の16更新binary準備は、4 scientific NCU cellsの反復とは区別する。

| セル | 開始状態 | Plan | allocation | kernel results | application counter passes |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | #132の8更新後 | #132 Plan | 1 | 48（各16） | 4 |
| 1 | 同じ状態 | #133 Plan | 2 | 48（各16） | 4 |
| 2 | #132の45k checkpoint | #133 Plan | 2 | 48（各16） | 4 |
| 3 | 同じ状態 | #132 Plan | 2 | 48（各16） | 4 |

counter pass数を独立run数とは数えない。全4セルで各passの開始・終了state/gradient/
metrics/order一致、fallback0、入力状態と過去保存済み16更新controlの一致を確認した。
異なる2 jobのSASS dump・resource dump・ELF listもfile SHA256で完全一致した。

## 観測事実と限界

保存済み#133の16更新profileを方向別に再集計すると、forwardは初期88.660→79.076us、
学習後89.996→80.152usと短縮し、reverseは97.124→120.232us、101.558→121.008usへ遅延。
これは既存profileの再集計で、fresh GPU timingや採用性能の測定ではない。

生成コードのresourceは次のとおり。

| contract | #132 Plan REG/thread | #133 Plan REG/thread | STACK bytes（両者） |
| --- | ---: | ---: | ---: |
| forward | 79 | 48 | 288 |
| reverse | 86 | 56 | 304 |

registerは減少しており、単純なregister増加説は支持されない。STACK増加もない。
resourceのLOCAL:0表記はlocalアクセスがないことを意味しない。実測local load/store
sector件数は各状態・方向で両Planが完全一致した。初期forwardはload1017450/store1137150、
reverseはload2084580/store1485180 sectors/kernel。同じ件数でもcache/依存待ちの費用が
同じとは限らない。

同じallocationで取った学習後のreverseでは、active warpsは36.406→39.321%、
eligible warpsは0.2635→0.2778、issue activeは21.388→20.607%。命令実行数は
7922591→7414141へ減る一方、long-scoreboard割合は29.654→42.058%だった。
割合を専有時間へ換算せず、load依存・命令配置・cache等の寄与を一意に断定しない。
「DRAM帯域律速」とも断定できない。原因は未特定のままとする。

#132 Planのforward/reverseそれぞれに`F2I.F64.CEIL`が2個あり、source上のR依存bank分類に
対応する仕事が残る。#133ではこれが0となり、key復元に伴う`F2I.U64.TRUNC`等が現れる。
known-bank案に削れる仕事がない、とは言えないが、これを速度予測には使わない。
独立CPU preflightではR1022個×中心78個＝79716条件で分類式の同値性を確認しただけで、
known-bankのCUDA実装やbyte検証・速度結果は存在しない。

clock-control baseを全セルで要求したが、初期classifierのSM frequency counter平均は
allocation1基準841.237MHz、allocation2候補822.088MHzだった。学習後のreverseは
同一allocationで基準809.377MHz、候補807.293MHz。memory frequencyは約6.242GHz。
clock counterは計測された値で、これだけで固定要求の失敗原因を決定できないが、
初期pairのdevice/clock差の影響を排除した因果比較とは言えない。
cache-control noneについてNCUの標準warningも記録されている。

したがって、取得されたcode・状態一致・counter件数の証拠と、条件を揃えた4セルの
定量診断としての合格を分ける。後者は未成立として停止し、追加計測で条件を救済しない。
NCUの時間・peakは採用判定には使用しない。

## host補強の検証と保全

診断専用入口の同一live axes検査は`1846a250`で別commitにした。
3拒否条件、周期境界payload、既存VJP fixtureを含むL4環境の12検証がpass。
host C++ syntax/Ruffと[CPU CI](https://github.com/takumiecd/torchcst/actions/runs/41791263609)もpass。
元の測定記録のsource/hashと判定は変更していない。

- 最初のjob `l4job-8f7074ff88994ca1913546b3f1589eef`、source archive
  `45535cb61ae26ae7321a6b776b5bcb133d6d51b6b031b6ecd9c914937338dae8`、result archive
  `15a6e32ade7f0b54fefee64649ae81acf522741946d32e7f64d7f05262df610b`、39 manifest filesを再hash。
- 継続job `l4job-a7297724e6634bc295e62397c291c004`、source archive
  `a9fa596db85f4597df06553dd964ddee47634a9c3493fdf4840a184b2f9e448f`、result archive
  `dc6ed95b719657a77126c4f776e2f79aeb410278cf32403be2fa715ebb4d38ba`、59 filesを再hash。
- driver sourceは1846a250、診断CUDA sourceは別archiveのff4ae86a。
  両driver returncode0/timed_out=false。rawと訂正後評価をともに保全。
  所有L4停止・supervisor/caffeinate正常終了を確認した。
- raw driver、NCU reports、全pass JSON、実.so、SASS/resource/ELF、query/help、元source archive、
  pool receiptはignored `output/contract-diagnostic/`と共有pool job directoryへ保管。
  source・counter設定・adapter hash、集計と停止理由は
  [processed JSON](owner-window-contract-diagnostics-results-20261104.json)に保存した。
