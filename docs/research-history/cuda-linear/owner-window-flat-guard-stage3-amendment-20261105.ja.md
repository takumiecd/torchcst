# G1 Stage3: 最後の追加試行の事前改訂

## 判断と上限

Pro回答`7e2a6c28-984d-46cc-a1f3-dfa4d493f7bd`を検討し、固定G1の正式比較を
最後の一度だけ追加する方針を選んだ。これは元のStage3のno-retry、およびStage1改訂の
「再びharness不備なら固定G1を未判定で保全終了」に対する**明示的な予算例外**である。
従来規則の内側の継続とは表示しない。

- 旧job `l4job-cf65034272e04e7dae022df1f47566a9`は基準1セル、候補0、完成pair0のまま保全する。
  基準0.753108ms/updateは観測済みだが、候補との比較結果は存在しない。
- 新しい1 allocation・1 jobだけ。最大48 fresh-processセル。formal累計最大49セル・2 job。
  開始数、process完了数、検証済み数を分けて記録する。Stage3の3回目は行わない。
- 旧基準時間は費用計上と保全にのみ使い、新allocationの候補とpaired比較へ接合しない。
- Stage1/2は再実行せず、検証済み証拠を継承する。追加の予備pairも作らない。
- 追加jobもdriver・環境・timeout・回収不備を含め一度で停止する。再失敗なら性能未判定で保全終了。
  有効なCST非勝利ならadmission不合格として停止し、測り直さない。

## 変更する検査だけ

旧driver SHA `70e399d6129c5d045607e4283ee50aa6ed55fa04222ebb7050ae2b4df4c2184d`と
原generatorは変更しない。別名の追加driverを作り、保存start/endのclock件数assertだけを
公開`validate_clock_schema`の呼出しへ接続する。CST6/Dense5のexact path、CPU FP32 scalar、
期待値・bytesとCST alias値を要求する。clockの切り捨てや`>=5`への緩和はしない。
追加した検査module/testのhashもsource manifestへ含める。

42/20状態、5勾配、metrics/order、finite/fallback、source/input/native backend、Plan/entry/実.so、
直接byte照合の既存検査は維持する。資格補強済G0とG1の核心4ファイル、数値body、src242ファイル、
bootstrap Sort32→測定FlatCのrouting、7入力、計時・capture peak scopeは変更しない。

## GPU確保前の条件

1. 公開clock検査のCPU回帰を通す。fractional値8.5、件数を維持したpath入替、CST alias欠落も拒否する。
2. 保存済み実基準セルについて、旧driverの誤assert再現を保持した上で、修正検査経路が他の全検査を
   外さず通ることをCPU assessmentとして別記録する。旧runの失敗・未完了flagは変更しない。
3. **実際に使う追加driver**のprocess起動をsynthetic fixture応答へ置換し、同じ順序・検証・比較・
   停止・集計コードをCPUで最後まで通す。48セル全通過、候補先行、最初のCST同値/負けで後続0、
   6勝でもGM短縮5%未満なら不合格、schema/数値/backend/来歴/memory不正、timeout/出力欠落/回収失敗の
   保存・停止を確認する。fixtureを数値・速度の実証に数えない。
4. 既存processログから初回compile、warm process、保存/検証のwall費用と900秒予算の整合を確認する。
   明らかに収まらないなら、このclock-only例外では起動しない。完走保証やGPU step時間の外挿にはしない。

追加CPU監査で候補/core/順序/採用条件の変更が必要と分かった場合も例外を広げず、固定G1を保全終了する。

## 維持する正式判定

元Stage3の順序と停止処理をそのまま使う。initial8/own45000の各6 balanced pair、各256実更新、
最大48セル。最初の有効CST `flat >= qualified`（同値含む）で後続denseも含め即停止する。
両stateそれぞれ6/6勝利、`1-exp(mean(log(flat/qualified))) >= 0.05`、byte一致、fallback0、
capture込みpeak増分allocated≤2MiB/reserved≤8MiBを要求する。denseの速度差でCST未達を救済しない。
hard timeout900秒、保存用45秒reserve、初回process最大500秒・以後最大150秒も維持する。
全短期条件を満たした場合だけ既定の二順序45,000更新へ進む。追加結果を得る前にこの改訂をcommitする。
