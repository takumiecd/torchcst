# G4での最終G1実行器の確認

目的は、L4で検証済みの最終版をG4でも測り、denseとの差と安全検査変更の効果を確認すること。
追加のkernel最適化、幅・batchの探索、公開defaultへの採用やmain統合は行わない。

## 測定前に固定する条件

- sourceは`9ee4642cad1beda9646101aa94e2bf88ca9bdc75`。既存L4測定の303 sourceファイル、
  runtime242ファイルとguard核心4ファイルをbyte単位で維持する。
- G4を1台・1 allocation。実hardware、Torch/CUDA/Triton、CPUを記録し、所有sessionを結果回収後に停止する。
  Torch2.11.0+cu130/CUDA13.0と既存資格境界を要求する。異なる場合は測定せず記録する。
- MNIST full train60,000/test10,000、seed41、N1024/A52428、真B4、中央1層CSTの3学習層。
  FP32 body＋既存FP64 geometry/Polar、TF32 off、math threads1、既存optimizer契約。
  初期値・データ・順序はL4と同じhashの入力を使う。
- 最大8 fresh process。AはCST qualified→CST G1→dense G1→dense qualified、Bは完全逆順。
  各3epoch/45,000実更新、Graph replay44,992回。候補とdenseに同じ実行方式を与える。
- 新GPUで既存FlatC kernel/whole-step契約テストを実行する。小規模31atomでは独立48点求積で
  Y/dX/全atom勾配とoptimizer参照を確認してから学習を測定する。
- 同一G4内では各epoch末のG0/G1・二順序の状態・勾配・指標・clockを直接byte比較する。
  L4との学習軌道のbyte一致は異世代間の合格条件にしない。モデル・入力・契約・データ順序は維持する。
- 既存runnerのtraining時間と準備・capture/再capture・復元・退役/GC込み時間を両方報告する。
  evaluation/checkpoint/import/data配置等は従来の時計外。cold compile時間は未分離。
- capture/評価込み累積peak allocated/reservedを記録し、G1増分は従来どおり2MiB/8MiB以内。
  L4 memory-v2.1合格をG4の合格として流用せず、今回は3epochの実測peakを報告する。
- G4の速度は記述的比較。5%以上というL4採用閾値をG4へ課して都合よく打ち切らず、
  正しさ・資格・メモリ違反または予算超過で停止する。新しい候補・救済再測定は行わない。
- 全体1200秒、回収用90秒を予約。GPUテスト最大500秒、最初の学習process最大500秒、以降180秒。
  source・入力・結果archiveのSHA256、実loaded binary、raw log/PTをignored outputへ保全する。

以上はGPU割当前に固定した仕様である。以下に実施結果を追記する。

## 結果

G4の確認は完了した。**N1024/B4では最終CSTはdenseの学習時間の3.14–3.15倍、
準備・退役込みで3.05–3.06倍だった。G1のguard変更によるG4上の実用的な短縮は確認できなかった。**
この確認をもって終了し、追加の最適化・GPU再測定は行わない。

実機はNVIDIA RTX PRO 6000 Blackwell Server Edition（SM120、101,974,081,536 bytes）、
AMD EPYC 9B45、driver580.82.07、Python3.13.16、Torch2.11.0+cu130
（git `70d99e998b4955e0049d13a98d77ae1b14db1f45`）、CUDA13.0、Triton3.6.0。
sessionは`cst-g4-g1-20261105`。一allocation内の8 fresh processであり、独立した8台のGPU測定ではない。

| 順序 | 最終CST 学習秒 | 最終dense 学習秒 | CST/dense | CST 準備・退役込み秒 | dense 同秒 | CST/dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 20.184048 | 6.420217 | 3.143827 | 20.657581 | 6.778694 | 3.047428 |
| B（全逆順） | 20.180700 | 6.405311 | 3.150620 | 20.671690 | 6.761934 | 3.057067 |

G0→G1のCST学習短縮はA0.01577%/B0.00654%、準備・退役込みはA0.08599%/B−0.05893%。
denseの学習短縮はA−0.22146%/B−0.00032%、準備・退役込みはA−0.30353%/B−0.11019%。
これらの微小差から速度改善や回帰の一般的な結論は出さない。L4の12.8–19.1%短縮を
G4へ一般化せず、G4で5%採用条件を通過したとも主張しない。
既存のL4研究opt-in判断は変更しない。GPU処理とCPU guardのどちらが律速かを調べる追加profileも行っていない。

| capture/評価込み累積peak | CST G1 | dense G1 |
| --- | ---: | ---: |
| allocated | 261,020,160 bytes（248.928 MiB） | 273,735,168 bytes（261.054 MiB） |
| reserved | 465,567,744 bytes（444 MiB） | 452,984,832 bytes（432 MiB） |

両順序とも同じpeakで、G0→G1増分は全4pairでallocated/reservedとも0 bytes。
CSTはallocatedが約12.126 MiB少ない一方、reservedは12 MiB多い。GPU process全使用量ではない。
G4で8cycle memory-v2.1を新たに実行したという意味でもない。

最終testはCST97.33%/loss0.084526062、dense97.79%/loss0.067210644。
同kindのG0/G1・二順序の各epoch末では状態・勾配・clock・指標・データ順序がbyte一致した。
異世代の学習軌道は同一とせず、この精度差を単独の比較実験から一般的性能差とも解釈しない。
大きい幅でのCST実用性やスケーリングは今回測定していない。

## 検証・保全

- 対象テスト130件pass、skip0（GPUを実行する126件、CPU4件）。
  FlatCのY/dX/全atom勾配、境界・空入力・部分勾配、whole-step capture/replayと変更時拒否を含む。
- 別processの31atom×3fixture（ordinary/constant/alternating）で独立48点求積を実施。
  Y/dX/全atom勾配が既存誤差条件を通過し、ordinaryではoptimizer参照も通過した。
  独立求積は52428atom全体のoracleではない。
- 回収後CPUでraw448ファイルをsize/SHAまで検証し、24 PTを元の全file bytesへ復元した。
  全24epoch snapshotのhash/clock/finiteを再確認し、G0/G1・順序間の18比較、延べ648 Tensor項目が直接byte一致した。
  CPU readerはTorch2.13で、GPU workerの実runtimeと区別した。
- source303/runtime242/guard核心4は元L4ソースと一致。実loaded拡張binaryも圧縮保存し、復元SHAを確認した。
  数値kernel、optimizer算術、学習runnerは変更していない。
- workerのdriver実行・保存は273.051秒。これは計測対象の学習時間とは別で、session確保・upload/downloadを含まない。
- 結果archiveのSHA/size確認後に所有sessionを停止し、`colab sessions`でactiveなしを確認した。

原本は研究worktreeのignored `output/g4-confirmation-20261105/`。
`source.tar.gz`、`inputs.zip`、`g4-driver.py`、`g4-oracle.py`、`launch.py`、
`adaptation.json`、`upload-manifest.json`、`receipt.json`、`results.tar.gz`、
`results/artifacts/`、`verify.py`、`verification.json`を保全する。
公開集計は同名`owner-window-g4-confirmation-20261105.json`。

| 原本 | SHA256 |
| --- | --- |
| source archive（9ee4642c） | `ea7e430fb746d54e6bdd04ec8ce699dc29c0584f0f5c2feb217463f2c7d91c0e` |
| input zip | `3bb366dfa1f20bc99dd0fe798a93eb0b4cbd536ce801f3688c1276858062685e` |
| frozen G4 driver | `c1eeadee77406e44dbad3e09ded6b316dbf6666bacadecf3d610d8c4ef6eef6d` |
| result archive（93,567,480 bytes） | `4b5254e04c0852b96cb76c9e824a8bc60a56b809d7206555f4ba5df2f6d19e89` |

再現は保全したupload-manifestのSHA/sizeを照合し、source archiveを新しいdirectoryへ展開する。
同じ入力zipとdriver/oracleをrepo rootへ置き、指定runtimeのG4上で`CST_JOB_OUTPUT`を新規出力先へ設定して
`python g4-driver.py`を実行する。元の出力へ再実行しない。転送・実行・回収の実コマンドは
`launch.py`とsessionログに保存し、回収検証はローカル`verify.py`で行った。
公開集計だけからraw checkpointや実binaryを復元することはできない。
