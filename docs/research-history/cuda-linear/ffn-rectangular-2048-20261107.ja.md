# 範囲を限定した2048幅FFNの検証

[English (canonical)](ffn-rectangular-2048-20261107.md) | 日本語

明示的な`GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048`で、従来未対応だった係数固定の
FFN条件を扱います。既存native演算を再利用し、元のPlan guard・recipe field・dispatcher
既定値は維持します。対応は512↔2048/A8192と128↔2048/A4096、FP32、最大256 token行、
Polar M8/base1/merge2/spacing1、radius bounds[.5,50]だけで、eager-onlyの範囲です。

## 実行した確認

R3拡大検証はRTX4090のnative全12条件を通りました。2048-siteの4方向、非連続B2T128、
chunk化したCPU FP64参照に対するY/dX/全Atom座標VJP、独立polynomial quadrature、
小さいcontrolled optimizer/modelのserialize後のresumeを含みます。

R8のrecipe検証では別途、512→2048の目標std.02、2048→512のstd.02/sqrt(8)、各8192 Atom、
実際のq0・振幅上限1・w_c.3・birth4を確認しました。両native方向、quadrature、普通の
ParameterとAtomが混在したoptimizer参照比較、小型モデルresumeを通りました。その後、
実際の4 block・幅512 causalモデルをDense/CSTで各20固定batch updateしました。CSTでは
FFN全8射影を置換し、attention/embedding/outputはDenseです。microbatch2×context128、
accumulation4、FP32、TF32off、seed17、AdamW基本LR3e-4、betas(.9,.95)、decay0、
global clip1です。LR・データの予定horizonは20M tokenで、8つの学習windowを繰り返します。
heldoutでの有効性を示す実験ではありません。

終了後の別監査で、checkpoint/curve/summaryのhash、optimizer状態・counterの有限性、
厳密なmodel load/cache generation、学習済みCST全8射影のB1T8 VJP、CEの独立再計算を
確認しました。使い捨てのコピー上で2 task updateを行い、serialize後の次の損失・Parameter・
moment・counterも確認し、両条件でParameterの最大差は0でした。元の保存runを再開したり
書き戻したりはしていません。

## 短いcomplete-stepの観測

| 幅512 FFN | update中央値 ms | Peak allocated MiB | Peak reserved MiB |
| --- | ---: | ---: | ---: |
| Dense | 18.40 | 952.66 | 1168 |
| CST標準 | 726.25 | 1187.04 | 1492 |

最初の10 updateを時間集計から除いた、同期したcomplete updateの10 sampleです。1つのjobで
Dense→CSTの順に実行し、振幅サンプリングはありません。20 updateすべてが予定200-updateの
LR warmup内です。データ転送と4 microbatchを測り、
別のprobe・評価は除外します。短い機能確認の観測で、独立反復やスケーリングの主測定では
ありません。FFN Parameter数はCST272384、Dense8398848ですが、実測GPU peakはCSTの方が
大きく、速度・メモリの実用上の優位は示していません。長いheldout学習、選択設定での幅・
縦横比の挙動、3 seed確認は[研究側の検証](https://github.com/takumiecd/cst-research/pull/29)に残ります。

## 来歴と再現

- Operator：`2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440`。
- R3拡大consumer：`036d0c5beb4a6f675ce1dbeafd4224e6c270f342`。
- R3 native receipt SHA256：`419ab7750182bca2302d7152c6fc01c1a961efe69f19bf5bb22355f3aa299ce6`。
- R8 consumer：`70e02d864e661d3a87cd8aea322b49ae3f4a1f8f`。
- R8 native receipt SHA256：`b40ac9ffa47814408eb22d61f65c2a0313f0420632566609f168c269131785d6`。
- R8終了receipt SHA256：`13e60f4b96786fb300c074891bbc02a1e6051c7fc2e934d4279c3f058b4348b0`。
- R8学習済み状態監査 SHA256：`08080c751cf5c2c80d57663daa4031bcb7fd231a401187ace7d16161511a9593`。
- 回収した小証拠archive SHA256：`932013048726fdc91c94f9c158ad39f96feee3884cb20511e0fad24327b21c37`。

実環境はNVIDIA GeForce RTX4090、24GB級、CC8.9、Torch2.11.0+cu126です。研究側の
`$CST_RESEARCH_EVIDENCE_ROOT/ffn-feasibility-20261106`から
`width512-reference-r8-01-evidence.tar.gz`、raw script、source manifestを辿ります。
bundle内のraw全7ファイルは回収したmanifestとhash照合済みです。大きなcheckpointは
hashとともにGPU hostに残します。検証したoperatorの既存CPU一式は2086 passed・2483 skipped
で、skipしたGPU条件は検証済みに数えません。

固定したソースと新しい出力先を使います。研究consumerで
`ffn_qualification --cohort recipe --width 512 --expansion 4 --coefficient 8`を実行し、
`ffn_run --kind dense`と`--kind cst`を、どちらも`--width 512 --tokens 20480
--schedule-tokens 20000000 --repeat-fixed-batches`で実行します。新しい実qualification・
data manifestとそのSHA256を引数で渡します。記録したbootstrap・audit scriptには環境設定、
正確なargv、PID、ソース確認を保持します。別consumer・scalar設定・GPU・Atom数をこの再現で
検証済みとは扱いません。
