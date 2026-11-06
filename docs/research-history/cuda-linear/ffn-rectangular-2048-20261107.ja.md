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
大きく、速度・メモリの実用上の優位は示していません。選択設定のheldout pilotは以下に記録し、
保存したseed29の観察も以下と[研究側の検証](https://github.com/takumiecd/cst-research/pull/29)に記録します。

## 選択設定のheldout pilot（R10）

対応する2048の両条件で、選択したAtom LR16を別途seed17で実行しました。
Dense・FFNなし・CSTをそれぞれ新規学習し、各976 update・999424 tokenまで完了しました。
共通の予定20M LR・データhorizonで、固定batchの繰り返しではありません。
4 blockのFFN全8射影をCSTにしています。通常ParameterのLRは同じで、16倍は振幅だけでなく
Atom group全体に適用します。上記と同じ固定R8 consumer・検証済みoperatorです。

| モデル幅 / Expansion | Dense DEV CE | CST DEV CE | FFNなし DEV CE | Dense / CST update ms | Dense / CST allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 4 | 3.28466 | 3.40110 | 3.71174 | 18.48 / 792.80 | 952.66 / 1187.04 |
| 128 / 16 | 3.91922 | 3.96799 | 4.35479 | 9.18 / 501.67 | 373.07 / 697.90 |

各条件で形状・Atom数・設定を厳密に合わせ、実際の両方向の目標標準偏差でnative検証を行い、
モデル全体のpreflightと上記の独立した終了checkpoint監査を通りました。
学習済みCST全8射影のY/dX/Atom VJP、有限なoptimizer/counter、CE再計算、serialize後の
次task stepを含みます。2048の4方向を、前の標準設定20-update smokeに加えて実タスク学習でも
確認できました。回収した両archiveとraw全22ファイルをmanifestとhash照合しました。

開発用1 seedでは両条件ともFFNなしより改善し、Denseとの差が残ります。
3 seed確認・収束・精度の漸近則を示す結果ではありません。同期した時間sampleは最初の10 updateを
除いた各966個です。reserved peakはN512でDense/CSTが1168/1492MiB、N128 expansion16で
460/788MiBでした。この逐次eager FP32観測でも速度・GPUメモリの優位はありません。
所有者の指示で未開始のseed43/71を止め、短期のフィールド別LR実験へ切り替えました。
主3幅のseed29は完了・監査・回収済みですが、予定した3seed確認は未完了です。
Attention置換は未着手です。

同じ研究証拠locator内のR10証拠 SHA256：

| 条件 | Native検証 | 学習済みcheckpoint監査 | 小証拠archive |
| --- | --- | --- | --- |
| N512 expansion4 | `c45dc18eaa0dd074d3f38e7c8488037ac48ae4d60b265d4f04c8e12903b58cf0` | `74ceefd58caa2dac38c5eaa77cf270d4c1176ad85d5d41b8c6cd3996ae22348a` | `e59e990cd947d5c3ac538e0cac28cf20b3b9ad0c9df1811fabcab18f4559c5c9` |
| N128 expansion16 | `78d37a661f5e01811c84f92fd2b76177f655d62f2595affa4502850906f7cc08` | `fbc172e1f38eef8d57efcee819391f5b4554d3f33cfda54165fb41feb1fa835f` | `0edb48531af793f9079d9911b9e9f3b06bbd24f321238f80b4f84c13a8bf732f` |

archive名は`width512-e4-lr16-r8-01-evidence.tar.gz`と
`width128-e16-lr16-r8-01-evidence.tar.gz`です。固定bootstrapの厳密なargvとsource manifestも
証拠locatorに保持し、再現では新しい出力先を使います。実タスクpilotを通っても、このoperatorの
形状・Atom数・精度のadmissionは拡大しません。

## 保存した幅512・seed29の観察（R11、所有者が停止した実験群）

同じ選択設定と予定20M期間で、Dense・FFNなし・CSTを新規に各4882更新・
4,999,168トークン学習しました。終端DEV CEは順に2.57642/2.83565/2.68898、
確認用CEは2.48887/2.74543/2.60475でした。学習済みCST全8射影のY/dX/全Atom VJP、
独立CE再計算、有限なoptimizer/counter、保存・復元後の次タスク更新の監査を通り、
復元後のパラメータ最大差は3条件とも0でした。回収したraw11ファイルはmanifestと一致しました。

Dense/CSTの更新中央値は18.57/822.90ms（最初の10回を除いた同期sample各4872個）、
allocatedは952.66/1187.04MiB、reservedは1168/1492MiBでした。この長めの1seed観察は
限定した経路の動作を裏づけますが、3seed再現性や速度・メモリ優位は示しません。
元のR11管理プロセスは意図的に停止しており、元のrunning journalは古い状態です。
最終owner-stop receiptを正とし、中止した条件を完了や数値的失敗とは扱いません。

- Native検証 SHA256：`3b3a80bf3c7dabf6fe1a4088dddf3f7e4c027eabee39fdb0746f79dc0a3fc309`。
- 学習済み監査 SHA256：`9764902e1b0acdee490f2a63f728a5369d6aeee2c17bf0a3604d08e324f78ec7`。
- Archive `width512-e4-lr16-seed29-r8-confirm-01-evidence.tar.gz` SHA256：
  `a1973aac3b176ba1e63a66373cd67a7305f9c9813335b7fbd09afb4709dbca74`。
- 最終owner-stop receipt SHA256：
  `59d7a4e28c60d84b8a92bdd1b442abbe74c81daeb8267fcaf8b51ff8674b7f27`。

Consumer70e02d8/operator2b587は変更していません。全曲線と停止群専用の観察レポートは
研究repoの`docs/ffn-feasibility-r11-seed29-20261107.md`と日本語版に記録しています。

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
