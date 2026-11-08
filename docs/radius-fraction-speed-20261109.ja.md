# 半径の区間内割合のAPIと短期FFN速度測定

[English (canonical)](radius-fraction-speed-20261109.md) | 日本語

現在の名前は`radius_fraction_init`、初期化の引数・reportの`radius_fraction`、
`field_lrs={"radius_fraction": ...}`です。所有者の指示により、merge前に公開q引数・
report項目・別名を削除しました。数式の記号としてqを使うことはあります。
過去の凍結source・証拠は記録どおり保全し、checkpointの読み替えは行いません。

## 状態を固定した測定

RTX6000 Ada Generation48GB、Torch2.9.1+cu128、CUDA12.8、FP32・TF32なし、
8192 Atom、256行、幅512・展開率4、tau0.03、初期割合0、半径[0.5,50]です。
各条件で初期の物理行・重み・入力のハッシュを揃えました。既存の
`benchmarks.cuda.linear.radius_cell_mass.train`でzero_grad、x.grad消去、
MSEのforward/backward、AdamWを実行します。step前後を同期し、setup時間は除外します。
各条件は別processで5 warmup・21測定です。順番を逆転した2組を平均せず別々に示します。

| 方向 / 順序 | Dense ms | physical本体 ms | 新方式 ms | 新方式＋公開CST更新則 ms |
| --- | ---: | ---: | ---: | ---: |
| 512->2048 / 正順 | 0.218 | 3.274 | 3.966 | 3.983 |
| 512->2048 / 逆順 | 0.310 | 3.277 | 3.980 | 3.975 |
| 2048->512 / 正順 | 0.210 | 3.259 | 3.975 | 3.959 |
| 2048->512 / 逆順 | 0.202 | 3.272 | 3.972 | 3.998 |

同じ通常のAdamWでは、新方式はphysical本体に対し21.1〜22.0%の追加コストでした。
Python写像、勾配の連鎖、内側のdispatchを含みます。単一のボトルネックを特定した結果でも、
半径をdetachする旧Polarの勾配・更新規則との直接比較でもありません。
この小型条件ではDenseの方が速い結果です。

| 方向 | Dense allocated MiB | physical allocated MiB | 新方式 allocated MiB |
| --- | ---: | ---: | ---: |
| 512->2048 | 45.25 | 66.94 | 69.50 |
| 2048->512 | 45.25 | 65.44 | 68.00 |

ピークにはwarmup・Adam状態を含み、setupは除きます。reservedピークはDense62〜66MiB、
physical78〜80MiB、新方式80〜82MiBでした。公開CST更新則を使うdown側は67.88MiB、
up側は69.50MiBです。新方式の21 step中の自然なallocated変動幅は1.88MiBでした。
長期のメモリリーク検査やGPU process全体の使用量ではありません。

## 範囲と証拠

本体比較の3条件はLR0で、勾配・モーメント・counterを計算し、作業負荷を固定しています。
公開CST条件は正でsub-ULPのLR1e-20を使い、射影・transportも通します。
測定後の物理行がbyte一致で変わらないことを検査しました。
固定状態のeager Linear単体の測定であり、LM収束、通常の非ゼロLRでの推移、Graph capture、
大規模幅の速度ではありません。この測定を根拠にruntimeの最適化は採用していません。

測定sourceは`2e41c3e8a50012cd23e53821bd4a4fbc956a9d29`、archiveは
`c618b50e7d67f0510c8c270d8e63612eb34587f677dd5f7bc58a32831c8d491e`です。
このsourceには改名と一時的な初期化の別名が含まれ、その後で別名を削除しました。
削除対象は初期化API・report・呼び出し側で、測定したtrain本体や境界写像・CUDA実行は不変です。
初期化の数式も変わりません。改名後のGPU検査41件が成功してから測定しました。
互換削除後は対象CPU90件とローカルCPU一式2233件が成功し、2613件はskipでした
（native opt-inは無効）。旧q引数は明示的に拒否するテストを置いています。

ジョブ`envelope-speed-20261109-01`で16個の測定workerと1個の検査processを実行しました。
回収後に37ファイル・source1310ファイルをローカルGit archiveと照合し、
全条件の物理値と入力のハッシュ一致、記録した全18 processの終了を確認しました。
GPUの事後確認は0%・39MiBです。証拠archive SHA256は
`5345513619a6c3a2e8904712cf20dbfd11ffe222c96aeeae8df996b3caf9226a`です。
[順序ごとの結果](radius-fraction-speed-20261109.json)を保存しています。
rawログ・source・buildはjob IDに対応するignoredな所有者用領域に保全しています。

固定sourceと対象環境で再現します。

```bash
python -m benchmarks.training.radius_envelope_probe \
  --arm envelope --ni 512 --no 2048 --output NEW_RESULT.json
```

他の条件はdense、physical、envelope_cstです。driverに全順序と各コマンドの実行記録があります。
setupには共通初期化と、物理行から作った重みによる初期出力照合を含みます。
このsourceはDraft PRに置く実験用であり、mainやdefaultへの統合は行っていません。
