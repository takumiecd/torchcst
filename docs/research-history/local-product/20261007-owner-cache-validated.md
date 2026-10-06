# 2026-10-07: exact cached permutation validation

Branch `kernel/order-validation-cache`, based on PR #39 `8c89cc88`.

N128 rho3では旧キーcacheが52 refresh中49/50回、方向全体を再sortした。
canonical keyが一つ変わっただけでも再sortする条件を緩める。

## 正確さの条件

cached IDsは初期identityとsort更新により常にcanonical IDsのpermutation。
現在の支持から各IDのキー `k(a) = (bucket(a)*(max(K,N)+1)+support_lo(a))*C+a`
を計算する。canonical IDによるtie-breakを含むのでキーは一意。
cached順の隣接キーに逆転がなければ、full sortとまったく同じ順序になる。
一つでも逆転があればfull keysをsortし、ID permutationを更新する。
近似・量子化・古い支持の流用はしない。

bucket membershipは順序を変えずに変化し得るためhistogram/offsetsは毎回更新。
係数・正規化・支持終端・owner envelopeも毎回更新し、forwardごとの独立snapshotを
backwardへ保存する。位置勾配・singleton・可変幅optimizerの数学は維持する。
Hのglobal保存は増やさない。物理cache residency/DRAM削減の測定証拠はまだない。

新routesはvalidated copy8/range8の2つ。固定output16、batch16行、atom chunk32、
owner split4を維持。canonical IDsはC<=32768ならint16、他はint32。
key/offset cache buffersは空。N128 A819でresident tensor budget3324 bytes
(2*C*2+2*3*8)だが、allocated GPUピークの実測値ではない。

## Host checkpoint 04:29 JST

changed-file ruff、5 CPU declaration/boundary tests、8 prepare/check成功。
全CPU suite 871 passed / 637 skipped、22.66s。GPU skipはGPU成功を意味しない。
wheel/sdist isolated build成功。最初の--no-isolationはhostにhatchlingがなく失敗し、
既存build workflowのisolated environmentで再実行した。
2 routesの20 GPU testsは独立FP64 gradients、slices、B1/32/64、20 captured
live optimizer updates/moments/steps、old backward snapshotsに加え、
キーが変わるが順序は変わらない場合・bucketのlive更新・片方向だけ逆転する場合を検証する。
GPU正しさ・性能は未検証。初期rho>1のN64/N128 rho1.25/3/8/mixedをcopy8と
compact keycacheとdenseで同時比較する。21 samples/execution、IEEE FP32/TF32off。
raw scripts/resultsはignored evidence/owner-cache-validated-20261007に保存する。

source `e0a0caf2362e831ecc7204d080b9fe5a57e9bd5f`、draft PR #40。
L4 check `l4job-1d0b7a0684f04bd2946f1d038430f274`、N64 `l4job-28b05068b1944d8cbaafbf80cdf1db71`、N128 `l4job-7f8cc4e12cab427d831e9fd4acc1d6d6` queued。最初のsubmit `l4job-61dd72df4d2d4f8dad21ce7a3de6412c` はsourceラベル誤記のため実行前にcancel、数値なし。数学とsource/result hashの整合を正式jobsで確認する。

GPU実行前に、他のinactive recordsの両方向loをclipped終端へ固定し、atom1のactivationが必ず逆転を作るようtest metadataを決定的にした。runtime不変。旧3 queued jobsはcancelし、source `321ef823c61b14a226842cf755285327119f90b4`でcheck `l4job-27edc1ed3886455aa1926156c720b24e`、N64 `l4job-2b8aa68e3bbb4f1d90ee44ec72864fdf`、N128 `l4job-16ee66bd4e4a46389de93ea3cb3d3bf4`へ置き換えた。CPU5 passed /20 skippedを再確認。

## 04:40 JST: GPU25 tests成功

`l4job-27edc1ed3886455aa1926156c720b24e`:25 passed (20 GPU +5 CPU)、74.17s。actual NVIDIA L4 / torch2.11.0+cu130 / CUDA13 / Triton3.6、source/result archivesと197 runtime hashes一致。キー変更なしだけでなく、キーは変わるが順序が変わらない場合、support endとbucket offsetのlive更新、両方向と片方向のみの逆転、slices/位置勾配/旧backward/20captured optimizer更新も通過した。独立snapshot保持を確認。screen実行中。

## N64 screen: 確認できた探索削減と費用

1 execution /21 samples、完全step us。source/result/197 runtime files、16 full-shape FP64比較、case内初期Parameter/input bytes一致。

|rho|copy8|compact keycache|validated copy8|validated range8|dense|
|---|---:|---:|---:|---:|---:|
|1_25|55.27|54.44|54.86|54.40|38.98|
|3|59.39|58.87|59.53|59.27|38.92|
|8|56.33|55.75|55.97|55.44|38.79|
|mixed|62.86|63.38|62.60|62.41|39.08|

mixed forwardは52 refresh中22 sorts→15 sorts、dX21→19。rho3 forward26→26/dX27→25、rho8は27/26で変わらない。単にkeyが変わっただけのsortを一部省けたが、常に省けるわけではない。
allocated peak118784(compact keycache)→117248(ID-only)、base115712より1536高い。reserved6291456 bytes。validated copy8のlayout eventはmetadata間接load/順序検査/histogram更新の費用を含み、compactより約1us高い場合がある。range8はN64で少し良いが、param2/atom16単独の方が完全stepでは速い。次はcanonical順の連続metadata load＋キーだけgatherで検査するPR41と比較する。N128 screenは進行中。

N64 screen4 artifactsをDB保存、byte-identical export/idempotent再取込成功。
