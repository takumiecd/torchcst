# N入力・N出力のStrip profile product

2026-10-08、所有者の指示でStripの大型化評価をN×Nに拡張する。
既存Strip測定の64出力×N入力から大型出力の速度を推定しない。

単一Euclidean Strip chart、shape=(N,N)、tile_shape=(N,64)、入力axis=1、
pitch68。Triweightのprofile product、Polar、全chart L2・floor1e-6一回。
B32、FP32 IEEE/no TF32、seed41、A=floor(N²/20)、AdamW+production Polar update。
初期rho3/8はwidth1..16内で学習する。N1024/2048/8192を宣言する。
分割するのは入力軸のみで、出力軸に独立Stripを追加する意味ではない。

既存CUDA Stripは出力128まで。専用研究Algorithmで出力8192まで明示対応し、
古いPlanの支持条件を変えない。executor/kernelと数学的契約は共通。
新fixtureはpolar_profile_product_square_strip、adapter revision7。
旧fixtureの初期化・64出力・adapter5、ordinary Productのadapter6を維持する。

FP64 oracleは全factor siteを列挙しatomのみchunkする。
独立の全行列L2 oracleとfloor・空支持を含めてCPU照合する。
GPUではY/dX/全atom勾配、20 captured更新、moments、可変幅、retained snapshot、
可変pitch、部分tile、非整列出力を検証する。
速度は既存runnerの全学習step21サンプル、capture/replay allocated/reserved peak。
prepared、split1/4/8/16、Torch contraction、denseを同じshapeで残す。

宣言・CPUチェックとGPUゲートを先に実行し、その後N1024/2048のrho3/8を測定する。
8192は追加段階として宣言済み、性能未測定。測定予算は各独立jobでゲートを含め1550秒、ゲート300秒、
各rhoのrunner600秒。ゲート失敗なら測定しない。性能による再試行・予算変更・
悪いCaseの除外は行わない。採用する改善は別jobの逆順比較で確認する。

CPU全体は1210 passed、2101 skipped、25.46s（追加adapterテスト前）。
変更はa5d40b5ae25d657bdf71fedff8a2852c4ea7512aで固定した。
追加adapter/dispatch/submissionを含むCPU対象は112 passed/32 skipped、Ruffと
wheel/sdist buildも成功。draft PR #72。

N1024とN2048の独立jobを同一sourceで投入した。各jobで45ケースのGPU/CPU
ゲートを先に実行するため、ゲート失敗時はそのjobで性能測定しない。
- N1024: l4job-ce28c141a62845cb8cb1af6265acb26c
- N2048: l4job-8fc462f00ea64dd28a6ca28765d6f2d2

どちらもrho3/8、prepared・split1/4/8/16・Torch・denseを残す。
初回は両jobとも9 failed/36 passedでゲートを停止し、性能測定は開始しなかった。
8件は共有テストのglobal対照にもStrip専用Planを渡していたfixtureの誤り、
1件はremote環境の第三者tests packageとテストhelperのimport衝突。
global対照は既存large Product Planへ対応させ、全45ケースを保持する。
helperはpytestのtests検索パスからimportする。kernel・oracle・許容誤差・測定条件は変更しない。
初回の失敗ログ・source・receiptは共有poolに保全する。修正版は別jobとして明示投入し、
各1550秒、ゲート300秒、各rho600秒、全route/Caseの既定予算を維持する。
raw evidenceはignored benchmarks/cuda/linear/evidence/profile-product-square-strip-20261008/
と共有poolの各jobディレクトリに保存する。

修正版605ccf2597aa692381904bb1c4a0c10cec5682e8の一次測定は両jobとも成功。
各jobで45 passed/zero skipped、各rhoで全6 CST routeの独立FP64 oracleと
optimizer検証、その後denseを含む全7 routeの21完全stepサンプルを保持した。

- N1024:l4job-ad8d3487f61c403ab37ecb3f1db88ab1
- N2048:l4job-2e72bc949dad425b9aec4c28903668f1

独立確認は同じsource・予算・全Case/routeでCST候補順のみ反転する。
denseは既存runnerの仕様で末尾のまま。rho3/8、B32、seed41、全chart正規化、
可変幅、21サンプル、同じAdamW/Polar更新を維持する。

- N1024逆順:l4job-94cd72e0a32f4991ac0d29dc40cad679
- N2048逆順:l4job-87bd7ad859f243829a35fa230c39cb19

mainのTorus referenceと空間順序の検証記録を研究branchへ取り込んだ後の
CPU全体は1255 passed/2137 skipped、24.86s。変更したPythonファイルのRuffはPASS。
一次測定sourceからCUDA kernel・benchmark・対象テストの内容は変わっていない。
逆順確認は両jobとも成功し、各45ケースと全routeの独立oracle/optimizerゲートを通過した。

## 完全stepの結果と採否

NVIDIA L4、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、FP32 IEEE、B32、
約5% atoms、学習で変わる幅。下表は一次/独立逆順の中央値ms。
各値は別jobの21同期サンプルであり、21独立実験とは数えない。
全route/誤差/時間サンプル/allocated/reserved/入力hash/source集合は
同名-summary.jsonに保持する。Case/初期atom/入力/target/optimizerは両順で一致。

| N | rho | split1 | split8 | Torch g8/p8 | dense |
|---|---|---|---|---|---|
| 1024 | 3 | 0.317362 / 0.316431 | 0.284640 / 0.284837 | 0.248118 / 0.247486 | 0.082396 / 0.082278 |
| 1024 | 8 | 0.459469 / 0.456707 | 0.424815 / 0.423211 | 0.387144 / 0.388838 | 0.081992 / 0.082223 |
| 2048 | 3 | 1.156789 / 1.165926 | 1.099792 / 1.096959 | 0.972311 / 0.977640 | 0.590531 / 0.592024 |
| 2048 | 8 | 1.801603 / 1.797618 | 1.759602 / 1.719056 | 1.622401 / 1.645099 | 0.589470 / 0.589922 |

Split8は全Caseの両順でsplit1より速い。N1024は約7.3..10.3%、
N2048は約2.3..5.9%の改善。N2048/rho3のsplit4とsplit8は0.4%未満の差で
順序によって逆転するため、細かな順位を断定しない。
Torch contractionは全Case・両順でCST最速、denseは全CSTより速い。

Capture/replayを含むpeak allocated/reserved bytesは両順・rho3/8で同じ。

| N | route | allocated | reserved |
|---|---|---|---|
| 1024 | split1 | 16689664 | 56623104 |
| 1024 | split8 | 16689664 | 58720256 |
| 1024 | torch-g8-p8 | 50768384 | 115343360 |
| 1024 | dense | 51382784 | 111149056 |
| 2048 | split1 | 66494976 | 163577856 |
| 2048 | split8 | 66494976 | 163577856 |
| 2048 | torch-g8-p8 | 99736064 | 205520896 |
| 2048 | dense | 102238720 | 148897792 |

Native split系のallocatedはN1024でdenseの32.5%、N2048で65.0%。
ただしN2048のreservedはdenseより9.9%大きい。全GPU process使用量は未測定。
Torchの速度優位とnativeのallocated優位を分けて研究基準として残す。

採否:単一chartのN入力/N出力を扱う専用研究capabilityとfixtureを統合する。
低allocatedの研究比較にはsplit8を、CSTの時間基準にはTorch contractionを使う。
公開Registry/CSTLinearの自動選択は変更しない。旧Strip Planの出力128上限と
旧64出力fixtureも維持する。8192は宣言・capabilityのCPU検査のみで性能未測定。
prepared contractionは全Caseで遅く、default候補にはしないが対照と証拠は保持する。

次の性能調査は、既存の正確な重み組み立て＋GEMMを基準に準備・組み立て・
atom VJPの別Graph診断を行う。完全stepにおける各部分の時間は今回未計装で、
どこが支配的かは未確定。空間順序の不採用実験とは独立の変更である。

## 証拠と再現

初回の失敗を含むraw source/results/logs/receiptは共有poolのjobs下へ保全した。
以下の4成功jobはarchive hashと全manifest file hashを再照合済み。

| job | frozen source SHA256 | result archive SHA256 |
|---|---|---|
| l4job-ad8d3487f61c403ab37ecb3f1db88ab1 | 6366baad89bb46053ec6775a6acba4aee5e832ff275812cc258d22ba08b84c95 | b1bfcc4285080049a4a14ea8176e50537571e1d8f643c49ab2249ea231a02ed3 |
| l4job-2e72bc949dad425b9aec4c28903668f1 | ed257d525471c1a7d0ffc745052e687f6bdc84e7f15c83853b16ce93e3250d5e | 1d6c4e860fcf5edd070f77e890108a0db4fe38a5c38029be205283c1125a3e5b |
| l4job-94cd72e0a32f4991ac0d29dc40cad679 | 1dc5e7eb7c173536363f3071a7971cc6bfb2011811f22fe28d4da108c0ef62b1 | a5a4852a728bfdaaccae6fd3230568807e73bdb0fa2dd8427469cb4a3046492d |
| l4job-87bd7ad859f243829a35fa230c39cb19 | ed8ff51ed62f6bc6a6cd23d6c4d12e339a7dc5bb4400615199ff06b765e9c675 | d1925b49fbcc3bd1794aeaf9f18696c89524f24e88c1f0a596861cb6a125a37b |

保全先:~/.local/state/colab-l4-pool/jobs/JOB_ID/
(spec.json,source.tar.gz,receipt.json,results.tar.gz,results/manifest.json,artifacts/)。
集約済み結果は同名-summary.json、収集driverと確認scriptはignored
benchmarks/cuda/linear/evidence/profile-product-square-strip-20261008/に保持する。
測定runtime sourceは605ccf2597aa692381904bb1c4a0c10cec5682e8。
統合branchへのmain取り込みでCUDA kernel・benchmark・対象テストは変更されていない。

再現は同じruntime commitのnamed kernel branchで、CUDA/Tritonを導入し、
pytest -q tests/test_square_strip_profile_product.pyを先に実行する。
その後、既存runnerで全case/routeを測定する:

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-profile-product-square-strip.json \
  --case benchmarks/cuda/linear/cases/profile-product-square-strip-1024-rho3.json \
  --polar-update fused --source-commit 605ccf2597aa692381904bb1c4a0c10cec5682e8 \
  --output output/square-strip-1024-rho3.json
python -m benchmarks.submissions check output/square-strip-1024-rho3.json
```

N2048、rho8は対応Caseへ置換する。逆順はCase JSONのplans配列だけを反転し、
同じ既存runnerを使う。baseline、dense、widths、optimizer、seedは変更しない。
独立runはsource/driverを別jobとして固定し、上記の元予算内で完了した。
