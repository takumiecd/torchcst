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
