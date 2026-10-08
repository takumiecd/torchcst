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
8192は追加段階として宣言済み、性能未測定。測定予算は独立jobごと1300秒、
各rhoのrunner600秒。ゲート失敗なら測定しない。性能による再試行・予算変更・
悪いCaseの除外は行わない。採用する改善は別jobの逆順比較で確認する。

CPU全体は1210 passed、2101 skipped、25.46s（追加adapterテスト前）。
GPU結果・性能は現時点で未検証。raw evidenceはignored
benchmarks/cuda/linear/evidence/profile-product-square-strip-20261008/へ保存する。
