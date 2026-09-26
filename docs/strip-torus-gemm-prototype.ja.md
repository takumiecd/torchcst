# Strip + Torus Linear の実行設計

`CSTLinear(..., backend="tiled")` は PyTorch 基準実装、`backend="triton"` は
重み生成と GEMM を融合する NVIDIA GPU 実装である。両方とも入力勾配と
アトム勾配を計算でき、`CSTParameterAdam` と `LinearJGAtomGrad` で使える。

## 責務と backend

| 場所 | 責務 |
| --- | --- |
| `nn/linear.py` | モデル・Parameter の所有、公開 API、AtomGrad との接続 |
| `nn/_backends/__init__.py` | backend の検証・選択・ディスパッチ |
| `nn/_backends/_torch.py` | materialized / factored / tiled の PyTorch 実行 |
| `nn/_backends/_preparation.py` | 固定設定の検証・座標計画と、毎回新しく行うアトム準備 |
| `nn/_backends/_triton.py` | autograd 接続。Triton は実行時だけ import |
| `nn/_backends/_triton_kernels.py` | forward、入力勾配、アトム勾配の GPU カーネル |
| `nn/_layout.py` | 不透明なアトム行の配置と repack 計画 |
| `nn/_strip_torus.py` | 所属判定、支持範囲オラクル、タイルの基準計算 |
| `DirectAmpWidth.tile_parameters` | 振幅 clamp、座標、帯域幅の stop-gradient の共通定義 |

`auto` は従来の選択を維持する。単一チャートは materialized、二チャートは
factor と dense のサイズから選ぶ。`layer.backend` の代入時に適用条件を検査する。
backend は checkpoint のモデル定義に含めず、同じパラメーターを切り替えて使う。

## 対象とデータ契約

- 単一の `StripChart` と `TorusGeometry`。出力行軸 `0` がステーションになる。
- 各 Chart タイルは入力列軸の全体を含む。`validate_support(sigma_max)` が成立する。
- `DirectAmpWidth` と正規化しないコンパクトプロファイル。
- Triton は `Biweight` / `Triweight` / `WendlandC2` / `Triangle` の標準実装に対応する。
- `offsets[G+1]` と `order[A]` がステーション順のアトム配置を記述する。

ステーションは円環 `0, 1, ..., G-1, 0` をなす。タイル `g` は格納ブロック
`g-1`, `g`, `g+1` を読む。ステーション数が 1 または 2 の場合は重複を除く。

所属判定は、アトムの円周座標から各ステーション内の最も近い行を求め、
その中で円周上の距離が最小のステーションを選ぶ。部分的な最終タイルと
円環の継ぎ目を含め、計算量と一時領域は `O(A G)`。

この判定が正しいのは、全ステーションで列の断面が共通だからである。
任意の列で、アトム中心とサイトの弦距離は、行の角度が中心角に最も近い
ときに最小になる。どこかのサイトに寄与するアトムは、この所属先にも
寄与する。`validate_support` の保証から、残る寄与先は隣接ステーションに
限られる。`support_mask` は全サイトを調べる独立な検証用オラクルとして残す。

## 再配置

1. 更新後の `destination[a]` を数え、累積和から新しい `offsets` を確定する。
2. 元の位置を新境界で見た所属を `new_block_at_old_slot[a]` とし、
   `adjusted_move[a] = new_block_at_old_slot[a] - destination[a]` を求める。
3. 行き先別の安定ソートで `order` を作り、衝突なくアトム行を配置する。

`destination` は任意のステーションを指定できる。移動量を `−1/0/+1` に
制限する必要はない。`max_arc_step` は学習時の幾何的な更新制約であり、
repack の正しさとは別の設定である。

現段階では forward ごとに所属と安定ソートを計算し、`index_select` による
一時配置を作る。モデルの Parameter や optimizer の moment は物理的に
並べ替えない。独立した `plan_repack` は将来の永続配置の基準実装である。

## Triton の計算

Chart の行を円周方向 `circle[N,2]`、列を断面 `section[K,D-1]` に分解する。
各サイトの埋込み座標は

```text
site[n,k] = concat(circle[n] * section[k,0], section[k,1:])
```

で生成する。全サイト分の `N×K×D` 座標表は保存しない。アトムの中心は
PyTorch で ambient 座標へ変換するため、intrinsic / ambient 両表現の勾配は
既存の Geometry の微分を通る。帯域幅からのタスク損失勾配は既存の定義どおり
遮断し、振幅と中心へ微分を戻す。

forward は各出力領域を一つのプログラムが担当し、入力列を小分けにして
重み生成と積の累積を行う。入力勾配は `dX = dY W` を同様に処理する。
アトム勾配は小さな `dW_tile = dY_tile.T @ X_tile` を作り、プロファイルの
微分と縮約してアトムごとに加算する。完全な `W` / `dW` は生成しない。

最初の実装は `BM=BN=BK=16`、アトムチャンク `BA=8`、4 warps。
これらは内部設定で、Chart の `tile_shape` を変更しない。浮動小数点演算は
float32、dot は `input_precision="ieee"`。autotune / Tensor Core 用の
精度緩和 / mixed precision はまだ適用していない。

アトム勾配は浮動小数点の atomic 加算を使う。決定的アルゴリズムモードで
その勾配を要求すると明示的にエラーになる。Triton 経路は一次微分専用で、
二次微分は PyTorch 経路を使う。初回の設定検証には CPU 同期が必要なため、
設定検証と固定座標の作成を warmup で済ませる。以降は、形状と設定を
固定した forward 全体を CUDA Graph に capture できる。入力とアトム値は
replay 間で更新できる。Chart・Kernel の設定や形状を変更した場合は再 capture
する。学習全体の capture と `torch.compile` 対応は保証していない。

固定座標と所属判定用の区間は実行計画として再利用する。buffer の identity・
version・device・dtype、および設定メタデータを検査し、変更時には再検証・再作成
する。checkpoint の読込みや `.to()` も対象となる。アトム値、並べ替え順序、
autograd graph は保存せず、毎回計算する。初回が inference mode でも固定座標は
通常の Tensor として作り、後の学習に使えるようにする。version のない
inference Tensor を設定に含むモデルでは計画をキャッシュしない。

## 検証と計測

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
# NVIDIA GPU + Triton 環境
python -m pytest -q tests/test_triton_linear.py
python -m prototypes.benchmark_triton_linear --output benchmark.json
```

GPU テストは密な計算との出力・入力勾配・アトム勾配の照合、両中心表現、
4 プロファイル、固定/可変帯域幅、振幅 clamp、円環の継ぎ目、端数タイル、
少数ステーション、空ブロック・空バッチ、optimizer 更新と AtomGrad 接続を
対象とする。

ベンチマークは準備処理、backend ごとの forward / forward+backward を分けて
測定する。加えて、同じ Triton の重み生成を使い、生成した重みを HBM に書いて
PyTorch の GEMM に渡す比較用経路も測る。融合版ではバッチ方向の担当ごとに
重み生成が繰り返されるため、融合が常に速いとは仮定しない。初回 JIT を除外し、
各ケースを複数回測定した中央値を JSON に保存する。

## 2026-09-26 A100 での結果：初版 `306d071`

NVIDIA A100 80GB PCIe の MIG 3g.40gb 区画、PyTorch 2.6.0+cu126、Triton 3.2.0。
リポジトリ全体は **371 passed**（skip なし）。うち Triton 関連は CPU 上の契約検査を含む 29 件。
ローカル CPU は 342 passed / 28 skipped。ruff と差分空白検査も通過。

以下は `[out,in]=[64,128]`、Chart タイル `[16,128]` の小規模測定。
全体は準備を含む壁時計時間の10回中央値、GPU欄は準備済みデータを使い
数値カーネルだけを CUDA Graph で反復して測った中央値で、同じ種類の時間ではない。

| 入力行 M | アトム数 | 準備 ms | PyTorch dense forward ms | Triton forward ms | Triton forward+backward ms | 融合 GPU ms | 分離 GPU ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 24.95 | 62.56 | 25.02 | 25.19 | 0.130 | 0.027 |
| 128 | 64 | 24.93 | 62.50 | 24.99 | 25.30 | 0.119 | 0.035 |
| 16 | 256 | 25.01 | 213.28 | 25.27 | 25.60 | 0.457 | 0.072 |
| 128 | 256 | 25.05 | 213.29 | 25.18 | 26.82 | 0.443 | 0.078 |

この4ケースの出力最大絶対差は `3.28e-7` 以下。初版では準備処理が全体時間の大半を占めた。
また、同じ重み生成を使った分離実行の方が GPU 時間では速い。融合版の
並列度・重み再生成・命令配置の改善余地を示す結果であり、広い形状や GPU での
速度優位を主張するものではない。この結果から、最初の改善対象を準備段階の
同期と配置更新にした。

全体検証で見つかった `device="cuda"` と `cuda:0` の初期化検査の不一致、
および `exp(log(bound))` の丸めで帯域幅が境界をわずかに越える問題も修正した。

再現用 snapshot は `srv11/cst-lab/torchcst-triton-20260926-impl-r6/`。
未コミットの実装を含むため、基準コミットと全109ファイルの SHA256 を
`manifest.json` に保存し、実行前に照合した。結果は `all-tests.xml`、
`all-tests.log`、`benchmark.json`、`benchmark.log`。ローカルの取得先は
`output/triton-a100-20260926/`。

## 同日の改善：固定設定の再利用と準備時の同期削減

設定検証と固定座標を再利用し、中心座標の重複 decode を除いた。所属数は
既知のステーション数の配列へ `scatter_add_` で数え、`bincount` の出力サイズ
決定に伴う同期を避ける。数値 GPU カーネル、FP32 の計算精度、Chart の定義は
変更していない。

同じ A100 区画・ソフトウェア・4形状で、warmup 後の10回中央値を比較した。
時間は準備と同期を含む壁時計時間。GPU カーネル単体の速度向上を表すものではない。

| 入力行 M | アトム数 | forward 初版 → 改善後 ms | forward+backward 初版 → 改善後 ms |
| ---: | ---: | ---: | ---: |
| 16 | 64 | 25.02 → 2.75 | 25.19 → 5.41 |
| 128 | 64 | 24.99 → 2.75 | 25.30 → 7.89 |
| 16 | 256 | 25.27 → 2.94 | 25.60 → 8.19 |
| 128 | 256 | 25.18 → 2.94 | 26.82 → 8.23 |

この範囲で forward は約8.6–9.1倍、forward+backward は約3.1–4.7倍になった。
出力最大絶対差は引き続き `3.28e-7` 以下。GPU カーネル単体では融合版が
0.126–0.458ms、分離版が0.027–0.078msであり、融合版自体には改善余地が残る。
ホストと同期の遅延も含む小規模な測定なので、大きな形状での倍率は別途測る。

A100 で全 **382 passed**（skip なし）、ローカル CPU で352 passed / 29 skipped。
追加検証には設定変更・checkpoint 読込み・dtype 変更時の無効化、inference 後の
学習、アトム値と入力を更新する CUDA Graph replay を含む。

再現用 snapshot は `srv11/cst-lab/torchcst-triton-20260926-perf-prep1/`。
基準コミットは `306d071`、変更を含む全110ファイルを manifest で照合した。
`manifest.json` の SHA256 は
`cc534514f899b6aaa6f31ef8938b8329bfeeadf26ebca2eb759cce0c8caa4648`。
取得結果はローカルの同じ出力ディレクトリに `*-perf-prep1.*` として保存した。
