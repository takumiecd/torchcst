# 宣言と実行の配置規約

数学的な意味を宣言する型と、それを Tensor 上で計算する実装を分ける。
宣言には backend の選択、forward / backward、GPU の launch 設定を入れない。
backend は宣言の意味を変えず、対応する Algorithm と Recipe で実行する。

## 現在の配置

```text
torchcst/
  geometry/
    spec.py                    Geometry / Pattern / Chart の不変な宣言
    _declarations.py           既存クラスからの設定 snapshot
    geometry.py                Geometry の設定・状態・互換入口
    chart.py / lazy_chart.py   Chart の設定・状態・互換入口
    pattern.py                 SitePattern の設定・状態・互換入口
  kernels/
    profiles/                  Triweight / Gaussian など関数の形の宣言
    parameterizations/         振幅・幅・activity の座標解釈の宣言
    normalization.py           正規化の規則・領域・床
    spec.py                    ProfileBinding / KernelSpec / 状態方針
    _declarations.py           既存クラスからの設定 snapshot
    既存の公開クラス            設定・状態・checkpoint・互換入口
  operators/
    spec.py                    単一 Chart / Chart の組 / linear Operator の宣言
    binding.py                 Module の live state を参照する Operator
  _backends/
    linear.py                  Module の backend 選択・遅延接続
    cuda/
      algorithm.py             Algorithm ABC
      context.py / schema.py   共通 OperatorSpec と実行条件・plan・診断
      registry.py / dispatch/  登録・検証・現行の選択方針
      algorithms/
        normalized_euclidean_strip/
          contract.py          対応する数学契約の判定・固定 site metadata
          _shared/             full / window が実際に共用する計算
          full/                algorithm / recipe / executor / kernels
          window/              algorithm / recipe / executor / provider / kernels
        strip_torus/fused/     現行の融合計算・準備・autograd・schedule
    torch/
      geometry/                埋め込み・距離・射影・更新・輸送
      patterns/                格子展開・点選択・bounds
      charts/                  Chart の座標生成・部分領域・支持域
      profiles/                Profile の評価・微分
      parameterizations/       振幅・幅・activity の実際の解釈
      kernels/                 atom 合成・重み生成・初期化・factor 微分
      updates/                 Kernel 座標の更新と optimizer 状態輸送
      operators/               Operator を受け取る Torch linear 実行
  nn/
    公開 Module                Parameter / buffer / checkpoint の所有
```

本体から `experiments/`、`benchmarks/`、`tests/` へ依存しない。Torch backend は
CPU 専用という意味ではなく、Torch Tensor の対応 device 上で計算する参照経路。
NVIDIA 向け Triton の最適化をここへ混ぜない。

## 宣言の区分

| 宣言 | 固定する意味 | 実行設定に移すもの |
| --- | --- | --- |
| Geometry | 空間、次元、metric、半径、center の保存表現 | 座標変換・距離・射影・更新の計算手順 |
| Chart / Pattern | shape、点配置、flatten 順序、原点、間隔、Strip の物理 pitch | 座標展開、GPU 上の配置・routing の手順 |
| Profile | 無次元距離に対する関数の形 | 関数評価・微分の実装 |
| Parameterization | atom 座標から振幅・幅・中心を解釈する規則と微分契約 | 解釈の計算・Tensor の pack |
| Normalization | 対象領域と L2 等の規則、ノルム床 | reduction、保存・再計算、作業領域 |
| Kernel | radial / separable / amplitude の合成 | materialize / full / window / fused の方式 |
| 初期化・更新方針 | 初期分布や座標更新の意味 | sampler、射影・retraction・状態輸送の計算 |

Sphere / Torus は現行の chord metric を維持する。正規化領域や amplitude-width の
結合微分を Recipe によって変えない。数値的な契約を変える場合は別の意味の版を
付け、既存 checkpoint や同じ演算の fallback と混ぜない。

## 共有処理の配置

一つの Algorithm で使う準備・計算は、その Algorithm 内に置く。同じ演算の複数
Algorithm が実際に共用するものは、演算グループ内の `_shared/` に置く。
別演算でも使う CUDA 専用処理は、必要性を確認して CUDA の中に共通化する。
将来の MPS / Radeon との共有を理由に、先回りして backend 共通の `shared/` を
作らない。共通の数学的意味は宣言で共有し、計算実装の共有とは区別する。

## 次の移行境界

公開クラスの既存の計算メソッドは、計算本体への遅延接続として残している。
Spec 単体から Module を生成する factory はまだない。Operator は既存 Module が
所有する Chart / Kernel / Atoms を参照し、Torch materialized / factored 実行の
共通入口になる。単一 Chart と入出力 Chart の組は layout の型で区別する。
宣言 snapshot を forward ごとに作らず、現在の Tensor / buffer を使う。
詳細は [Operator](../operators/README.md)を参照。

CUDA Registry も共通 `OperatorSpec` を受け取る。対応する数学契約の判定は
各 Algorithm にあり、Registry 自体は normalized Strip の shape や 5 列の atom を
前提にしない。Recipe は対応する Algorithm のディレクトリに置く。数学的な宣言は
Kernel / Geometry / Chart にあり、`contract.py` はその対応判定と metadata 抽出だけを
行う。`operators/` に normalized Strip 専用の実装や互換 adapter は置かない。

Strip + Torus の現行 `backend="triton"` は `_backends/linear.py` から fused executor
へ接続する。Module の live Chart / Kernel を使う既存経路で、共通 Spec による
Registry 登録はまだ行っていない。設定 snapshot を forward ごとに作らず、既存の
buffer version による固定座標計画の invalidation と fresh な atom 準備を維持する。
この接続は次の統一段階で扱う。新しい既定選択や性能による自動採用は行わない。

演算の契約は backend の外に置く。Algorithm / Recipe の設定と kernel 技術の名前を
区別し、Triton というだけの理由で複数の計算方式を一つのファイルへ集めない。
未実装のディレクトリや空の雛形は作らない。

検証は `tests/`、継続する正しさ・時間・メモリ測定は `benchmarks/`、未採用方式は
`experiments/` に置く。登録、対応判定、承認、既定 dispatch への採用は別の段階とする。

詳細は [Kernel の移行](../kernels/README.md)と
[Geometry と Chart の移行](../geometry/README.md)を参照。
