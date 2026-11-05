# 大域正規化Stripの実行

[English (canonical)](README.md) | 日本語

`torchcst.nn.CSTLinear`は、signed amplitude、log bandwidth、3つの連続center座標からなる
`Atoms`表を所有する。両実装は演算子全体の有限な点配置について、joint 3D Triweightの
支持と離散L2ノルムを評価する。norm floorは`1e-6`、bandwidthは`.03–3.25`にclampする。

- `full/executor.py`と`full/kernels.py`: 既定のAtlasStream実装。全FP32重みを保持し、
  全parameter VJPのための随伴を構築する。
- `window/executor.py`、`window/provider.py`、`window/kernels.py`: opt-inのrolling実装。
  `min(512, N) × K`のFP32 weight/adjoint bufferと`8 × K`のhaloを再利用する。
  各forwardで大域norm・現在の支持・row bucketを再構築する。
  backwardではdX用重みを再構築し、各atomの結合した完全VJPを一度処理する。
- `_shared/atlas_table.py`と`window/narrow_table.py`: 保守的な候補表であり、固定centerの
  近似ではない。実際のsite値には現在のparameterを使う。
- `_shared/common.py`: 支持offset、guard、GPU bucket permutation。

CUDAはFP32、spacing `(1, .5, .5)`、atom programあたり1 warp、FP fusionを使う。
Window経路はquarter-grid originと32で割り切れるrow数も要求し、満たさなければ
検証済みfull経路を使う。座標guardは実証済みFP32 routing範囲外を拒否する。
atomic accumulationはbit単位に決定的ではない。autocastとTF32は拒否する。
CUDAは一次微分に対応し、CPUは`_backends/torch/algorithms/normalized_radial/executor.py`を使う。

独立oracleとfull/window Graph/optimizer統合の検査は
`tests/test_normalized_strip_public.py`と
`benchmarks/cuda/linear/check_normalized_strip_public.py`にある。
過去の測定とその限界は`benchmarks/cuda/linear/results/normalized-strip-history-20261001.json`
に記録する。[利用と測定上のtradeoff](../../../../../../../docs/normalized-strip.ja.md)を参照。
CPU importにTritonは不要。CUDA moduleはすべて遅延importし、wheelにも含む。

`full/algorithm.py`と`window/algorithm.py`はmetadata-onlyのAlgorithm登録で、
launch設定は各`recipe.py`にある。registryは共通の`operators.OperatorSpec`を受け取り、
`contract.py`が固定kernelとEuclidean siteの契約を確認する。
登録metadataのimportでTritonをimportしない。
