# CUDA Linear 開発候補

このディレクトリは未昇格の実装置き場。`torchcst` の配布 wheel や
`CSTLinear(backend="auto")` に含まれない。現在残す候補は、正確な mapped
学習経路 (`block_streamed_backward.py`) と、別演算である固定アンカー近似
(`anchor_atom_training.py`)。どちらも移植・採用の判断は
[kernel 選別表](../../../docs/cuda-kernel-shortlist.ja.md)と
[dispatch 設計](../../../docs/cuda-dispatch-design.ja.md)に記録する。
旧 wrapper の静的 import とテストが参照する比較実装も残っている。
このディレクトリ内の全 kernel を採用候補とみなす意味ではない。

アンカーは開発途中。近似品質と長期学習の検証が終わるまで既定にはしない。
8192²では融合 Torus decode の精度問題があるため、対応する測定は PyTorch
decode を使う。古い探索コードの参照方法は
[旧 `prototypes/` 索引](../../../docs/legacy-prototypes.ja.md)を参照。

このディレクトリのコードはCUDA/Tritonを要する。GPU のない環境では
関連テストが skip されるため、GPU 上の正確性・速度は別途測る。
