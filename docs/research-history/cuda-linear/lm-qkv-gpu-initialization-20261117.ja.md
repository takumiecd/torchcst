# RTX 4090 での幅2048 QKV の FP32 分散初期化

[English](lm-qkv-gpu-initialization-20261117.md) | 日本語

幅2048の TinyStories QKV では q=0 と指数型の半径下限で重みを初期化する。
この研究用経路は CPU の乱数列と既存の有界な振幅倍率探索を維持する。
候補ごとの物理半径と階層タイルを CPU FP64 で求め、物理 Atom 行とタイル番号を
CUDA に送る。FP32 カーネルは、局所周期が16サイトより長ければ Fourier 係数から
セル質量の二乗和を求め、短ければセルを直接積分する。片側の格子は2048サイトで、
最大の局所周期は1536サイト。

最初の全 FP32 試作 lm-qkv-gpu-init-20261117-01 では、0.5よりわずかに大きい半径を
0.5に丸めてからタイルを選んでいた。一部の局所周期が変わり、72射影の初期 Atom 行が
CPU と一致しなかった。修正版は丸め前にタイルを選び、短い周期では中心を周期で
剰余化してからセル質量の二乗和を計算する。

| 射影あたり Atom 数 | CPU の倍率探索 | GPU 併用の倍率探索 | 標本エネルギーの相対 L2 誤差 |
| ---: | ---: | ---: | ---: |
| 131,072 | 33.376秒 | 31.824 ms | 2.178e-7 |
| 262,144 | 70.465秒 | 59.923 ms | 2.531e-7 |

seed 17 の先頭ブロックの Q で、同じ入力を使って倍率探索の同期済み実時間を測った。
GPU 版は CPU の半径とタイル選択、転送を含み、カーネルのウォームアップ後に測った。
計測区間はモジュール構築、CUDA ビルド、LM 学習の前後と分けた。両方の Atom 数で
CPU 版と GPU 版の初期 Atom 行がビット単位で一致した。seed 17・23・42 の72射影でも、
学習実験に記録した CPU 版の倍率と同じ乱数列から復元した Atom 行に一致した。
72射影での適合エネルギーの相対差は最大1.643e-7。

保存済み lm-qkv-2048-20261117-01 の torchcst ソースに、9c0cf9979dde545999b290f6e0ec13ff93ecf2b5
から 27ca1390e24fbb5f3f0c854b09c1fc0585472add までの差分を重ねる。検証済みの学習
summary を入力に使う。PYTHONPATH をそのソースに向け、新しい TORCH_EXTENSIONS_DIR を
選び、PyTorch 2.9.1+cu128 と TORCH_CUDA_ARCH_LIST=8.9 を使う。RTX 4090 で研究用
ベンチマークを実行する。

~~~sh
python -m benchmarks.cuda.linear.lm_qkv_initialization \
  --study "$TRAINING_EVIDENCE/summary.json" --output "$CALIBRATION_RESULT"
~~~

検証済みの実行 ID は lm-qkv-gpu-init-20261117-02。結果の SHA-256 は
be0e4c189193ed67c4ebe22e9261e926ae974f05ee924f352e06c3b35363e78a。
入力の学習 summary の SHA-256 は
15ce5cc578b817fe94b39e0dfd84ac268520d6a36a615c2e4c23e8ff23fbceda。
