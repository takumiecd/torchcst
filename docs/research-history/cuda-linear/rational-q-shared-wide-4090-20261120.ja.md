# RTX 4090での有理式半径Queryのタイル集約共有

[English](rational-q-shared-wide-4090-20261120.md) | **日本語**

実験用の明示Plan `cuda_rational_radius_wide_shared` に、
[PR #168の集約共有](https://github.com/takumiecd/torchcst/pull/168) を
幅4096の正方射影・32N Atom向けに移植した。64サイト以上の階層では、
8行の組ごとにフーリエ係数タイルを一度計算し、32サイトを担当するブロック間で
共有する。forwardとdXの両方に適用し、応答の再計算と物理半径の勾配は既存実装を
使う。Planはベンチマーク用レジストリに登録した。

## 比較

TinyStoriesのシード17・4,882更新時点の先頭層Query、131,072 Atom、128行。
チェックポイント、Atom、シード生成したx・dY・目標値のハッシュは
[前回の比較](rational-wide-fourier-4090-20261120.ja.md)と一致した。
RTX 4090 SM89、PyTorch 2.9.1+cu128、CUDA 12.8、Triton 3.5.1、eagerの
IEEE FP32。時間は順序を反転した2測定で、各8回のウォームアップ後に24回を測り、
中央値を平均した。forwardとbackwardは固定dYでCUDAイベントを使用した。
学習更新はMSEとCSTAdamWまで含む同期済み実時間。先行Planの常駐バッファの影響を
除くため、ピークは各Planを別プロセスで起動して測定した値を掲載する。

| Plan | Forward (ms) | Backward (ms) | Forward + backward (ms) | 1更新 (ms) | Forward + backwardのピーク割当 / 予約 (MiB) | 更新のピーク割当 / 予約 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| フーリエ重み展開 | 3.73 | 2.35 | 6.08 | 6.19 | 588.11 / 648 | 592.11 / 648 |
| 直接 | 13.63 | 25.17 | 38.80 | 40.50 | 233.50 / 246 | 239.49 / 310 |
| 直接・集約共有 | 7.39 | 18.61 | 26.01 | 27.50 | 233.50 / 246 | 239.49 / 310 |

集約共有でforwardとbackwardの合計が33%、学習更新が32%短くなった。
別プロセスで測ったピークは直接版と同じだった。展開版に対する集約共有版の
学習更新は4.4倍の時間で、ピーク割当は40%だった。ピークはPyTorchの
アロケータで計測した。

別に採取したCUDAトレースでは、係数集約と展開カーネルの合計が
forward/backwardの1組あたり17.74から4.98 msに短縮した。応答カーネルは
約12.5 msだった。全Atomを使った照合で、出力・入力勾配・Atom勾配の
4項目すべてが直接版とバイト単位で一致した（テンソルのSHA-256で確認）。展開版のq勾配の相対L2差は3.29e-6で、
診断用の要素別許容差を超えた点は1件だった。4090で対象テスト8件が通過した。
CPU FP64との照合は128個の有効Atomを使い、全サイトを確認した。
CPUの全体スイートは2,203件通過、2,582件スキップ。

## 再現と証拠

比較元の実装は `4b36c781`、計測した候補は
`37d0f55ae348fab57b533503c3bc321d05678100`。スナップショットと
ネイティブバイナリのSHA-256はそれぞれ
`806d22b890d05dbc3154e2ce60a6a68145616f6805413075c148190756cb7a91`、
`41ae64a0b48abee08cb77124db74cb1a271d7e2c7734d04755aa00715fd48c9f`。
記録は `output/rational-q-shared-wide-20261120-01/` に保存した。
クラスタの採用スナップショットは `rational-q-shared-wide-20261120-01-v2`、
時間と独立プロセスのピークを測定したジョブはPID 67709と68024。
スナップショットと同じ環境・ソースを使い、前回の再生ディレクトリも
PYTHONPATHに追加する。

```bash
export CST_Q_SOURCE_COMMIT=37d0f55ae348fab57b533503c3bc321d05678100
python replay-matched.py
for plan in weight direct shared; do python isolated_peaks.py "$plan"; done
```

時間計測JSONのSHA-256は
`f829f8500447750e92c199587d399f53c2dd04e1c8a4881c588f14f7a947f2e2`。
各Planの独立プロセスの結果は3件の `isolated-*.json` に、
一致したテンソルのハッシュは `byte-parity.json` に保存した。
検証範囲はeagerの単一射影で、CUDA Graphおよびモデル全体の学習は対象外とした。
