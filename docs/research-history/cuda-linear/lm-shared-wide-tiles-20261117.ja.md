# TinyStories FFN down の広い階層タイルの集約共有

[English](lm-shared-wide-tiles-20261117.md) | 日本語

レベル6以上では、階層タイルと8行の組ごとに応答係数を一度だけ集約する。
対象となる単位幅の二分グリッドでは、これらは64サイト以上のタイルに相当する。
従来の展開ブロックはそれぞれ32サイトを受け持ち、セルを評価する際に共有係数を
読む。集約先ごとに元の owner 順とチャンク境界を使う。forward の出力と
dX の逆方向変換の両方に適用し、一時的な FP32 係数は変換1回の間だけ保持する。
Atom 勾配は同じ応答テンソルを使い、既存の実装で計算する。

非公開の包絡 Plan は `radius_cell_mass/lm_width.py` の
`RADIUS_ENVELOPE_SHARED_WIDE_LM`。対象は許可済み FP32 グリッド上の
4096→1024 射影と196,608 Atom。半径の復号とその VJP は両条件で共通。

## RTX 4090 でのチェックポイント再生

TinyStories の seed 17、4,882更新のチェックポイントから FFN down 4層を再生した。
4,882更新の最初のマイクロバッチを保存済みモデルに通し、256行の活性値と
上流勾配を採取した。同じプロセスで学習時のバイナリと比較した。PyTorch
2.9.1+cu128、CUDA 12.8、TF32 無効。各層で両方の計測順を回し、順序ブロック
ごとに5回ウォームアップして24回計測した。forward は半径復号と準備を含む。
backward は dX と Atom の4列の勾配を含む。合計は同期済みの実時間。

| 条件 | Forward（ms） | Backward（ms） | 合計（ms） | 常駐分からのピーク allocated 増分（MiB） | ピーク allocated（MiB） | ピーク reserved（MiB） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 学習時の版 | 40.19 | 46.35 | 86.42 | 536.22 | 986.71 | 1,006 |
| 広いタイルの集約共有 | 21.25 | 18.10 | 39.34 | 537.56 | 984.55 | 1,006 |

時間は各層の標本中央値のうち4層の中央値。メモリは条件ごとに、全計測ブロック
と全層を通した最大値。測定範囲は eager での単一射影の再生。

初回は Y の完全一致検査で停止した。完了した比較では既存の最大絶対誤差、
相対 L2 誤差、要素ごとの基準を適用した。4層で最大絶対差は Y が3.81e-6、
dX が2.91e-11、最大相対 L2 差は Y が9.22e-9、dX が1.02e-8だった。
Atom 勾配の4列はビット単位で一致した。先頭層の R1・R7・R9 も同じ比較に
通過した。候補はさらに先頭層の全196,608 Atom を R3 で独立 FP64 因子参照と
照合し、最大絶対誤差8.71e-6、最大相対 L2 誤差3.26e-7、要素別の基準超過は
0件だった。

## 再現と証拠

保存済み学習ソースの隔離コピーに、有理式下限を保持したまま
`31f40e64241eb866e95305401b94cc8da4ac5a7e` から
`d3e2bee6accab8ef23e9a8b1568256b4ff793658` までの差分を適用する。RTX 4090
環境で、保全した driver に保存済み実験と採取スクリプトを渡して実行する。

```sh
export PYTHONPATH="$SOURCE/torchcst/src:$SOURCE/torchcst:$SOURCE/cst-research/src"
export TORCH_CUDA_ARCH_LIST=8.9
python "$EVIDENCE/compare_shared_wide.py" \
  --training-run "$TRAINING_RUN" --source "$SOURCE" \
  --capture-driver "$PROFILE_RUN/profile_down.py" --output "$EVIDENCE/result.json"
```

初回比較とソース差分の保存先は `lm-shared-wide-tiles-20261117-01`、完了した
比較は `lm-shared-wide-tiles-20261117-02`。driver、結果、チェックポイントの
SHA-256 は順に
`b0bda6f8a8c4e3e99c52bf80acbd01bdd6ed9721dc9b8bec9f9fb926b4e5fed0`、
`2131814e75c0a88504103eab6398b7b4227d762bed687892a1a25a3a01c3d1f9`、
`8409ddbbfb6e7b0a2a9706952657156bec52bf3074e136cb49c68467a1e4dce3`。
