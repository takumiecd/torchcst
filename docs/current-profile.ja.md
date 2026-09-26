# 現在の融合版・atom直接版のプロファイル

2026-09-26。PyTorch/CUPTIでカーネル別時間を取得した。
これ以前の工程別ベンチマークとコードの分析だけでは、カーネル内部の
律速要因は特定できていなかった。今回も命令・メモリの性能カウンタは未取得。

## 条件と集計方法

- NVIDIA A100 80GB PCIe **MIG 3g.40gb**、Torch 2.6.0+cu126、Triton 3.2.0。
- Strip + Torus、raw Triweight、FP32、TF32無効。
- Wの形256×512、入力行数M=128、atom数512、seed=21。
- 既存benchmarkの通常fixture。I/B混在ケースや他の形状への一般化は未検証。
- 各経路3回warmup、3回をcaptureし、各stepの後でCUDA synchronize。
- forwardは分類・sort・packを含む。学習側はforwardとX・pへの一次勾配。
  optimizer更新は含まない。
- 表の値はChrome traceの `cat=kernel, ph=X` の `dur` を合計して3で割った値。
  GPU annotation範囲は重なるので加算しない。memcpy、起動間の空白、CPU時間も含まない。
- **end-to-end時間ではない。** CUDA Graphによる既存ベンチマークとは
  実行条件・集計方法が異なるため、そのdense時間との比は計算しない。

## Forward

単位ms。「その他」は準備などのGPUカーネルの合計。

| 経路 | 本体 | その他 | 合計 | 本体の割合 |
| --- | ---: | ---: | ---: | ---: |
| 融合 BM16 | 0.359 | 0.102 | 0.460 | 77.9% |
| 融合 BM64 | 0.194 | 0.101 | 0.296 | 65.7% |
| 直接 BM4 | 1.832 | 0.114 | 1.946 | 94.2% |
| 直接 BM16 | 0.892 | 0.112 | 1.004 | 88.8% |

sortのカーネルは各経路で約0.036ms。
直接版の遅さにはGPU本体の処理が大きく寄与しており、準備の改善だけでは解決しない。
融合BM64ではその他の割合が34.3%あり、準備処理も改善対象になる。
Python・起動待ちのend-to-endへの寄与をこの表だけで否定することはできない。

## Forward + backward（BM16）

| 処理 | 融合 ms | 直接 ms | 直接版の合計に対する割合 |
| --- | ---: | ---: | ---: |
| forward本体 | 0.359 | 0.889 | 7.1% |
| 入力勾配 dX | 0.297 | 7.021 | 56.0% |
| atom勾配本体 dP | 0.088 | 4.389 | 35.0% |
| その他 | 0.219 | 0.231 | 1.8% |
| 合計 | 0.962 | 12.530 | 100% |

「その他」には準備とdecode・pack等のautogradカーネルが含まれる。
直接版の学習ではdXとdPが合わせて約91%を占める。
学習高速化の最初の対象はdX、その次がdP。

## コンパイラのリソース情報（forward）

全経路128 threads/block。

| 経路 | grid | registers/thread | compiler n_spills | shared bytes/block |
| --- | --- | ---: | ---: | ---: |
| 融合 BM16 | 8×16×1 | 116 | 0 | 2,048 |
| 融合 BM64 | 2×16×1 | 145 | 0 | 5,120 |
| 直接 BM4 | 32×256 | 39 | 0 | 512 |
| 直接 BM16 | 8×256 | 64 | 0 | 512 |

今回取得したforwardのコンパイル結果にspillは報告されていない。
これは達成occupancy、キャッシュhit率、帯域使用率、backwardのspillの測定ではない。

## 次に検証する仮説

コード上、直接版dXは各入力要素の担当が全stationとその候補atomを巡回する。
forwardは行の外接boxで除外できなければ全K区間を評価する。
dPはatomごとに対象行・列区間・M方向の縮約を行う。
これらの走査と反復計算を減らすことを候補にするが、
どの命令やメモリアクセスが律速かを今回の時間だけで断定しない。

Nsight Compute/Systemsの実行ファイルはPATHと通常のCUDA・NVIDIAの
インストール先で見つからなかった。詳細な律速判定には性能カウンタを取れる環境、
または走査範囲・再利用単位を個別に変える比較実験が必要。
TritonからCUDA C++への移植だけで解消するという根拠も得られていない。

## 再現情報

GPU実行ソース: `026e847d83534ccefb4c487a61d2b44005fb8449`。
archive SHA256:
`3b8b26ec72acfd9bf8318dc46871750e946b460500826dfe2ded5dc336182534`。
リモート: `srv11/cst-lab/torchcst-current-profile-20260926/`。

```bash
python -m prototypes.profile_current_paths --output-dir results --source-commit COMMIT
```

初回の集計はCUDA device eventにGPU annotationも含めて二重加算していた。
実行後に `_profile_trace.kernel_summary` で元のChrome traceから実カーネルだけを
再集計した。GPUカーネルや実行条件の変更・再実行はしていない。
初回 `profile.json` とログのdevice event合計は使用しない。

ローカルの生traceと修正集計:
`output/triton-a100-20260926/current-profile/results/`。
正しい集計は `profile.corrected.json`。
6本のtraceを再集計し、annotationが混入していないことを確認。Ruff通過。
