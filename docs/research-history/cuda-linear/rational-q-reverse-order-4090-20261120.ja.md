# RTX 4090でのrational-radius Linearのbackwardにおける出力タイル順の再利用

[English](rational-q-reverse-order-4090-20261120.md) | **日本語**

[毎回のAtom並べ替え](rational-q-atom-order-4090-20261120.ja.md)では、
半径バンクと入力タイルでAtomをまとめる。研究用の明示的なPlan
`cuda_rational_radius_wide_reverse_ordered` は、forwardで既に求めた
出力タイル順も保存し、逆向きの応答を計算するCUDAレーンの割り当てに使う。
応答は入力タイル順に並んだ配列上の対応する位置に書き戻すので、
勾配計算に渡る入力と加算の順序は維持される。

## 時間と検証

RTX 4090、PyTorch 2.9.1+cu128、eagerのIEEE FP32、M8 cell mass、128行、
幅Nにつき32N Atom。両Planで同じネイティブバイナリとAtomを用いた。
各Planを独立した2プロセスで測り、各プロセスで8回ウォームアップ、
24回計測した中央値を平均した。更新時間にはforward、backward、
CSTAdamWを含む。ピークは更新中のPyTorch割当メモリ。

| 幅 | Forward (ms) | Backward (ms) | F+B (ms) | 更新 (ms) | ピーク (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | 3.23 → 3.24 | 5.98 → 4.48 | 9.21 → 7.71 | 9.33 → 7.71 | 241.74 → 242.74 |
| 8192 | 4.52 → 4.57 | 11.50 → 8.04 | 16.03 → 12.60 | 16.86 → 12.88 | 484.49 → 486.49 |
| 16384 | 7.56 → 7.57 | 23.63 → 15.70 | 31.19 → 23.25 | 32.87 → 24.16 | 966.99 → 970.97 |

幅16384の個別プロファイルで逆向き応答は9.66 → 1.69 ms。
`batched_parameter_kernel` は9.23 msで、新しいbackward時間の約59%を占めた。
3条件すべてで出力、入力勾配、Atomの4成分の勾配がビット単位で一致した。
7行で入力勾配のみ、Atom勾配のみを計算した場合も一致した。
対象のCUDAテストは14件成功。CPUテストは2,206件成功、2,585件スキップ。

幅16384の7層の容量検証では、3更新すべての損失が一致した。
更新2–3の平均は0.2200 → 0.1636秒、ピーク割当は4.06 → 4.08 GiB。
先のFP32 Dense計測では最初のoptimizer更新でメモリ不足になった。
この検証では合成入力を用い、eager学習で収まるかを調べた。

## Atomの物理配置と更新番号

幅16384で `AtomState.relayout()` を実行し、Parameter、既存勾配、
AdamWの2つのモーメントが揃って移動することを確認した。出力の相対差は0だった。
CSTAdamWの更新でParameterのPyTorch `_version` は1 → 3に進み、
`layout_version` は0のままだった。その後のrelayoutで
`_version` は4、`layout_version` は1になった。PolarのCUDA更新も
直接書き込みの後に `torch.autograd.graph.increment_version` を明示的に呼ぶ。

従来のPlanで32更新した別計測では、毎回並べ直す場合、計測した24更新の中央値は
32.89 msだった。最初に一度だけ物理的に並べ替えて使い続ける場合は37.00 msで、
初回のrelayoutにかかった109.73 msはこの値に含まない。ウォームアップ済みの
単独relayout検証では51.91 msだった。32更新後には50.58%のAtomでキーが
初期値から変わり、物理順で隣接するキーが逆順になった割合は0 → 24.21%だった。

今回のPlanは各forwardで現在の値から順序を計算し、対応するforwardとbackwardの間で
データを共有する。eagerでforward間の順序をキャッシュする場合は、Parameterの
同一性とストレージ、PyTorchの更新番号、Atomの配置バージョン、ソート条件を
照合すれば、Atomの値を共有するマイクロバッチで再利用できる。
生ポインタから直接書くネイティブ実装では、更新を明示的に通知する必要がある。
物理配置を固定してもoptimizer更新後にAtomの対応は保たれ、キーの変化により
メモリの局所性は落ちうる。

## 成果物

計測したソースは `570c740f25a1d97dca9c2fb44d7bef2941bd16d5`。
ローカルの `output/rational-q-reverse-order-20261120-01` に `summary.json`、
プロセスごとの計測、`reverse-profile.json`、`physical-{verify,online,physical}.json`、
および再現・プロファイル・容量・物理並べ替えの各スクリプトを保存した。
ネイティブバイナリのSHA-256は
`4394778ee16ad722f768ea7ae6cf16813d991aabd88ad199410bc6b269259758`。
ソーススナップショットのSHA-256は
`9ee7763583f4484621f10fae01b6b88438b000e0d1f1dae62c0ae7c1f5637e56`。
主要な測定結果アーカイブのSHA-256は
`aa3e2ad6b4005347285de262a727272b5e79d9926fa6570bb5a7ae808568c225`。
