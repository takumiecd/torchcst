# L4・1024²小バッチCSTの現行ボトルネック（2026-09-29）

> この文書に残る `prototypes` の実行例は commit `190a1bb3fa6259fde493a30d901d5cab7c2ebb82` 時点の
> 歴史的な再現手順です。削除したコードの参照方法は
> [旧実験コード](../legacy-prototypes.ja.md)を参照してください。

## 測定条件

Colab ProのNVIDIA L4、PyTorch 2.11.0+cu128、Triton 3.6.0。論理重み1024×1024、52,429 atoms（5%）、64×64 station、入力行数16/128、単層AdamW。現行の速い実験経路、すなわち**候補リスト付きW 8×64、atom勾配16×64、512行窓、FP16全行Wキャッシュ、CUDA Graph、IEEE FP32 GEMM**をプロファイルした。atomと入力はseed 21。`benchmarks/cuda/linear/profile_mapped_training_kernels.py`のGraph再実行1ステップをwarmup後に`torch.profiler`で採取した。

下表は**GPU kernel実行時間の合計**であり、CPU時間やkernel間の隙間を含むwall時間ではない。Profiler実行の単発値である。別の交互測定では同じ方式の完全ステップ中央値がM=16で約1.03 ms、M=128で約1.11 msだったが、両測定の数値は直接加減しない。

## 完全ステップの時間内訳

| 工程 | M=16 | 構成比 | M=128 | 構成比 |
| --- | ---: | ---: | ---: | ---: |
| atom配置・候補リスト準備 | 141.7 µs | 13.2% | 142.5 µs | 12.6% |
| forward: W生成・GEMM・FP16保存 | 240.8 µs | 22.4% | 264.3 µs | 23.3% |
| backward: 入力勾配dX | 28.9 µs | 2.7% | 49.4 µs | 4.4% |
| backward: 局所dW・atom勾配 | **437.5 µs** | **40.8%** | **451.9 µs** | **39.9%** |
| atom勾配から元パラメータへの逆変換 | 101.0 µs | 9.4% | 101.6 µs | 9.0% |
| AdamW | 123.2 µs | 11.5% | 122.5 µs | 10.8% |
| **GPU kernel合計** | **1,073.2 µs** | 100% | **1,132.2 µs** | 100% |

**backward全体**（dX、dW＋atom、元パラメータへの逆変換）はM=16で567.5 µs（52.9%）、M=128で602.9 µs（53.3%）。一番重いのはdXでもdWのGEMMでもなく、`mapped_backward_atoms_listed`である。

さらにkernel単位へ割ると次の通り。

| 主要kernel・処理 | M=16 | M=128 | 回数/step |
| --- | ---: | ---: | ---: |
| `mapped_backward_atoms_listed` | **419.0 µs** | **420.6 µs** | 2 |
| `materialize_listed`（W 8×64） | **219.4 µs** | **222.7 µs** | 2 |
| 局所dW GEMM | 18.6 µs | 31.2 µs | 2 |
| forward GEMM＋split-K縮約 | 約17.0 µs | 約37.3 µs | 各2 |
| `support_buckets_batched_box` | 38.5 µs | 38.4 µs | 1 |
| 候補リスト構築8×64・16×64 | 15.6 µs | 15.4 µs | 2 |
| bucket histogram＋scatter | 24.5 µs | 24.5 µs | 各1 |

入力を16行から128行へ8倍にしても、atom勾配とW生成はほぼ一定で、kernel合計の増分は約59 µs。増えるのは主にGEMMである。小バッチで固定費が目立つ理由はこれである。Graph内のkernel数はM=16で130、M=128で128。配置準備、勾配逆変換、AdamWにも細かいkernelが多数ある。

## atom勾配kernelの内部で何が重いか

候補リストは16×64タイル当たり平均146.77、最大176 atomで、192枠からのあふれは0。全1024²で**約1.539億 candidate×site訪問**を行う。W生成の8×64タイルは平均132.27 atomで、全体で**約1.387億訪問**。サンプル64個の16×64タイルでは、候補×siteの62.48%がTriweightの支持内、37.52%は支持外だった。論理Wは全要素非ゼロ。したがってWを疎行列として扱う方策は合わない。

Nsight Compute 2025.1.1で、現在のkernelをそれぞれ1つ採って測った。計測時のclockとcache状態はGraph traceとは異なるため、**Nsightの絶対時間を上の表へ混ぜない**。

| 指標 | W生成8×64 | atom勾配16×64 |
| --- | ---: | ---: |
| registers/thread、spill | 128、0 | 168、0 |
| 実効occupancy | 23.3% | **14.6%** |
| SM compute throughput | 67.3% | 45.4% |
| DRAM throughput | 1.06% | 0.93% |
| L2 hit rate | 98.98% | 98.96% |

この一回のNsight測定ではDRAM帯域は飽和せず、atom勾配はレジスタ制約で理論occupancyも25%、実効値は14.6%だった。NsightはこのkernelのgridがSM資源を埋め切れていないとも報告した。候補数115～176のばらつきがtailを作る可能性はあるが、その寄与率は未測定。Nsightはcache状態が制御されていないと警告しており、L2 hit率は単発の定性的な手掛かりとして扱う。

atomic加算の寄与を切り分けるため、全1024行のatom勾配kernelで最終出力をatomic加算から**タイルごとの部分結果書き込み**へ替えた。単独kernelは0.467→0.430 msで約8%短縮。ただし後者には部分結果をatomごとに集約する費用が入っていない。**atomicだけを除いても主な約0.42 msは消えない**という下限診断である。

## 改善の優先順位

1. **atom勾配の候補×site評価を減らす。** 419～421 µs、全GPU kernel時間の約40%。候補数を少し下げるだけでは足りず、支持内の値と導関数をより少ない評価点・係数でまとめる方法が必要。
2. **W生成の評価を減らす。** 219～223 µs、約20%。全Wを書かないFlash型融合はL4で現行分離方式より遅かったため、書き込み削減だけを第一目標にしない。
3. **残る固定費を整理する。** 準備約142 µs、勾配逆変換約101 µs、AdamW約123 µs。M=16ではW生成とatom勾配のkernel時間を仮にゼロにしても約435 µs残る。denseの約0.14 msに近付けるには、これらのkernel数・パラメータ変換も対象になる。

生データは [`benchmarks/cuda/linear/evidence/l4-bottleneck-20260929/`](../../../../benchmarks/cuda/linear/evidence/l4-bottleneck-20260929/) のGraph trace、kernel要約、Nsight CSV、候補支持率、atomic対部分書き込みのJSON。`prototypes/probe_l4_cst_kernel_metrics.py`がNsight用の単一kernel起動を再現する。基準コミット`aeacc11`の隔離archive SHA256は`dcf20cd6ff4e3e2135a13381d191275b7e75ead2e20c1084bf2dcf11c7afe351`。Nsight probeのみ別途転送したファイルSHA256は`4eddb00a79617608ebab92ca4eec983731ec7d96fddd26578ffde43c203b65f2`。Colab sessionは停止し、active sessionなしを確認した。
