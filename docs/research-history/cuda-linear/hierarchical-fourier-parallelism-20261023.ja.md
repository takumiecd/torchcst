# 多段の支持分類と階層Fourierの並列性を検討する

2026-10-23。既存数値実装 `ecb6987c`、調査時HEAD `606e9e8f`。
ユーザーと設計を検討するための最初の構造診断。CUDA runtimeの変更・GPU実験は行っていない。
CPU上の支持/routing再現とlaunch式の集計であり、GPU correctness・時間・occupancy・cacheの測定ではない。

## 今回の設計の前提

- 幅・支持によって多段に分類する。小/大の二分や、新方式とH方式の二者択一にはしない。
- 局所処理で完結する群では独立したglobal Hの生成・保存を省く。
  block内で一時的に縮約値を持つことは可能。広い支持ではZ/Hの再利用を検討する。
- 各群の配置、入力/出力の参照順序、blockの担当範囲を合わせて考える。
- この会話のrhoは `sigma/(2*spacing)`。実装のlive幅Iや成分の支持半径Lとの対応を
  追加の定義なしに同一視しない。今回の集計は現コードのI/Lをそのまま使う。
- 有限Fourier profile、中心・幅の連続性、混合後の離散L2/floor、全微分を維持する。
  既存の出力残差はatomic加算を使う。説明用の出力所有block案とは区別する。

## 依存関係とlaunch式

同じforwardのrouting/demandを作り、必要なら親から子へ需要を伝播する。
Zは階層ごとに子から親へ生成し、その後atom chunkごとに
`prepare/norm/pack/order -> H -> C集約 -> 残差Y加算`を行う。
全chunkのC蓄積後にYへ展開する。C更新は既存順序では逐次であり、
chunkを別streamへ移すだけではCのread-modify-write競合を解決できない。

階層lの候補node数をJ_l、総数をJ、batchをB、chunk atom数をA_c、
Hのatom/batch blockをg_a/g_bとする。現在の主なTriton program数は

\[
P_{Z,l}=J_l\lceil B/16\rceil,\qquad
P_{H,c}=\lceil A_c/g_a\rceil\lceil B/g_b\rceil,
\]
\[
P_{C,c}=J\lceil B/16\rceil,\qquad
P_{R,c}=4A_c\lceil B/4\rceil,\qquad
P_Y=\lceil B/4\rceil\lceil N/32\rceil.
\]

起動program数には空の仕事も含まれる。Zは需要がなくてもゼロの書込みを行う。
Hのprogram内には成分・支持/tileのループ、Cにはbank候補のループがある。
Y展開の各programは全bankを順に走査する。program数だけでは実仕事量や時間は決まらない。

## CPUで集計した例

N=1024、B=32、A=1024、M8、seed=41。
既存small/medium/large fixtureの初期Parameterを使い、catalogの
`tile32-boundary-h-count8`を明示的に選択した。
各Caseの既定plan集合による実測結果ではない。
atom chunk=256、g_a=4、g_b=4、minimum_sharing=8、sample_cache=512、reuse=h。
候補は14 bank、232 node。routingは現行の候補共有数gateであり、
選択後の共有数が8以上になる固定点保証ではない。

各Z階層の入力計算があるprogram数（子への依存伝播を含む）は以下。

| tile sites | 候補program数 | small | medium | large |
| --- | ---: | ---: | ---: | ---: |
| 32 | 256 | 0 | 184 | 192 |
| 64 | 128 | 0 | 64 | 96 |
| 128 | 64 | 0 | 0 | 48 |
| 256 | 16 | 0 | 0 | 8 |

起動数はZ合計464。Hは1 chunkあたり512、全4 chunkで2048。
Cは1 chunkあたり464、全体1856。残差depositは全体32768。
Y展開は256。これらは同時に走るblock数でも、SM稼働率でもない。

smallでは共有成分が選ばれないが、現在も全atomのHを生成する構造である。
今回の「局所群はglobal Hを保存しない」という案は未実装。
reuse=hのHには入力中心/幅微分もあり、確保形状は `[B,A_c,3]`。
今回の全chunk保存量は393216 bytes。Z/Cはそれぞれ504832 bytes。
これはTensor容量で、完全stepの測定ピークではない。

C集約はnodeごとに、そのchunk内で同じbankに属する全成分を走査し、
支持がnodeを完全に含むか判定する。以下はbatch方向の反復を掛ける前の数。

| fixture | 成分所属の判定数 | 実際の成分→出力tile関係数 | 比率 |
| --- | ---: | ---: | ---: |
| medium | 30960 | 1800 | 17.2 |
| large | 11796 | 2154 | 5.4763 |

この比率は候補判定数の比であり、転送量・実行命令数・速度改善率ではない。
32成分単位のpaddingやbatch、基底数、リスト構築費用も別にある。

## 次に比較したい設計候補

1. 粗いZ階層でbatchまとめ幅を16から小さくする。
   独立仕事を増やせるが、基底の読込共有が減る。例えばlargeの最上位は、
   batch幅4なら仕事数が8から32へ増える。GPUで速くなるかは未確認。
2. Cの出力tileごとの成分参照リストを作り、bank全体の走査を減らす。
   リスト構築・prefix/並べ替え・保存・動的更新の費用を含める。
3. 各仕事の重さの分布を記録する。Zはsites/新規周波数/継承、Hは直接site数/共有tile数、
   Cは候補走査数/実寄与数、残差は出力先ごとの加算の集中を見る。
   重いCだけ分割する場合は部分Cと追加縮約の費用を明示する。

まず各段の並列性と仕事量を把握し、その後に段間の重ね合わせを検討する。
GPUで確認するのは段別traceと、未計装の完全step時間・allocated/reservedピーク。
診断時間の合計を完全step時間の代わりにしない。
新候補は独立Y/dX/全atom勾配、可変幅、retained backward、Graph replayを検証する。

## 証拠と再現

ignored `output/hierarchical-fourier-parallelism-20261023/` の `audit.py` / `audit.json`。
CPU実行は支持のstrict境界を全対象成分についてsite列挙と照合し、
完全tileが支持内に収まることをassertした。runtime/driver/case/planのSHA256をJSONに保存。

```bash
PYTHONPATH=src:. /Users/ware10sai/Desktop/personal_research/torchcst/.venv/bin/python \
  output/hierarchical-fourier-parallelism-20261023/audit.py
```

CPU再現の算術であり、CUDA libdeviceとの境界でのbit一致やGPU経路の照合ではない。
集計した初期状態から、学習後の分布や他条件へ一般化しない。
