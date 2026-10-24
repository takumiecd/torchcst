# 128サイト版：atom群の並列処理と集団単位のjob割り当て

2026-10-23。ユーザーとの設計相談をまとめたもの。
**新しいCUDA forwardの設計案であり、この融合構成は未実装・未検証。**
本書の追加ではruntimeを変更せず、GPU実験も行っていない。

追記：担当の割り当てと人数調整は、最初から**thread単位**で行う。
各関数の中身は[thread単位の関数別擬似コード](hierarchical-fourier-thread-functions-20261023.ja.md)を参照。
一つのatomのHを複数threadで協調縮約する旧案は撤回し、
各threadが別のatomのHを計算する。群の全atomを一括して並列処理の対象にする。

## 1. 今回の方針

thread集団のメンバーと出力blockの所有範囲は最初に決める。
空いた集団の代表が、依存の揃ったjobを取得し、集団全体へ渡す。
jobにはZ一枚の生成と、同じ階層・参照タイルを使うatom群の処理がある。
Zは協力して作るが、atom群のHはthreadごとに別のatomを担当して作る。
Z不要のatom群は、共通の準備が済み次第、Z生成と独立に開始できる。

```text
共通準備：係数・正規化・支持・所属リスト・job依存を決める
                          ↓
                  block内を初期化
                 ／              ＼
       Z_TILEを取った集団         ATOM_GROUPを取った集団
       子やXからZを作る           直接計算から開始できる
       作成担当内で同期           必要なZが完成してから取得
       タイルの完成を通知 ─────→ 群のHを並列生成 → Cの集約
       集団全員が完了したら空きへ戻り、次のjobを取得する
                 ＼              Cを最後に出力へ展開
                  ＼             ／
                    最後に全員同期
                    部分和を合算 → Y
```

毎回「全員で作業 → block全体の同期」を繰り返すround方式は、今回の基本構成にはしない。
入力Zには子から親への依存があるが、別の直接計算や、完成済みZを使う計算まで
同じ全体同期で止めない。出力側の階層も、すべてを階層順に直列実行する意味ではない。

## 2. 最初に扱う大きさと担当

| 項目 | 最初の候補 |
| --- | --- |
| 入力・出力 | 各128サイト |
| atom数 | 128 |
| 1 blockの担当出力 | 32サイト × 4バッチ |
| 1 blockのthreads | 128 threads、4 warps |
| 各集団の人数 | `worker_group_sizes`。1 thread刻みで指定し、合計128とする |
| jobの開始割合 | 階層・種類ごとの選択重みと同時担当人数の目安で調整する |
| 最初から実行可能な仕事 | 子依存のないZ生成、Z不要のatom群 |
| 空き管理・job queueの範囲 | このblock内の固定集団。GPU全体の空きthread表ではない |
| 最後のY書き込み | 出力要素ごとに一つのblockが所有 |

バッチ4なら出力4 block、バッチ32なら4 × 8 = 32 blockになる。
これは担当を理解するための初期候補で、GPUを十分使える最適な分割という意味ではない。

タイルはデータの領域であり、warpやthreadと一対一には対応させない。
atom群のjobには所属する全atomを入れる。例えば128 atomsを128 threadsで担当するなら
各threadが別のatomを担当する。thread数が足りなければ、各threadがstrideで次も処理する。
全員で一つのatomを処理してから、次のatomへ進む方式にはしない。

集団の人数は任意に指定できるようにするが、容量・実行可能性・速度は未検証。
Cと出力部分和、一時H、job・集団の管理状態は、実際の必要数で予算を取る。
warp数が4だから部分和も4担当分で済む、という仮定は置かない。

この最初の候補では、各blockが必要なZ・H・Cをblock内で作る。
複数の出力blockに必要な値には重複計算が生じる。
広域Hなどをglobalに一度保存して共有する方式は、後で比較する候補として残す。
これとは別に、forwardからbackwardへのH保持は、再計算・保存・一部保存を選択可能とする。
2026-10-24に追加した[backwardの保持方針](hierarchical-fourier-backward-128-20261023.ja.md)を参照。

## 3. 短い擬似コード

`team`は固定メンバーの作業集団であり、担当するjobは空くたびに取得する。
同じ集団の全員が同じjobを受け取る。atom群の中では各threadの担当atomが異なる。
このコードは処理の約束を表すもので、そのままコンパイルするCUDAコードではない。
**入口の`while`は配分を調べるための暫定構成で、最終仕様として固定しない。**
人数配分・jobの発行順が決まり、全仕事と依存を直接の割り当てで扱えるなら、
空き探索だけでなく入口の繰り返し自体をなくす方式も検討する。

```cpp
init_block();

while (true) {                      // 調査用。後に除去する可能性がある
    job = acquire_ready_job(team);   // 一時的に候補がなければ再試行
    if (job == NO_READY_JOB)
        continue;
    if (job == ALL_DONE)
        break;

    if (job.kind == Z_TILE) {
        build_z_tile(job.tile);
        finish_z_tile(team, job.tile);
    } else {
        do_work(team, job);
    }
    complete_job(team, job);
}

expand_c_parts(team);
sync_finish();
finish_output();
```

```cpp
do_work(team, job) {
    compute_h_batch(team, job);       // 各threadが別のatomを最後まで計算
    group_sync(team);                 // job内の全Hが完成
    add_direct_parts_batch(team, job);
    add_c_parts_batch(team, job);
}
```

`compute_h_batch()`は、各threadが自分のatomの入力縮約を最後まで行う。
Hを作るためのthread間の部分和合算はない。必要なZはjob取得時に完成確認する。
一時Hはjobごとのsharedへ書き、全Hの完成後にCと出力を集約する。
未完成Zを待つjobで全集団を占有しない。

## 4. 関数名と具体的な役割

| 関数 | 読むもの・行うこと | 完成させるもの |
| --- | --- | --- |
| `init_block()` | Xの必要領域をsharedへ読み、C・出力部分和・通知用状態を初期化する | 全担当が使える初期状態。最後にblock全体で同期 |
| `acquire_ready_job(team)` | 集団の代表が依存の揃ったjobをatomicに取得し、全員へ渡す | 一つの集団だけが担当するjob |
| `build_z_tile(tile)` | X、基底、必要な完成済み子Zを読み、協力して縮約する | 指定タイルのZ値。通知前の状態 |
| `finish_z_tile(team, tile)` | Zの書き込みを作成担当内で同期し、完成を公開する | そのZを安全に読めるという通知 |
| `compute_h_batch(team, job)` | threadごとに別のatomを担当し、入力支持全体を縮約する | job内全atomのHを一時sharedへ保存 |
| `add_direct_parts_batch(team, job)` | jobの全Hを使い、直接区間への出力を要素ごとに集約する | 集団専用の出力部分和 |
| `add_c_parts_batch(team, job)` | jobの全Hと出力係数を、階層・タイル・基底ごとに集約する | 集団専用のC部分和 |
| `complete_job(team, job)` | 全員の読み書き完了後、job完了と集団の空きを公開する | 次のjobを取得できる状態 |
| `expand_c_parts(team)` | 自分の集団が蓄積したCを有効な出力領域へ展開する | 集団専用の出力部分和へ加算した値 |
| `sync_finish()` | すべての担当の出力部分和が完成するまで待つ | block全体の最終合算の開始条件 |
| `finish_output()` | 各担当の出力部分和を足し、担当するYへ保存する | 完成した出力32サイト × 4バッチ |

`build_z_tile()`では、親の全周波数を単純に子Zの和へ置き換えない。
既存階層方式で継承できる偶数周波数と、Xから計算する残りを区別する。

入力でZを使うかどうかと、出力でCへ集約するかどうかは別の判定である。
一つのatomにも、階層を使う部分と直接評価する部分が共存する。

## 5. Z・H・Cの数式

入力の生profileを \(v_a(i)\)、出力の生profileを \(u_a(j)\)、
振幅を \(A_a\) とする。本書ではHを未スケールの入力縮約として固定する。

\[
H_{b,a}=\sum_i X_{b,i}v_a(i),\qquad
\alpha_a=\frac{A_a}
{\max(\|u_a\|_2\|v_a\|_2,\epsilon)},
\]

\[
Y_{b,j}=\sum_a \alpha_a H_{b,a}u_a(j).
\]

正規化は既存の混合後profile・全観測サイトに対する離散L2とfloorを使う。
タイルごとに正規化し直さず、\(\alpha_a\) も一度だけ掛ける。

基底の種類をk、空間タイルをt、基底番号をrとして、入力Zは

\[
Z_{b,k,t,r}=\sum_{i\in t} X_{b,i}\phi_{k,r}(x_i).
\]

入力支持を階層で処理する部分と直接処理する部分へ重複なく分ければ、
Hは必要なZと入力側係数の縮約、および直接部分の和で作れる。
出力blockが32サイトでも、Hの入力支持をその32サイトへ切り詰めない。

出力側では、atomのFourier成分cの位相・混合を含む係数を
\(\beta^u_{a,c,k,r}\) とする。担当wのC部分和は

\[
C^{(w)}_{b,k,t,r}=
\sum_{(a,c)\in\mathcal A_{w,k,t}}
\alpha_a H_{b,a}\beta^u_{a,c,k,r}.
\]

\(\mathcal A_{w,k,t}\) は「担当wが処理し、出力タイルtへ割り当てられた成分」の集合である。
担当wは固定されたthread集団であり、warp番号ではない。
どのatom群を担当wが処理するかは、実行可能jobの取得によって決まる。
\(C^{(w)}\) はタイルtと担当blockの出力範囲が重なるサイトだけに展開し、
最後に各担当の出力部分和を足す。異なる周期のC同士を、そのまま係数として足さない。

## 6. 出力側も階層タイルで仕分ける

Fourier成分の式には周期があるが、実際のprofileは有限支持である。
現行成分の支持条件は \(|y-\mu_a|<L_{a,c}\)、Fourier式の基本周期は \(2L_{a,c}\)。
支持外へ周期的な式を展開してはならない。

Cの仕事は「基底の種類 × 階層 × 出力タイル」で区切る。
そのタイル内の全サイトで有効な成分だけを、タイルのCへ集約する。
端はさらに小さいタイルへ割り当てるか、直接計算する。
現行bankに小さいタイルがない場合は、対応する直接経路で全支持を保持する。

```text
出力128サイト
├─ 128サイトのC
├─ 64サイトごとのC
└─ 32サイトごとのC、必要ならさらに小さい領域／直接部分

出力32サイトを所有するblock
  → 自分の範囲と重なる各階層のCを使う
  → 自分の範囲の直接部分を足す
  → 最後にYへ一度保存する
```

各atom成分・各出力サイトの寄与は一度だけ数える。
同じ成分の同じ出力位置を、親タイルと子タイルの両方や、Cと直接部分の両方へ登録しない。
中心が担当出力blockの外でも、支持が重なるatomは拾う。

会話で合意した \(\rho=\sigma/(2\,\mathrm{spacing})\) のspacingはサイト間隔。
このsigmaと現行実装のlive幅I・成分半長Lの対応は、別途明示する。
rhoだけで所属サイト数を断定せず、中心位置と厳密な支持境界も使う。

## 7. 同期の範囲を分ける

| 同期・通知 | 参加する範囲 | 関係のない担当 |
| --- | --- | --- |
| 初期化完了 | block全員 | 最初だけ全員待つ |
| Z作成担当内の同期 | そのZを書く固定thread集団 | 直接計算などを続けられる |
| Zの完成通知・待機 | Zの作成側と、そのZを読む側 | そのZを使わなければ待たない |
| atom群のH完成 | 異なるatomのHを書き、その群を集約する固定集団 | 別のjobを続けられる |
| 最終合算前 | block全員 | 全員の出力部分和を待つ |

指定人数の集団だけを同期させる概念例は次のとおり。
参加者はthread番号から決め、人数を32の倍数へ丸めない。
これは作成担当内の同期であり、別集団への完成通知までは含まない。

```cpp
if (threadIdx.x == 0) {
    for (group : plan.worker_groups)
        init(&prep_bar[group.id], group.thread_count);
}
__syncthreads();

if (job.kind == Z_TILE) {
    build_z_tile(tile);
    prep_bar[ctx.team.id].arrive_and_wait();
} else {
    compute_independent_work();
}
```

実装時には次を満たす。

- 固定集団の全員が同じjob、同じphaseの同期に参加する。
  job取得・継続・終了の判断を代表から全員へ渡し、threadごとにばらばらにしない。
- barrierや集団は分岐に入る前に正しく初期化する。
  集団がjobを完了して空きへ戻るのも、全員の読み書き完了後に揃える。
- 完成通知はZの全書き込みの後、読み手のアクセスは通知の確認後とする。
  通常のboolや`volatile`だけで受け渡しを済ませない。
  `cuda::barrier`のarrive/wait、または集団内同期とrelease/acquireの公開手順で実装する。
- 親Zの作成に必要な子も完成していることをjob取得時に保証する。
  実行可能なjobだけを取得し、未準備jobを持ったままの待機や、依存の循環を作らない。
- 最初はZを最後まで保持し、上書き・再利用しない。
  再利用を入れるときは、親Z生成と全H読者の利用完了も別に通知する。
- `compute_h_batch()`では各threadが別のatomを最後まで計算する。
  出力・Cの集約へ進む前に、そのjobの全Hがsharedへ書かれたことを集団内で確認する。
  一つのatomのHをthread間のshuffle・部分和合算で作る方式は使わない。
  H生成・出力集約・C集約の間で、threadが担当する軸は変わってよい。

参照：NVIDIA公式の[非同期barrierとproducer/consumer](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)、
[Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)。

## 8. メモリの置き場所と寿命

| 値 | 最初の配置案 | いつまで保持するか |
| --- | --- | --- |
| X・Parameter・準備済み係数・norm・所属表 | global。Xの小バッチ部分はsharedへ読む | 既存のforward/backward契約に従う |
| block内で読むX | shared、4 × 128 | そのblockの入力利用完了まで |
| Z | shared、需要と子依存から枠を決める | 最初はblock終了まで |
| Hの計算中の値 | 担当threadのregisterを意図。別threadと入力部分和を合算しない | そのatomのHをsharedへ書くまで |
| job全atomの一時H | jobごとの重ならないshared領域 | jobの直接出力とCへの集約が終わるまで。初期版は固定slotを確保 |
| C部分和 | 担当別のshared領域 | その担当の全atom集約と出力展開が終わるまで |
| 出力部分和 | 集団別のshared領域。各要素の書き手をthread番号で一意にする | `finish_output()`まで |
| 完成通知・barrier・job状態・集団の空き状態 | shared | block終了まで |
| Y | global | forwardの出力として返す |

Cは一つの担当領域への書き込みを一意にするため、初期案では担当別に分ける。
同じ担当内でも各係数をどのthreadが書くかを固定する。
各担当は自分のCだけを展開してから、最終的に出力値として合算する。
途中で担当間のCを共有更新するatomicは、この案では必要ない。
job取得・完了公開などの制御用atomicは別に必要である。

Hはjob内のatom数 × batch_size個をsharedで一時保持する。
各atomを一つのjobへ所属させるため、初期版のjob別固定slotは合計A × batch_size個で収まる。
A128・batch_size4・FP64なら4,096 bytes/block。これはglobal Hの保存量ではなく、
他のshared領域や完全stepのピークを含んだ数値でもない。

`recompute`ではHをglobalに保持しない。`save`・`hybrid`では、選択したatomの
H項目を一意な書き手から別のglobal領域へ保存してbackwardへ渡す。
Hのvalueだけを保存するか、入力中心・幅の微分縮約値も保存するかを区別する。
この保持はforward内のblock間H共有とは別の選択である。W全体は作らない。
ただし準備情報のglobal保存や、block間の再計算費用は残る。
sharedへ置くことはL2常駐の指定ではなく、registerを意図した変数も退避の有無を要確認。

枠数・dtype・部分和の担当数を決めて、初期化・通知・paddingまで含めた容量を数える。
配列容量の計算と、完全stepのallocated/reservedピーク測定は区別する。
共有メモリを増やせば速くなると仮定せず、同時実行できるblock数への影響も測る。

## 9. 実装する順番

一度に全部を書かず、次の単位で確認する。

1. **仕事と所属を決める。** 入力Zの子依存、出力のC所属と直接部分、
   固定作成集団、計算担当を明示する。CPUで支持の被覆・重複なし・境界を確認する。
2. **Z一枚の生成と通知を作る。** `build_z_tile()`と`finish_z_tile()`を実装し、
   別の利用担当が通知後に正しい値を読めることを確認する。
   作成担当内の同期と利用側への通知は別々に確認する。
3. **子Zから親Zを作る。** 継承する周波数とXから読む周波数を独立参照と照合する。
   最初は作成順を固定し、Zのメモリを再利用しない。
4. **atom群のHを並列生成する。** 各threadが別のatomを担当し、
   直接入力とZ経由の両方を確認する。全Hの完成後に群として出力・Cへ集約する。
5. **出力階層のCを作る。** 担当別Cの集約・展開と直接境界を組み合わせ、
   最終のYを独立参照と照合する。
6. **学習経路へ進む。** backwardの再計算・保存方針を実装し、
   独立したY/dX/全atom勾配、支持・正規化、可変幅、retained backward、Graphを検証する。
7. **配分を調べ、規則へ移す。** 集団の動的job取得で階層ごとの発行比率と待ちを調べる。
   準備・所属構築・forward/backward/更新を含む完全stepで既存方式と比較し、
   有効な配分が分かったら、仕事量から人数・順序を求める規則や固定割り当てへ進む。

forwardの試作が通っても、学習経路や速度が検証済みになったとは扱わない。
GPUでの並列実行の有無はtraceで、採用判断は未計装の完全stepと総ピークで確認する。
本書の整理を理由にGPUジョブを自動再開しない。

## 10. 決まったことと、これから決めること

| 決まった方向 | 実装・計測で詰めるもの |
| --- | --- |
| 初期はthread単位の固定集団で、空きを管理してready jobを取る | 集団人数・job発行比率、後の直接割り当てと入口ループの除去 |
| atom群全体を並列対象にし、各threadが別のatomのHを計算する | atom数と担当人数の釣り合い、C・出力縮約内部の高速化 |
| 入力は階層Z、出力は階層・空間・基底別Cを使う | 各階層のタイル幅、直接計算との選択 |
| jobのHはsharedで群の集約に使い、backwardへの保持は3方式から選ぶ | 保存するatom群・項目、保存を省く融合、forward内のblock間共有 |
| backwardのH/Gは同じthreadまたは別集団で計算する方式を選ぶ | 別集団ではsharedで受け渡し、H/G完成後の勾配jobとGだけで進むdX jobを比較 |
| 有限支持、混合後の離散正規化を保つ | sigma・I・Lの表記の対応、厳密な所属算出 |
| 初期案は担当別部分和、最終出力は一意な所有者が保存 | shared容量、dtype、後のbuffer再利用 |
| 数学を保って128から確認する | 実際の速度、GPU並列性、完全stepの総メモリ |

job件数だけで配分を決めず、人数・処理時間・依存待ち・集団の空き時間を分けて記録する。
空き管理は各集団の代表が仕事の受け渡し時に行い、全threadを常時走査しない。
動的な配分で傾向を調べた後、job量から人数・順序を求める規則へ置き換えられる構成にする。
配分の決め方は未定であり、まずは記録を取れることと、取得・完了の正しさを優先する。
空き探索を省く段階でも、計算上の依存完了の確認は残す。
最終的には動的job取得を繰り返す入口へ固定せず、全jobの処理を保証する別の割り当てで
入口の`while`をなくせる可能性も残す。対象はjob管理の繰り返しであり、
計算内部の積和ループや必要な同期を無条件になくす方針ではない。

backwardでは、launch前に集団・atom群・Z作成順・勾配の書き手を決める固定担当表も候補とする。
別集団ならH担当とG担当を固定し、G担当がdX側の処理後にHを受け取って勾配まで計算する。
「Gが軽ければH/Gを同じthreadへまとめる方が有利かもしれない」という仮説も含め、
[backwardの固定担当表](hierarchical-fourier-backward-128-20261023.ja.md)に擬似コードと比較項目を記した。

既存の[Hの寿命を軸にした設計](../../h-lifecycle-kernel-design.ja.md)に従う。
[128で実行済みの保持・受け渡し](hierarchical-fourier-128-execution-20261023.ja.md)は
別のTriton実装の測定であり、本書のCUDA融合案の実測ではない。
[局所処理のCUDA読み合わせ例](examples/output_local_walkthrough.cu)も、
局所経路の説明用で、今回のZ/H/C通知方式を実装したものではない。
