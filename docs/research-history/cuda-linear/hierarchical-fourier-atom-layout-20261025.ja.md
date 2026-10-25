# 階層Fourier：atom全体のソートとglobal配置の設計案

2026-10-25。[ライブラリ全体の状態契約](../../atom-state.ja.md)に従う方式固有の設計案。
本書は配置の検討と合意した処理方針を記録する。2026-10-26にソート付き128版を追加した。
実装した配列・関数・保持量と検証は[プログラムの読み方](hierarchical-fourier-block128-reading-20261026.ja.md)に分ける。
多次元・周期格子・差分更新等の汎用案は、全てが実装済みという意味ではない。
[現在のblock128実装](hierarchical-fourier-block128-cuda-20261024.ja.md)は比較基準として残す。

## 決定済みの方針と、これから決める値

- まず子タイル2枚を親へマージする階層を考える。各軸でT_h = 2^h T_min。
- atomは全体を一つの順列でソートする。境界グループ内では出力側の中心から得たタイル座標、
  次に入力側のタイル座標を優先する。blockごとに別々の本体を保存することは要求しない。
- 入力・出力はそれぞれ複数座標軸を持てる。product profileの支持は各軸の区間の直積で表す。
- T_min、周期2Lが何タイル分か、blockの担当幅Q_O、小バッチ幅B_tの具体値は未確定。
  階層タイル幅T_hとblock担当幅Q_Oを同じものとして固定しない。
- 支持・周期・混合の数学的な意味はSpec側にある。deviceに合わせて配置を変えても
  同じWを保つ。数学的なLの選択則を変える場合は別の宣言・意味の版として扱う。

単位を明示する。T_hは物理的な幅、軸dのsite間隔をs_dとし、タイル内site数は
t_{h,d}=T_{h,d}/s_dという整数にする。Q_Oはblockが担当する出力site数。
会話で検討した周期とタイル幅の関係は、次の設計条件である。

\[
2L_h=\left(\left\lceil\frac{2I}{T_h}\right\rceil+1\right)T_h,
\qquad m_h=\frac{2L_h}{T_h}.
\]

m_hやL_h=f(T_h)の具体形はまだ選んでいない。この式だけで既存Specの周期・混合を
置き換えない。階層によって分かる枚数の上限と、中心位置によって変わる実際の枚数も区別する。

既存block128は入力・出力各1次元の有限な中心支持を扱い、端でclipする。
各軸をwrapする周期格子は[FlatTorusの契約](../../periodic-grid.ja.md)に従う別の対応対象。
本書の周期所属表現だけで、既存CUDAに周期評価を追加したことにはならない。
空間の周期P_dと、Fourier基底の周期2Lを区別する。

## 座標とタイル番号：横に読んでから次の行へ進む

次の配置なら、A, B, C, D, E, Fと横に読んでから次の行へ進む。

```text
A B C
D E F
```

既存Chartと同じrow-majorをカーネル内部でも使う。
最後の座標軸の添字を先に進め、その軸を数え終えたら一つ外の軸を進める。
軸dの要素数をN_d、添字をx_dとして、

\[
\operatorname{flat}(\mathbf x;\mathbf N)
 =\sum_{d=0}^{D-1}x_d\prod_{e=d+1}^{D-1}N_e.
\]

空積は1。掛けるN_dは最大添字ではなく要素数であり、最大添字+1。
2軸ならx_0 N_1+x_1、3軸ならx_0 N_1 N_2+x_1 N_2+x_2。
例えば入力shapeが(8,16)なら入力は128 sitesで、(5,9)の番号は5×16+9=89。
入力側と出力側で独立にこの式を使い、番号をi、oとする。
論理行列の向きはW[out,in]のままで、行優先の行列添字は

\[
\operatorname{address}(o,i)=oN_I+i,
\qquad N_I=\prod_dN^{\mathrm{in}}_d.
\]

これはCST重みを密な配列として保存する指示ではなく、論理添字の対応である。
内部番号と公開site番号を一致させ、batchを最外側に置く連続X/Yなら、
Xの添字はbN_I+i、YはbN_O+o。
座標の一次元化でEuclidean/FlatTorus等のmetricやperiodic支持を変更しない。

siteと同じ規則でタイル番号も作る。ただしstrideに使うのはsite数ではなくタイル数。
軸dのタイル内site数をt_dとするとM_d=ceil(N_d/t_d)で、

\[
tile\_id=\operatorname{flat}(\mathbf q;\mathbf M).
\]

例えば(8,16) sitesを(4,4)で区切ればタイル格子は(2,4)で、(1,2)のタイル番号は1×4+2=6。
siteの番号、タイルの番号、タイル内の局所番号は別々にこの式を適用する。
タイル番号を作っただけでX全体がtile-majorにpackされたとはしない。
X/Yのglobal配置と、atomのソート順・sharedでの配置をそれぞれ明示する。

[既存のRegularGrid](../../regular-grid.ja.md)と[PeriodicGrid](../../periodic-grid.ja.md)は、
このrow-majorを契約としている。今回も同じ軸順・site順で扱い、
入力/出力の座標軸を逆順へ振り直す変換は追加しない。
この配列の読み順は、W[out,in]とW[in,out]を入れ替える転置とは別の規則である。

## ソートkey：境界グループ、出力タイル座標、入力タイル座標

slotは並べ替え後の行、IDは物理行と独立したatomの識別子とする。
境界グループgは、live幅Iと数学宣言・実行planから算出する分類である。
具体的な境界値はT_min等を決めた後に確定する。

一つのatomに混合成分が複数ある場合、atomを複製して別threadへ分配しない。
gは必要な成分の階層・bankの組を区別できる分類にし、各threadは担当atomの成分をまとめて処理する。
値または必要な微分へ寄与する成分を対象にし、振幅0を理由にatomやdAmpを削除しない。
Hは入力側の必要成分を合算し、出力側の必要成分に使う。入力と出力で同じ成分番号の項だけを
掛けて、混合profileの交差項を落としてはいけない。

出力座標数をd_O、入力座標数をd_Iとして、中心を各軸のタイル幅で量子化する。
共通の階層hを用いる場合も、最小幅は軸ごとに持つ。

\[
T_{h,d}=2^hT_{\min,d},\qquad
q_{a,d}=Q\!\left(\frac{\mu_{a,d}-o_d}{T_{h,d}}\right).
\]

o_dは軸の原点で、入力数・出力数ではない。Qはplanで定める量子化関数であり、
以下の範囲探索例はfloorを使う。周期軸の中心は原点基準の代表範囲へwrapしてから量子化する。
各側のタイル座標を上のrow-majorの規則で一次元化し、

\[
\begin{aligned}
o^{tile}_a&=\operatorname{flat}(\mathbf q_{a,\mathrm{out}};\mathbf M_{h,\mathrm{out}}),\\
i^{tile}_a&=\operatorname{flat}(\mathbf q_{a,\mathrm{in}};\mathbf M_{h,\mathrm{in}}),\\
\boxed{key(a)}&=\boxed{\big(g_a,o^{tile}_a,i^{tile}_a\big)}.
\end{aligned}
\]

の三つ組を辞書式に比較する。同じgの固定shape内では、元の(q_0,q_1,...)の
辞書順と、この一次元タイル番号の順序は一致する。
gは階層hと、必要なら周期/bankの分類を含む。
IDはソートkeyに必須ではなく、同値の順序を決定的にしたい方式だけで追加する。
どちらでも元のatomへのID・行対応は保持する。
支持端は中心のkeyとは別に保ち、量子化されたタイル全域を非ゼロとみなさない。
支持枚数は所属情報であり、位置より先にソートするkeyにしない。

例えば1次元ずつの模式的なkeyなら、次のように全体をソートする。

| atom ID | g | 出力中心のタイル番号 | 入力中心のタイル番号 |
| --- | ---: | ---: | ---: |
| 9 | 1 | 0 | 0 |
| 3 | 0 | 1 | 0 |
| 7 | 0 | 0 | 2 |
| 1 | 0 | 0 | 1 |

結果はID順列 [1, 7, 3, 9]。出力block別のcontainerを作る指定ではない。
タイル座標は各軸で算出し、行列のflatten添字の単一区間から支持を推定しない。

## 各軸の支持：周期をまたいでもstart/count

所属情報の基本を、各成分・各座標軸の有効なsiteのstart/countとする。
これから開始タイル、またがる枚数、端の有効部分を算出できる。
タイルのstart/countだけでは、端のタイル内の有効siteまでは分からない。

周期軸のsite数をN_dとすれば、短い連続支持を

\[
j_r=(start_d+r)\bmod N_d,\qquad 0\le r<count_d
\]

で表せる。開始位置は周期上の区間開始であり、wrap後の番号の最小値ではない。
例えば16タイル上のstart=15、count=3なら15, 0, 1。
実際のloadは末尾側と先頭側の連続区間へ分けられる。
全周に支持がある軸は全域を一度ずつ評価し、周期コピーの重複加算をしない。
wrapの両側が同じタイルに入る場合も、有効部分を扱い、同じsiteを重複加算しない。
値と微分は宣言された最短周期差等の規則で評価する。

## globalに持つ配列：汎用案

まず方式専用のソート済みコピーを使い、正本は任意順のままとする。
以下は汎用案の候補。実装した128版ではsorted_P・元行対応・atomのgroup・rangesを持ち、
支持端等は現在値から評価する。[実装配列の表](hierarchical-fourier-block128-reading-20261026.ja.md)を正本とする。
Aはatom数、Pは座標Parameter幅、Sは成分枠数、D=d_O+d_I。
block128の現在の数学ではP=4、S=4、D=2。出力係数Cと成分枠数Sを区別する。

| 配列・情報 | 形・所在 | 用途・寿命 |
| --- | --- | --- |
| 正本P | [A,P]、AtomState／呼出側global | 学習情報。初期順の前提なし |
| sorted_P | [A,P]、global | key順の実行用コピー。値の更新に追従 |
| slot_atom_id | [A]、global、int64 | slotから不変IDへの対応 |
| slot_source_row | [A]、global、int64 | 正本行への対応。gatherと勾配のscatterに利用 |
| group_offsets | [G+1]、global、整数 | 幅・階層分類gごとの連続区間。出力block別の表ではない |
| group_descriptor | G件のmetadata | gの分類・基底・経路の解釈。kernelが読む部分だけglobalへ置く |
| support_start / support_count | 各[A,S,D]、global、整数 | 成分・軸ごとのsite支持。端の処理と周期wrapにも使う |
| fit | 現行[32,9] FP64、global | 形状の固定係数表。現在のblock128と同じ意味の範囲で利用 |
| norm | 現行[A,2,3] FP64、global | 混合後の実site上のnormと微分。forward値に対応させる |

slot_source_rowは順列なので、slotで計算したdPを元の各行へ一度ずつ戻せる。
backwardは元のParameter・入力の変更検査と呼出し固有の保存状態を保つ。
sorted_Pだけを保存して、forward後の元Parameterの変更検査を失ってはいけない。

明示的な逆順列は最初の必須配列にしない。sort key、sort workspace、変更検出用の
一時配列の確保量と寿命は実装時に数える。成分情報の圧縮や配列の共有は後で検討する。
全体の所属情報はO(ASD)で、密なWや全siteのPhiを作ることは要求しない。

正本そのものを並べ替える別案ならAtomState.relayoutを使い、gradient・momentも追従させる。
正本と実行用コピーの二方式を曖昧に混ぜない。slotの変更だけでは正本のlayout_versionを進めない。

## H・Z・Cとblockへの割り当て

局所経路ではZ/Cと利用するHをblock内に保持し、Hをglobalへ保存することは要求しない。
広域Hの保存は明示的に選んだ経路だけに追加し、対象atom・batch・寿命を定める。
H/ZはXに依存する。各入力からの準備と、backwardでの保存／再計算を区別する。

blockへの渡し方はソート済み配列を使う実行手順として後で定める。
最初からregion_atom_refs等のblock別配列をglobal配置の必須条件にしない。
中心順でも、担当領域の外や周期末尾に中心があるatomを含めて探索する必要がある。
get_rangeはグループの支持半幅上限を使い、必要な全寄与を含む保守的なbegin/endを返す。
この範囲をそのままsharedへ読み、ゼロ寄与を含めて一括計算する。
最初の案では、範囲取得後のatomごとの追加選別・compactionは行わない。
直接評価の支持外ゼロやタイル表現の有効範囲・境界残差は、元の数学に従って扱う。
周期端は最大二つの範囲、全周なら全体を一度だけ取る。これは出力1次元の範囲探索であり、
出力多次元の候補箱では複数の連続範囲が必要になる。

## block内の見取り図：Z → H → C → Y

2026-10-26、この「入力タイル → atom → 出力タイル」を基本の処理方式として採用する。
Z・H・Cを別の役割として実装し、最初はHとCを同じblockの段階処理でつなぐ。
関数を分けることと、kernel起動を分けることは区別する。

一つの出力担当と、現在処理するatom集団を取り出した後の関係を示す。
実際の保持場所は経路ごとに選ぶ。この図だけで全Z・全Hを同時にsharedへ置くとはしない。

```mermaid
flowchart LR
    X["入力タイル X"] --> Z["Z：タイル・基底ごとの入力要約"]
    Z --> H["H：atomごとの完成した入力縮約"]
    X -->|直接・境界の入力寄与| H
    H --> C["C：出力タイル・基底ごとの係数"]
    C --> Y["担当する出力 Y"]
    H -->|直接・境界の出力寄与| Y
```

| 段階 | 担当の単位 | 途中値と完了条件 |
| --- | --- | --- |
| Z生成・親子マージ | 必要なタイル・bank・基底・batch | 必要な子と同一周波数のZだけ継承し、未対応の周波数は別に生成 |
| H縮約 | 各threadが別atomを担当 | 必要な入力タイルの寄与をthread専用値へ累積。全支持の寄与が揃ってHが完成 |
| Hの公開 | 現在のatom集団 | 完成したHをsharedへ置き、利用者との同期を取る。別atomのHを一つへ合算しない |
| C集約 | 出力タイル・bank・基底・batchの各C要素 | 各要素の書き手を一意にし、全対象atomのHと出力係数の積を加算 |
| Y保存 | 小バッチ・担当出力site | Cの展開と直接部分を累積し、各Y要素を一意な書き手が保存 |

同期は完了・可視性を保証する手順であり、加算そのものではない。
書き手を上表のように分ければblock内の計算にatomicAddを要求しない。
異なるbankのCは分けて保持し、その基底で出力へ展開する。

atom集団を入れ替える場合の短い骨組みは次の通り。全員が必要な同期へ到達する。

```cpp
Cと出力累積を初期化();
for (atom集団 : get_rangeで取った範囲) {
    入力タイルと必要なZを準備しながら、各threadのHへ累積();
    完成したHをsharedへ書く();
    sync();
    Cと直接出力へ寄与を足す();
    sync();  // H領域を次の集団で上書きする前
}
Cを展開して担当Yを保存();
```

Zの階層依存とCの支持条件は既存の数式に従う。この骨組みは、全階層の準備を
毎atom集団で必ず作り直すという指定ではない。準備したZの再利用範囲・寿命は別に定める。

### 現在のCUDAとの対応

block128は既にこの段階構造で計算している。2026-10-26の整理では、入口にあった
atom縮約をcompute_atom_responsesへ切り出し、段階と書き手を明示する。
演算式・支持・正規化・同期位置・shared配置・起動数を変更するものではない。

| 方針上の処理 | 現在のCUDA |
| --- | --- |
| 入力タイルからZを作る | build_z_tile。親子依存ごとにblock同期 |
| atomごとのH/Gを完成させる | compute_atom_responses → contract_component。各threadが別atom |
| H/Gを同じblockのCへ渡す | sharedのvaluesへ保存してblock同期 |
| 出力タイルのCを作る | build_c。各C要素の書き手は一意 |
| Y/dXを保存する | store_output。Cの展開と直接/境界部分を加算 |

dX側はdY → 出力側Z → G → 入力側C → dXで対応する。
Gは未スケールで、dH = alpha × G。入力側Cを元の出力側Cの勾配と混同しない。
atom勾配の担当はHとGおよび必要な微分項目を使う。

全128 atomを処理する比較基準に加え、別Recipeのソート版へsort key・global pack・
get_rangeによる範囲取得を追加した。比較基準は残す。
下記5 passedの記録は、その追加前のH/G関数分離に対する確認である。
確認したのは、移動したH/G計算本体のsource一致、同期位置の維持、専用テストの
CPU部分5 passed / CUDA依存6 skipped、既存Plan/Caseの宣言検査PASS、文書リンク。
この変更後のCUDAコンパイル・実機検証は未実施。

### Hの容量とthread配分の未確定事項

forwardのH値の要素数はB_t × A_t。B_tは小バッチ幅、A_tは現在のatom集団の大きさ。
FP64なら8 B_t A_t bytesで、B_t=4、A_t=128は4,096 bytes。
全8,192 atomを一度に持てば262,144 bytesだが、集団ごとにHを使い切れば
その全量をsharedへ置く必要はない。Z/C・入力・出力累積・sort workspace等の容量は別に数える。
backwardでHの中心/幅微分も保持する場合は、必要なfield数をさらに掛ける。

「大きいsigmaでは階層タイルの再利用で計算が軽くなり、多くのthreadを割り当てなくても
済むのではないか」を検討する仮説として残す。小さいsigma側への増員を前提にはしない。
sigmaだけで割合を固定しない。各群のatom数、小バッチ数、
必要なZ・基底数、直接/境界site数、Cの縮約量、他blockとの再利用も配分に関わる。
今回は具体的な割合や速度の結論を置かず、まず各段階の処理量と保持量を紙上で数える。

## 配置情報の版と更新

実行状態のmetadataには、Algorithm ID/revision・semantics、方式固有の配置形式ID/版、
planの内容、座標shape/spacing/周期・stride規則、device/dtype、元の配置世代を対応付ける。
これらは実行情報であり、別deviceで復元が必須なモデルの学習情報ではない。
既存のAlgorithmStateだけで以下の数値更新を自動追跡できると仮定しない。

```text
初回／方式・device変更：分類 → key生成 → 全体sort → pack・支持情報の準備
optimizer更新後：現在値から分類・key・支持端を再判定
    keyが不変：順列を再利用し、変わった値と支持情報を更新
    keyが変化：最初は全体sortをやり直す
forward：現在のPとXに対応するnorm・H/Z等を生成
backward：このforwardのslot対応・保存値で計算し、正本行へ勾配を戻す
```

最初はeagerの安全なstep境界で更新する。未完了／retained backward中の正本relayoutや、
保存中のslot対応・数値をworkspace更新で壊すことは許さない。
差分sort・変更部分だけの転送・CUDA Graph内更新の実装や性能は未検証。

汎用版のT_min等を含むplanの調整、差分sort、多次元と周期格子の実装は今後の対象。
現在の128版の実装と検証範囲は[プログラムの読み方](hierarchical-fourier-block128-reading-20261026.ja.md)に記録する。
