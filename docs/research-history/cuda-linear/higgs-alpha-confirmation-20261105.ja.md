# HIGGS低密度CST: 長期学習・3 seed確認

## 実行前の条件

N1024/2048でalphaを0.001〜0.01へ下げ、調整したdenseに対して
品質を保持する最小の検証点と不確実性を調べる。
2026-11-05、所有者はaccuracyの許容差−0.2 percentage pointと
ROC-AUCの許容差−0.002を明示的に承認した。結果を見て許容差を変更しない。

モデルは28→N→N→2、ReLU×2。中間層だけをCSTまたはdenseにする。
入力・出力の種類はdenseだが、**全3層を学習**する。中間biasは両方式で無し。
alphaは{0.001,0.002,0.005,0.01}。整数divisor{1000,500,200,100}を使う。
確認用結果を見た後に別の密度を追加する場合は探索扱いとし、今回の正式判定へ混ぜない。

$$
A=\lfloor\alpha N^2\rfloor,\qquad
P_{\mathrm{dense}}=N^2+31N+2,\qquad
P_{\mathrm{CST}}=4A+31N+2.
$$

半径セル積分M8・spacing1・physical radius[0.5,50]、public初期化、
AdamW proposalとCST Polar更新を前回から維持する。
denseは標準初期化。同一N/seedで入力・出力層の初期値とデータ順を共有。
真batch4、FP32、TF32 off、CPU math1 thread、L4×1。
alphaはPolar内部のactivity座標とは別のatom密度を表す。

## データの分離

前回検証済みの公式HIGGSを再利用する。学習は先頭100,000行。
標準化統計は全training10,400,000行だけから計算したものを維持する。
公式validation100,000行を次のように分ける（zero-based半開区間）。

| 用途 | validation内index | 元CSV行index | 使用方法 |
|---|---|---|---|
| development | [0,50000) | [10400000,10450000) | lrとcheckpoint選択 |
| confirmation | [50000,100000) | [10450000,10500000) | 全選択凍結後に一度だけ評価 |
| official test | — | [10500000,11000000) | 今回は評価せず保持 |

前回使用した先頭8,192 validation例は全てdevelopment側に入る。
調整seedは17。確認seedは**29,43,71**。
confirmationのaccuracy/AUCをlr・予算・checkpoint選択に使わない。
NPZのdecompressはvalidation全配列に及ぶが、training/selectionの参照sliceはdevelopmentだけ。

## 学習率と学習予算

各Nのdenseと、各(N,alpha)のCSTをそれぞれ同じ候補集合
lr={0.0001,0.0003,0.001,0.003}で比較する。
普通層/dense WD0.01、atom raw WD0、logical amplitude WD0.01等は変更しない。
各候補を**10 epoch**学習し、各epochのdevelopment全50,000例をnative B4で評価する。
checkpointはdevelopment cross-entropyが最小のepoch、完全tieなら早いepoch。
lrもbest-development lossで選び、tieなら小さいlrを優先する。

最良lrが候補集合の端なら、その方向へ**1回だけ**追加する
（下端0.0001なら0.00003、上端0.003なら0.01）。追加点も同じ10epoch。
追加点がなお最良ならedgeフラグを残す。失敗や非有限値を成功結果へ置き換えない。

全5構成（dense＋4密度）の調整後、選択された候補のいずれかが
best epoch>=9かつepoch8 loss−epoch10 loss>0.001なら、
そのNの確認用学習予算を**全構成共通で20epoch**へ延長する。
この場合、各構成の10epoch上位2lrを20epochで再比較してから凍結する。
20epochでも改善が続く場合は未収束フラグを残す。
Nごとの最終予算T_Nはconfirmationを開く前に確定する。

各確認seedでdense＋4密度を同じT_Nだけ学習し、checkpointをdevelopmentだけで選択する。
**30個のcheckpoint、lr、予算、データSHAを全部凍結してから**
confirmation50,000例を評価し、logitsとlabelsを元の順序で保存する。

## 非劣性判定

各(N,alpha)、seed s、指標m（accuracy/ROC-AUC）についてpaired差を取る。

$$
\Delta_s^{(m)}=Q_{\mathrm{CST},s}^{(m)}-Q_{\mathrm{dense},s}^{(m)},\qquad
\epsilon_{\mathrm{acc}}=\epsilon_{\mathrm{AUC}}=0.002.
$$

accuracy0.002は**0.2ポイント**であり、相対差ではない。
主判定は両指標が同時に非劣性を示すintersection-union test（IUT）。
3つのseed差に対して正規IID近似のpaired t（df=2）を使う。

$$
z_m=\frac{\bar\Delta_m+\epsilon_m}{s_m/\sqrt3},\qquad
p_m=1-F_{t_2}(z_m),\qquad
p_{\mathrm{config}}=\max(p_{\mathrm{acc}},p_{\mathrm{AUC}}).
$$

2幅×4密度の**8構成**にBonferroniを使い、
8*p_config<=0.05ならその構成の非劣性を支持する。
両指標を必ず同時に通すIUTなので、さらに指標数2を掛けない。
SE=0の場合、mean+margin>0ならp=0、それ以外はp=1。
境界mean=-marginのゼロ分散を自動合格にはしない。

次は主判定と分けて診断として残す。

- 16 endpointのtwo-sided同時95% seed-t Bonferroni band（critical約17.8466）。
  いずれかの上限が−marginより小さい場合は悪化を支持する。
- 各指標のpointwise one-sided95% seed-t lower bound。
- 学習済み6モデルを固定したpaired-example bootstrap95%区間。
  両方式・全3seedで同じexample再標本化weightを使い、AUCのtieを正確に扱う。
  2,000 draw、乱数seed20261105。seedとexampleの両方に対するjoint95%とはしない。
- 3seed全ての観測差が許容差内か。これは候補の指標で、正式な非劣性判定とは分ける。

**3seedでは小さい差への検出力が低く、正規近似の妥当性も検証できない。**
「非劣性を支持」「悪化を支持」「判定保留」を区別する。
主IUTが支持しても、より広い診断用two-sided bandの下限は許容差を跨ぎ得る。
診断bandを主IUTの閾値に取り違えない。

最小supported tested alphaを報告するが、小さい点がinconclusiveなら境界未確定。
0.001が支持された場合も**今回の範囲で最小、必要密度は0.001以下の可能性**とする。
品質のalpha単調性や普遍的なCST最適alphaは仮定しない。

統計の根拠:
[NIST t confidence interval](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)、
[NIST Bonferroni](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm)、
[all co-primary endpointsのIUT原理](https://www.fda.gov/media/162416/download?attachment=)。

## 資格・実行・保存

低密度専用tag/private Graph入口を追加し、旧MNIST・旧HIGGSのcohortは維持する。
8個の(N,alpha)でY/dX/全dPをpublic CPU FP64と比較し、独立quadratureも使う。
両幅・最小/最大密度・CST/denseのeager/Graphを2epoch×16更新し、
Parameter/optimizer/gradient/clockを実Tensorのbyteで比較する。
短い資格試験だけdev32例を許可。本番HPO/確認ではdev50,000例を必須にする。

L4共有pool1台を使い、stageごとに結果を回収する。
同じcommitの検証済みimmutable source cacheでCUDA compileを共有する。
campaign stateはsource/data/runtimeに結び付けたhash付きrecovery archiveに保存する。
poolのDB/slot記録は直接変更せず、他のjobやVMを停止しない。
source freeze後は算術や判定を途中変更しない。資格・source identity不一致は停止する。

各epochのcurveとstate/gradient hash、best-dev/finalの実checkpoint、
lr全候補、選択規則、confirmation predictions、seed順、データ順SHA、
同期完全step時間、準備込み時間、allocated/reserved peakを保持する。
GPU process全使用量とallocator peakは別に扱う。
raw証拠はignored output/higgs-alpha-confirmation-20261105/とpool job directoryへ保存する。

この文書は実行前のprotocol。結果・完了・採用を意味しない。
