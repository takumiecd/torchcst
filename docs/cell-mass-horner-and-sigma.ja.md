# 係数と微分のHorner評価・sigma分布の確認

2026-10-27、Issue #120 / Draft PR #121。runtime source `7c1404c1`。
数学契約・精度・sigma分布を変えず、1024の完全layer stepを約22%短縮した。
allocated/reservedピークは増えず、workspaceなしdense以下を維持する。
通常denseの約3倍の時間なので、約2倍の目標はまだ未達。

## sigmaは全atomで一定ではない

測定fixtureは `sigma = torch.linspace(0.2, 12.0, 52428)`。
乱数からの標本でも実際の学習モデルから取り出した分布でもなく、等間隔の初期値である。
全52,428個は初期状態で異なり、平均6.10、中央値6.10、四分位点3.15/9.05。

| 初期sigma範囲 | atom数 | 割合 |
|---|---:|---:|
| [0.2, 0.25) | 223 | 0.43% |
| [0.25, 1) | 3,332 | 6.36% |
| [1, 4) | 13,329 | 25.42% |
| [4, 8) | 17,772 | 33.90% |
| [8, 12] | 17,772 | 33.90% |

sigmaもAdamWで更新する。warmup10と測定30step後のHorner run0では
min/maxは0.19754882/12.00350475、四分位点3.149783/6.098838/9.049680。
sigma全ベクトルを測定後にCPUへ回収し、集計は時間・ピーク測定の外で行った。
min/maxだけを保存した過去runから、過去の全分布を復元したとは扱わない。

### 幅パラメータと実際の集中半径を区別する

現profileのsigmaは標準偏差や支持半径そのものではない。
`src/torchcst/_backends/torch/profiles/fourier.py`の`effective_radius`と
CUDAの`make_profile`は、Mとtail_boundから集中半径の下限を設ける。
profile周期P、M=8、tail_bound epsilon=1e-3、t=P/3として、

\[
a=\frac{\operatorname{acosh}(\epsilon^{-1/2})}{M},\quad
r_{\min}=\frac{P}{\pi}\arccos(1/\cosh a),\quad
q=\left(\frac{r_{\min}}t\right)^4,
\]
\[
r(\sigma)=t\left[q+(1-q)(\sigma/t)^4\right]^{1/4}.
\]

| 軸 | profile周期P | 初期sigmaによる集中半径の範囲 |
|---|---:|---:|
| 入力 | 48 | 7.5872 .. 12.3151 |
| 出力 | 96 | 15.1744 .. 16.4209 |

出力側の幅はかなり近い。sigmaが広く分布することだけを根拠に、非常に局所的な
profileが多い場合も検証済みとは言えない。spacing0.125なのでrho=sigma/(2*spacing)は
0.8..48だが、今回のprofileでrho<1を「単一siteだけ」と読み替えない。
集中半径と、whole-tile窓の範囲と、厳密なcompact支持も別の量である。

現在は全atomが1軸1bank、同じM8・K3・tileサイズを使う。
sigmaに応じた複数階層への分類、小sigma直接計算はまだ実装していない。
今回の高速化は同じprofileを評価する算術の改善であり、その分類の代替ではない。
幅の意味や分類を変更する場合は別の数学・測定条件として扱う。

## 何を軽くしたか

新しい研究設定 `coefficient_method="polynomial_horner"` を追加した。
従来の正の多項式と同じFourier係数・幅微分を、Horner法で一緒に評価する。
既定値`polynomial`と、対照の`polynomial_recurrence`は変更しない。

\(x=\beta^2,\ t=x-1,\ n=M-k\)として、k<Mでは

\[
P_k(t)=\sum_{j=1}^{n}c_jt^j,\qquad
c_j=\binom{M-k-1}{j-1}\binom{M+k+j-1}{j}.
\]

共通の正規化前因子を消去したraw係数は

\[
r_k=\frac12\left[\frac{M}{M+k}x^kP_k(t)+\mathbf1_{k=0}\right],
\qquad r_M=\frac14x^M.
\]

以前は隣の項を作って値と微分へ加算した。新方式は定数c_jを読み、
PとPのt微分を次のように同じループで求める。

```cpp
value = c[n];
slope = 0;
for (int j = n - 1; j >= 1; --j) {
    slope = slope * t + value;
    value = value * t + c[j];
}
slope = slope * t + value;
value = value * t;             // 定数項は0
// slope = dP/dt。prefixの微分と dt/dsigma を掛けて幅微分を返す。
```

`horner_coefficient`がこの処理、`raw_coefficient`が方式選択を担当する。
`project_atom`と`cache_expansion_coefficients`は、harmonicを進める際に
scalarのx^kを一回ずつ掛けて更新する。M8では各kでの累乗作り直し計36回を8回へ減らす。
phase漸化式・面積P*c0による正規化・セル積分・中心微分・窓の選び方は同じ。
全内部演算とZの保存精度はdoubleを維持し、許容差も変更しない。

## 何をどこに保持するか

- `HornerBanks`: M2,4,...,16の整数係数444個をdoubleのCUDA constant memoryに置く。
  追加の論理容量3,552bytes。元の表9,864bytesと合わせ13,416bytes。
  cuobjdumpでもCONSTANT[3]=13,416を確認。atom数に比例する表ではなく、
  Torch allocatorピーク外のmodule定数なので別記する。
- thread register: beta、t、value/slope、現在のx^kをscalarで持つ。
  全modeの配列や全atomのFourier係数/Jacobianをglobalへ保存しない。
- shared/globalの配列と寿命: [C_in融合](cell-mass-cin-fusion.ja.md)の通り。
  32atom chunk、C_in部分和544KiB、C_out部分和272KiBを維持し、G表は作らない。
  今回の変更によるatomコピーや新しいTensor bufferはない。

定数生成はint64の漸化式で行う。全444係数をPythonの厳密整数と照合し、
最大係数23,728,968,900、生成中の最大整数3,132,223,894,800が各型の範囲内であることを確認した。

## 完全stepと診断

N1024/A52428/B4、FP32、Q128、spacing0.125、input tile128/output tile256、
M8、K3、tail_bound1e-3、seed120。前回と同じfixture、検査・sort込み、
固定X/dYのforward/backward/fused AdamW。Graph/TF32なし、JIT/oracle/初期W構築は測定外。
L4 / Torch2.11.0+cu130 / CUDA13.0。2fresh process、2巡目は逆順、warmup10/同期wall30標本。

| 同じjobの比較 | step ms run0 / run1 | peak allocated MiB | peak reserved MiB |
|---|---:|---:|---:|
| 従来recurrence・C_in融合 | 1.7401 / 1.7315 | 4.260742 | 6 |
| Horner・C_in融合 | **1.3536 / 1.3540** | **4.260742** | **6** |
| dense通常 | 0.4459 / 0.4491 | 32.312988 | 42 |
| dense workspaceなし | 0.6384 / 0.6330 | 16.062988 | 22 |

完全stepを22.21% / 21.80%短縮。通常denseとの時間比は3.036 / 3.015で目標未達。
時間は通常dense、メモリはworkspaceなしdenseという別の比較点。
workspaceなしは`:0:0` / cuBLASLt workspace0。両allocator指標ともdense以下だが、
CUDA Graph capture/replayと全GPU processメモリは未測定。

以下は別のevent診断。確保を含むnative呼び出し単位であり、和や差を完全stepの代用にしない。

| 処理 | recurrence ms run0 / run1 | Horner ms run0 / run1 |
|---|---:|---:|
| prepare | 0.1618 / 0.1597 | 0.1567 / 0.1588 |
| H生成 | 0.1731 / 0.1761 | 0.1321 / 0.1351 |
| C_out生成 | 0.2181 / 0.2191 | 0.1731 / 0.1741 |
| G/Parameter勾配のみ（対照） | 0.7188 / 0.7404 | 0.4311 / 0.4321 |
| G/Parameter勾配/C_in融合 | 0.8458 / 0.8458 | 0.5489 / 0.5478 |
| 融合＋dX展開pair | 0.9021 / 0.9041 | 0.6154 / 0.6082 |
| dX展開のみ | 0.0696 / 0.0696 | 0.0686 / 0.0696 |

次の主対象はなお融合producer。算術削減によりここも約35%短くなったが、
prepare、C_out、H生成も残る。profileの幅分布を変えて速く見せる比較は行っていない。

研究設定は[実測状態](cell-mass-status.ja.md)の1024設定に
`coefficient_method="polynomial_horner"`を加える。公開dispatcherの採用判断とは別。

## 正しさと保全

CUDA・数学 **1816 passed**（45.50秒）、失敗/skipなし。
CPU全体 **2006 passed / 4564 skipped / 既存警告18件**（45.47秒）。
Hornerを既存のM2..16、FP32/64、DFT参照、投影微分、C_in融合の検査へ追加した。
別のCPU式検査では正規化係数とbeta微分をChebyshev密度のDFTと比較し、
最大差1.01e-14 / 4.98e-13以下だった。

全52,428atomの独立CPU float64 oracleへ、対照と候補をそれぞれ照合した。
候補のY/dX最大差は各1.49e-8、全4列Parameter勾配2.91e-11。
同じGPU fused AdamWの更新delta/step差0、moments差3.64e-12/2.78e-17。
Ruff、clang-format、差分検査、wheel/sdist buildも通過。
321個のtracked runtime sourceはsnapshotとgit revisionをbyte照合。
両packageはsrc内の330ファイル（README 9個を含む）を現runtimeへbyte照合した。
128は既存のCUDA回帰を含むが、この回に128の完全stepを再測定していない。

job `l4job-c038c89feb9146b48e2d07a9fc626ebc`:

- source SHA256 `3fb31813a3696fcbc6e62f07daca4f727c0d86e026c27a09ca7706fefbe87c1b`
- result SHA256 `997377d58bfad4f38ae0dd0db71360b651aaa874777938b0a97fea22eca0e6e5`
- driver SHA256 `ae7c99a92780204f464f29e0d0c6407559cc3f3bbcd1e49a822f925f24e2dc18`
- binary SHA256 `858d269cf560df9a756a417fb568af1be331a7be6ba80f23fecbb6a037fb97f5`

62 manifest files、driver、binaryを照合。raw/source/result/spec/receiptと検証結果は
ignored `output/cell-mass-horner-20261027/verified/`、分布と図はその親directoryへ保存。
共有L4 poolの所有runtime停止も確認した。
再現は固定source archiveの`__pool_driver__.py`にsrc/rootのPYTHONPATHと
空directoryの`CST_JOB_OUTPUT`を渡す。共有L4ではpool submit/serveを使う。
GPU検査は`python -m pytest tests/test_cell_mass_coefficients.py tests/test_cell_mass_reference.py tests/test_cell_mass_cuda.py -q`。
