# 1024×1024・5% atoms の有限 Fourier 比較

2026-10-14。前の [N300・A1024 の実測](fourier-shared-tiles-triton.ja.md)と区別する。
今回は dense の自由な重み1048576個に対し、CST は
`int(0.05 * 1024 * 1024) = 52428` atoms、209712個の学習可能な atom coordinates。
batch は32、TF32/autocast は使わない。

## 合意した格子

ユーザーが計算・数学の整合性を優先して選んだ等幅tile配置を使う。

| 項目 | 値 |
| --- | ---: |
| 入力／出力 sites | 1024／1024 |
| spacing | 0.1 |
| global Torus period | 102.4 |
| tile 数 | 8 |
| sites / tile | 128 |
| tile 幅 | 12.8 |
| profile の局所周期 | 38.4 = 3 tile |
| live sigma 範囲 | 0.1〜10 |
| K=H² の最高正周波数 M | 32、64 |

tile=10 / localP=30 / spacing=0.1 は1 tile100 sitesなので、1024 sitesを等分できない。
端数tileの特殊処理を避けるため、tile幅と局所周期を明示的に変更した。
sigma 最大10はtile幅12.8以下で、ownerと左右の計3 tilesという支持定義を保つ。
全3-tile局所周期は384 sitesで、`384 > 2M` の離散直交条件を満たす。
M64の実特徴数129は一つのtileの128 sitesより多いが、この条件には反しない。
再利用の対象は全atomの入力読み取りと出力集約である。

immutable geometry の周期・原点は元の宣言値からFP64で保存する。
atoms、入力、Triton contractionsはFP32。係数・L2準備はGPU上のTorch FP64。
`.cuda()`等のdevice-only移動はこの配置を保つ。
`.float()`でgeometryまで丸めると、正確な一様周期を要求する研究routeの対応判定から外れる。
整数tile番号にPython floatを掛ける際の既定FP32への丸めは、FP64 castで修正した。
元の整数tile幅10の測定条件はその修正で変わらない。

## 比較と検証

既存 `benchmarks.cuda.linear.periodic_comparison` の明示的rev5 caseを使う。
必須cohortは shared Triton、native direct Triton、dense。
巨大なsite×atom×harmonicのautograd scratchを作るTorch-factored経路はこのcaseでは使わない。
directは全global sitesのfactorsを保存する現実装であり、最適compact-support baselineではない。

新しい独立CPU FP64 oracleは、NumPyで宣言した次数M/2のnative Hを全sitesで評価し、
H²と実際の観測siteノルムを微分する。GPUのK係数、tile moments、中心非依存L2式は使わない。
atom chunk64で全52428 atomsを確認し、サンプリングしない。
同一X/target/dyと初期atoms、更新前／24実更新後のY・dX・全atom勾配、
固定max_abs／relative_L2 `4e-4` gate、live幅、Graph/public eager更新のmoments／clockを確認する。

未計装の完全stepは係数・L2準備、forward、MSE loss、backward、fused AdamW、
Polar projection、周期中心更新を含む。各routeを隔離subprocessで実行し、
正逆のworker順で二つの独立cohortを測る。21 replay samplesは一つのrunのサンプル。
allocator allocated／reserved peakはwarmup・Graph capture・replayを含む。
独立CPU oracleとGPUの照合scratchは測定から除き、GPU process全体のVRAMは未測定。

```bash
python -m pytest -q tests/test_fourier_tiles_cuda.py tests/test_fourier_density_cuda.py
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/fourier-density-chebyshev-1024-a52428-b32-m32.json \
  --plans fourier-shared fourier-direct dense \
  --output output/fourier-density/cohort.json
```

## 実測と採否

**この条件で現在のshared実装をdenseの代替として採用しない。**
Chebyshev/Kaiser、M32/64の4条件ともdenseが時間・allocated・reservedで優位だった。
各条件2独立cohortの完全step中央値を併記する。MiBは2^20 bytes。

| profile | M | shared ms（2 runs） | native direct ms（2 runs） | dense ms（2 runs） | shared allocated / reserved MiB | direct allocated / reserved MiB | dense allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chebyshev | 32 | 20.937 / 21.160 | 123.495 / 124.008 | 0.08542 / 0.08574 | 174.85 / 330 | 892.20 / 1210 | 33.00 / 86 |
| Chebyshev | 64 | 78.426 / 78.508 | 329.306 / 331.968 | 0.08650 / 0.08543 | 341.32 / 620 | 903.20 / 1494 | 33.00 / 86 |
| Kaiser | 32 | 14.311 / 14.435 | 117.724 / 118.410 | 0.08580 / 0.08551 | 174.85 / 232 | 891.00 / 1112 | 33.00 / 86 |
| Kaiser | 64 | 26.670 / 26.890 | 284.031 / 285.327 | 0.08530 / 0.08504 | 341.32 / 438 | 903.00 / 1364 | 33.00 / 86 |

actual hardwareはNVIDIA L4、Torch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0。
sharedは現native direct比で約4.2〜10.6倍速いが、dense比で約167〜919倍遅い。
allocatedはdenseの5.30〜10.34倍。runnerの`assessment.qualifies`はnative directに対する
改善判定なので、denseに勝った意味に読まない。
denseは自由な重み行列、CSTはprofile productという異なるモデルであり、学習品質は未比較。

24 workers = 4条件 × 2独立cohort × 3 routes。全CST workersの初期／24更新後、
計32回の独立全演算oracle gateがPASS。最大絶対誤差はY `3.00944e-5`、
dX `3.17594e-5`、全dP `6.26042e-5`。各relative L2は`1.59271e-6`以下。
GPU回帰43 testsはskipなしでPASS。CPUは1748 PASS、CUDA/DB未接続2745 skips、
既存warnings18。wheel/sdist build、独立wheel import、変更箇所Ruff、catalog検査もPASS。

## どこが重いか

主測定とは別のmodel/Graphで、clock24から30まで計装して診断した。
以下は最後のdiagnostic replayのCUDA event値であり、主測定から差し引かない。
`coefficients`は`prepare`の内側なので二つを足さない。

| stage | Kaiser M32 ms | Chebyshev M64 ms |
| --- | ---: | ---: |
| coefficients（prepare内） | 5.940 | 64.795 |
| prepare全体（coefficients・L2・中心位相・sort） | 9.189 | 72.783 |
| input tile moments | 0.0092 | 0.0113 |
| input atom contraction（H、Hprime） | 2.057 | 2.188 |
| output restore | 0.280 | 0.413 |
| output-gradient tile moments | 0.0061 | 0.0092 |
| output atom contraction（G、Gprime） | 2.058 | 2.177 |
| input-gradient restore | 0.276 | 0.412 |

tile momentsの再利用は成立している。重いのは全52428 atomsのlive係数準備と、
再利用後に残るatomごとの縮約。係数準備だけを速くしても縮約が残る。
別Graphのoptimizer中央値も両条件で約0.166〜0.168 msで、現在の更新実装まで含めた
構造変更が必要になる。この診断値を普遍的な速度下限としては扱わない。

`beta`だけでも、2軸分のFP32 arraysは
`2 * A * (2M+1) * 4 bytes` = M32で26.00 MiB、M64で51.60 MiB。
これに全atom/batchのH・Hprime等、FP64準備scratch、optimizer、Graphの保存が加わる。
単なるlaunch/recipe調整で今回の差を解消できる見込みは低い。

## 狭いprofileに対する数学的な制約

L2自体は扱いやすい。局所周期を全384 sitesで観測し、`384 > 2M`なら
`q_j = sum_{k=-M}^M c_k exp(2 pi i k (j/384 - mu/P))`について

```text
||q||_2^2 = 384 * sum_k |c_k|^2
```

が成立し、中心をsite間で動かしてもノルムは一定。
一方、低いMでsite程度まで狭くするには制約がある。
正規直交DFT基底の各行の二乗ノルムは`(2M+1)/384`なので、Cauchy–Schwarzより

```text
|q_j|^2 <= ((2M+1)/384) * ||q||_2^2
sum_{j in S} |q_j|^2 / ||q||_2^2 <= |S| * (2M+1)/384
tail_L2_energy >= max(0, 1 - |S|*(2M+1)/384)
```

中心`mu=0.05`、半径0.1なら内側sitesは2個。
外側のL2エネルギー比の下限は**M32で約66.15%、M64で約32.81%**。
これは個別のprofile選択、非負性や係数準備の実装によらず、最高K周波数Mという
制約から出る。値の最大漏れや連続積分の下限ではなく、全384観測sitesの二乗和比の下限。
whole-productのL2スケーリングでも比は変わらない。
局所周期や基底、支持の定義を変えた別演算にはそのまま当てはめない。

したがって、現在の固定P=38.4・M32/64のまま「半径0.1の外をほぼ0に貼り付ける」
という要件は満たせない。sigmaは既存定義でもconcentrationのパラメータで、
実際のFWHMやGaussianの標準偏差と同一ではない。

## 係数の安い新profileを小さく試した結果

CPU FP64で、既存2 profilesに加え、2種類の幅テンプレートprofileとFejer型を定義して試した。
新profileは既存形への近似誤差を目的に最適化していない。2つのM、9つの幅、
5つのsub-site中心、計90 profiles / 450 normsのnative-site二乗和とParsevalは一致し、
中心移動のL2相対幅は最大`1.182e-14`だった。裾の領域は閉じた半径とし、
今回の格子・中心・半径を1/40単位の整数に直して判定した。FP64の周期wrap誤差で
半径ちょうどのsiteを外側に数えることは避ける。GPU速度や全operator VJPの検証ではない。

幅テンプレートはknots `[0.1, 0.3, 1, 3, 10]`のHを静的に用意し、隣接knots間で
log(sigma)による`t`を使い、新しい定義`H=(1-t)H0+tH1`、`K=H²`とする。
事前計算した`c00=h0*h0`、`c01=h0*h1`、`c11=h1*h1`で

```text
c = (1-t)^2 c00 + 2t(1-t)c01 + t^2 c11
v = [(1-t)^2, 2t(1-t), t^2]
Gamma_ab = 384 * <c_a, c_b>
||K||_2^2 = v^T Gamma v
```

となり、per-atom FFT・Bessel・Chebyshev recurrenceなしに係数とL2を作れる。
平方なので非負、帯域と中心微分は保てる。幅knotsでの幅微分の滑らかさは別課題。
現在のbandwidth微分契約はstop-gradientだが、新しい数学的profileとしての定義・
公開spec・全VJP・GPU完全stepの検証が必要になる。

Fejer型はHの次数`r`まで一定係数、隣の`r+1`係数だけ分数weight`t`で加え、H(0)=1とする。
Kの係数は三角形のconvolution-countと端点補正から直接計算できる。
ここでは`r+t = (M/2)/sqrt(1+(2 pi (M/2) sigma/P)^2)`という新しい幅定義を試した。

M64/sigma0.1のFWHMはChebyshev約0.865、Kaiser約1.099、Fejer約0.591。
Fejerは狭くなるが、半径0.1外のL2エネルギー比は中心により38.6〜56.9%残った。
sigma6ではこのFejer定義のFWHMは約11.65で、Chebyshev約1.74、Kaiser約1.82とは
異なる幅校正になる。幅パラメータの数値だけで形の優劣を判定しない。
テンプレート型は係数準備を簡単にできるが、同じMの幅下限とatom縮約・betaメモリは残る。
両案とも、今回の狭さ・dense速度・メモリ要件を解決した候補とは扱わない。

## 次の判断

現在の方式の細かなCUDA調整は止める。共有集計やFourier全般が不可能という結論ではない。
次に調べるなら、狭い支持を直接持つprofileと、少数の共通幅basis／局所補間で
atom当たりの作業を抑える演算を、数学的な定義・裾・L2・中心VJPから評価する。
この測定からそれらの速度勝利は主張できない。
今回のPRは比較case・独立oracle・非整数tileの精度修正・負の実測を残すためのもの。
実装を公開dispatcherの既定に採用しない。

## 証拠と再現

GPU測定source commitは`34ec2890b65f2bdd55eb2b6c5c6e4da483162b05`。
全workerのPython source hashesは同一328 files。raw source/result archives、receipts、
driver、CPU profile図とJSONはignored `output/fourier-density-1024-20261014/`に保全する。
source tarにはdriverも入るため、jobごとのsource archive hashは異なる。
job IDと完全SHA256は同ディレクトリの`evidence-manifest.json`に保存する。

| scope | pool job ID |
| --- | --- |
| GPU43 tests + Chebyshev M32 2 cohorts | l4job-6d999c8c545e4ec0bebfbbfbf8308ec1 |
| Chebyshev M64 2 cohorts | l4job-c80160f9f2b14f2386c51e555b75f3be |
| Kaiser M32 2 cohorts | l4job-09c23bac81b04a2e92aa97623451bbe6 |
| Kaiser M64 2 cohorts | l4job-34ff72ea63e545a5b5be26c96b612acf |
| separate phase diagnostics | l4job-686500375a0c4d3ba5ea9a4021c021a3 |

primary source SHA256（Chebyshev M32 job）:
`42cce1303bb5a5fdc75d40ee2362f979d962dd14a8c9b52ed6f42531af109684`。
diagnostic result archive SHA256:
`d4e45120b7777ddab9af7c2fdbc2120fe7c497dc6a11c4328fb1b407d61bb50a`。
全jobのarchive hashes:

```text
Chebyshev M32 source 42cce1303bb5a5fdc75d40ee2362f979d962dd14a8c9b52ed6f42531af109684
              result 4f9960dd4199f38a2b556d515f2f4d62d53432e483e1e20157ede7a8fd9f9d09
Chebyshev M64 source cce99e77b2185dc3793167ca3a9c473c527fd7669277e7fa736e92f73f4a7834
              result c0927277ad5549f070ffd00b53cf94f1101faec8659330116c4eb435279d4d09
Kaiser M32    source e35ce6e33c4f4ef1bd5895ef6a3460f1e32ed5f8145e9ab9b9d679fb2468c212
              result cd834adce2f913dccb5f38580f05f9d38837a25f2008f151945d043549cbe70f
Kaiser M64    source 3bd80616cd9fedd7be11bea0775d679b3286db97d0866f73dd413c636a0e78c0
              result 08585ecbca840ed5c42b3e8badf08eece86abc340eae2eafb14dc1d7cdd5e2b0
diagnostic    source 7e95b8c2febe625d6a9510a65f8fc0f6294884d7ecea6af4c716f27c38d57bc5
              result d4e45120b7777ddab9af7c2fdbc2120fe7c497dc6a11c4328fb1b407d61bb50a
```

24 workers・32全oracle gates・全receiptとsource hashes・集計値を独立read-only auditで確認した。
GPU poolの全owned slotsは停止確認済み。

```bash
PYTHONPATH=src:. python output/fourier-density-1024-20261014/summarize.py
PYTHONPATH=src:. python output/fourier-density-1024-20261014/profile_feasibility.py
```
