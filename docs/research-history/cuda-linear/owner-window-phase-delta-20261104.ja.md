# FP32局所変位共有の研究（#132基準）

## 事前固定した契約と判定

基準は#132 Flat C（fba2e72d、runtimeは34b44a0cと同じ）。
候補は`GLOBAL_OWNER_WINDOW_FP32_PHASE_DELTA`のみ。
分類kernelが現在のParameterからFP64 geometryとint64 owner keysに加え、
FP32 `[2,A]`の局所変位を生成する。

\[
\delta_{i/o,a}=\mathrm{float32}\left(
\operatorname{wrap}_{i/o}(\mu_{i/o,a})
-t^{\mathrm{signed}}_{i/o,a} T_{i/o,a}\Delta_{i/o}\right).
\]

引き算までは既存FP64、既存cast直後の値を共有する。signed first tileは
ownerの周期moduloより前の値を使う。逆方向は保存したfull keyからsource bankを
復元し、同じ局所変位を使う。位相乗算、Fourier/profile、cell mass、正規化、
C縮約順、全atom勾配、Polar pullbackとoptimizerは維持する。
追加payloadはA52428で419,424 bytes（約0.40MiB）。追加launch/barrierはない。
通常のSavedVariableとしてforward単位に保持・検証し、replayごとに上書きする。
公開dispatcherとGraphのguard/hook/live settings契約は変えない。

N1024/A52428/真B4/seed41、L4、FP32 body・FP64 geometry、TF32 off、
native CSTAdamW、MNIST全モデルGraphで比較する。モデルは重み付き3層、CSTは中央1層。
基準の8実更新後と#132自身の45,000更新checkpointを凍結する。
初期値・最終checkpoint・入力gzipのSHA256をdriverに固定する。

短期は各状態256更新×6 balanced-order fresh-process pairs。
同等条件のdense Graphを含める。全12 CST cellの42 state、5 gradient、
metrics/data orderをbyte一致、fallback 0とし、過去#132 numeric controlとも照合する。
両状態でpaired geometric reduction >=5%、6/6勝利を要する。
captureを含むpeak増加allocated<=2MiB、reserved<=8MiB。
memory-v2.1の8 cycle×16実更新（128実更新/120replay）でphase・epoch peak・
cycles3–8のallocated/reserved plateau、全状態・勾配・metricsも確認する。

すべて通過した場合のみ、基準→候補と候補→基準の2順序、各45,000更新を実行する。
準備・capture/restore・再準備・旧Graph解放GC・最終retirementを含む時間で各順序>=5%、
各3 epochの47 Tensor byte一致と長期peak予算を要求する。
3 epochだけを8 cycle lifecycleの証拠とはしない。
独立FP64 oracleは既存`3e-5 + 3e-4 abs(ref)`を維持し、byte不一致の救済に使わない。
別processの16更新profileは診断専用。短期償却推定を長期実測と混同しない。

## ローカル事前検証

CPU suite: 1742 passed / 2529 skipped。host C++ syntax、Ruff、Plan/Case宣言検査pass。
GPU correctness、短期・長期の結果は取得後に追記する。
