# 接続M16のFP32化・再利用：事前の最適化条件

[English](connected-m16-reuse-20261109.md)

完了：[実測結果](connected-m16-reuse-results-20261109.ja.md)を参照。
基準は5ade3859/PR159の固定した接続M16。
数学的revision9、セル全体のK3窓、半径/座標の五次混合、正規化と混合比の全微分を維持。
公開登録・既定値・optimizerは変更しない。

既存Linear runnerに2つの研究用recipeを登録する。

- `connected16-fp32`：Chebyshev16の正の同次多項式係数と半径の解析微分をFP32で計算。
  有効な階層ごとに1回作り、入力/出力と両窓で再利用する。
  タスク演算と保存TensorにFP64を必要としない。
- `connected16-reuse`：同じ準備に加え、8行のresponse再利用、子タイルのDC/偶数周波数
  モーメント再利用、K3の定数回転、タイル所有者による係数集約を使う。
  広い階層は所有者が一度書き、Atomごとのglobal atomicを使わない。
  小さいタイルは局所集約後にbank/siteあたり1回のatomicを使う。

Hは振幅を掛ける前の完全なAtom応答。forwardはX/P、FP32幾何、呼出しごとのrouting範囲を
保存し、backwardはHと微分を再計算する。ParameterはA×4を維持し、永続的仮想Atomや
密行列Wは作らない。毎forwardの現在値からroutingを作り直す。
sort/range構築と全作業領域もstep全体の測定に含める。

M16係数は同次化したChebyshev漸化式を二乗し、各Fourier係数を非負の基底
z^(16-k-j)(1-z)^j、z=cos²(pi R/P)に変換する。共通係数は正規化で消える。
生成器と独立したCPU FFT/数値積分の照合を残す。FP32丸めと加算順序の変更は、
従来と同じY/dX/全フィールドの絶対最大誤差・相対L2が3e-4以下という条件で確認する。

各幅の計測前に、既存fixtureと独立FP64 oracleでN128,2048,8192,16384を照合する。
AdamWの4step状態一致、端数行、retain/multiple forward、保存Parameterの変異検出も確認。
Graphは未認証のままとし、captureメモリの主張はしない。

予約済みで空いているRTX4090で、基準、FP32準備、全再利用、最適化M8比較、Denseを比べる。
B256,A8N、P/X/targetのhash一致、physical半径[.5,32]の対数一様、FP32 IEEE、共通の
fused AdamW/射影。5回準備し、同期した15回のeager step全体を測る。
allocated/reservedは常駐Parameter/勾配/Adamと一時幾何/layout/momentを含む。
profileは別計測。標本数を独立実験数と数えない。有望なら別実験で再確認し、失敗とsourceも保存する。

追加の`connected16-owner`比較はタイル所有/結合モーメントを維持し、行単位responseと
通常の勾配照会を使う。row8/高速勾配照会の組との違い、実行sourceは結果に記録する。
