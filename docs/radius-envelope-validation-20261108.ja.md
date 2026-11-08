# 半径の包絡線の実装検証

[English (canonical)](radius-envelope-validation-20261108.md) | 日本語

明示的な追加方式`kernels.radius_envelope`は、以下の検査を通りました。
[APIと数式](radius-envelope.ja.md)、[コンパクトな検証記録](radius-envelope-validation-20261108.json)。
演算・optimizerの検証であり、LM精度や最適tauの検証ではありません。

## 実施した検査

- ローカルCPU一式は`a86db78dc4fab6ad35c782d619a46b7bd3bfdcd6`で2433成功・2408skipでした。
  GPU/live環境が必要なskipは検証済みとは数えません。最終変更はCUDAの条件追加だけで、
  最終sourceの対象CPU検査は24成功・CUDA8skipでした。
- 最終RTX6000 Adaジョブは40件成功で、内訳は**実CUDA14条件**とCPU/metadata26条件です。
  skipはありません。新方式は512->2048と2048->512の両方向、8192 Atom、非連続入力の
  7/256行、最大半径4.7/50、q0/q0.25/q1と振幅0の混在を含みます。
- 写像が作るFP32物理行はCPU/GPUで完全一致しました。同じ符号化値を保ち、physical参照の
  因子と集計をFP64で計算します。Y・dX・Atom全4項目の勾配はatol1e-4/rtol4e-3を通り、
  許容誤差は広げていません。繰り返したbackwardは完全一致しました。モデルとoptimizerを
  実際にtorch.save/loadし、次のParameter・モーメント・counterまで一致しました。
- CPUでは、独立した多項式のGauss積分によるY/dX、全Atom項目の有限差分、q両端のa/q微分、
  tau0.01/0.03/0.1、振幅0、極端に小さい有限tau、符号化半径の範囲、qの学習とfield LR、
  射影、固定定数の精度保持、モーメント初期化、明示的Planの分離を検査しました。
- 従来のphysical FFN経路も、両方向・7/65/256行で成功しました。
  既存physical/Polar Planは新しいq保存形式を拒否します。

CUDA演算は既存physicalのままで、Python境界写像はFP64、物理行はFP32です。
default selectorや従来Polarの更新・勾配規則は変更していません。

## 比較参照の修正と保全した試行

| 試行末尾 | 結果 | 原因・位置づけ |
| --- | --- | --- |
| 01 | 25成功・8失敗 | system環境にNinjaがなく、CUDA build/importで失敗。 |
| 02 | 25成功・8失敗 | CUDAヘッダの探索パスにcusparse.hがなかった。 |
| 03 | 31成功・2失敗 | CPUの内部変数modelをdouble化し、符号化半径の階層分岐が変わった。 |
| 04 | 36成功 | FP32物理行を揃えた参照で比較。optimizer・復元も確認。 |
| 05 | 40成功 | 同じ検査を最大半径50へ拡張。 |

最大半径4.7では、FP32の内部変数modelをdoubleへ変換することで、470/8192 Atomの
半径0.5の分岐が変わりました。集計精度だけを上げる比較ではなく、異なる関数を比べていました。
符号化値を揃えた診断の両射影で、最大絶対差はY5.54e-7、dX9.09e-7、Atom勾配2.86e-6でした。
これは診断ケースの最大値であり、最終検査全体の最大値とは主張しません。
境界の挙動と比較参照は回帰テストへ追加しました。
最初の実装commit`ee9c89556e99b09ed993e3a1a8aff874e3a1ee99`以降、runtime sourceは
変更していません。その後の変更はテスト・文書です。

## 来歴と再現

実機はNVIDIA RTX6000 Ada Generation・48GB級・CC8.9、Python3.13.11、
Torch2.9.1+cu128、CUDA12.8でした。最終sourceは
`40e120c9f1b7260364f08bf6723185f62bf31ac6`、source archive SHA256は
`c9e11d23d94a8e83f1d257af35f380a58d78076ec6c46828f18940ff0d3aedcf`です。
ジョブ`radius-envelope-6000ada-20261108-05`は専用のextensionディレクトリを使い、
Ninjaのある検証済みvenvで実行しました。NVIDIA wheelのincludeをCPATHへ設定しました。
正確な環境とdriverの記録はignoredな所有者用証拠に保存し、マシン固有の絶対パスは
この文書へ複製していません。

```bash
python -m tools.kernel_dev test --suite cpu
python -m pytest tests/test_radius_envelope.py tests/test_lm_physical_radius.py -q
```

2つ目のコマンドには対象CUDA GPUとextensionのbuild環境が必要です。
このホストでは、インストール済みNVIDIA wheelのincludeをCPATHへ設定する必要がありました。
packageやruntimeへの直接の変更は行っていません。

コンパクトなarchiveのSHA256は
`99d6f25b5295642b011a60387345bb399ef4fd917c0aeee9d523e8af28a1c24e`です。
回収後、28ファイル、各試行の凍結source1306ファイルと対応するローカルGit archive、
driverのハッシュ、runtime sourceが不変であることを独立に照合しました。
記録したcontroller5個はすべて停止済みで、GPUの事後確認は使用率0%・39MiBでした。
失敗試行、archive、native buildも保全しています。ローカルの証拠は所有者の共通torchcst
checkoutのignoredな`output/radius-envelope-6000ada-20261108-01`〜`-05`です。
LM学習、tau探索、速度・ピークメモリ比較、他GPUの条件、CUDA Graph captureは検証範囲に含みません。
