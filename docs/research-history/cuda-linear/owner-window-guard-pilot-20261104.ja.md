# Graph guard費用pilotの結果

## 判定

事前screenの全5 blockでsignature-only / G0完全step中央値が26.57–26.83%となり、
「一つのflat実現性prototypeを検討する価値あり」を満たした。**flatは未実装であり、
26.7%の高速化、CPU律速、採用を意味しない。** eligible時間からsignature時間を引いた値や、
全stepからguard時間を引いた値を予測として使わない。K/Oは待機とする。

## 実施と正しさ

測定sourceは`72419a6db15e4c99d1fd3448b8fdadad303a07a8`。R0は#132
`fba2e72d733325fbd7be5aac334a0a7463d96b27`の`PreparedStep`を独立moduleとして読み込んだ。
G0はweak ordered root edgeだけを補強した。両方式でruntime `src`242ファイルは#132と一致し、
`_signature/body/capture/step`のASTも一致する。新CUDA/optimizer variantはない。

L4 `GPU-4ad46e5a-eddf-19b7-042c-726008bfe5d0`、Torch2.11.0+cu130、CUDA13.0、
Python3.13.16、Xeon2.20GHz/12logical/math1。Torch gitは
`70d99e998b4955e0049d13a98d77ae1b14db1f45`。

- GPU対象testは10 passed。alias拒否の状態不変と、replay未到達かつ現在のeagerと一致するfallbackの2件も実行済みで、skipなし。
- 固定12 referenceセルを全て実施。clock8→264、fallback0、CST42状態+5勾配とmetrics/orderを凍結#132へ照合し一致。denseも自身のschemaで全反復一致。
- guard-onlyはclock8のまま。CST42状態+5勾配・totals・対象version・既存signature範囲のlive設定が前後不変。capture仮更新後のgradを基準にした不変検査であり、仮更新のgrad復元まで主張しない。
- 取得後のdirect byte再照合は746 Tensor field、復元checkpoint14個。raw manifest1200ファイルとarchive/source/実loaded `.so`のSHAを照合した。
- ローカルCPU正本は明示的な対象`src`のPYTHONPATHで1746 passed/2398 skipped。最初のローカルrunはvenvのeditable mainを誤参照したため採否証拠にせず、環境を直して全体を再実行した。

## 費用とメモリ

| 対象 | signature-only 5 block（µs/call） | eligible-only中央値（µs/call） | R0完全step中央値（ms） | G0完全step中央値（ms） |
| --- | --- | ---: | ---: | ---: |
| CST | 155.432, 154.177, 155.063, 153.962, 154.553 | 435.121 | 0.578668 | 0.579423 |
| dense | 71.869, 73.952, 72.713, 72.575, 73.021 | 127.196 | 0.208797 | 0.211828 |

eligibleはsignatureを含むため両者を足さない。referenceは各3セルの工程確認であり、
6 pairのadoption試験でも小差の有意性検定でもない。CSTのenqueue区間はR0で
0.133856–0.135827秒、G0で0.139283–0.141140秒、最終sync区間はそれぞれ
0.012311–0.014283秒 / 0.007280–0.008876秒だった。全step区間は同程度であり、
host発行とGPU処理の重なりやbackpressureをこの2区間だけから分離しない。

capture込みpeak allocated/reservedは、全CST referenceで263,752,704 / 463,470,592 bytes、
全dense referenceで277,450,752 / 452,984,832 bytes。guard前後のcurrentも各cohort内で一致した。
optimizer構築前に一回だけpeak resetし、途中reset/empty_cacheなし。退役後currentはJSONへ別記する。
guard-onlyとreferenceはtotalsの退役時所有範囲が異なるため両runnerのcurrentの直接同値判定はしない。
新数値cacheはなく、この限定pilotを**memory-v2.1合格とは呼ばない**。

## 証拠と残る工程

成功jobは`l4job-ca1928ca101640c4beb0462c5d16f20e`。source archive SHA256は
`f34f1274b92c79bd4ddfda5613c4c31b96ec4fa74bfebbc7b94573948a78f38a`、result archiveは
`488c69b19145d89d40e85e28fd96da4c6f4a64aac3add377c26ca1d51d4730b3`。
loaded extension SHA256は`7b4968eb1643a0660bea2fc6b39480da629a7744beb7e2049239891113796476`。
数表・12セルのenqueue/sync・退役currentは[結果JSON](owner-window-guard-pilot-20261104.json)、
事前手順は[screening記録](owner-window-design-screening-20261104.ja.md)にある。
raw logs/PT/build/driver/source copiesはignored `output/design-search-20261104/`へ保全し、pool VMは停止確認済み。

先行job `l4job-ffd37b0e88af48f0b84d7da05f662bed`は埋め込んだcheckpoint helperのCLI entryが先に動き、
driver main前に失敗した。test/probe/referenceは0。原本を保全し、helper entryを除去して一回再投入した。
性能上の負けを捨てた再測定ではない。

次は既知のcallable cold/live穴、weak lifetime、capture後grad更新を含め、flat certificateの最小範囲を
Proと確認する。screen通過を理由に大きな実装へ自動移行しない。prototypeを作る場合も既存の
full-model byte/速度/両状態6 pair/prospective lifecycle/45,000更新の条件は維持する。
