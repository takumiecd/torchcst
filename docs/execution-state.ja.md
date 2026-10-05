# Algorithmの実行メモリと所有者

2026-10-05。実装はAlgorithm単位に置き、演算の意味・計算方式・実行メモリの
所有を分離する。現段階では既存の演算内の配置を維持し、異なる演算が実際に
同じ契約を共有する段階で共通部分を切り出す。

## 分ける責務

| 要素 | 決めること・所有するもの |
| --- | --- |
| 演算の契約 | 入出力、正規化、支持、微分、状態更新の意味 |
| `Atoms` | 固定shapeの学習Parameter。列の意味と実行配置は持たない |
| Algorithm / Recipe | 計算方式、対応条件、実行設定。RegistryのAlgorithmはimmutableでTensorを保持しない |
| model / optimizer / 呼び出し側 | 選択したAlgorithmが必要とする実行状態の所有者 |
| Algorithmのexecutor | 実行状態を使った計算とautogradの接続 |
| forward / backwardごとの状態 | 呼び出し固有のmetadata、保存値、scratch、gradient |

実行状態が不要なAlgorithmは常駐bufferを持つ必要がない。必要なAlgorithmだけが
自分の実装で容量・配置・更新手順を定め、所有者へ状態を持たせる。登録インスタンスへ
入力やlayoutを保存しない。Parameter・AdamW moments・stepなどの学習状態と、
再構築可能な実行配置は区別する。

`Atoms`の行番号をcanonical atom IDとする。実行slotは別の位置であり、移動しても
Parameterとoptimizer状態の行番号は変えない。共有Atomsに対して、異なるAlgorithmや
modelが異なる実行配置を持つこともできる。値が変わるときは各配置の分類・数値を更新する。

## 現在の小さいlocal product

実装場所は `src/torchcst/_backends/cuda/algorithms/local_product/`。

| ファイル | 責務 |
| --- | --- |
| `layout.py` | metadataの説明、呼び出しごとの`LayoutSnapshot`と確保 |
| `persistent.py` | model所有のID/slot対応表・bucket容量・空きslot・counter、refreshのlaunch |
| `layout_kernels.py` | 現在の支持/ρからの所属判定、差分移動、overflow時の再構築、snapshotへの書き込み |
| `kernels.py` | 正規化・因子計算・縮約・勾配。compact配置の`pack_tiles`もここに残る |
| `executor.py` | 準備・配置・計算を接続し、そのforwardのTensorをbackwardへ保存 |

`LayoutSnapshot`は二方向の数値view、canonical ID、境界を一緒に返す。
compactでは`ends=None`、persistentでは別の使用末尾`ends`を持つ。
snapshotは常駐layoutから独立して確保し、複数forwardやretained backwardで
古い配置が必要な間は上書きしない。Pythonのfrozen dataclassはTensorの内容を
書込み禁止にはしないので、executorがこの寿命を守る。

常駐layoutはstepを跨いで再利用するが、振幅・幅・支持・正規化係数を固定する
cacheではない。Graph replayでも現在値を準備する。H/Gはcanonical IDを使う
別の呼び出しbufferであり、空きslotの容量に合わせて学習状態を増やさない。
`persistent=False`の登録bufferはmodel上で生存し、checkpointには入らない。

新しいρ区分やtile別配置を追加する具体的な箇所と検証は
[local-productの説明](../src/torchcst/_backends/cuda/algorithms/local_product/README.md#layout-ownership-and-extension)
を参照する。

## Polar・更新演算とTorch fallback

現在の`OperatorSpec`はlinear専用で、汎用`CSTOperation`はまだ実装していない。
Polar decode・Polar更新・optimizer proposalなどを将来同じ実行選択の枠組みへ
載せる場合も、演算ごとに入力・出力・微分・更新対象の契約を定める。
optimizerの呼び出し制御とmoments所有は維持し、その中で必要な更新Algorithmを呼ぶ。
linearのchart/shape契約へ更新演算を押し込まない。

対応する最適化実装がなければ、同じ演算契約を実装するTorch経路を選ぶ方針とする。
Torch backendはCPU専用ではなく、対応するTensor device上で動く参照実装である。
更新演算は更新前に対応判定・選択を完了する。部分的な更新後の例外を捕まえて
Torch更新を重ねて実行しない。

汎用Operation、共通実行状態interface、Polar/optimizerのRegistry接続、fallback
変更は今後の設計範囲。この変更では系統別の実行階層や汎用layout APIを追加せず、
既存Algorithm内で所有と寿命を明示する。
