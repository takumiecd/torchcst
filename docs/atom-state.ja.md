# AtomとAlgorithmの状態所有

`CSTModule`は`atom_state: AtomState`を所有する。`layer.atoms`はその中の既存
`Atoms`への参照であり、`atoms.py`の固定shape・不透明なParameter契約は維持する。
共有Atoms、または同じParameterから作るModuleは同じAtomStateを使う。
atom IDはAtomState内の固定集合`[0, count)`で、物理行とは独立する。
別のAtomState間でIDは比較しない。

## ライブラリ全体の契約：任意のatom順と再構築可能な実行状態

2026-10-25に確認した、全backend共通の所有・配置の契約。

- atomの初期順・現在順には、幅・中心・タイル等による規則性を要求しない。
  「任意順を受け付ける」は初期化時にランダムshuffleする指示ではない。
- atom和による重みの学習情報の正本はAtomsのParameterとする。演算を復元するには、
  Chart・Geometry・Kernel・Profile・正規化の宣言と、それらが所有する状態も必要である。
  optimizerのmoment等は学習継続用の状態であり、実行用cacheとは区別する。
- ソート順・支持分類・タイル配置・係数表・norm・H等は、それぞれの寿命に応じて
  Algorithm側で準備する派生情報である。global memoryへの追加保存を許すが、
  これらをモデルの意味の復元に不可欠な学習パラメータにはしない。
- 実行方式はatomの任意順を受け付け、必要なら自分の方式へソート・packする。
  その具体的な配置や補助配列はAlgorithm/Recipeの契約で定める。
  backend共通の必須形式として、出力tile別の参照表や特定の階層を要求しない。
- AlgorithmStateはdevice・方式・dtype・shape等に依存する再構築可能な状態とする。
  別device・方式への変更時に、元の実行bufferをモデルの一部として引き継がない。
  必要な派生情報は移動先の方式で再準備する。checkpointの扱いは末尾に記す。

atom数をA、全IDの置換をpiとすれば、数学的には

\[
W=\sum_{a=0}^{A-1}K(p_a)
 =\sum_{s=0}^{A-1}K(p_{\pi(s)})
\]

である。物理行順の変更で演算・正規化・勾配の意味を変えない。
浮動小数点では加算順による丸め差があり、bitwise一致までは意味しない。
正本の並べ替えではID・gradient・宣言されたoptimizer状態を同じ写像で移す。
Algorithm専用コピーなら、勾配を元のatomへ戻す対応を持つ。

### 配置形式の版・配置世代・数値更新

配置形式のID/revisionは「配列の読み方」、AtomStateのlayout_versionは
「正本の行配置の世代」を表す。通常のoptimizerによる値の更新とは別である。
方式固有の形式IDやkeyの規則を追加する場合は、そのAlgorithmの契約に記録する。
新しい共通の永続storeや全方式共通の固定pack形式は要求しない。

更新後もソートkeyが不変なら、順列と有効な構造情報は再利用できる。
中心・幅・原点等が変われば、keyだけでなく支持端や境界残差の情報も検査・更新する。
norm・係数・H等の数値の再利用可否は、それぞれの依存値で判定する。
特にH/Zは入力Xにも依存し、layout_versionが同じという理由で次の入力へ再利用しない。
更新通知を持たない方式は、実行時に現在値から必要な情報を準備する。

deviceごとに実行方式を選んでも、同じ演算の数学的な意味は維持する。
Fourier支持・周期選択・混合・正規化そのものを変える設定は、単なる配置変更とは
区別し、対応する数学宣言・semanticsの版を明示する。

## 配置と学習状態

| クラス | 配置 | 所有するもの |
| --- | --- | --- |
| `Atoms` | `atoms/atoms.py` | 固定shapeのParameter |
| `AtomState` | `atoms/state.py` | Atoms、ID/行対応、配置version、AtomOptimizerState |
| `AtomOptimizerState` | `atoms/optimizer_state.py` | optimizer契約、名前付き学習状態、atom軸の宣言 |
| `OptimizerFieldSpec` | 同上 | `atom_axis`。`None`なら配置から独立した状態 |
| `Algorithm` | `_backends/algorithm.py` | 状態の作成・準備・処理の共通interface |
| `AlgorithmState` | `_backends/state.py` | AtomStateへの参照、専用配置、再利用buffer、参照世代 |

状態の数値計算・更新はoptimizer、状態の寿命・配置はAtom側が管理する。
`CSTOptimizer`はTorch optimizerの遅延初期化をAtomOptimizerStateの辞書に接続する。
既存の学習済みTorch状態もコピーせず接続できる。wrapperとbaseのstate/groupは共有し、
atomごとの辞書は`layer.atom_state.optimizer_state.fields`と同じ実体になる。
空の状態は最初のstepまでTorchのstate mapに登録しない。通常の非CST Parameterは
従来のTorch状態を使う。一つのAtomStateに接続できる学習状態の更新元は一つ。

組込みSGD・Adam・AdamW・RMSprop・Adamax・Adagrad・Adadelta・Rpropについて
状態項目のatom軸を明示する。二次moment、AMSGrad最大momentも配置に追従する。
scalar step counterは動かさない。custom optimizerは`state_specs=`で
`OptimizerFieldSpec`を追加できる。未宣言の状態がある間はrelayoutとmodel変換を拒否する。
座標更新のvector transport用`OptimizerStateAdapter`と、配置の軸宣言は別の契約である。

## 正本の配置変更

```python
ids = torch.arange(layer.atoms.count, device=layer.atoms.p.device).flip(0)
layer.atom_state.relayout(ids)
```

引数は「変更後の各行に置くatom ID」のint64 Tensor。全IDが一度ずつ必要で、
追加・削除・shape変更は扱わない。IDとその逆対応、Parameter、既存gradient、
宣言された学習状態を同じ写像で移し、完了後に`layout_version`を進める。
既に同じ配置ならversionは進めない。Parameter/gradient/momentのオブジェクトを
置き換えない。移動先は全て確保・検証してから書き込む。重なるstorage範囲は拒否するが、
一つのslab内の重ならないmoment領域は利用できる。copy中に例外が起きた場合は状態を
使用不可にし、部分的な移動を新世代として公開しない。

relayoutはeagerのstep境界専用で、CUDA Graph capture中には実行できない。
CUDAでは全処理を同じcurrent streamに直列化する。異なるstream/threadと同時に
正本を変更することはサポートしない。Module forwardとAlgorithm.runはautogradの
保存期間を保護し、未完了またはretained backwardがある間はrelayout・変換・読込みを
拒否する。保護用のゼロ要素CPU markerは外側のsaved-tensor offload hookから独立する。
frozen atomでもdXのために保存された状態を保護する。

直接Parameterを使う実装は`parameters_for_execution()`または`protect_tensor()`を
使い、推論・独自CUDA Graphなどの生存期間とstep境界は呼び出し側で管理する。
生のTorch optimizerの状態は自動発見できないので、その状態も動かす場合は
`CSTOptimizer`経由で接続する。CSTOptimizerのcoordinate policyは引き続き
CUDA Graph captureに未対応であり、この変更でcapture対応を追加してはいない。

Algorithm専用slotの並替えは、そのAlgorithmState内で完結させる。正本を変更しない
コピーの更新ではAtomStateのversionを進めない。backwardの呼び出し固有の保存値を
再利用workspaceへ上書きしない。

## Algorithmの処理と状態

共通Algorithmは`create_state(binding, recipe=..., **configuration)`、`prepare(state)`、
`run(state, inputs)`を持つ。runはAlgorithmの同一性と準備済み世代を確認し、
必要ならprepareしてexecuteへ接続する。Tensor結果のautograd寿命も保護する。
Tensor以外の複合結果を持つ実装は、個々のTensorの寿命を自身で保護する。

具体的な配置・buffer・準備・演算は各Algorithmのディレクトリ内に実装する。
Algorithmのimmutable ID/revision/operation/semantics、recipe検証、support、workspace上界と
処理/state lifecycleはbackend共通のABCで定義する。CUDA専用ABCは廃止する。
公開CSTLinearはInputsとbinding interfaceをDispatcherに渡し、Module所有のAlgorithmStateを
取得して実行する。強制PlanもDispatcher.runの同じ検証を使う。
StateはbindingとRecipeを保持し、atomを使う方式ではbindingのAtomStateを参照する。
atomを使わない方式は同じStateを空の保持メモリとして使える。

AlgorithmStateは配置version、Parameterとoptimizer状態Tensorの実体/device/dtype/
shape/stride/storageを確認する。checkpoint復元・model変換・moment遅延初期化でも
古いviewを再利用しない。Contextのshape・dtype等やbinding.state_signatureの変更も
再準備の条件になる。数値の変更は配置versionで判断せず、Polar decodeなどは
execute時に現在値を準備する。Polarやoptimizer proposalの具体的なAlgorithmは
その演算契約を定めて同じRegistryへ追加できる。DispatcherにもLinearやCSTOperationの入力形を
要求しない。具体的なPolar/optimizer Algorithmの本番登録はこの変更に含めない。

## checkpoint

Module形式はversion2、CSTOptimizer manifestはversion3。旧形式の互換loaderはない。
Module checkpointにID対応と学習状態を含め、optimizer checkpointにはTorchのgroup/
stateと数学契約、field specs、atom行順を記録する。通常どおりmodelを先に、optimizerを
次に復元する。行順の異なるmodelへoptimizer checkpointだけを読み込むことは拒否する。
model-only復元から新しい同契約optimizerへ接続することもできる。

AlgorithmStateは再構築可能な実行状態なのでcheckpoint・Module deepcopyに含めない。
modelの再読込みは配置世代を進め、既存のAlgorithmStateを無効化する。
モデルとoptimizerを一つのcheckpointに保存する場合、学習状態への参照は両方の
state dictに含まれる。runtimeで二重の更新元やmomentコピーを持つわけではない。
