# Algorithm実行の境界

2026-10-05の境界方針を実装した契約。公開APIは[README](../README.md)、
所有関係は[AtomとAlgorithmの状態所有](atom-state.ja.md)を参照する。
DispatcherとExecutionBindingはbackend共通、LinearInputsとLinearBindingは
演算側のinterfaceである。Concrete Polar/optimizer Algorithmsの本番登録は含まない。

## 責務

| 要素 | 担当するもの |
| --- | --- |
| Inputs | 一回の呼び出しで渡す実データ。必須・任意・出力の演算契約 |
| bindingを提供する呼び出し元 | 演算設定とliveな所有状態への参照、入力検証、Context構築、AlgorithmStateの取得 |
| Context | 選択・対応判定に必要な、不変の実行条件metadata |
| Registry | Algorithmの登録・取得、PlanとRecipeの宣言検証、PlanのJSON読み書き |
| Selector | Contextから候補Planを選ぶ方針。未選択を表せる |
| Dispatcher | 検証・選択・状態取得・実行の順序と、実行前のfallbackを管理する |
| Algorithm | 入力契約への適合宣言、対応判定、専用状態の作成・準備・演算 |
| AlgorithmState | bindingへの参照、そのAlgorithmの配置・再利用buffer・準備済み世代 |
| AtomState / AtomOptimizerState | Atoms、ID/行対応、配置世代と配置に追従する学習状態の正本 |

共通部分にLinear/Polar/optimizerの名前で分岐を追加しない。具体的な準備・配置・演算は
各Algorithmのディレクトリで完結させる。共通入力型を使うことは系統別のAlgorithm継承を
要求しない。入力型は演算の呼び出し契約、Algorithmはその計算方式である。

## 毎回の入力・設定・状態

毎回変わる実データをInputs、結び付けておく演算設定をbinding、保持するメモリをStateに分ける。
LinearではInputsは`x`、bindingは既存Operatorへの参照、StateはAtomStateへの参照と
Algorithm専用の配置・bufferになる。Chart/Geometry/Kernelは既存Operatorから取得し、
normalized専用のOperatorやadapterを追加しない。

実行入口で`x, parameters, operator, site, state`を独立に組み合わせない。Parameterの
正本はbindingのAtomStateから、liveなChart/KernelはそのOperatorから得る。宣言snapshotと
実際のTensor参照を区別し、trainableなChartの値を宣言snapshotで置き換えない。
Algorithmが使う連続化・pack・decodeはAlgorithm側の処理であり、学習状態の別の正本にはしない。

必須入力は入力型のデフォルトなしfield、任意入力は明示的なデフォルト付きfieldで定義する。
「省略」と`None`が異なる場合はその入力型で区別し、共通Dispatcherに暗黙の補完を持たせない。
未知の入力field、型やshapeの不正、必須入力の不足は選択前に拒否する。
Registryにrequired/optional一覧を複製したり、全演算に任意の`x`を用意したりしない。

同じ演算を実行する候補は同じInputsと出力・更新契約を使う。各Algorithmは受け入れる
入力型を宣言し、演算側の共通検証に加えて自身の制約を判定する。fast pathだけに必要な
packやsupport表を利用者の必須入力にしない。それらはAlgorithmが準備する。
Polarやoptimizerの具体的なInputsは、それぞれの演算を実装するときに定める。

## bindingと所有者

Contextを作る時点ではAlgorithmはまだ選ばれていない。したがって、選択用Contextの
構築を選択後のAlgorithmStateに依存させない。

bindingは新しいParameter所有者ではなく、Moduleまたは直接実行の呼び出し元が提供する
結び付けのinterfaceとする。最低限、次の能力を持つ。

- 入力型と演算契約を定め、Inputsを共通検証する。
- Inputsとliveな所有状態からContextを構築する。
- 選ばれたAlgorithm/Recipeに対応するAlgorithmStateを取得する。

公開Moduleはこのinterfaceを提供し、Moduleが引き続きAtomStateとAlgorithmState cacheを
所有する。Dispatcher、Selector、RegistryはParameter・moment・所有者別State cacheを
保持しない。共通部分からCSTModuleの具体型を要求しない。直接Plan実行の呼び出し元も
同じinterfaceを提供し、Module外のAtomsを利用できる。新しい具体binding classは必要な
演算でだけ作り、各演算にwrapper classを必須にしない。

## Context

ContextはInputsとbindingから得た、今回の実行条件の記述。Tensor、Parameter値、
moment、可変なStateそのものを保持しない。入力の内容をhostへ読み戻したり、decodeや
kernel実行を行ったりしない。必須入力の検証はInputs/bindingが行い、Contextは自身の
metadataの型・範囲・整合性だけを検証できる。

共通境界は`operation_id`と`workspace_limit_bytes`に絞る。それ以外は演算ごとの
必要なmetadataとする。LinearContextならOperator宣言、入力shape/stride、dtype、
device、atom数、座標幅、必要勾配、精度・決定性・eager/Graph設定を持つ。
PolarやoptimizerにLinearのshapeやOperatorSpecを要求しない。

公開実行では利用者がContextを手入力せず、Dispatcherがbindingを通して生成する。
宣言検査・候補の比較ではmetadataから構築してよいが、そのContextを実データの
検証済み証明として使わない。直接Plan実行でもInputs/bindingから再構築する。
宣言metadataの更新・必要なdevice情報取得はCUDA Graph captureの前に済ませる。

Contextの一致はParameter値やdecode結果の一致を意味しない。配置versionも数値cacheの
鮮度を保証しない。atom値に依存するdecodeや支持域の準備はAlgorithmが実行ごとに更新する。

## 選択から実行まで

公開実行は次の順序で進む。

```text
dispatcher.run(binding, inputs)
  1. bindingがInputsを共通検証する
  2. bindingからContextを作る
  3. Selectorが候補Planを提案する
  4. RegistryでID/revision/Recipeを検証し、Algorithmを取得する
  5. Algorithmの対応条件とworkspace上限を確認する
  6. 非対応/未選択なら、明示したfallback候補に対して4・5を行う
  7. bindingから選ばれたAlgorithm/RecipeのStateを取得する
  8. AlgorithmがState・実入力の整合性を確認し、必要な準備を行う
  9. Algorithm.run(state, inputs)で実行する
```

候補ごとの対応検証は共通の一つの処理にまとめ、SelectorとDispatcherに別々の検証規則を
実装しない。Selectorは順位・探索・完全一致表などの方針を担当する。未選択・非対応と
不正なPlanを区別し、未知のID、revision、壊れたRecipeをfallbackで隠さない。
対応条件を満たした候補だけを実行し、通常呼び出しでは利用者に演算結果を返す。
選択理由やPlanはDispatchDecisionとして診断できるが、Tensor/Stateをそこに保存しない。

強制Plan実行は同じDispatcher入口に`plan=`を指定する。Selectorを省略するだけで
入力検証・対応判定・状態準備を省略せず、非対応ならエラーにする。実行経路を
Registry.executeとDispatcher.runの二つに増やさない。

## fallbackと状態準備

fallbackは正しいInputsに対して候補が非対応、またはSelectorが未選択の場合に行う。
同じ演算の意味・正規化・必要勾配・出力・更新契約に対応するTorch Algorithmを明示する。
Torchも非対応なら理由付きで拒否する。演算名だけの一致で別の意味へ落とさない。

任意入力の省略は入力型のデフォルトの規則で処理し、fallbackとは無関係にする。
State準備や演算の開始後の例外、OOM、optimizer更新の失敗を捕まえて別Algorithmで
再実行しない。失敗時の部分更新を隠したり、optimizer stepを二度進めたりしない。
Algorithm固有の実入力制約が選択に必要なら、事前にContextへ表現できるmetadataを使う。

保持状態の型もAlgorithmの契約とする。Atomを使う方式は共通AlgorithmStateからAtomStateを
参照するが、全演算にAtomsやOperatorを必須にはしない。保持メモリが不要な方式は自身の
空Stateを使える。共通Registryが`state=None`で異なる実行経路へ切り替える設計にはしない。

AlgorithmStateの再利用keyはAlgorithm実体/版、Recipe、演算bindingと所有者の同一性を
区別する。入力shape/device/dtypeなど準備が依存する条件はAlgorithmが確認し、変化時は
専用配置・bufferを再準備する。デフォルトの配置世代確認だけで準備済みと見なさない。
Recipeや演算設定の変更は別Stateの取得または明示的な無効化を必要とする。

Algorithmは保持するメモリと一時メモリを区別し、workspace上界には準備・実行で管理する
対象を明示する。上界の検証はGPU全体の実測peakの保証ではない。所有者が保持する他の
AlgorithmState cacheの費用も別途存在する。未知の上界に対して指定上限を満たすとは扱わない。

## 配置・backward・Graph

Algorithm専用の並替えやコピーはAlgorithmState内で完結し、正本の配置versionを進めない。
Atomsの正本を移す場合は、明示的なeager step境界でAtomState.relayoutを使う。
Dispatcherの選択、通常のprepare、forwardの副作用として正本を移さない。
Parameter・gradient・宣言済みoptimizer状態は同じ写像で移し、次のrunで古いStateを再準備する。

backwardはforwardで使ったAlgorithmと呼び出し固有の保存状態を使い、再dispatchしない。
再利用bufferに未完了backwardの保存値を上書きしない。AtomStateの寿命保護を維持する。
CUDA Graph replayはcaptureした実行を使い、Selectorやbindingの変更後は再captureする。
この契約はoptimizerやrelayoutのGraph対応を新たに保証しない。

## 配置と移行

共通実装は`_backends/dispatch/`に置く。Registry、Algorithm、AlgorithmStateは
既存の共通ファイルを使う。Inputsとbindingの具体処理は演算の契約を定義する側に置き、
Algorithm固有の配置・kernel・実入力制約は各Algorithm内に置く。
Linearの呼び出し元は既存Module/Operatorを利用する。共通Dispatcherはoperatorsやnnの
具体型に依存しない。演算契約をbackendの都合で変えない。

移行した境界は次のとおり。

| 移行前 | 現在 |
| --- | --- |
| Context.validate_inputsがTensor検証を持つ | Inputs/bindingが共通検証し、Contextはmetadataだけ |
| Registry.validateが対応条件/workspaceも確認する | 共通dispatch処理がAlgorithmへ問い合わせる |
| Registry.executeが検証と実行を接続する | Dispatcherが唯一の公開実行経路になる |
| Selector.selectがRegistry検証とfallbackも持つ | 選択方針と共通の候補検証/fallback処理を分離する |
| ModuleがPlan選択とnamed inputsの組立てを行う | Moduleがbindingを提供し、DispatcherにInputsを渡す |
| create_state(atom_state, **configuration) | binding/Recipeから参照と専用Stateを構築する |

公開READMEを更新し、Plan/selector artifactの既存形式を維持した。Registry.validate/executeと
Context.validate_inputsは削除し、互換wrapperは追加しない。Selector.selectはmetadataだけの
便宜APIで、Dispatcherと同じ候補検証/fallback処理を呼ぶ。強制PlanはDispatcher.runのplan引数を使う。

回帰確認項目は、必須/任意/未知入力、Contextと実データの一致、custom Registry、
無関係な演算入力、Torch fallback、強制Plan、不正Planの拒否、状態の再利用/無効化、
共有Atomsとmoment、backward/Graphの寿命、宣言importの計算コード非読込みとする。
既存CPU回帰に加え、GPU oracle・完全step・peakを比較する。
