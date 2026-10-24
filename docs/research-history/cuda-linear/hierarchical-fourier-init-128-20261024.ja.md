# init128：最小比較構成の担当・shared配置・同期

2026-10-24。[forwardの関数](hierarchical-fourier-thread-functions-20261023.ja.md)と
[backward128](hierarchical-fourier-backward-128-20261023.ja.md)を、最初に確認する構成へ絞る。
本書の数値はCPU上のbank表照合、容量・offset・担当範囲の検査から得た設計値。
後続の[CUDA C++実装と検証記録](hierarchical-fourier-block128-cuda-20261024.ja.md)を別に作成した。
以下の擬似コードや容量だけをGPUの正しさ・進行・速度・ピークの実測として扱わない。

## 1. 比較基準として固定するもの

| 項目 | 今回の比較基準 |
| --- | --- |
| site数 / atom数 | 入力128、出力128、atom128 |
| thread / 小バッチ / 担当site数 | 128 threads、batch_size=4、出力YまたはdXの32 sites |
| thread集団 | 一つのblock全員を一集団にする |
| H/Gの担当 | `same_thread`。各threadが別atomの入力・出力縮約を最後まで行う |
| Hの保持 | `recompute`。backwardへ渡すglobal Hは作らない |
| 実行順 | 全員でZ準備 → 各threadで別atom → 全員でC集約 → 出力 |
| 値のdtype | global X/dY/Y/dX/PはFP32、Z/H/G/Cと縮約はFP64 |
| Fourier次数 | 8。basis_count=17 |

これは同期と寿命を確認する**比較用の単純な構成**。直接計算の先行実行・複数集団・
動的job管理を採用しないと決めたわけではない。複数集団版と比較する基準を先に作る。
ここでは空き管理・job選択・ready flag・集団barrierの領域は不要で、block全体の同期を使う。
128 threadsを一集団にした最適性や、SMの高い稼働率は主張しない。

最初は全blockで全128 atomを対象にする。`atom_id = threadIdx.x`とし、
空支持・振幅0も削除しない。階層の違いは各atomの成分と参照bankで表し、
同じatomの縮約を複数threadに割らない。入力支持と重ならないatomの除外や並べ替えは次の比較対象。

## 2. block番号と保存する要素

```cpp
tile_id = blockIdx.x % 4;          // 0,1,2,3
batch_group = blockIdx.x / 4;
batch_begin = batch_group * 4;
site_begin = tile_id * 32;
param_owner = backward && (tile_id == 0);

atom_id = threadIdx.x;             // 各threadが別atom
output_batch = threadIdx.x / 32;   // 0..3
output_site = site_begin + threadIdx.x % 32;
```

最後の2行は最終出力を分担するときの番号で、atom計算時の担当とは別。
warpへ役割を固定する指定ではない。
batch4ならforward 4 blocks、backward 4 blocks。batch32ならそれぞれ32 blocks。
端数batchはload・H/G・dPの和・storeをmaskし、無効batchを0として扱う。

forwardでは各blockが自分のYだけを保存する。
backwardでは各blockが自分のdXだけを保存し、tile 0だけが全atomのdP部分和を保存する。
パラメータ担当のH/Gは両側の支持全体を読む。担当32 sitesへ縮約を切り詰めない。
複数小バッチのdPは全backward完了後に別kernelで合算する。

## 3. Zの枠とCの枠を数える

CPUで既存の`FourierLinearSpec(128)`・`PackedFourierRecipe()`のbank構造を確認した。
ここではそのbank集合を比較用の容量上限として利用し、既存のGPU recipe自体は変更しない。

| タイルのsite数 | 異なる基底のbank数 | bankごとのnode数 | 合計node数 |
| --- | ---: | ---: | ---: |
| 32 | 4 | 4 | 16 |
| 64 | 4 | 2 | 8 |
| 128 | 4 | 1 | 4 |
| 合計 | 12 | | 28 |

同じ32 sitesにも周期の違うbankが4つある。site領域だけでnodeを同一視しない。
この上限ではZ一組が`28 × 4 × 17`個のdouble、15,232 bytesとなる。
支持外・bankがない成分は既存の直接経路で計算し、周期コピーで置き換えない。

担当32 sitesは各bankの一つのnode内に収まるので、出力へ展開するCは最大12 node分。
ただしCへ入れるのは、**そのnode全域が当該成分の有限支持に含まれる場合だけ**。
残りは直接成分／境界残差へ回す。32-siteの担当範囲に入るだけで、
支持からはみ出す64/128-site nodeへ成分を集約してはいけない。
C一組は`12 × 4 × 17`個のdouble、6,528 bytes。

初期の比較基準は28 nodeをすべて準備する。これは並列配分を調べる前の上限であり、
live需要だけのnode生成・入力側と出力側の別々の圧縮は次に比較する。
全域の`Phi[bank,site,basis]`は追加保存せず、Z生成・C展開のthreadが必要な基底を評価する案とする。
その追加計算費用も測定対象。既存のPhi全保存方式からの速度改善を主張しない。

## 4. sharedのoffsetと寿命

16-byte単位で領域を配置する。基底・norm・係数・支持表等を無条件に全量sharedへ複製しない。
以下は新しい比較構成で必要な配列payloadの設計値であり、既存GPU経路の測定値ではない。

### forward

| view | byte offset | bytes | 最後の利用 |
| --- | ---: | ---: | --- |
| X `[4,128]` FP32 | 0 | 2,048 | 全Hの生成 |
| Z_X `[28,4,17]` FP64 | 2,048 | 15,232 | 全Hの生成 |
| H_value `[128,4]` FP64 | 17,280 | 4,096 | C／直接出力の集約 |
| C_output `[12,4,17]` FP64 | 2,048 | 6,528 | 最終Yの生成 |

**全Hが完成したblock同期の後**にZ_Xの先頭をC_outputとして再利用する。
確保量は合計21,376 bytes。ZとCを同時確保した27,904 bytesから6,528 bytes減る。

### backward：パラメータ担当block

| view | byte offset | bytes | 最後の利用 |
| --- | ---: | ---: | --- |
| X `[4,128]` FP32 | 0 | 2,048 | 全atomのH・dP計算 |
| dY `[4,128]` FP32 | 2,048 | 2,048 | 全atomのG計算 |
| Z_X `[28,4,17]` FP64 | 4,096 | 15,232 | 全atomのH・dP計算 |
| Z_dY `[28,4,17]` FP64 | 19,328 | 15,232 | 全atomのG計算 |
| G_value `[128,4]` FP64 | 34,560 | 4,096 | C／直接dXの集約 |
| C_input `[12,4,17]` FP64 | 4,096 | 6,528 | 最終dXの生成 |

H/Gのvalue・center・widthはatom担当threadのローカル値。
dPを計算して保存した後はHもGの微分項目も不要で、Gのvalueだけを他threadへ渡す。
**全atomのH/G・dP計算終了のblock同期後**に、Z_XをC_inputとして再利用する。
確保量は38,656 bytes。同時確保の45,184 bytesから6,528 bytes減る。
registerへ置く意図のローカル配列は、実際に退避がないかコンパイル後に確認する。

### backward：dX専用block

XとZ_Xを作らない。dY offset0、Z_dY offset2,048、G_value offset17,280とする。
Gの全縮約後にZ_dYをC_inputとして再利用でき、必要payloadは21,376 bytes。

### 起動時に予約する量は別に考える

`<<<grid,128,shared_bytes>>>`のshared_bytesは、**その起動の各block**に予約する量。
一つのbackward起動へ上記二種類のblockを混ぜるなら、全blockで38,656 bytesを予約する。
dX専用blockの有用なpayloadが21,376 bytesでも、そのblockの予約量が自動で減るわけではない。
forwardの起動は21,376 bytes。追加の静的sharedを置く場合はその分も数える。
[CUDA公式の起動設定](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#execution-configuration)に従い、対象deviceの上限を確認する。

パラメータ担当とdX専用を別起動に分ければ、それぞれの量を指定できる。
その場合は追加起動・実行順との比較になるため、初期案の一つのbackward起動と分けて評価する。
総sharedの予約は同じSMに同居できるblock数へ影響する。配列のpayloadだけで占有率を断定しない。

## 5. initがすること

次はCUDA風擬似コード。addressは上表のoffsetに対応し、動的sharedの先頭を16-byte整列させる。

```cpp
init128(X, dY, total_batch, plan, shared_bytes) {
    ctx = decode_block_and_thread(plan);      // 第2節の番号。tidもここで取得
    bind_shared_views(ctx, shared_bytes);     // 第4節のoffset。ZとCは同じarena

    for (q = ctx.tid; q < 4 * 128; q += 128) {
        b = ctx.batch_begin + q / 128;
        i = q % 128;
        if (ctx.needs_X)
            shared.X[q] = b < total_batch ? X[b][i] : 0;
        if (ctx.backward)
            shared.dY[q] = b < total_batch ? dY[b][i] : 0;
    }
    __syncthreads();                          // 全員のsource読込が完了
    return ctx;
}
```

各threadはsource一組あたり4要素を読む。
Zは生成時に全要素を上書きするのでinitでは0にしない。
H/Gも、空支持や無効batchを含む全128 atom・4件を担当threadが必ず書く。
Cをinitで消すと後でZに上書きされるため、Cの計算段階で各要素を0から作る。
block内malloc、全threadの空き走査、jobカウンタの初期化はこの比較基準では行わない。

## 6. init以降の短い入口

```cpp
forward128(...) {
    ctx = init128(...);
    for (node : bottom_up_nodes) {
        build_Z_elements(ctx, node);         // (batch,basis)をthreadへ分配
        __syncthreads();                     // 子Z → 親Z、最後はZ → H
    }
    atom = prepare_atom_in_this_thread(ctx);  // P・norm参照、係数・支持を準備
    shared.H_value[ctx.atom_id] = compute_H_value_for_atom(ctx, atom);
    __syncthreads();                         // 全H完成。全員がZを読み終えた
    build_C_elements_from_H(ctx);            // Z arenaをCへ変更。各要素を一度書く
    __syncthreads();                         // C完成
    compute_direct_plus_C_and_store_Y(ctx);  // 各threadが出力1要素を保存
}

backward128(...) {
    ctx = init128(...);
    for (node : bottom_up_nodes) {
        build_Z_elements_for_required_sides(ctx, node);  // ownerだけX側も作る
        __syncthreads();
    }
    atom = prepare_atom_in_this_thread(ctx);
    g = compute_G_for_atom(shared.dY, shared.Z_dY,
                           atom.output_plan, ctx.param_owner); // dX専用はvalueだけ
    shared.G_value[ctx.atom_id] = g.value;
    if (ctx.param_owner) {
        h = recompute_H_fields(shared.X, shared.Z_X, atom.input_plan,
                               {value, center, width});
        grad = atom_parameter_vjp(atom, h, g, atom.norm);
        store_parameter_partial(ctx, grad);
    }
    __syncthreads();                         // 全員がZを読み終えた
    build_C_elements_from_G(ctx);            // ownerはZ_X、他はZ_dYのarenaを使う
    __syncthreads();
    compute_direct_plus_C_and_store_dX(ctx);
}
```

各nodeのZは`q=tid; q<4*17; q+=128`で分担する。
backwardの二側を同じnode段階で作る場合は`side,batch,basis`を平坦化して分担する。
親の偶数modeだけを、互換な子bankの対応modeから継承する。残りはsourceから評価する。
bank ID順で子が先になることをCPUで確認したが、実装時も依存順を検査する。

Cは`q=tid; q<12*4*17; q+=128`で分担し、各要素の担当threadが全対象atom・成分を足す。
H/G生成時とは分担軸が異なる。全threadが一つのatomのH/Gを協調計算する方式には戻さない。
最終出力は第2節のbatch/site対応で一要素ずつregisterのaccへ直接部分とC展開を足し、globalへ保存する。
一集団なのでYpart/dXpartのshared配列や集団間の最終部分和は不要。

全blockの全threadが全block同期へ参加する。無効batch・空支持を理由にthreadを途中returnさせない。
param_ownerはblock全員で同じ値であり、その分岐内部へblock同期を追加しない。

## 7. globalに残すものと、まだ決めていない実装

X・dY・P・Y/dX、normとその微分、bank/node等の不変な記述はglobalから参照する。
生profileの混合後normは各側128 sites全体から準備し、tileごとに正規化しない。
例えばnorm `[128,2,3]` FP64だけで6,144 bytes。振幅0でもdAmpを計算する。
再計算はforward時点のP等を使い、autogradの元Tensorのversion検査を保つ。

係数は担当threadがatomの成分ごとに用意して使う案とし、全atomのNativeやBetaを
新たな常駐配列として必須にしない。このcompactな準備は未実装。
既存`executor._prepare()`をそのまま呼ぶとCo/Ci/Beta/Native等のglobal確保は残るので、
上のshared配置を実装しただけで総メモリが減ったとは扱わない。
既存準備を再利用する比較と、compactな準備へ替える比較を区別して記録する。

batch32のdP部分和をFP64 `[8,128,4]`で持つなら32,768 bytes。
最終dPはPと同じdtypeへ変換する。FP64 scratchをFP32最終勾配とそのままaliasしない。
batch4ならatom担当threadからFP32最終dPへ一度保存する構成にできる。
optimizer状態・snapshot・Graph pool・register退避等は上表のshared値に含まれない。

Z/Cのarena再利用は、全atomの縮約が終わるという**このphase型構成の条件**を使う。
元の動的job案でZ利用とC蓄積を重ねる場合は寿命が重なるため、このoffsetを流用しない。
H保存・split_groups・複数集団へ広げるときは、それぞれの寿命から配置を作り直す。

## 8. CPUで確認した範囲と次の検証

[容量計算の実行例](examples/block128_layout.py)は、配列のoffsetとphaseごとの寿命を検査する。
全要素の一意なthread担当と端数batch、dPのtile 0所有もCPU上で確認する。

```bash
python3 docs/research-history/cuda-linear/examples/block128_layout.py
```

forward 21,376 / backward owner 38,656 / dX専用payload 21,376 bytesを出力する。
これはallocatorピークや、CUDA同期の実行成功の検査ではない。
次はこの小さい構成でCUDAのprepare・Z・atom縮約・C・最終出力を実装し、
独立FP64参照でZ/H/G/C、Y/dX/全パラメータ勾配を順に照合する。
その後にGPUの同期・退避・予約量と完全stepを確認してから、集団数・発行方法を比較する。

CPU上の照合では、現在のbank表が12 banks / 28 nodesであること、
各親の子nodeが先に並ぶこと、各32-site担当範囲を含むnodeがbankごとに一つあることを確認した。
容量例ではnode数0/1/7/28とC枠0/1/12の組み合わせについて、生存期間が重なるviewの
addressが重ならないことを確認した。例のPythonに対するRuff検査も行う。

数え方の参照：
[executorのbank_families / tables](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/executor.py)、
[既存の有限支持とVJP](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/kernels.py)。
