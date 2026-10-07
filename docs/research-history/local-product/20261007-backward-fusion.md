# Small Linear: owner-local backward fusion

## 狙いと数式

N64/N128の正規化local polar productを保ち、dXが計算するGをsource VJPでも使う研究候補。
`kernel/fused-backward-vjp` / [PR #57](https://github.com/takumiecd/torchcst/pull/57)。
公開選択器の変更はない。

各atomの正規化した入力・出力profileをv、u、振幅をaとすると、

```text
H[b,t]   = sum_j X[b,j] v_t[j]
G[b,t]   = sum_i dY[b,i] u_t[i]
dH[b,t]  = sum_j X[b,j] (dv_t[j]/dc_in)
dG[b,t]  = sum_i dY[b,i] (du_t[i]/dc_out)
Y[b,i]   = sum_t a_t H[b,t] u_t[i]
dX[b,j]  = sum_t a_t G[b,t] v_t[j]
da_t     = sum_b H[b,t] G[b,t]
dc_in_t  = a_t sum_b dH[b,t] G[b,t]
dc_out_t = a_t sum_b H[b,t] dG[b,t]
```

full-domainのnormとgammaを含む既存factor微分と既存source Polar cotangent変換を共有する。
可変幅を含むproduction AdamW/Polarの更新方式を変えない。

## 実装と不変条件

dXを受け取る16入力siteのownerのうち、clipped input_loを含む一つがsource VJPを書く。
atomが複数ownerにまたがっても、source VJPは各batch tileで一回だけ書く。
owner内のatom block分割も排他的。入力方向のorderからcanonical Parameter IDへ戻す。
batch tileごとのsource partialをゼロ初期化し、未訪問の空支持atomは正確なゼロを保つ。
最後にbatch partialをreduceする。直接singleton prefixもdaを保存し、certified singletonの
center微分をゼロにする。norm-floorが効いた一点支持や片側だけのsingletonは一般経路の微分を保つ。

両方の勾配を要求したときだけ融合し、dXのみ・sourceのみは既存経路へ戻る。
forwardごとの13-field snapshotを保持するため、古いforwardと新しい支持更新は干渉しない。
追加のglobal H/G bufferは作らない。実際のL2/shared/register residencyやDRAM trafficは未測定。

explicit `_fusedback` routeはuncached atom16 parameter4/batch2とcounterfree repair4の二つ。
既存routeの意味・設定はそのまま。fusion部分のconstexprは既存recipeではFalse。

## 検証と比較

CPU最終検査: 894 passed / 1038 skipped。8件の宣言・paired prepare、Ruff、wheel build PASS。
L4 gate: 53 passed (51 CUDA + 2宣言)。切り出し/矩形domain、支持境界・singleton・空支持、
Y/dX/全source VJP、batch1/19/32/64、retained/outstanding backward、片方だけの勾配、
20回のcaptured production更新とAdamW moments/stepsを検査。
追加の完全空支持partialゼロ検査は性能jobの前に独立して実行する。

初期実rho>1、N64/N128 B32 A204/A819、FP32 IEEE、seed41、21 timing samples/execution。
各Planを全サイズの独立FP64 oracleで照合してから既存runnerで完全stepを測る。
現行・候補・denseのParameter/input/target hashを照合し、capture/replayを含む
peak allocated/reservedを報告する。phase/compiler診断を完全stepの代用にしない。

性能結果: 回収中。初回N64 attemptは5 Plan全サイズFP64と2件のexact-zero検査がPASSした後、
独立parameters区間のevent未記録により診断でFAIL。失敗した比較から性能を採用しない。
融合用dx_source_vjp/source_partial_reduce区間を明示する修正を`ceed4417`に記録。
レジスタ/spill、zero-fill launch、dX scratchとsource partialの同時生存、
owner間のsource仕事量の偏りを採否判断で確認する。GPU gateだけでは高速化を主張しない。

## 保全先

- ignored raw: `benchmarks/cuda/linear/evidence/backward-fusion-20261007/`
- pool: `~/.local/state/colab-l4-pool/jobs/JOB_ID/` のsource/result archiveとreceipt
- gate job: `l4job-aa863e77ea7f4d6fa5647f8c3fa465ba`
- gate source commit: `838d94da83c4903eecce19ed28969959fc2f2b19`
- gate source SHA256: `62ff1f171709678f050e7932c84a01e962cd9b31b5998da4c9abcdc6d2f8b847`
- measurement source: `42202e8c` (追加testのみ、GPU gateと197 runtime files一致)
- initial screen N64: `l4job-5afa74ee13e2416698bc88a6f0cf49c4`
- initial screen N128: `l4job-e6bff3f46879445f89a0a5e3d1448728`
