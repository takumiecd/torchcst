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

## 初回の結果: 32 atoms/iteration

同一構成のmatched controlとの完全step時間（us）。各条件は1 execution / 21 samples。

| size / 初期rho | uncached現行 → 融合 | repair4現行 → 融合 | dense |
| --- | --- | --- | --- |
| N64 / 3 | 51.49 → 50.62 | 52.38 → 50.47 | 38.79 |
| N64 / 8 | 51.15 → 53.20 | 52.06 → 51.93 | 39.26 |
| N128 / 3 | 68.31 → 70.38 | 68.60 → 71.90 | 45.80 |
| N128 / 8 | 70.25 → 80.82 | 65.54 → 77.47 | 45.69 |

matched allocatedはN64で+5120 B、N128で+1024 B。reservedは6291456 Bで同じ。
N64の融合は201 registers / 0 spill slots、N128は255 / 62。幅広N128は15〜18%悪化。
32-atoms版は採用しない。N64/rho3の小さい改善だけを一般化しない。
数値・compiler・job/source/archive hashは`20261007-backward-fusion-screen.json`に保存。
DBへ4 artifactをimportし、byte-identical exportと同じprovenanceでのidempotent再importを確認。

初回のN64/N128 attemptは5 Plan全サイズFP64と各2件のexact-zero検査がPASSした後、
廃止した独立parameters区間のevent未記録により診断でFAIL。失敗した比較から性能を採用しない。
融合用dx_source_vjp/source_partial_reduce区間を明示する修正を`ceed4417`に記録。
失敗archiveとreasonは`20261007-backward-fusion-failed.json`に保存。

## 次の候補: 融合だけ16 atoms/iteration

forwardのatom_block32、batch16、owner16入力siteはそのまま。
両勾配を融合する場合だけatom blockを16へ変え、ライブ値とspillを減らす。
新しい明示的route `_fusedback16` を追加。32-atoms routeの意味は変えない。
`b0bbc45d` のCPU 894 passed / 1064 skipped、8 paired prepare、宣言、Ruff PASS。
L4 `l4job-0c05ac42b2e344c68543db8bb6fe8c55`: **26 CUDA tests PASS / 27 deselected**。
197 runtime filesのcommitted/submitted/worker SHA256一致を確認。

性能jobはN64/N128の初期rho1.25/3/8/mixedを全サイズFP64 oracleで再確認してから測る。
`l4job-b1d6c9360d6142d1a622c0159d3605d3` / `l4job-98a44095052f48d08a9ce05b11912dbc`。
初回8条件すべてPASS。N64/rho1.25とmixedは独立jobでcase/Planを逆順に再測定した。
`l4job-77fdb17e50724f53915c1eae2ec6f030`も197 runtime files一致、各Plan全サイズoracleと
4 exact-zero tests、標準runner PASS。narrow/mixedは2 executions/condition、他は1 execution。

| N64 / 初期rho | uncached現行 → 融合16 | repair4現行 → 融合16 | dense |
| --- | --- | --- | --- |
| 1.25 (2 runs median) | 50.16 → 47.61 | 50.17 → 46.71 | 38.81 |
| 3 | 51.01 → 52.67 | 52.21 → 52.15 | 39.07 |
| 8 | 51.14 → 55.11 | 51.81 → 59.34 | 38.79 |
| mixed (2 runs median) | 55.95 → 54.93 | 56.84 → 54.33 | 38.90 |

repair4のmatched時間短縮はrho1.25で7.15% / 6.65%、mixedで4.23% / 4.59%。
最速の現行controlと比べても両runで改善する。N64は96 registers / 0 spill、allocated +5120 B。
N64/rho3はほぼ同等〜悪化、rho8は8〜15%悪化するため一般採用しない。

| N128 / 初期rho | uncached現行 → 融合16 | repair4現行 → 融合16 | dense |
| --- | --- | --- | --- |
| 1.25 | 64.14 → 64.56 | 60.34 → 61.04 | 45.49 |
| 3 | 67.41 → 68.75 | 68.80 → 69.82 | 45.52 |
| 8 | 70.01 → 76.77 | 65.64 → 73.15 | 45.50 |
| mixed | 69.44 → 73.78 | 66.55 → 71.05 | 45.48 |

N128は255→128 registers、62→0 spillにできたが、全条件で悪化。allocated +1024 B。
両サイズのmatched reservedは6291456 Bで同じ。N128は現行を維持する。
compilerのspill除去だけでは完全stepの改善を証明できない。
原票とpaired statisticsは`20261007-backward-fusion-tuned.json`。

L4 drain後に全owned slots停止を確認。再現したN64のrho1.25/mixedだけをG4の1 short jobで比較する。
`colabjob-d5c79375cfd840f8aa270ebb032ddb6a`。結果回収中。公開dispatchやmain統合は未変更。

## 保全先

- ignored raw: `benchmarks/cuda/linear/evidence/backward-fusion-20261007/`
- pool: `~/.local/state/colab-l4-pool/jobs/JOB_ID/` のsource/result archiveとreceipt
- gate job: `l4job-aa863e77ea7f4d6fa5647f8c3fa465ba`
- gate source commit: `838d94da83c4903eecce19ed28969959fc2f2b19`
- gate source SHA256: `62ff1f171709678f050e7932c84a01e962cd9b31b5998da4c9abcdc6d2f8b847`
- measurement source: `42202e8c` (追加testのみ、GPU gateと197 runtime files一致)
- initial screen N64: `l4job-5afa74ee13e2416698bc88a6f0cf49c4`
- initial screen N128: `l4job-e6bff3f46879445f89a0a5e3d1448728`
