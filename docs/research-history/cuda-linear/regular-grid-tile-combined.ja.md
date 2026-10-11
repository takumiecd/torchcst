# RegularGrid: 支持幅生成と dH/dG 即時消費の組み合わせ

## 比較の目的

支持幅に合わせた16-site panelの生成と、dH/dGを生成したkernel内で
パラメータ勾配へ消費する方式を組み合わせる。
現行 `input-site-reciprocal-bm16-g16-h32`、固定幅 `tile-dot32-bm16`、
支持幅のみ `tile-window16-bm16`、即時消費のみ `tile-derivative-bm16`、
組み合わせ `tile-combined-bm16` を独立したplanとして比較する。
既存単独版の演算・保存方式は変えない。

組み合わせでは各BM16 chunkで H → G/dG → H/dH → dX の順に進める。
Hは支持幅panelで先に完成させ、G/dGの生成中にdA/dc_outを完成させる。
続く入力側の生成ではdHをGと縮約してdc_inへ消費する。
Hの値だけを作る追加passは残る。H/Gの二つのslabを使い、dH/dGは保存しない。
全支持範囲を縮約してからパラメータ勾配を作り、tile間の交差項を保持する。
全operator-siteに対する正規化、floor、singleton、source VJP、広支持fallback、
32-site ownerと集約方式は既存版に揃える。

## 固定プロトコル

RTX 6000 Ada、N=2048、B=32、A=209715、rho初期値3/8、seed41。
IEEE FP32、TF32無効。24回の実AdamW/Polar更新、21個の完全step計測。
各rhoで独立2cohortを実行し、2回目はplan順を反転する。
各planはfresh process。準備・routing sort・生成・集約・backward・optimizerを含む。
capture/replayを含むpeak allocated/reservedをdenseと比較する。
全224件のGPU qualification後、初期とpost24で独立FP64の
全軸・全atom・全4パラメータ勾配を4e-4の既存基準で検証する。
失敗した比較を再試行して選び直さない。

matrix/factor/onchipおよびdense controlも同じcohortで測る。
9 workers × 4 cohorts = 36 workers、756 samples、64 full FP64 gates。
詳細phaseは別の固定post24 snapshotで計測し、primary stepへ足し合わせない。
結果が確定するまでは研究用planに留める。
