# 2026-10-07: exact order cacheの保存表現を小さくする

Branch `kernel/order-cache-compact`、PR #37 checkpoint `dee61370`を基点とする。
PR #36/#38のtile/VJP変更は含まない。sortを省くexact-key条件と最新payload/支持更新は
PR #37と同じで、cacheだけから不要なcanonical ID項を除く。

## Equalityとbound

canonical row aとatom数 Cはcacheの生存中に固定される。full sorting keyは
`k_a = L_a*C+a`、`L_a = bucket_a*span+position_a`、`span=max(K,N)+1`。
したがって同じaの前回/現在で`k_a==old_k_a` iff `L_a==old_L_a`。
cacheへLだけ保存してもrebuild/reuse判定は完全に同じ。rebuild時のsortは引き続き
full keyを使いcanonical IDのtie-breakも維持する。support endだけの変化は毎回の
owner範囲更新で扱い、coefficients/normalizers/幅/位置も省略しない。

Domainはcount<=128、interval開始位置は0..countへclipされる。安全な上限
`(ceil(max(K,N)/16)+5)*(max(K,N)+1)`はN128でも1677。Lの符号付きint16保存は
量子化を含まない。cache IDはC<=32768ならint16 (最大ID32767)、それ以上はint32。
より広いlogical boundへ備えてkey dtypeはint32/int64 fallbackを持つ。
default routeのfull composite/int32 IDsは変更しない。

cacheからIDを読むときはint32へ戻し、per-forward Ordersは従来のint32 snapshot。
mutable cacheをbackwardへ保存せず、全位置・dX・atom gradientsを維持する。

## 候補とtensor予算

copy8/range8の2 compact routes、controlsはcopy8とfull-key-cache copy8。
N128/A819のresident buffer予算: keys3276 + IDs3276 + offsets104 + counters48
=6704 bytes (full-key cache13256 bytes)。N64/A204では1752 bytes (full3384)。
これはbuffer予算であり、実測capture/replayピークの短縮とは区別する。
物理L2常駐やDRAM削減をこの表現から主張しない。Hのglobal tensorは追加しない。

同じN64/N128 B32 A204/A819、seed41、FP32 IEEE、TF32 off、初期decoded rho>1、
rho1.25/3/8/mixed、21samples、production fused AdamW/Polarを比較する。

## Host検証 (03:29 JST)

CPU:866 passed / 617 skipped (23.99s)、変更対象ruff、8 cases prepare/check、
wheel/sdist build成功。GPU skipは成功証拠ではない。
4 CPU boundary testsはC0/1/32768/32769のID rangeと全logical stampsの無損失保存を確認。
新20 GPU tests + declarationはFP64 Y/dX/全atom・位置、slices/B1/32/64、
N64/N128の20 Graph更新・moments/step、old backward、key reuse/end更新/
band/singleton/inactive changesとfresh full sortのexact snapshot一致を確認する。
GPU結果は未回収。raw driver/analyzer/DBはignored
`benchmarks/cuda/linear/evidence/owner-cache-compact-20261007/`へ保存。
