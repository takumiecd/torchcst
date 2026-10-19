# 係数とHの保持量を減らす比較

目的はB32/B64混合σ条件の完全学習step時間をDenseの2倍以内、capture/replay込みのallocated/reserved両peakをDense以下にすること。未達時は目標を達成扱いにしない。

v17は同じforward状態のMeta（FP64）から `_jump` を呼び、FP64の係数を利用直前に生成する。係数全体とCutsのsnapshotは保持しない。support・正規化・Polar・task幅stop-gradient・全atom勾配契約は変更しない。H/Hμ/G/GμのFP32丸め境界も維持する。

Hは非保存・Hのみ・H/Hμ保存を比較し、係数保存かつH非保存も対照にする。非保存のbackward H再計算にも明示的FP64 fmaを使う。v14/v16等の既存Planの意味は保つ。配置は各forwardで検証し、古いHや古いσをstep間で使い回さない。

独立FP64 Y/dX/全atom勾配のmaxabs/relative L2閾値はともに4e-4。まず回帰と混合B32/B64の既存runnerを実行する。現時点で速度・メモリ改善は未検証。

正規化の追加候補 `norm_fma` は、区間内のquartic sampled normとその中心微分の積和を明示的FP64 fmaへ書き換える。積分近似やdtype低下は行わない。通常版は従来の演算順序を維持し、既存recipeにこの設定を暗黙適用しない。診断のnorm replayも選択した同じ算術を使う。
