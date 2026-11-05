# 検証済みCUDA研究の統合

[English (canonical)](pr-consolidation-20261105.md) | 日本語

2026-11-05、所有者はPRの整理と統合を要求した。PR #140の対象をmainへ変更し、
squashや研究履歴の書き換えをせず、各commitを保持する。
整理前のheadは76204ef47505f1945cdaba4de581b2ce4aa377c7。

## 統合する実装と証拠

- #128〜#132: benchmark内のguard付き完全step Graph、明示的FP32
  owner-window Plan、geometry共有、一時sort keyのint32化、Cの並列化。
- #136〜#137: 順序付きweak-root topologyの正しさの修正、stock/callableの
  共通資格付け、限定したweak-leaf guard最適化。
- #139: 大規模shape対応と、修正後のG4での16384までの資格・測定。
- #140: 完了済み1epoch HIGGS中間層の密度・幅比較とsource/data/result来歴。
  後続の確認実験とは区別する。

いずれも明示opt-in。自動Plan選択と通常入口のGraph拒否は維持する。
benchmarkのGraph cohortは汎用の公開CUDA Graph学習APIではない。
FP32 bodyは同じ演算則に従うが、FP64評価とのbit一致を主張しない。

記録したgateには独立なY/dX/全atom勾配のFourier・quadrature参照、
optimizer・勾配・時計比較、lifecycle・メモリ確認、両順序45000更新がある。
各revisionの範囲を読むこと。#131は元のlong-memory失敗flagと事後の補足を保持し、
補足を事前gate成功へ読み替えない。#139の大規模側は768更新の短期実験とhash比較で、
直接の状態byte比較は2048で行った。長期品質の証拠ではない。
兄弟branchのG4確認#138は文書として別途統合する。

## 統合しないものと継続するもの

#133〜#135は未mergeでcloseする。完全stepはそれぞれ2.1〜3.2%悪化、
固定5%基準に対して2.40〜2.58%改善のみ、約6.8%悪化で停止条件に到達した。
branch・commit・保持した証拠は残す。PRのcloseはそれらの削除ではない。

#141は10/20epoch・3seedのHIGGS確認を実行中。#143は別のDST pilotでGPU資格待ち。
準備や1つのdevelopment seedを、完了した確認結果と扱わない。
mainへの統合にかかわらず、既存runの固定source identityを保つ。

モデル単位の計画と今後のrunnerは[cst-research](https://github.com/takumiecd/cst-research)
で整理する。kernelの正しさ・実装検証はtorchcstに残す。
mainを対象とする統合PRでCIを再実行する。この整理文書は新しい算術、dispatcherへの
採用、branch削除、worktree整理や証拠削除を導入しない。
