# 有限Fourier profileへの移行と旧研究の整理

2026-10-13、main `641df5aa1ccf98cbaa654469fa965958fe2c88b0` を基準に
既存PR・研究worktreeを整理した。既に統合済みのAPI・実装・研究記録は維持する。
新しいprofileの採用やGPU性能の判断を、この整理で行うものではない。

| 対象 | 判断 | 根拠・保存 |
| --- | --- | --- |
| PR #100 / RegularGrid H/G研究 | runtimeをmergeせずclose | 方針終了。共通支持モードは現行late経路より1.26–8.35%遅く、H/G容量を削減しない。rho .25は全Parameter勾配gate未通過。ソース・成功/失敗・以前のメモリ改善も保存 |
| Product CSR / fused backward | runtimeを採用せず研究終了 | 既にmainへ統合した研究記録を維持。融合によるCSR比改善はnative/Torch比較の完全step採用gateを満たさない |
| Sphere direct / recompute / gather / adaptive precision | runtimeを採用せず研究終了 | 負の時間/メモリ結果と精度gate失敗を維持。FP64 floor修正は新しい研究ファイルの修正であり、mainの独立したruntime修正ではない |
| Periodic comparison / Sphere baseline / Torus成果 | 再mergeせず旧checkoutを整理 | 対応PRは既にmerge済み。mainの実装・記録を維持 |

## 保存と復元

所有者のローカルarchiveはrepoルートから
`output/research-archives/profile-transition-20261013/`。
`README.md` に復元コマンド、`inventory-before.json` に旧checkout/HEAD、
`branch-dispositions-inventory.json` に全71ローカルbranchとPRの対応を残す。

- `history.bundle`：全branch/remote ref/tagとstashを保存。
  SHA256 `2618a3d85179cc2d3261b41f34ca0afae4ba32866625feff6635564dfa35c220`。
  正確なhashはarchiveの `PRESERVATION_VERIFIED.json` を正本とする。
- 独立bare cloneで `git fsck --full --strict` PASS。全元refのobjectと
  9 checkoutのHEADを復元先で確認。stash refの移動により失われる履歴も
  recovery refに含めてbundle化した。
- 9個の `*-evidence.tar` は20,360ファイル、15,624,785,920 bytes。
  各memberをsourceのSHA256/sizeと照合した後に旧checkoutを削除。
  manifestにはsymlinkのリンク先も記録する。
- mainの既存 `uv.lock` 変更は専用stash、`uv.lock.before` とbinary patchへ保存。
  元の3件のstashも保持。mainの既存ignored evidence・資格情報は変更しない。
- cache、再生成可能なvenv/build/pycを除き、研究ログ・tensor・source snapshot・
  driver・receiptを保存。`research-notes/` は9 checkoutの読み取り用ノート写し。
- 70ローカルbranchと15リモートbranchはbundle保存後に整理。
  remote削除には保存済みHEADを条件とするleaseを使用。

共有Colab poolの3 slotは停止状態、旧checkoutを参照するローカルcontrollerはなし。
pool jobの原本と過去のremote Hub証拠は元の保存先に維持し、今回削除しない。
この確認に新規GPU割当や実験は含まない。

## 次の実装

有限Fourier級数そのものを新profileの定義とし、旧profileへの近似を目的にしない。
最初にFlatTorusの局所周期・tile選択・幅下限・M・L2・task微分を宣言し、
PyTorchの直接参照と共有タイル集計を照合する。Triton/CUDAの最適化・完全step
時間とallocated/reserved peak・dispatcherへの採用はその後の別段階である。
