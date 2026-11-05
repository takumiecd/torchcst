# G1の最終判断: 固定研究実行器のopt-inとして採用可

## 判断

固定stock MNIST、N1024/A52428/真B4、L4・Torch2.11.0+cu130の全step Graph研究実行器について、
**安全補強済みqualified G0に対するG1を、benchmark-localな明示opt-inとして採用可能**と判断する。
正式短期、二順序45,000更新、数値・optimizer状態・実行契約・capture込みmemoryの事前条件を通過した。
Proの公開コード・結果レビューも同じ限定結論で、今回の採用を阻む具体的不具合は指摘されなかった。
「有望案を絞り、採用可否まで検証する」今回のGoalの研究作業をここで終了する。

採用判断とmergeは分ける。PR #136/#137はDraftのままコード・結果を保全する。
公開dispatcher/default、汎用Graph API、他モデル/seed/batch/device/Torchへの採用を意味しない。
有限stock資格は、指定wheelと信頼できるclean bootstrapという既存の信頼境界を前提にする。
未知callbackのordinary eagerへのfallbackを含む契約を維持し、任意の先行monkeypatchを認証する機能とはしない。

## 判断を支える観測

- 正式短期: initial8/own45kとも6/6勝利、paired幾何平均短縮12.387%/13.048%。
- 二順序45,000更新: CST training短縮A19.715%/B13.189%、定義した準備・退役込み短縮A19.135%/B12.786%。
  両順序の両時計が個別5%条件を通過した。二順序平均で未達を救済していない。
- 保存した全epoch末の状態・勾配・clock・metrics/orderは、対照/候補・順序間・元#132参照と直接byte一致。
  先行all-atom Y/dX/dP Gateも通過し、src242/数値body/optimizer arithmeticは変更していない。
- 全pairのcapture込みallocated/reserved増分0/0。Stage2のprospective CST memory-v2.1とdense手動継続条件を継承。
- 正式短期raw4334/56 PT、長期raw441/24 PTを原SHA/sizeまで回収検証し、欠落・未検証・errorは0。

詳細・全セル値・source/result/driver/検証SHAは、
[正式短期](owner-window-flat-guard-stage3-amendment-20261105.ja.md)と
[長期結果](owner-window-flat-guard-long-20261105.ja.md)、それぞれの同名JSONを参照する。
元のharness失敗、CPU reader修正、事前に記録した一度限りの予算例外も残す。
旧unpaired基準を新pairへ接合せず、失敗を消した再測定や反復追加は行っていない。

## 帰属と限界

比較対象は共通安全補強済みG0であり、未変更#132からの純高速化を示したものではない。
denseにも同じ実行基盤変更を与え、training7.677%/7.032%、準備・退役込み7.290%/6.617%の短縮を観測した。
CSTとdenseの百分率を引いて固有效果へ分解せず、CUDA算術削減や全面的なCPU律速とも解釈しない。

長期は一つのL4 allocation内の二順序・8 fresh process、同じmodel/seedである。
A/Bの基準時間は変動し、その原因は未特定。短縮率の範囲は今回の観測値で、信頼区間や最低保証ではない。
job内cache共有、cold compile費用未分離を維持する。準備・退役込み時計はprocess全wallではなく、
evaluation/checkpoint/provenance I/O/import/data配置等の既存時計外費用を同率短縮したとも主張しない。
全45,000時点の全中間値やCPU metadata費用ゼロを証明したものではない。

Pro reviewはhead `8d70e8e29755773da2c9ce25a057799cd4ca627c`の公開差分・runner・結果を対象とした。
非公開raw PT/trace/実.so/最上位driverをPro自身が直接再検証したわけではなく、GPU独立再現にも数えない。
review turnは`6ceee6c2-d823-45fc-99d1-9e9c35167e37`、回答原文はignored evidenceへSHA付きで保全した。

## 保全して終了する範囲

K（限定periodの厳密FP32整数算術）とO（同一AdamW functorのview分割）は、source/CPU段階の待機案として分離する。
G1の成功はK/OのGPU正しさ・利益を保証せず、今回の完了のために追加実装・GPU探索は行わない。
#133–#135の負結果も変更しない。

worktree整理では12 full snapshot・復元manifest/helperを保全し、作業checkoutは4つとした。
元payload削除0。archiveは`~/.codex/research-archives/torchcst/worktrees-20261104T130657Z/`、
復元証拠はmainのignored `output/worktree-reorganization-20261104/`にある。
所有L4の結果回収・停止を確認済み。終了に伴う追加cleanup、merge、新しいGoal作成は行わない。
