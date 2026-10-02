# Benchmarks

本体の正しさ・baseline・速度・メモリを継続して確認する測定コード。
現在の入口と測定範囲は [CUDA Linear](cuda/linear/README.md) にまとめる。
試作コードは本体の測定ツリーへ残さず、独立した研究 branch で扱う。
過去の実験・採否は [研究履歴](../docs/research-history/cuda-linear/README.md) に保存する。

完成した結果JSONはignored `output/` に出力し、[提出用Issue](../docs/benchmark-contributions.ja.md)
から共有Neonへ送る。結果をGitへコミットする必要はない。
一般は1日10実行・1点、管理者が認定したアカウントは件数上限なし・既定10点。
設定の正本は [公開policy](../.github/benchmark-submissions.json)。
