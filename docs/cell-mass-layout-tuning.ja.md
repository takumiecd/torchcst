# begin/end生成とthread数の最後の切り分け

[共有Z窓](cell-mass-owner-windows.ja.md)の後に、prepare・C producer・小batchの
Parameter勾配を切り分けた。数式・whole-tile窓・double内部精度・許容差は変えない。

## 採用した変更

32bitの整数keyを作り、stable argsortで元atom行の順序を得る。
この研究入口は各軸1024以下なので、owner pairのキーは32bitに収まる。
次のkernelでは各threadがgroup境界を一つ担当する。

```cpp
parallel_for(group = 0 .. group_count) {
    first = 0; last = atom_count;
    while (first < last) {                 // このthreadだけの二分探索
        middle = first + (last-first)/2;
        if (keys[sorted_rows[middle]] < group)
            first = middle + 1;
        else
            last = middle;
    }
    offsets[group] = first;
}
// group g の範囲は [offsets[g], offsets[g+1])。
```

sorted keyのコピーとquery番号の配列は不要になる。
空groupやatom0でも同じコードで空範囲を返す。
値検査・エラーの優先順・1回のhost確認・stableな同値key順は維持する。
64bitキーもnative検証用に残し、順序と全境界を32bitと照合する。

C producerのsharedは、最大mode数の配列ではなく、実際のbasis数Dに合わせて確保する。
32atom chunkなら \((32+4\times32+32D)\times8\) bytes。
M8/D17で5,632bytes、M16/D33で9,728bytes。
C producerのthread数は32/64/128で指定できるが、site展開は128threadのまま。
thread数で数学のtile・窓・係数・atom所属は変わらない。

## 採用しなかった小batch専用化

batch<=4の場合、入力profileを使い終えてから出力profileを作り、
両profileのregister生存期間を短くする候補を試した。
全勾配・更新の検証は通ったが、step時間の改善は確認できなかった。
最終source `c9b67dd4`ではこの専用分岐を外し、汎用の同一thread再投影を残した。
候補の復元先はcommit `f742151a`。

cuobjdumpではFP32入力/FP64 Zの融合kernelは、汎用REG236/STACK40、
専用REG140/STACK232、どちらもLOCAL0と表示された。
register数の減少だけから高速化とは判断しない。
この表示だけで実際のmemory trafficやstall原因も断定しない。

## 同じsourceでの比較

候補source `f742151a`、L4 / Torch2.11.0+cu130 / CUDA13.0。
N1024/A52428/B4、FP32、M8、sigma初期0.2..12、Q128。
Z窓・owner C・G/Parameter勾配融合、double Z、幅256/split128/atom thread32。
CSTは検査・sort込みのforward/backward/fused AdamW。
2fresh process、逆順、warmup10/同期wall30標本、Graphなし、TF32無効。

| 経路 | step ms run0 / run1 | allocated MiB | reserved MiB |
|---|---:|---:|---:|
| 汎用勾配・C128thread | 1.847 / 1.848 | 5.003 | 8 |
| 小batch専用勾配・C128thread | 1.877 / 1.873 | 5.003 | 8 |
| 小batch専用勾配・C64thread | 1.872 / 1.868 | 5.003 | 8 |
| 小batch専用勾配・C32thread | 1.854 / 1.857 | 5.003 | 8 |
| dense通常 | 0.429 / 0.439 | 32.313 | 42 |
| dense workspaceなし | 0.615 / 0.613 | 16.063 | 22 |

Cのthread数による差も小さく、普遍的な32thread優位とはしない。
既定128を維持し、設定だけ独立して選べるようにした。
汎用勾配の時間比は通常denseの4.30/4.21倍で約2倍の目標には未達。
前のjobからallocatedは約5.00MiBのままだが、reservedは6→8MiBとなった。
allocatorの確保履歴も含む値なので、配列容量から同じreservedになるとは主張しない。
それでもworkspaceなしdenseのallocated/reservedをともに下回る。
moduleの定数表9864bytesと全process消費量はこのallocator値に含まない。

別のevent診断でprepareは0.16..0.17ms、H単独0.17ms。
小batch専用G/Parameter融合は0.69..0.71msで、引き続き重い。
C128ではowner Y0.35/dX0.29ms、C32では0.32/0.30ms。
全stepの判断にphase時間の和やregister数だけを使わない。

## 検証・証拠

CUDA・数学1,301 passed（36.84秒）、failed/skippedなし。
新しいC thread32/64、FP32/64、M8/M16、batch0/末尾、atom0を含む。
全52,428atomのY/dX/全4列勾配・AdamW更新delta/moments/stepは全対象設定で通過した。
更新deltaとstepの最大差は0。許容差を緩めていない。
CPU2006 passed / 4049 skipped / 既存警告18件（43.26秒）。
wheel/sdistは両packageと321runtime sourceがbyte一致した。

job `l4job-7c675b51261644dd905a24651e4101b9`:

- source SHA256 `e3a2344a08988abff8ff99413388aaef4faa8d7b6d14f04fcc6fad18ca8b637c`
- result SHA256 `7e04abcde10fec115c7307a6704dbdba852ede90a7f628dab9efc9291f9ff37a`
- driver SHA256 `5da26bdb3153d2e62cfd2cd5cb8053f0919ae11ae68a2a4660ebd4355f996b84`
- binary SHA256 `94e9483e813d421b64e70043607978950a500fe502956f4200beb1cda1c2a2e7`

165manifest files、321runtime source、driver、128helper、binaryを照合。
ignored `output/cell-mass-1024-20261027/launch-tuning/`へ保全。
最終sourceの再確認と採用設定は[実測状態](cell-mass-status.ja.md)を参照。

次に大きく減らす対象は、投影とC生成でのatom係数の再評価。
同じ32atom chunkでH/G投影とC部分和まで融合する案はあるが、現時点では未実装・未測定。
全atom係数表をglobalに保存する案や精度を下げる案は、この結果へ含めていない。
