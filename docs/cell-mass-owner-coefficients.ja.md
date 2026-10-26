# 所属タイルのCを作り、近隣タイルで再利用する

1024の初回完全stepでは、再計算案9.64〜9.65msのうち、別測定のYが3.73ms、
dXが3.17msだった。同じatomがK=3タイルへ寄与するため、その係数を何度も生成し、
各splitでsiteへ展開してから部分Y/dXを合計していた。
この重複を、所属タイルごとのCとbegin/endで減らす。
数学のprofile・窓・誤差基準・入出力dtypeは変えない。

実行設定`owner_coefficient_reduction=True`の候補として実装し、既定値はFalseのまま比較する。
1軸1bankという現在の範囲で使う。同じ周波数・周期のCを合計する設計なので、
異なるbankの係数をそのまま混ぜる一般化はしない。

## ソート済み行から、必要な区間だけ読む

入力のタイル数をnI、出力をnOとすると、キーは従来通り

```text
key = output_owner * nI + input_owner
```

stable sortした行番号`rows[A]`は維持する。
ソート済みキーに`searchsorted`を適用し、`offsets[nO*nI+1]`を追加する。
atom本体も、入力用の第2の行番号配列も複製しない。

```cpp
// 出力owner qが使うatomは、入力方向をまとめた一つの区間。
begin = offsets[q * nI];
end   = offsets[(q + 1) * nI];

// 入力owner rが使うatomは、出力ownerごとの連続区間。
for (q = 0; q < nO; ++q) {
    begin = offsets[q * nI + r];
    end   = offsets[q * nI + r + 1];
    consume_contiguous_range(begin, end);
}
```

1024のcaseはnI=8、nO=4。
forwardでは1区間、dX側では4区間を読む。
後者もatomごとのランダムな検索リストにせず、それぞれの連続区間をchunkで読む。
空groupのbegin/endは等しい。範囲は元の行番号ではなく、ソート済みのslot番号。
H/Gもslot順なので、そのまま対応する値を読める。

## Producer: 各atomは所属ownerのCへ一度だけ寄与

出力側のownerをq_aとし、そのownerのunwrappedなchart frameでのatom係数をd_a,mとする。

\[
C^{owner}_{q,b,m}=\sum_{a:q_a=q}\alpha_a H_{ba}d_{a,m}.
\]

入力側はHをGへ、出力profileを入力profileへ置き換える。
blockは`(owner, batchの4行, atom split)`を担当する。
複数の連続区間がある場合は、それらのchunk数の合計をsplit数で割り、
区間をまたいで作業量を分担する。

```cpp
ranges = get_owner_ranges(owner, offsets);
chunk_range = split_total_chunks(ranges, split_id);
C_in_registers = 0;

for (atom_chunk : chunk_range) {
    // thread 0..31が、それぞれ別のatomの係数を生成する。
    load_atom_and_make_owner_coefficients_into_shared();
    load_H_or_G_chunk_into_shared();
    __syncthreads();

    // batchとFourier modeの担当を分け、32atom分をCへ足す。
    accumulate_owned_C_elements_in_registers();
    __syncthreads();
}
store_partial_owner_C();
```

H/Gの生成では引き続き各threadが異なるatomを担当する。
このC producerでも係数生成は1thread/atomで、32atomを同時に準備する。
Cの各要素の和は別々のthreadが担当する。
正確なowner区間を取得済みなので、atomごとの支持所属判定をCの内側で繰り返さない。

producerのsharedは32atom分の係数・振幅・4batchのH/Gだけ。
Cの和はregister。グローバルへ保存するのは、後段が合計する小さなC部分和。

## Consumer: K個のownerを合計し、siteへ一度だけ展開

出力タイルtは、固定されたK個の近隣ownerからCを受け取る。
chartの端をまたがなければ、Cをそのまま足す。
端をまたぐ場合だけ、owner frameとtarget tile frameの差をFourierの位相で直す。

harmonic kについて、\(\theta=2\pi k\,image\,Q/P\)なら

\[
\begin{bmatrix}C_{cos}^{target}\\C_{sin}^{target}\end{bmatrix}
=
\begin{bmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}C_{cos}^{owner}\\C_{sin}^{owner}\end{bmatrix}.
\]

この回転は個々のatomに対して繰り返さず、集約済みCへ適用する。
QがPの整数倍とは仮定しない。DC成分は回転しない。

```cpp
for (owned_batch_mode) {
    C = sum_splits_and_neighbour_owners_with_seam_rotation();
    shared_C[owned_batch_mode] = C;
}
__syncthreads();

for (owned_site) {
    Y_for_4_batches = 0;
    for (mode) {
        basis = integrated_site_basis(site, mode);
        add_shared_C_times_basis_to_4_batches();
    }
    store_Y_for_4_batches();
}
```

各siteの基底値は4batchで共用し、splitごとには展開しない。
出力は一意writerで書く。同じstream上のproducer→consumerという依存で、
blockをまたぐ待機ループや出力atomicを追加しない。

## メモリの寿命と上限

| 配列 | 保存先・寿命 |
|---|---|
| 元atom、X、dY、H/G、Z | 従来の経路と同じ |
| rows[A] | global、forwardからbackwardまで |
| offsets[nO*nI+1] | global、forwardからbackwardまで |
| 32atom分の係数・H/G | producerのshared、現在のchunkだけ |
| Cの部分和 | producerのregister |
| partial_owner_C | global、YまたはdXの呼び出し内だけ |
| 合計したtarget C | consumerのshared |
| 最終Y/dX | global、一意writer |

C scratchはdoubleなので、対象軸のタイル数nT、基底数D=1+2Mなら

\[
bytes(partial\_owner\_C)=8\,S\,B\,nT\,D.
\]

N1024/B4/M8/S128ではY側272KiB、dX側544KiB。
同じS128のsite部分出力なら各2MiB必要になる。
両scratchを同時に保持せず、autogradにも保存しない。
offsetsはこのcaseでは33要素・264bytes。
実際のallocated/reserved peakはallocatorと他のbufferも含めて別に測る。

この経路だけ`atom_splits=128`まで比較する。
既存のsite部分和経路は引き続き64までで、既定の実行方法は変えない。

## 検証と実測

source `799a9013`。CPU2006 passed / 3258 skipped / 既存警告18件（42.81秒）。
CUDA・数学510 passed、failed/skippedは0。
1/3/5タイル窓、周期端、M16、batchとatomの0・端数、FP32/64、
両Z方式・両係数方式、微分保存とdouble保持の有無、owner offsetsの境界を含む。
全52,428atomのY/dX/全Parameter勾配・同一GPU fused AdamW更新を独立CPU oracleへ照合した。

L4、Torch2.11.0+cu130、CUDA13.0。N1024/A52428/B4、FP32、double Z、
微分再計算、値検査とsortを含むforward/backward/fused AdamW。
2fresh process、逆順、warmup10/同期wall30標本。同じsourceでの比較。

| 方式 | step ms run0 / run1 | allocated MiB | reserved MiB |
|---|---:|---:|---:|
| site部分和・幅128/split32/128thread | 9.619 / 9.632 | 5.5298 | 8 |
| owner C・幅256/split32/128thread | 3.839 / 3.840 | 5.5303 | 8 |
| owner C・幅256/split128/128thread | 3.788 / 3.785 | 5.5303 | 8 |
| owner C・幅256/split128/64thread | 3.746 / 3.754 | 5.5303 | 8 |
| dense通常 | 0.447 / 0.458 | 32.313 | 42 |
| dense workspaceなし | 0.619 / 0.665 | 16.063 | 22 |

最短案は約2.57倍の改善。追加allocatedはoffsetsの512bytes分のみで、reservedも維持。
時間は通常denseの8.37/8.20倍で、まだ目標の約2倍には届いていない。
別測定のowner Yは約0.52ms、owner dXは約0.47ms。
これらは準備済みfloat H/Gを使用し、旧Y/dXの別測定はdouble H/Gを共通使用している。
上の完全step比較では両方式ともfloat H/Gで、条件をそろえている。

producerはSHARED9728bytes/block・REG94/thread、consumerはSHARED1056bytes/block・REG60/thread。
LOCALは0。グローバルにC部分和を置くことで、係数の再生成とsplitごとのsite展開を減らした。
その後の[G/Parameter勾配の融合](cell-mass-fused-pullback.ja.md)は別の変更として測定する。

job `l4job-c908feb32ca34c47866c2c274ff38f89`:

- source SHA256: `71ee7e95b33a15454d45b967b010788eded0a19844d23bb54e8b2bdfa3c0b6e3`
- result SHA256: `d5e774335ab59b044a31b00104bed928acdc1a5ea228930b8ee5da6f2b955aef`
- driver SHA256: `26d17d6349b096ca01f2fa7fe68d42de5d23f46e4b806259ff86aeda54cdccdb`
- binary SHA256: `268d602eb206066383062d0b0f62b2e1ae4a0461919f00d222dd44b72b600456`

manifest117ファイル、321runtime source、driver、128 helper、binaryを照合した。
wheel/sdistも321ファイル一致。rawはignored `output/cell-mass-1024-20261027/owner/`。
再現はそのsource archiveのsrcとrootをPYTHONPATHへ追加、空directoryへ`CST_JOB_OUTPUT`を設定し
`__pool_driver__.py`を実行する。公開dispatcher・複数bank・多次元・Graphは未実装。
