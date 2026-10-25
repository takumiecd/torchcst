// 128-site版の「並べる規則」と「blockへ渡す範囲」。
// CUDAの処理とCPU検査が、同じ境界計算を使うための小さな関数だけを置く。
// ここでは有限区間の中心支持を扱う。空間の周期wrapは追加していない。
#pragma once

#include "block128_math.h"

namespace cst128 {

constexpr int HIERARCHY_GROUPS = 12;
constexpr int OUTPUT_TILE_SITES = 32;
constexpr int OUTPUT_TILES = N / OUTPUT_TILE_SITES;
constexpr int RANGE_AXES = 2;
constexpr int OUTPUT_AXIS = 0;
constexpr int INPUT_AXIS = 1;
constexpr int BLOCK_RANGE_COUNT = RANGE_AXES * OUTPUT_TILES * HIERARCHY_GROUPS;
constexpr int SORTED_PARAMETERS_BYTES = A * 4 * sizeof(float);

// 一つのatomを並べるための情報。Parameter本体はこのkeyには入れない。
// q_out/q_inは床関数で得た整数値だが、遠方の中心も扱えるようdoubleで持つ。
struct AtomSortKey {
    int group;
    double output_tile;
    double input_tile;
    int source_row;
    double support_radius;
};

// group内の連続区間と、group全体で必要な支持半幅の上限。
struct GroupSpan {
    int begin;
    int end;
    double maximum_radius;
};

// [begin,end)はソート後のslot番号。endそのものは含めない。
struct AtomRange {
    int begin;
    int end;
};

CST128_HD double group_tile_width(int group) {
    // 現行Specの最小幅0.2を、階層ごとに2倍にする。単位は座標と同じ。
    return 0.2 * (1 << group);
}

CST128_HD AtomSortKey make_atom_sort_key(const float *parameters, int source_row) {
    // P[row] = (振幅, 出力中心, 入力中心, 幅logit)。
    // 振幅が0でも勾配が必要なので、ソートから除外しない。
    const float *atom = parameters + source_row * 4;
    double width = 0.1 + 9.9 * sigmoid(atom[3]);
    int group = hierarchy_level(width);
    double tile_width = group_tile_width(group);

    // 混合している成分すべてを覆う上限。group内でさらに最大を取って使う。
    double maximum_radius = 0.0;
    for (int component_index = 0; component_index < 4; ++component_index) {
        Component part = component(width, component_index);
        bool needed = part.weight > 0.0 || part.width_derivative != 0.0;

        if (needed) {
            maximum_radius = ::fmax(maximum_radius, part.length);
        }
    }

    AtomSortKey key;
    key.group = group;
    key.output_tile = ::floor(static_cast<double>(atom[1]) / tile_width);
    key.input_tile = ::floor(static_cast<double>(atom[2]) / tile_width);
    key.source_row = source_row;
    key.support_radius = maximum_radius;
    return key;
}

CST128_HD bool atom_key_less(const AtomSortKey &left, const AtomSortKey &right) {
    // 主keyは(group, output_tile, input_tile)。
    // 全部同じなら元の行順を保つ。source_rowは空間の分類には使わない。
    if (left.group != right.group) {
        return left.group < right.group;
    }
    if (left.output_tile != right.output_tile) {
        return left.output_tile < right.output_tile;
    }
    if (left.input_tile != right.input_tile) {
        return left.input_tile < right.input_tile;
    }
    return left.source_row < right.source_row;
}

CST128_HD GroupSpan find_group_span(const AtomSortKey *sorted_keys, int group) {
    GroupSpan span{0, 0, 0.0};

    // groupが昇順なので、そのgroupの全atomは一つの区間に並んでいる。
    while (span.begin < A && sorted_keys[span.begin].group < group) {
        ++span.begin;
    }
    span.end = span.begin;

    while (span.end < A && sorted_keys[span.end].group == group) {
        span.maximum_radius = ::fmax(span.maximum_radius, sorted_keys[span.end].support_radius);
        ++span.end;
    }
    return span;
}

CST128_HD int lower_bound_output_tile(const AtomSortKey *sorted_keys, int begin, int end,
                                      double tile_number) {
    // 「出力タイル番号がtile_number以上になる最初のslot」を二分探索する。
    while (begin < end) {
        int middle = begin + (end - begin) / 2;
        if (sorted_keys[middle].output_tile < tile_number) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin;
}

CST128_HD AtomRange get_range(const AtomSortKey *sorted_keys, GroupSpan group_span, int group,
                              int axis, int output_tile) {
    // このblockが保存するのは、32 sites。X/Yともsite間隔は0.1。
    // group内の最大半幅だけ両側へ広げると、必要なatomを落とさず候補を取れる。
    double first_site = output_tile * OUTPUT_TILE_SITES * 0.1;
    double last_site = (output_tile * OUTPUT_TILE_SITES + OUTPUT_TILE_SITES - 1) * 0.1;
    double tile_width = group_tile_width(group);
    double first_bin = ::floor((first_site - group_span.maximum_radius) / tile_width);
    double end_bin = ::floor((last_site + group_span.maximum_radius) / tile_width) + 1.0;

    if (axis == OUTPUT_AXIS) {
        // forward：出力順のkeyなので、そのまま一つの連続範囲を取れる。
        int begin =
            lower_bound_output_tile(sorted_keys, group_span.begin, group_span.end, first_bin);
        int end = lower_bound_output_tile(sorted_keys, begin, group_span.end, end_bin);
        return AtomRange{begin, end};
    }

    // dX：入力keyはgroup全体では昇順ではない。
    // 最初の128版では、入力候補の最初〜最後を含む「広めの一つの範囲」を返す。
    // 途中に不要なatomが混ざってもそのまま計算し、追加の選別・詰め直しはしない。
    int first_match = group_span.end;
    int end_match = group_span.begin;

    for (int slot = group_span.begin; slot < group_span.end; ++slot) {
        double input_bin = sorted_keys[slot].input_tile;
        if (first_bin <= input_bin && input_bin < end_bin) {
            if (first_match == group_span.end) {
                first_match = slot;
            }
            end_match = slot + 1;
        }
    }

    if (first_match == group_span.end) {
        return AtomRange{group_span.begin, group_span.begin};
    }
    return AtomRange{first_match, end_match};
}

CST128_HD int block_range_index(int axis, int output_tile, int group) {
    // ranges[axis][output_tile][group][begin/end]の、beginの配列添字。
    return ((axis * OUTPUT_TILES + output_tile) * HIERARCHY_GROUPS + group) * 2;
}

CST128_HD AtomRange read_block_range(const int *ranges, int axis, int output_tile, int group) {
    int index = block_range_index(axis, output_tile, group);
    return AtomRange{ranges[index], ranges[index + 1]};
}

} // namespace cst128
