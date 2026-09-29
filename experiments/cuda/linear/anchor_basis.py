"""Fixed local interpolation bases used by the anchor experiment."""

import torch


def evenly_spaced_indices(length, count):
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def interpolation_matrix(length, anchors):
    """Use four neighboring anchors for piecewise cubic Lagrange interpolation."""
    matrix = torch.zeros((length, len(anchors)), dtype=torch.float64)
    for row in range(length):
        if row in anchors:
            matrix[row, anchors.index(row)] = 1.0
            continue
        selected = sorted(range(len(anchors)), key=lambda j: abs(anchors[j] - row))[:4]
        for j in selected:
            value = 1.0
            for k in selected:
                if k != j:
                    value *= (row - anchors[k]) / (anchors[j] - anchors[k])
            matrix[row, j] = value
    return matrix.float()


def column_basis(anchors_per_segment):
    local = evenly_spaced_indices(16, anchors_per_segment)
    base = interpolation_matrix(16, local)
    result = torch.zeros((64, 4 * anchors_per_segment), dtype=torch.float32)
    anchors = []
    for segment in range(4):
        result[
            segment * 16 : (segment + 1) * 16,
            segment * anchors_per_segment : (segment + 1) * anchors_per_segment,
        ] = base
        anchors.extend(segment * 16 + index for index in local)
    return result, anchors
