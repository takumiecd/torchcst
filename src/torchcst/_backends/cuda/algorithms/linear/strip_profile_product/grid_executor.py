"""Strip coordinates for the shared whole-chart profile-product grid."""

from ..profile_product_global.executor import grid_product


def strip_grid(x, p, value, chart, declaration, recipe):
    sizes = (
        len(x),
        declaration.shape[1],
        declaration.shape[0],
        declaration.tile_shape[1],
        declaration.axes[1].spacing[0],
        declaration.axes[1].start[0],
        declaration.axes[0].start[0],
    )
    return grid_product(x, p, value, chart.tile_pitch, sizes, recipe)
