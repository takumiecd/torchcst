"""Select registered plans in fixtures without reconstructing registries."""

from torchcst import Dispatcher, FixedSelector
from torchcst._backends.catalog import get_registry
from torchcst.nn.conv import _default_selector as default_conv2d_selector
from torchcst.nn.linear import _default_selector as default_linear_selector
from torchcst.operators.conv2d import Conv2dInputs
from torchcst.operators.execution import LinearInputs


def fixed_plan(plan):
    return FixedSelector(plan, registry=get_registry())


def default_conv2d_plan(layer, x):
    dispatcher = Dispatcher(selector=default_conv2d_selector())
    return dispatcher.prepare(layer, Conv2dInputs(x)).plan


def default_linear_plan(layer, x):
    dispatcher = Dispatcher(selector=default_linear_selector())
    return dispatcher.prepare(layer, LinearInputs(x)).plan
