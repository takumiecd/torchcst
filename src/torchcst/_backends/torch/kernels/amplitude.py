"""Signed amplitude composed with a radial or separable atom function."""

import math

import torch

from . import execution as _kernel


def initialize(state, *charts_and_atoms, mode):
    atoms = charts_and_atoms[-1]
    inner = _kernel.initialize(state.inner, *charts_and_atoms, mode=mode)
    amplitude = inner.new_empty(atoms, 1).normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
    return torch.cat((amplitude, inner), -1)


def _split(state, *charts_and_p):
    *charts, p = charts_and_p
    width = _kernel.parameter_dim(state, *charts)
    if p.ndim != 2 or p.shape[1] != width:
        raise ValueError(f"p must have shape [atoms, {width}]")
    return p[:, :1], p[:, 1:]


def materialize_atoms(state, *charts_and_p):
    *charts, p = charts_and_p
    amplitude, inner = _split(state, *charts, p)
    return amplitude[:, None] * _kernel.materialize_atoms(state.inner, *charts, inner)


def weight(state, chart, p):
    return materialize_atoms(state, chart, p).sum(0)


def factors(state, input_chart, output_chart, p):
    amplitude, inner = _split(state, input_chart, output_chart, p)
    phi_input, phi_output = _kernel.factors(
        state.inner, input_chart, output_chart, inner
    )
    return phi_input, phi_output * amplitude.T


def tangent_backend(state, input_chart, output_chart):
    inner = _kernel.tangent_backend(state.inner, input_chart, output_chart)
    if inner is None:
        return None
    from .tangent import amplitude

    return lambda p: amplitude(state, inner, input_chart, output_chart, p)
