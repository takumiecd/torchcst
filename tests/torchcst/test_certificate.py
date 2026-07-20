"""CertificateSubspace aggregation, SVD timing, and audit contracts."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from torchcst.instruments import CertificateSubspace


def test_scripted_certificate_has_analytic_subspace_ratio_and_participation() -> None:
    root_two = 2.0**0.5
    left = torch.tensor(
        [[1 / root_two, -1 / root_two], [1 / root_two, 1 / root_two], [0.0, 0.0]],
        dtype=torch.float64,
    )
    certificate = CertificateSubspace(rank=2)
    certificate.G = left @ torch.diag(torch.tensor([3.0, 1.0], dtype=torch.float64))

    reading = certificate.snapshot()

    projection = reading.U_r @ reading.U_r.T
    torch.testing.assert_close(projection, left @ left.T)
    torch.testing.assert_close(reading.S, torch.tensor([3.0, 1.0], dtype=torch.float64))
    assert reading.sigma2_over_sigma1 == pytest.approx(1 / 3)
    assert reading.participation_ratio == pytest.approx(2.0)


def test_outer_products_accumulate_without_svd_until_snapshot_and_reset() -> None:
    certificate = CertificateSubspace(rank=1)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    g = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    original = torch.linalg.svd

    with patch("torch.linalg.svd", wraps=original) as svd:
        certificate.accumulate(x, g, weight=0.5)
        assert svd.call_count == 0
        torch.testing.assert_close(certificate.G, 0.5 * g.T @ x)
        certificate.snapshot()
        assert svd.call_count == 1

    certificate.reset()
    assert certificate.G is not None
    assert torch.count_nonzero(certificate.G) == 0
