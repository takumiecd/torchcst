"""Same-profile local/direct and shared contractions for CPU diagnosis."""

from collections import defaultdict

import torch

from torchcst._backends.torch.kernels.hierarchical_fourier import components
from torchcst._backends.torch.profiles.fourier_peak import coefficients

from .hierarchical_fourier_cpu import contract_tiles, make_layout, sampled_norm_squared


def site_intervals(spec, centers, widths, sites, spacing=0.1):
    L, tiles, w, _, _, _ = components(spec, widths)
    begin = (
        (torch.floor((centers[:, None] - widths[:, None]) / tiles) * (tiles / spacing))
        .ceil()
        .clamp(0, sites)
        .long()
    )
    end = (
        (torch.ceil((centers[:, None] + widths[:, None]) / tiles) * (tiles / spacing))
        .ceil()
        .clamp(0, sites)
        .long()
    )
    return L, w, begin, end


def direct_contract(profile, spec, inp, centers, widths, *, spacing=0.1):
    """Evaluate the same Fourier function only on each atom's local sites.

    The already evaluated f also supplies its exact sampled norm. No new
    profile, polynomial approximation, or X-dependent L2 approximation.
    """
    L, w, begin, end = site_intervals(spec, centers, widths, inp.shape[1], spacing)
    b, _ = coefficients(profile, widths[:, None] / L)
    n = torch.arange(profile.modes + 1, device=inp.device, dtype=torch.float64)
    values, norms = [], []
    saved = 0
    max_scratch = 0
    for atom in range(widths.numel()):
        active = w[atom].detach() > 0
        first = int(begin[atom, active].min())
        last = int(end[atom, active].max())
        idx = torch.arange(first, last, device=inp.device)
        x = idx.to(dtype=torch.float64) * spacing
        phase = (x[:, None, None] - centers[atom]) * torch.pi * n / L[atom, :, None]
        mask = (idx[:, None] >= begin[atom]) & (idx[:, None] < end[atom])
        f = ((phase.cos() * b[atom]).sum(-1) * w[atom] * mask).sum(-1)
        values.append(inp[:, first:last] @ f)
        norms.append(f.square().sum())
        saved += f.numel()
        max_scratch = max(max_scratch, phase.numel())
    return (
        torch.stack(values, -1),
        torch.stack(norms),
        {"direct_profile_reals": saved, "direct_max_phase_scratch_reals": max_scratch},
    )


def contraction_counts(
    profile, spec, centers, widths, *, sites=1024, spacing=0.1, layout=None
):
    _, w, begin, end = site_intervals(spec, centers, widths, sites, spacing)
    active = w.detach() > 0
    union = (
        torch.where(active, end, 0).amax(-1)
        - torch.where(active, begin, sites).amin(-1)
    ).clamp_min(0)
    component = ((end - begin) * active).sum(-1)
    return {
        "support_sites": union,
        "component_sites": component,
        "direct_profile_harmonic_evaluations": int(component.sum())
        * (profile.modes + 1),
        "direct_X_real_multiply_adds": int(union.sum()),
        "direct_norm_real_multiply_adds": int(union.sum()),
        "active_gram_complex_terms": int(active.sum(-1).square().sum())
        * (2 * profile.modes + 1) ** 2,
    }


def choose_direct(
    profile,
    spec,
    centers,
    widths,
    *,
    sites=1024,
    spacing=0.1,
    cutoff=8,
    keep_shared=1e9,
):
    """Diagnostic selector; discrete routing never changes the function."""
    counts = contraction_counts(
        profile, spec, centers, widths, sites=sites, spacing=spacing
    )
    mask = counts["support_sites"] <= cutoff
    if keep_shared < 1e9:
        layout = make_layout(
            profile, spec, centers, widths, sites=sites, spacing=spacing
        )
        groups = defaultdict(set)
        for (j, t), requests in layout["requests"].items():
            for atom, _, k in requests:
                groups[j, t, k].add(atom)
        total = widths.new_zeros(widths.numel())
        number = widths.new_zeros(widths.numel())
        for atoms in groups.values():
            idx = torch.tensor(sorted(atoms), device=widths.device)
            total[idx] += len(atoms)
            number[idx] += 1
        mean = total / number.clamp_min(1)
        mask = mask & (mean < keep_shared)
    return mask


def hybrid_contract(
    profile, spec, inp, centers, widths, *, cutoff=8, keep_shared=1e9, spacing=0.1
):
    mask = choose_direct(
        profile,
        spec,
        centers,
        widths,
        sites=inp.shape[1],
        spacing=spacing,
        cutoff=cutoff,
        keep_shared=keep_shared,
    )
    direct, shared = mask.nonzero().flatten(), (~mask).nonzero().flatten()
    H = inp.new_zeros(inp.shape[0], widths.numel())
    norm = widths.new_zeros(widths.numel())
    payload = {"direct_atoms": int(mask.sum()), "shared_atoms": int((~mask).sum())}
    if len(direct):
        h, n, p = direct_contract(
            profile, spec, inp, centers[direct], widths[direct], spacing=spacing
        )
        H = H.index_copy(1, direct, h)
        norm = norm.index_copy(0, direct, n)
        payload.update(p)
    if len(shared):
        layout = make_layout(
            profile,
            spec,
            centers[shared],
            widths[shared],
            sites=inp.shape[1],
            spacing=spacing,
        )
        h, p = contract_tiles(
            profile, spec, inp, centers[shared], widths[shared], layout
        )
        n = sampled_norm_squared(
            profile,
            spec,
            centers[shared],
            widths[shared],
            sites=inp.shape[1],
            spacing=spacing,
        )
        H = H.index_copy(1, shared, h)
        norm = norm.index_copy(0, shared, n)
        payload.update(p)
    return H, norm, payload
