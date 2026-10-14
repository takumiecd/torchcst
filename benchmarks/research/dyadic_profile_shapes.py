"""CPU shape and sampled-norm study of an integer-power rounded profile bank.

User sigma is decay length. Public MaternMixtureSpec takes sqrt(sigma).
Power each native periodic image before summation, never the periodic profile.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

from torchcst import MaternMixtureSpec
from torchcst._backends.torch.profiles.matern_mixture import fixed_values, values
from torchcst._backends.torch.profiles.matern_moments import evaluate_gram

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PERIOD = 102.4
SPACING = 0.1
SIZE = 1024
BMAX = 10.0
POWERS = np.array([128, 64, 32, 16, 8, 4, 2, 1])
DECAYS = BMAX / POWERS
SHOWN = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
IMAGES = np.arange(-8, 9)


def native(d, b, *, powered=True, derivative=False):
    """Independent image sum; omitted image height < 1e-32 for the bank."""
    signed = np.asarray(d, dtype=np.float64)[..., None] + PERIOD * IMAGES
    r = np.abs(signed)
    if powered:
        e = np.exp(-r / BMAX)
        for _ in range(round(math.log2(BMAX / b))):
            e = e * e
    else:
        e = np.exp(-r / b)
    peak_r = PERIOD * np.abs(IMAGES)
    peak = ((1 + peak_r / b) * np.exp(-peak_r / b)).sum()
    if derivative:
        return (signed / b**2 * e).sum(-1) / peak  # centre derivative
    return ((1 + r / b) * e).sum(-1) / peak


def weight(sigma):
    j = np.clip(np.searchsorted(DECAYS, sigma, side="right") - 1, 0, len(DECAYS) - 2)
    t = np.clip(np.log(sigma / DECAYS[j]) / np.log(DECAYS[j + 1] / DECAYS[j]), 0, 1)
    return int(j), float(t)


def profile(d, sigma):
    j, t = weight(sigma)
    return (1 - t) * native(d, DECAYS[j]) + t * native(d, DECAYS[j + 1])


def half_radius(sigma):
    left, right = 0.0, PERIOD / 2
    if profile(right, sigma) >= 0.5:
        return None
    for _ in range(52):
        middle = (left + right) / 2
        if profile(middle, sigma) > 0.5:
            left = middle
        else:
            right = middle
    return (left + right) / 2


def verify():
    spec = MaternMixtureSpec(width_knots=tuple(float(v) for v in np.sqrt(DECAYS)))
    points = np.linspace(-PERIOD, PERIOD, 2003)
    power_abs = closed_abs = 0.0
    for b in DECAYS:
        independent = native(points, b)
        power_abs = max(
            power_abs, float(abs(independent - native(points, b, powered=False)).max())
        )
        closed = fixed_values(
            torch.tensor(points), torch.tensor(float(b), dtype=torch.float64), PERIOD
        ).numpy()
        closed_abs = max(closed_abs, float(abs(independent - closed).max()))
    offsets = np.linspace(0, SPACING, 41)
    delta = (-offsets) % SPACING
    gram, dgram = evaluate_gram(spec, torch.tensor(delta), PERIOD, SIZE)
    sites = np.arange(SIZE) * SPACING
    d = (sites[None, :] - offsets[:, None] + PERIOD / 2) % PERIOD - PERIOD / 2
    bank = np.stack([native(d, b) for b in DECAYS], axis=-1)
    dbank = np.stack([native(d, b, derivative=True) for b in DECAYS], axis=-1)
    grams, derivatives = [], []
    for j in range(len(DECAYS) - 1):
        a, b = bank[..., j], bank[..., j + 1]
        da, db = dbank[..., j], dbank[..., j + 1]
        grams.append(np.stack([(a * a).sum(1), (a * b).sum(1), (b * b).sum(1)], -1))
        derivatives.append(
            np.stack(
                [(2 * a * da).sum(1), (da * b + a * db).sum(1), (2 * b * db).sum(1)], -1
            )
        )
    gram_abs = float(abs(gram.numpy() - np.stack(grams, 1)).max())
    derivative_abs = float(abs(-dgram.numpy() - np.stack(derivatives, 1)).max())
    mixture_abs = norm_abs = norm_derivative_abs = 0.0
    for sigma in np.geomspace(0.1, 10, 71):
        j, t = weight(sigma)
        independent = (1 - t) * bank[..., j] + t * bank[..., j + 1]
        package = values(
            spec,
            torch.tensor(d),
            torch.tensor(math.sqrt(sigma), dtype=torch.float64),
            PERIOD,
        )
        mixture_abs = max(mixture_abs, float(abs(package.numpy() - independent).max()))
        w = np.array([(1 - t) ** 2, 2 * t * (1 - t), t * t])
        moment_norm = gram[:, j].numpy() @ w
        norm_abs = max(
            norm_abs, float(abs(moment_norm - (independent**2).sum(1)).max())
        )
        derivative = (1 - t) * dbank[..., j] + t * dbank[..., j + 1]
        moment_derivative = -dgram[:, j].numpy() @ w
        norm_derivative_abs = max(
            norm_derivative_abs,
            float(abs(moment_derivative - (2 * independent * derivative).sum(1)).max()),
        )
    centre_slope_abs = monotonic_rise = peak_abs = 0.0
    minimum = 1.0
    radius = np.linspace(0, PERIOD / 2, 2001)
    for sigma in np.geomspace(0.1, 10, 161):
        y = profile(radius, sigma)
        j, t = weight(sigma)
        slope = (1 - t) * native(0.0, DECAYS[j], derivative=True) + t * native(
            0.0, DECAYS[j + 1], derivative=True
        )
        centre_slope_abs = max(centre_slope_abs, abs(float(slope)))
        monotonic_rise = max(monotonic_rise, float(np.diff(y).max()))
        minimum = min(minimum, float(y.min()))
        peak_abs = max(peak_abs, abs(float(y[0]) - 1))
    assert minimum >= 0 and centre_slope_abs < 1e-14 and monotonic_rise < 1e-14
    checks = {
        "power_vs_direct_image_max_abs": power_abs,
        "centre_slope_max_abs": centre_slope_abs,
        "peak_one_max_abs": peak_abs,
        "outward_monotonic_max_rise": monotonic_rise,
        "sampled_min_height": minimum,
        "native_images_vs_closed_form_max_abs": closed_abs,
        "sampled_gram_max_abs": gram_abs,
        "sampled_gram_centre_derivative_max_abs": derivative_abs,
        "mapped_public_mixture_max_abs": mixture_abs,
        "sampled_norm_squared_max_abs": norm_abs,
        "sampled_norm_squared_centre_derivative_max_abs": norm_derivative_abs,
    }
    assert power_abs < 2e-14 and closed_abs < 2e-14
    assert gram_abs < 2e-10 and derivative_abs < 2e-9
    assert mixture_abs < 2e-14 and norm_abs < 2e-10 and norm_derivative_abs < 2e-9
    return checks


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    checks = verify()
    offsets = np.linspace(0, SPACING, 201)
    sites = np.arange(SIZE) * SPACING
    d = (sites[None, :] - offsets[:, None] + PERIOD / 2) % PERIOD - PERIOD / 2
    r = abs(d)
    sigmas = np.unique(np.r_[np.geomspace(0.1, 10, 161), SHOWN, DECAYS[1:]])
    records, norms = [], []
    for sigma in sigmas:
        y = profile(d, sigma)
        norm2 = (y * y).sum(1)
        norm = np.sqrt(norm2)
        tail = ((y * y) * (r >= 3 * sigma - 1e-12)).sum(1) / norm2
        outside10 = ((y * y) * (r >= 10 - 1e-12)).sum(1) / norm2
        j, t = weight(sigma)
        records.append(
            {
                "sigma_decay_length": float(sigma),
                "active_decay_lengths": [float(DECAYS[j]), float(DECAYS[j + 1])],
                "broad_weight": t,
                "half_height_radius": half_radius(sigma),
                "l2_norm_min": float(norm.min()),
                "l2_norm_max": float(norm.max()),
                "l2_norm_relative_span": float(norm.max() / norm.min() - 1),
                "tail_l2_squared_fraction_outside_3sigma_min": float(tail.min()),
                "tail_l2_squared_fraction_outside_3sigma_max": float(tail.max()),
                "tail_l2_squared_fraction_outside_10_min": float(outside10.min()),
                "tail_l2_squared_fraction_outside_10_max": float(outside10.max()),
                "antipode_raw_height": float(profile(PERIOD / 2, sigma)),
            }
        )
        norms.append(norm)
    norms = np.asarray(norms)
    colors = plt.cm.viridis(np.linspace(0, 1, len(SHOWN)))
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.2})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    x = np.linspace(-12, 12, 4801)
    z = np.linspace(0, 8, 1601)
    for color, sigma in zip(colors, SHOWN):
        ax[0].plot(x, profile(x, sigma), color=color, label=f"sigma={sigma:g}")
        valid = z * sigma <= PERIOD / 2
        ax[1].plot(z[valid], profile(z[valid] * sigma, sigma), color=color)
        ax[2].semilogy(z[valid], profile(z[valid] * sigma, sigma), color=color)
    ax[0].set(
        xlabel="Distance from centre",
        ylabel="Peak-one profile",
        title="Rounded dyadic mixture",
    )
    ax[0].legend(ncol=2, fontsize=8)
    ax[1].set(
        xlabel="Distance / sigma",
        ylabel="Peak-one profile",
        title="Shape at each width",
        ylim=(0, 1.02),
    )
    ax[2].set(
        xlabel="Distance / sigma",
        ylabel="Peak-one profile (log)",
        title="Tails",
        ylim=(1e-4, 1.05),
    )
    fig.savefig(output / "shapes-and-tails.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for color, sigma in zip(colors, SHOWN):
        row = int(np.argmin(abs(sigmas - sigma)))
        ax[0].plot(
            offsets / SPACING,
            norms[row] / norms[row, 0],
            color=color,
            label=f"sigma={sigma:g}",
        )
    ax[0].set(
        xlabel="Centre offset / spacing",
        ylabel="L2 norm / norm at site",
        title="Centre between sites",
    )
    ax[0].legend(ncol=2, fontsize=8)
    ax[1].loglog(sigmas, norms.max(1) / norms.min(1) - 1, color="tab:blue")
    ax[1].set(
        xlabel="sigma (decay length)",
        ylabel="Relative norm span",
        title="Sampled L2 variation",
    )
    ax[2].semilogx(
        sigmas, [v["half_height_radius"] / v["sigma_decay_length"] for v in records]
    )
    ax[2].set(
        xlabel="sigma (decay length)",
        ylabel="Half-height radius / sigma",
        title="Width progression",
    )
    fig.savefig(output / "centre-l2-and-width.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    z = np.linspace(-5, 5, 2001)
    for color, sigma in zip(colors, SHOWN):
        row = int(np.argmin(abs(sigmas - sigma)))
        ax[0].plot(
            z,
            profile(z * sigma, sigma) / norms[row, 0],
            color=color,
            label=f"sigma={sigma:g}",
        )
        radius = np.linspace(0, PERIOD / 2, 2001)
        ax[1].semilogy(radius, profile(radius, sigma) / norms[row, 0], color=color)
    ax[0].set(
        xlabel="Distance / sigma",
        ylabel="Sampled-L2-normalized profile",
        title="Normalized rounded peaks",
    )
    ax[0].legend(ncol=2, fontsize=8)
    ax[1].set(
        xlabel="Distance from centre",
        ylabel="Normalized profile (log)",
        title="Full Torus tails",
        ylim=(1e-10, 2),
    )
    fig.savefig(output / "normalized-shapes.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    zoom = np.linspace(-0.5, 0.5, 2001)
    for color, sigma in zip(colors[:4], SHOWN[:4]):
        row = int(np.argmin(abs(sigmas - sigma)))
        ax[0].plot(zoom, profile(zoom, sigma), color=color, label=f"sigma={sigma:g}")
        ax[1].plot(
            zoom,
            profile(zoom, sigma) / norms[row, 0],
            color=color,
            label=f"sigma={sigma:g}",
        )
    ax[0].set(
        xlabel="Signed distance from centre",
        ylabel="Peak-one profile",
        title="Narrow peaks: rounded centre",
    )
    ax[0].legend()
    ax[1].set(
        xlabel="Signed distance from centre",
        ylabel="Sampled-L2-normalized profile",
        title="Narrow peaks after sampled L2",
    )
    fig.savefig(output / "narrow-centre-zoom.png", dpi=150)
    plt.close(fig)
    result = {
        "definition": "periodic peak-normalized adjacent log-decay mixture of (1+n*r/10)*exp(-r/10)**n",
        "sigma_means": "decay length b, not previous sigma squared",
        "period": PERIOD,
        "sites": SIZE,
        "spacing": SPACING,
        "centre_offsets": offsets.tolist(),
        "powers": POWERS.tolist(),
        "fixed_decay_lengths": DECAYS.tolist(),
        "validation": checks,
        "records": records,
        "discrete_l2_norms": norms.tolist(),
        "scope": "CPU FP64 shape and Gram validation only; GPU memory measured separately",
    }
    (output / "shapes.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = ""
    for sigma in SHOWN:
        v = records[int(np.argmin(abs(sigmas - sigma)))]
        rows += (
            f"<tr><td>{sigma:g}</td><td>{v['half_height_radius']:.5g}</td>"
            f"<td>{100 * v['l2_norm_relative_span']:.5g}%</td>"
            f"<td>{100 * v['tail_l2_squared_fraction_outside_3sigma_min']:.3f}–"
            f"{100 * v['tail_l2_squared_fraction_outside_3sigma_max']:.3f}%</td>"
            f"<td>{v['antipode_raw_height']:.3g}</td></tr>"
        )
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><title>累乗で作る丸いprofile</title>
<style>body{{font:17px/1.6 system-ui;margin:32px auto;max-width:1200px;padding:0 24px;color:#182331}}
img{{width:100%;height:auto}}code,pre{{background:#eff3f8;padding:4px}}table{{border-collapse:collapse;width:100%}}
td,th{{padding:9px;border:1px solid #d6dde5;text-align:right}}h1,h2{{line-height:1.3}}</style>
<h1>σmaxから累乗で作る丸いprofile</h1><p>2026-10-14。形を優先する新しいprofileの実験。既存形への近似誤差を採用条件にしていません。</p>
<h2>定義とσの意味</h2><p>今回のσは<strong>減衰長b</strong>です。以前のσ²とは異なります。最大減衰長10から、n=1,2,4,…,128の整数累乗で8幅を作ります。</p>
<pre>u(r)=exp(-r/10)
f_n(r)=(1+n r/10)u(r)^n, b_n=10/n
G_n(d)=sum_k f_n(|d+kD|) / sum_k f_n(|kD|)
g_σ=(1-t)G_j+tG_(j+1)
t=log(σ/b_j)/log(b_(j+1)/b_j)</pre>
<p>このレポートのtは<strong>太い側の重み</strong>、細い側は1−tです。以前の式で細い側をtと呼んだものとは表記が逆になりますが、両方の重みを同時に変えています。</p>
<p>周期化したuやGを累乗しません。各画像の直線profileを整数累乗で作ってから周期化します。
各固定山は中心の傾きが0で、混合後も頂点に角はありません。隣接する幅だけを混ぜるので、0.1と10を直接混ぜた広い肩は避けられます。
公開MaternMixtureSpecではwidth_knots=sqrt(b_n)、live intrinsic sigma=sqrt(今回のσ)として同じ数学的形を表現できます。</p>
<h2>狭い山の頂点を拡大</h2><img src="narrow-centre-zoom.png"><p>距離±0.5でσ=0.1、0.2、0.3、0.5を表示。幅全体の図で細く見える最小幅でも、中心は角のない丸い山です。</p>
<h2>山と裾</h2><img src="shapes-and-tails.png"><p>横軸をσで割ると、すべての幅で丸い山になります。円周上の距離は反対側D/2までに制限し、周期を越えた再上昇を裾として表示しません。整数累乗は鋭い角を新しく作りません。混合率はlog幅なので、σに対して連続です。固定幅の境目では幅微分に折れがあります。</p>
<h2>中心をサイト間で動かした離散L2</h2><img src="centre-l2-and-width.png">
<p>D=102.4、1024sites、spacing=0.1。中心はsiteから次のsiteまで201点。ノルムは実際のsitesでsqrt(sum_i g²)を取り、連続積分へ置き換えていません。</p>
<table><tr><th>σ</th><th>半値半径</th><th>L2ノルム変動</th><th>距離3σ外の二乗和割合</th><th>反対側の高さ</th></tr>{rows}</table>
<h2>L2正規化後</h2><img src="normalized-shapes.png"><p>正規化は山全体に一つの係数を掛けるので、形と裾の二乗和割合は変わりません。</p>
<h2>数学的検証</h2><pre>{json.dumps(checks, indent=2)}</pre>
<p>直接指数と整数累乗、独立周期画像と閉形式、実siteのGramとmoment式、中心微分、公開profileへのsqrt幅変換をCPU FP64で確認しました。0.1〜10の161幅で非負・peak1・外側への単調減少・中心の傾き0を確認しています。
メモリはこのCPU配列サイズからGPUのピークを推測せず、別実装・別計測の結果で判断します。</p>
<p><a href="shapes.json">全測定JSON</a></p></html>"""
    (output / "report.html").write_text(html)
    print(
        json.dumps(
            {
                "validation": checks,
                "shown_records": [
                    v for v in records if v["sigma_decay_length"] in SHOWN
                ],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("output/dyadic-profile-streaming-20261014")
    )
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
