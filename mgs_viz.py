#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MGS 3D visualization (compact-bump window).

Kernel:
    rho(u,v,t) = o * W(u,v) * exp(-phi^2 / (2 s_n^2))
    phi(u,v,t) = t - h(u,v)
    h(u,v)     = kappa (sqrt((eta_u u)^2 + (eta_v v)^2 + delta_0^2) - delta_0)
    W(u,v)     = exp(-c * r / (1 - r))   if r < 1,  else 0
                where r = u^2/s_u^2 + v^2/s_v^2

Hyperparameters:
    kappa    -- signed curvature
    eta_u, eta_v -- anisotropic slope magnitudes
    s_u, s_v     -- in-plane footprint scales
    s_n          -- normal-direction (transverse) thickness
    c            -- window flatness  (smaller c -> flatter top, sharper drop)
    delta_0      -- apex smoothing (fixed small)

Five figures:
    A.  kappa sweep:           transitions through PGSR (kappa=0)
    B.  c sweep:               flatness vs sharpness trade-off
    C.  anisotropy sweep:      eta_u / eta_v ratio
    D.  3DGS vs MGS:           the key thesis figure
    E.  s_n sweep:             normal-direction thickness
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


DELTA_0 = 1e-3      # fixed apex smoother

# ===========================================================================
# Primitive
# ===========================================================================
class MGSPrimitive:
    """MGS primitive with compact-bump in-plane window."""
    def __init__(self, s_u=0.7, s_v=0.7, s_n=0.10,
                 eta_u=1.0, eta_v=1.0, kappa=0.0, c=0.10,
                 opacity=1.0):
        self.s_u, self.s_v = float(s_u), float(s_v)
        self.s_n = float(s_n)
        self.eta_u, self.eta_v = float(eta_u), float(eta_v)
        self.kappa = float(kappa)
        self.c = float(c)
        self.opacity = float(opacity)
        self.delta_0 = DELTA_0


def chart_h(prim, U, V):
    """h(u,v) = kappa (sqrt((eta_u u)^2 + (eta_v v)^2 + delta_0^2) - delta_0)."""
    A = (prim.eta_u * U) ** 2 + (prim.eta_v * V) ** 2 + prim.delta_0 ** 2
    return prim.kappa * (np.sqrt(A) - prim.delta_0)


def phi(prim, U, V, T):
    return T - chart_h(prim, U, V)


def in_plane_r(prim, U, V):
    """r = u^2/s_u^2 + v^2/s_v^2."""
    return (U / prim.s_u) ** 2 + (V / prim.s_v) ** 2


def window(prim, U, V):
    """Compact-bump window: exp(-c r/(1-r)) inside footprint, 0 outside."""
    r = in_plane_r(prim, U, V)
    inside = r < 1.0
    # Guard against r exactly 1 -> divide-by-zero
    safe = np.where(inside, r, 0.0)
    W = np.zeros_like(r)
    W[inside] = np.exp(-prim.c * safe[inside] / np.maximum(1.0 - safe[inside], 1e-12))
    return W


def density(prim, U, V, T):
    """rho / opacity = W * exp(-phi^2 / 2 s_n^2)."""
    W = window(prim, U, V)
    P = phi(prim, U, V, T)
    return W * np.exp(-0.5 * P ** 2 / prim.s_n ** 2)


# ===========================================================================
# Plot helpers
# ===========================================================================
def plot_chart_surface(ax, prim, n=100, color='#3aa860', show_boundary=True,
                       alpha_max=0.9):
    """The chart surface { t = h(u,v) } inside the footprint disk, with
    transparency modulated by W (so visible 'density' on the manifold equals W)."""
    u = np.linspace(-prim.s_u, prim.s_u, n)
    v = np.linspace(-prim.s_v, prim.s_v, n)
    U, V = np.meshgrid(u, v)
    r = in_plane_r(prim, U, V)
    in_disk = r < 1.0
    W_vals = window(prim, U, V)
    T = chart_h(prim, U, V)

    # color by W (density value ON the chart)
    facecolors = np.zeros((n - 1, n - 1, 4))
    Wmid = 0.25 * (W_vals[:-1, :-1] + W_vals[1:, :-1] +
                   W_vals[:-1, 1:] + W_vals[1:, 1:])
    in_mid = 0.25 * (in_disk[:-1, :-1].astype(float) +
                     in_disk[1:, :-1] + in_disk[:-1, 1:] + in_disk[1:, 1:])
    base_rgb = np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    facecolors[..., 0] = base_rgb[0]
    facecolors[..., 1] = base_rgb[1]
    facecolors[..., 2] = base_rgb[2]
    facecolors[..., 3] = alpha_max * Wmid * (in_mid > 0.99)

    Tplot = np.where(in_disk, T, np.nan)
    ax.plot_surface(U, V, Tplot, facecolors=facecolors,
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False)

    # Footprint boundary curve
    if show_boundary:
        theta = np.linspace(0, 2 * np.pi, 200)
        bu = prim.s_u * np.cos(theta); bv = prim.s_v * np.sin(theta)
        bA = (prim.eta_u * bu) ** 2 + (prim.eta_v * bv) ** 2 + prim.delta_0 ** 2
        bt = prim.kappa * (np.sqrt(bA) - prim.delta_0)
        ax.plot(bu, bv, bt, color='black', linewidth=1.4,
                linestyle='--', alpha=0.65, zorder=18)

    # Apex marker (center of footprint)
    ax.scatter([0], [0], [0], color='#1a4525', s=25, zorder=20)


def plot_density_shells(ax, prim, level_fracs=(0.85, 0.5, 0.15),
                         face_alphas=(0.92, 0.45, 0.18),
                         N=60, color_base='#3aa860'):
    """Marching-cubes isosurfaces of rho/opacity at several levels."""
    box_u = 1.05 * prim.s_u; box_v = 1.05 * prim.s_v
    cone_h = abs(prim.kappa) * (np.sqrt((prim.eta_u * box_u) ** 2 +
                                          (prim.eta_v * box_v) ** 2
                                          + prim.delta_0 ** 2) - prim.delta_0)
    t_pad = max(1.2 * cone_h + 3 * prim.s_n, 0.3)
    if prim.kappa >= 0:
        t_min, t_max = -3 * prim.s_n, t_pad
    else:
        t_min, t_max = -t_pad, 3 * prim.s_n

    u = np.linspace(-box_u, box_u, N)
    v = np.linspace(-box_v, box_v, N)
    t = np.linspace(t_min, t_max, N)
    U, V, T = np.meshgrid(u, v, t, indexing='ij')
    F = density(prim, U, V, T)

    spacing = np.array([2 * box_u / (N - 1),
                        2 * box_v / (N - 1),
                        (t_max - t_min) / (N - 1)])
    origin = np.array([-box_u, -box_v, t_min])
    base_rgb = np.array([int(color_base[i:i + 2], 16) / 255 for i in (1, 3, 5)])

    for level, falpha in sorted(zip(level_fracs, face_alphas),
                                 key=lambda x: x[0]):
        try:
            verts, faces, _, _ = marching_cubes(F, level=level)
        except (ValueError, RuntimeError):
            continue
        verts = verts * spacing + origin
        mesh = Poly3DCollection(verts[faces], alpha=falpha,
                                linewidth=0, antialiased=True)
        tri = verts[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nl = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
        normals /= nl
        shade = 0.55 + 0.45 * normals[:, 2]
        rgba = np.concatenate([
            np.clip(base_rgb[None, :] * shade[:, None], 0, 1),
            np.full((len(shade), 1), falpha),
        ], axis=1)
        mesh.set_facecolor(rgba)
        ax.add_collection3d(mesh)

    ax.set_xlim(-box_u, box_u); ax.set_ylim(-box_v, box_v)
    ax.set_zlim(t_min, t_max)


def style_ax(ax, view=(20, -58), title=""):
    ax.set_xlabel('u', fontsize=8, labelpad=-6)
    ax.set_ylabel('v', fontsize=8, labelpad=-6)
    ax.set_zlabel('t', fontsize=8, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect((1, 1, 0.6))
    if title:
        ax.set_title(title, fontsize=10)


def cap_zlim(ax, prim, pad=0.10):
    box_u = 1.05 * prim.s_u
    cone_h = abs(prim.kappa) * (np.sqrt((prim.eta_u * box_u) ** 2 +
                                          (prim.eta_v * box_u) ** 2 +
                                          prim.delta_0 ** 2) - prim.delta_0)
    cone_h = max(cone_h, 0.20)
    if prim.kappa >= 0:
        ax.set_zlim(-pad * cone_h, (1 + pad) * cone_h)
    else:
        ax.set_zlim(-(1 + pad) * cone_h, pad * cone_h)


# ===========================================================================
# Figure A:  kappa sweep through PGSR (kappa=0)
# ===========================================================================
def fig_kappa_sweep(out_path):
    kappas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    colors = ['#c41e3a', '#c41e3a', '#1f5fb4', '#3aa860', '#3aa860']

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        r"Smooth transition through PGSR as $\kappa$ crosses zero "
        r"($\eta^u=\eta^v=0.85$, $c=0.10$, $s_n=0.08$)."
        "\n"
        r"Top: chart $t=h(u,v)$ shaded by $W$.  "
        r"Middle: $\rho/o$ density shells.  "
        r"Bottom: profile at $v=0$.",
        fontsize=12,
    )

    for k, kappa in enumerate(kappas):
        prim = MGSPrimitive(eta_u=0.85, eta_v=0.85, kappa=kappa,
                            s_u=0.7, s_v=0.7, s_n=0.08, c=0.10)
        color = colors[k]

        # Top row: chart surface
        ax = fig.add_subplot(3, 5, k + 1, projection='3d')
        plot_chart_surface(ax, prim, color=color)
        cap_zlim(ax, prim, pad=0.10)
        title = rf"$\kappa={kappa:+.1f}$"
        if kappa == 0:
            title += "   (PGSR limit)"
        style_ax(ax, title=title)
        ax.set_xlim(-1.05 * prim.s_u, 1.05 * prim.s_u)
        ax.set_ylim(-1.05 * prim.s_v, 1.05 * prim.s_v)

        # Middle row: density shells
        ax = fig.add_subplot(3, 5, 5 + k + 1, projection='3d')
        plot_density_shells(ax, prim, color_base=color)
        style_ax(ax)

        # Bottom row: profile at v=0 showing chart, W, and W * exp slice
        ax = fig.add_subplot(3, 5, 10 + k + 1)
        u_line = np.linspace(-prim.s_u * 1.02, prim.s_u * 1.02, 300)
        A = (prim.eta_u * u_line) ** 2 + prim.delta_0 ** 2
        t_chart = prim.kappa * (np.sqrt(A) - prim.delta_0)
        W_line = window(prim, u_line, np.zeros_like(u_line))

        # Top axis: chart
        ax.plot(u_line, t_chart, color=color, linewidth=2.5,
                label=r'chart $t=h(u,0)$')
        ax.axhline(0, color='black', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.set_xlabel(r'$u$  ($v=0$)', fontsize=9)
        ax.set_ylabel(r'$t = h(u,0)$', fontsize=9, color=color)
        ax.tick_params(axis='y', labelcolor=color, labelsize=7)
        ax.set_ylim(-0.85, 0.85); ax.set_xlim(-1.05 * prim.s_u, 1.05 * prim.s_u)

        # Second axis: window W on the chart
        ax2 = ax.twinx()
        ax2.fill_between(u_line, 0, W_line, color='gray', alpha=0.18,
                         label=r'$W(u,0)$  (density on chart)')
        ax2.plot(u_line, W_line, color='gray', linewidth=1.2)
        ax2.set_ylabel(r'$W$  (density on chart)', fontsize=8, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
        ax2.set_ylim(0, 1.15)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure B:  c sweep -- window flatness
# ===========================================================================
def fig_c_sweep(out_path):
    cs = [0.02, 0.05, 0.10, 0.30, 1.00]

    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(
        r"Window flatness parameter $c$  (smaller $c$ $\to$ flatter top, sharper boundary)."
        "\n"
        rf"Fixed $\kappa=+0.6$, $\eta=0.85$, $s=0.7$, $s_n=0.08$.  "
        r"Top: chart shaded by $W$.   Bottom: radial profile of $W$ at the apex.",
        fontsize=12,
    )

    for k, c_val in enumerate(cs):
        prim = MGSPrimitive(eta_u=0.85, eta_v=0.85, kappa=0.6,
                            s_u=0.7, s_v=0.7, s_n=0.08, c=c_val)

        ax = fig.add_subplot(2, 5, k + 1, projection='3d')
        plot_chart_surface(ax, prim, color='#3aa860')
        cap_zlim(ax, prim, pad=0.10)
        style_ax(ax, title=rf'$c={c_val}$')
        ax.set_xlim(-1.05 * prim.s_u, 1.05 * prim.s_u)
        ax.set_ylim(-1.05 * prim.s_v, 1.05 * prim.s_v)

        ax = fig.add_subplot(2, 5, 5 + k + 1)
        rr = np.linspace(0, 1.05, 400)
        inside = rr < 1
        W_rad = np.zeros_like(rr)
        W_rad[inside] = np.exp(-c_val * rr[inside] / np.maximum(1 - rr[inside], 1e-12))
        ax.plot(rr, W_rad, color='#3aa860', linewidth=2.5)
        ax.fill_between(rr, 0, W_rad, color='#3aa860', alpha=0.18)
        ax.axvline(1.0, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel(r'$r = u^2/s_u^2 + v^2/s_v^2$', fontsize=9)
        ax.set_ylabel(r'$W(r)$', fontsize=9)
        ax.set_xlim(0, 1.05); ax.set_ylim(-0.03, 1.08)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure C: anisotropy sweep (eta_u vs eta_v)
# ===========================================================================
def fig_anisotropy_sweep(out_path):
    pairs = [(1.0, 1.0), (1.0, 0.6), (1.0, 0.3), (0.6, 1.0), (0.3, 1.0)]

    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(
        r"Anisotropic slope magnitudes $(\eta^u, \eta^v)$ at fixed $\kappa=+0.7$, "
        r"$c=0.10$, $s_u=s_v=0.7$."
        "\n"
        r"Aspect ratio between $\eta^u$ and $\eta^v$ controls the chart's "
        r"principal-curvature ratio.",
        fontsize=12,
    )

    for k, (eu, ev) in enumerate(pairs):
        prim = MGSPrimitive(eta_u=eu, eta_v=ev, kappa=0.7,
                            s_u=0.7, s_v=0.7, s_n=0.08, c=0.10)

        ax = fig.add_subplot(2, 5, k + 1, projection='3d')
        plot_chart_surface(ax, prim, color='#3aa860')
        cap_zlim(ax, prim, pad=0.10)
        style_ax(ax, title=rf'$\eta^u={eu},\, \eta^v={ev}$')

        ax = fig.add_subplot(2, 5, 5 + k + 1, projection='3d')
        plot_density_shells(ax, prim, color_base='#3aa860')
        style_ax(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure D:  3DGS vs MGS (the key thesis figure)
# ===========================================================================
def fig_3dgs_vs_mgs(out_path):
    """The structural-claim figure.

    Top row:  density shells.   3DGS shells are nested ellipsoids;
              MGS shells nest around a curved 2-manifold.
    Middle row:  v=0 slice colored by density.   3DGS shows blob centered on
                 the point mu_i; MGS shows a curved ridge along the chart.
    Bottom row:  density along the chart curve in MGS, density along v=0 in
                 3DGS, plotted on the same axes -- shows MGS is nearly flat
                 across the chart while 3DGS decays from a single point.
    """
    s = 0.7; s_n = 0.08

    # 3DGS: axis-aligned anisotropic Gaussian
    def density_3dgs(U, V, T, s_u=s, s_v=s, s_n=s_n):
        return np.exp(-0.5 * ((U / s_u) ** 2 + (V / s_v) ** 2 + (T / s_n) ** 2))

    # MGS with the compact-bump window
    prim_mgs = MGSPrimitive(eta_u=0.85, eta_v=0.85, kappa=1.0,
                            s_u=s, s_v=s, s_n=s_n, c=0.10)

    fig = plt.figure(figsize=(13, 12))
    fig.suptitle(
        r"3DGS  (point-centred)   vs.   MGS  (surface-centred, compact-bump window)."
        "\n"
        r"MGS density is high on the entire chart inside the footprint and "
        r"drops sharply at the boundary.",
        fontsize=12,
    )

    # --- Top row: 3D density shells -----------------------------------------
    box_u = 1.05 * s; box_v = 1.05 * s; box_t = 4 * s_n
    N = 70
    uu = np.linspace(-box_u, box_u, N)
    vv = np.linspace(-box_v, box_v, N)
    tt = np.linspace(-box_t, box_t, N)
    U, V, T = np.meshgrid(uu, vv, tt, indexing='ij')

    F_3dgs = density_3dgs(U, V, T)
    F_mgs = density(prim_mgs, U, V, T)
    spacing = np.array([2 * box_u / (N - 1), 2 * box_v / (N - 1), 2 * box_t / (N - 1)])
    origin = np.array([-box_u, -box_v, -box_t])

    for col, (F, label, color) in enumerate([
        (F_3dgs, '3DGS  (point-centred)', '#3a6db0'),
        (F_mgs, 'MGS  (chart $\\kappa=1.0$, $\\eta=0.85$)', '#3aa860'),
    ]):
        ax = fig.add_subplot(3, 2, col + 1, projection='3d')
        base_rgb = np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        for level, falpha in [(0.85, 0.95), (0.5, 0.40), (0.15, 0.15)]:
            try:
                verts, faces, _, _ = marching_cubes(F, level=level)
            except (ValueError, RuntimeError):
                continue
            verts = verts * spacing + origin
            mesh = Poly3DCollection(verts[faces], alpha=falpha, linewidth=0,
                                    antialiased=True)
            tri = verts[faces]
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nl = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
            normals /= nl
            shade = 0.55 + 0.45 * normals[:, 2]
            rgba = np.concatenate([
                np.clip(base_rgb[None, :] * shade[:, None], 0, 1),
                np.full((len(shade), 1), falpha),
            ], axis=1)
            mesh.set_facecolor(rgba)
            ax.add_collection3d(mesh)

        if col == 0:
            ax.scatter([0], [0], [0], color='#101a30', s=45, zorder=20)
            ax.text(0.05, 0.05, 0, r'  $\mu_i$ (peak)', fontsize=9, color='#101a30')
        else:
            theta = np.linspace(0, 2 * np.pi, 150)
            bu = s * np.cos(theta); bv = s * np.sin(theta)
            bA = (prim_mgs.eta_u * bu) ** 2 + (prim_mgs.eta_v * bv) ** 2 + prim_mgs.delta_0 ** 2
            bt = prim_mgs.kappa * (np.sqrt(bA) - prim_mgs.delta_0)
            ax.plot(bu, bv, bt, color='#101a30', linewidth=1.8, alpha=0.85,
                    label=r'chart $\{\phi=0\}$')

        ax.set_xlim(-box_u, box_u); ax.set_ylim(-box_v, box_v)
        ax.set_zlim(-box_t, max(box_t, 0.65))
        style_ax(ax, title=label)

    # --- Middle row: v=0 slice density ---------------------------------------
    Ns = 300
    us = np.linspace(-1.1 * s, 1.1 * s, Ns)
    ts = np.linspace(-4 * s_n, max(4 * s_n, 0.6), Ns)
    Us, Ts = np.meshgrid(us, ts)
    Vs = np.zeros_like(Us)

    F3_slice = density_3dgs(Us, Vs, Ts)
    Fm_slice = density(prim_mgs, Us, Vs, Ts)

    for col, (F, label, cmap, peak_marker) in enumerate([
        (F3_slice, '3DGS  density at $v=0$', 'Blues', 'point'),
        (Fm_slice, 'MGS  density at $v=0$', 'Greens', 'chart'),
    ]):
        ax = fig.add_subplot(3, 2, 2 + col + 1)
        im = ax.pcolormesh(Us, Ts, F, cmap=cmap, vmin=0, vmax=1, shading='auto')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r'$\rho/o$', fontsize=9); cb.ax.tick_params(labelsize=7)

        # Footprint vertical lines
        for ub in [-s, s]:
            ax.axvline(ub, color='black', linestyle='--', linewidth=0.7, alpha=0.5)

        if peak_marker == 'point':
            ax.scatter([0], [0], color='black', s=40, zorder=10)
            ax.annotate(r'$\mu_i$', (0, 0), xytext=(0.07, 0.05), fontsize=10)
        else:
            u_line = np.linspace(-s, s, 200)
            A = (prim_mgs.eta_u * u_line) ** 2 + prim_mgs.delta_0 ** 2
            t_chart = prim_mgs.kappa * (np.sqrt(A) - prim_mgs.delta_0)
            ax.plot(u_line, t_chart, color='black', linewidth=1.8,
                    label=r'chart $\{\phi=0\}$')
            ax.legend(fontsize=8, loc='upper left')

        ax.set_xlabel('u', fontsize=10); ax.set_ylabel('t', fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_xlim(us.min(), us.max()); ax.set_ylim(ts.min(), ts.max())

    # --- Bottom row: density on the peak locus -----------------------------
    ax = fig.add_subplot(3, 1, 3)
    u_line = np.linspace(-s * 1.05, s * 1.05, 600)

    # 3DGS: density at (u, 0, 0) -- on its peak plane t=0
    F3_peak = np.exp(-0.5 * (u_line / s) ** 2)
    # MGS: density at (u, 0, h(u,0)) -- on its chart, which is its peak locus
    A_line = (prim_mgs.eta_u * u_line) ** 2 + prim_mgs.delta_0 ** 2
    t_line = prim_mgs.kappa * (np.sqrt(A_line) - prim_mgs.delta_0)
    W_line = window(prim_mgs, u_line, np.zeros_like(u_line))
    Fm_peak = W_line  # phi=0 on the chart, so exp factor = 1
    # The MGS density on the chart equals W(u, 0) exactly.

    ax.plot(u_line, F3_peak, color='#3a6db0', linewidth=2.5,
            label=r'3DGS: $\rho(u, 0, 0)/o$  (along the flat peak plane)')
    ax.plot(u_line, Fm_peak, color='#3aa860', linewidth=2.5,
            label=r'MGS:  $\rho(u, 0, h(u,0))/o = W(u, 0)$  (along the chart)')
    ax.fill_between(u_line, 0, Fm_peak, color='#3aa860', alpha=0.15)
    ax.axvspan(-s, s, color='black', alpha=0.04)
    for ub in [-s, s]:
        ax.axvline(ub, color='black', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_xlabel(r'$u$  (along the peak locus)', fontsize=10)
    ax.set_ylabel(r'$\rho/o$  on the peak locus', fontsize=10)
    ax.set_title(r'Density along each method''s peak locus.  '
                 r'MGS is nearly flat across the footprint; '
                 r'3DGS falls off as $e^{-u^2/2s_u^2}$.',
                 fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.25)
    ax.set_xlim(u_line.min(), u_line.max()); ax.set_ylim(0, 1.08)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure E:  normal-direction thickness s_n sweep
# ===========================================================================
def fig_sn_sweep(out_path):
    sns = [0.03, 0.06, 0.10, 0.20, 0.40]

    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(
        r"Normal-direction thickness $s_n$  (controls shell thickness around the chart)."
        "\n"
        rf"Fixed $\kappa=+0.6$, $\eta=0.85$, $c=0.10$, $s_u=s_v=0.7$.",
        fontsize=12,
    )

    for k, sn in enumerate(sns):
        prim = MGSPrimitive(eta_u=0.85, eta_v=0.85, kappa=0.6,
                            s_u=0.7, s_v=0.7, s_n=sn, c=0.10)

        ax = fig.add_subplot(2, 5, k + 1, projection='3d')
        plot_chart_surface(ax, prim, color='#3aa860')
        cap_zlim(ax, prim, pad=0.10)
        style_ax(ax, title=rf'$s_n={sn}$')

        ax = fig.add_subplot(2, 5, 5 + k + 1, projection='3d')
        plot_density_shells(ax, prim, color_base='#3aa860')
        style_ax(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main():
    out_dir = "mgs_3d_viz"
    os.makedirs(out_dir, exist_ok=True)

    figs = [
        ("mgs_kappa_sweep.png", fig_kappa_sweep),
        ("mgs_c_sweep.png", fig_c_sweep),
        ("mgs_anisotropy.png", fig_anisotropy_sweep),
        ("mgs_3dgs_vs_mgs.png", fig_3dgs_vs_mgs),
        ("mgs_sn_sweep.png", fig_sn_sweep),
    ]
    for fname, fn in figs:
        path = os.path.join(out_dir, fname)
        fn(path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()