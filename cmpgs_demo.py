#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:47:57 2026

@author: hong
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curved Mixed-Primitive Gaussian Splatting (CMPS) -- primitive visualization.

Two primitive classes, both surface-centred (kernel evaluated AT the ray-
surface intersection, depth = intersection depth, NO off-surface thickness
in the alpha and NO line-integral):

    Curved disk      (2-manifold patch, quadratic height-graph offset)
        surface : h(u,v) = a u^2 + b u v + c v^2
        kernel  : G(u,v) = exp(-((u/s_u)^2 + (v/s_v)^2) / 2)
        Recovers PGSR exactly at a = b = c = 0.
        Curvature gradients dh/da = u^2, dh/db = u v, dh/dc = v^2 are all
        non-degenerate at the flat init -- no bootstrap problem.

    Curved capsule   (1-manifold sweep along a parabolic spine)
        spine   : gamma(t) = (t, beta * t^2, 0),  t in [-L/2, L/2]
        kernel  : G(t,w) = exp(-((t/L)^2 + (w/sigma_perp)^2) / 2)
        Recovers MPGS's straight Gaussian line at beta = 0.
        Curved 1-manifold primitives are novel; no prior splatting work has
        bent 1D primitives.

Five figures:
    A.  Curved-disk quadratic coefficient sweep                (a, b, c)
    B.  Curved-capsule bend sweep                              (beta)
    C.  Taxonomy:  2 x 2 over (intrinsic dimension, curvature)
    D.  Disk -> capsule conversion via aspect-ratio gating
    E.  Comparison with 3DGS / PGSR / MGS / CMPS
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


# ===========================================================================
# Primitive classes
# ===========================================================================
class CurvedDisk:
    """2-manifold patch.  h(u,v) = a u^2 + b u v + c v^2.  Kernel = 2D Gaussian."""
    def __init__(self, s_u=0.7, s_v=0.7, a=0.0, b=0.0, c=0.0, opacity=1.0):
        self.s_u, self.s_v = float(s_u), float(s_v)
        self.a, self.b, self.c = float(a), float(b), float(c)
        self.opacity = float(opacity)

    def chart_h(self, U, V):
        return self.a * U**2 + self.b * U * V + self.c * V**2

    def kernel(self, U, V):
        return np.exp(-0.5 * ((U / self.s_u)**2 + (V / self.s_v)**2))

    def surface_normal(self, U, V):
        """Unit normal at chart point (u, v).  Returns (nx, ny, nz) in local frame."""
        nu = -(2.0 * self.a * U + self.b * V)
        nv = -(self.b * U + 2.0 * self.c * V)
        nz = np.ones_like(U)
        norm = np.sqrt(nu**2 + nv**2 + nz**2)
        return nu / norm, nv / norm, nz / norm


class CurvedCapsule:
    """1-manifold sweep.  gamma(t) = (t, beta t^2, 0).  Kernel = G(t) x G(w)."""
    def __init__(self, L=1.0, sigma_perp=0.08, beta=0.0, opacity=1.0):
        self.L = float(L)
        self.sigma_perp = float(sigma_perp)
        self.beta = float(beta)
        self.opacity = float(opacity)

    def density(self, P, n_t=200):
        """Evaluate density at 3D points P (shape (..., 3)).  Brute-force closest-
        point on a dense sampling of the spine -- adequate for visualization."""
        ts = np.linspace(-self.L * 0.65, self.L * 0.65, n_t)
        spine = np.stack([ts, self.beta * ts**2, np.zeros_like(ts)], axis=-1)
        P_flat = P.reshape(-1, 3)
        diff = P_flat[:, None, :] - spine[None, :, :]            # (N, n_t, 3)
        d2 = np.sum(diff**2, axis=-1)                            # (N, n_t)
        idx = np.argmin(d2, axis=-1)
        t_star = ts[idx]
        w_star = np.sqrt(d2[np.arange(len(idx)), idx])
        density_flat = np.exp(-0.5 * ((t_star / self.L)**2 +
                                       (w_star / self.sigma_perp)**2))
        return density_flat.reshape(P.shape[:-1])


def density_3dgs(U, V, T, s_u=0.7, s_v=0.7, s_n=0.08):
    """Anisotropic 3D Gaussian, for the comparison panel."""
    return np.exp(-0.5 * ((U / s_u)**2 + (V / s_v)**2 + (T / s_n)**2))


def density_mgs_chart(U, V, T, s_u=0.7, s_v=0.7, s_n=0.08,
                        kappa=1.0, eta=0.85, delta0=1e-3, c_bump=0.10):
    """MGS density (compact-bump window + off-surface Gaussian), for comparison."""
    A = (eta * U)**2 + (eta * V)**2 + delta0**2
    h = kappa * (np.sqrt(A) - delta0)
    r = (U / s_u)**2 + (V / s_v)**2
    inside = r < 1.0
    safe = np.where(inside, r, 0.0)
    W = np.zeros_like(r)
    W[inside] = np.exp(-c_bump * safe[inside] /
                        np.maximum(1.0 - safe[inside], 1e-12))
    phi = T - h
    return W * np.exp(-0.5 * (phi / s_n)**2)


# ===========================================================================
# Plot helpers
# ===========================================================================
def plot_curved_disk_surface(ax, prim, n=100, color='#3aa860',
                              alpha_max=0.95, ext_sigma=2.0):
    """Render { (u, v, h(u,v)) } shaded by kernel value G(u,v)."""
    u = np.linspace(-ext_sigma * prim.s_u, ext_sigma * prim.s_u, n)
    v = np.linspace(-ext_sigma * prim.s_v, ext_sigma * prim.s_v, n)
    U, V = np.meshgrid(u, v)
    K = prim.kernel(U, V)
    T = prim.chart_h(U, V)
    mask = K > 0.011                       # ~3-sigma cutoff

    base_rgb = np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    facecolors = np.zeros((n - 1, n - 1, 4))
    Kmid = 0.25 * (K[:-1, :-1] + K[1:, :-1] + K[:-1, 1:] + K[1:, 1:])
    mask_mid = 0.25 * (mask[:-1, :-1].astype(float) + mask[1:, :-1] +
                       mask[:-1, 1:] + mask[1:, 1:])
    facecolors[..., 0] = base_rgb[0]
    facecolors[..., 1] = base_rgb[1]
    facecolors[..., 2] = base_rgb[2]
    facecolors[..., 3] = alpha_max * Kmid * (mask_mid > 0.99)

    Tplot = np.where(mask, T, np.nan)
    ax.plot_surface(U, V, Tplot, facecolors=facecolors,
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False)

    # Centre marker (mu_i)
    ax.scatter([0], [0], [0], color='#1a4525', s=22, zorder=20)

    # 2-sigma footprint curve on the surface
    theta = np.linspace(0, 2 * np.pi, 200)
    bu = 2 * prim.s_u * np.cos(theta)
    bv = 2 * prim.s_v * np.sin(theta)
    bt = prim.chart_h(bu, bv)
    ax.plot(bu, bv, bt, color='black', linewidth=1.1,
            linestyle='--', alpha=0.55, zorder=18)


def plot_capsule_shells(ax, prim, color='#c84d3c', N=60,
                         level_fracs=(0.85, 0.5, 0.15),
                         face_alphas=(0.92, 0.45, 0.18)):
    """Marching-cubes isosurfaces of the capsule's 3D density."""
    bx = prim.L * 0.55 + 3 * prim.sigma_perp
    by_top = max(prim.beta * (prim.L * 0.5)**2 + 3 * prim.sigma_perp,
                 3 * prim.sigma_perp)
    by_bot = -3 * prim.sigma_perp if prim.beta >= 0 else \
             prim.beta * (prim.L * 0.5)**2 - 3 * prim.sigma_perp
    bz = 3 * prim.sigma_perp

    x = np.linspace(-bx, bx, N)
    y = np.linspace(by_bot, by_top, N)
    z = np.linspace(-bz, bz, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    P = np.stack([X, Y, Z], axis=-1)
    F = prim.density(P, n_t=120)

    spacing = np.array([(x[-1] - x[0]) / (N - 1),
                        (y[-1] - y[0]) / (N - 1),
                        (z[-1] - z[0]) / (N - 1)])
    origin = np.array([x[0], y[0], z[0]])
    base_rgb = np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])

    for level, falpha in sorted(zip(level_fracs, face_alphas), key=lambda x: x[0]):
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

    # Spine curve
    ts = np.linspace(-prim.L * 0.5, prim.L * 0.5, 100)
    ax.plot(ts, prim.beta * ts**2, np.zeros_like(ts),
            color='black', linewidth=2.2, alpha=0.95, zorder=25)
    ax.scatter([0], [0], [0], color='#1a4525', s=22, zorder=26)

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_zlim(z[0], z[-1])


def style_ax(ax, view=(22, -58), title=""):
    ax.set_xlabel('u', fontsize=8, labelpad=-6)
    ax.set_ylabel('v', fontsize=8, labelpad=-6)
    ax.set_zlabel('t', fontsize=8, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect((1, 1, 0.6))
    if title:
        ax.set_title(title, fontsize=10)


def cap_disk_zlim(ax, prim, pad=0.08, ext=2.0):
    """Set a sensible z-range for a curved-disk axis.  Samples h on a 5x5 grid
    so saddle and ridge cases (where extrema sit on the axes, not corners)
    are captured correctly."""
    box_u = ext * prim.s_u
    box_v = ext * prim.s_v
    us = np.linspace(-box_u, box_u, 5)
    vs = np.linspace(-box_v, box_v, 5)
    U, V = np.meshgrid(us, vs)
    h_vals = prim.chart_h(U, V).flatten()
    h_vals = np.concatenate([h_vals, [0.0]])
    lo, hi = float(np.min(h_vals)), float(np.max(h_vals))
    span = max(hi - lo, 0.20)
    ax.set_zlim(lo - pad * span, hi + pad * span)
    ax.set_xlim(-box_u, box_u)
    ax.set_ylim(-box_v, box_v)


# ===========================================================================
# Figure A : curved-disk quadratic coefficient sweep
# ===========================================================================
def fig_curved_disk_sweep(out_path):
    """Sweep (a, b, c).  Show how the quadratic offset realises different
    local shapes (flat, bowl, saddle, asymmetric ridge)."""
    cases = [
        (0.0,  0.0,  0.0,  "flat (PGSR)",                '#1f5fb4'),
        (0.4,  0.0,  0.0,  r"$a=0.4$  parabolic-u",      '#3aa860'),
        (0.4,  0.0,  0.4,  r"$a=c=0.4$  bowl",           '#3aa860'),
        (0.4,  0.0, -0.4,  r"$a=0.4,c=-0.4$  saddle",    '#c84d3c'),
        (0.2,  0.5,  0.2,  r"$b$ large  twisted ridge",  '#a060c0'),
    ]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        r"Curved disk  $h(u,v) = a\,u^2 + b\,u v + c\,v^2$  "
        r"with Gaussian kernel $G(u,v)$.   "
        r"Sweep over $(a,b,c)$.   "
        r"$\partial h/\partial a = u^2$, $\partial h/\partial b = uv$, "
        r"$\partial h/\partial c = v^2$  are non-degenerate at the flat init.",
        fontsize=12,
    )

    for k, (a, b, c, label, color) in enumerate(cases):
        prim = CurvedDisk(s_u=0.7, s_v=0.7, a=a, b=b, c=c)

        # Row 1: 3D surface coloured by kernel
        ax = fig.add_subplot(3, 5, k + 1, projection='3d')
        plot_curved_disk_surface(ax, prim, color=color)
        cap_disk_zlim(ax, prim)
        style_ax(ax, title=label)

        # Row 2: top-down view (kernel * surface coloured contour)
        ax = fig.add_subplot(3, 5, 5 + k + 1)
        n = 200
        u = np.linspace(-3 * prim.s_u, 3 * prim.s_u, n)
        v = np.linspace(-3 * prim.s_v, 3 * prim.s_v, n)
        U, V = np.meshgrid(u, v)
        K = prim.kernel(U, V)
        H = prim.chart_h(U, V)
        # Contours of h colored on top of kernel intensity
        ax.pcolormesh(U, V, K, cmap='Greys', vmin=0, vmax=1, shading='auto', alpha=0.7)
        cs = ax.contour(U, V, H, levels=10, colors=color, linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
        ax.set_aspect('equal')
        ax.set_xlabel('u', fontsize=9)
        ax.set_ylabel('v', fontsize=9)
        ax.set_title(r"kernel (grey) + chart contours", fontsize=9)
        ax.tick_params(labelsize=7)

        # Row 3: 1D profile  h(u, 0) and G(u, 0)
        ax = fig.add_subplot(3, 5, 10 + k + 1)
        u_line = np.linspace(-3.2 * prim.s_u, 3.2 * prim.s_u, 400)
        t_line = prim.a * u_line**2
        K_line = np.exp(-0.5 * (u_line / prim.s_u)**2)
        ax.plot(u_line, t_line, color=color, linewidth=2.5,
                label=r'$h(u, 0)$')
        ax.axhline(0, color='black', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.set_xlabel(r'$u$', fontsize=9)
        ax.set_ylabel(r'$h$', fontsize=9, color=color)
        ax.tick_params(axis='y', labelcolor=color, labelsize=7)
        ax.set_xlim(u_line.min(), u_line.max())

        ax2 = ax.twinx()
        ax2.fill_between(u_line, 0, K_line, color='gray', alpha=0.18)
        ax2.plot(u_line, K_line, color='gray', linewidth=1.2)
        ax2.set_ylabel(r'$G(u,0)$', fontsize=9, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
        ax2.set_ylim(0, 1.15)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure B : curved-capsule bend sweep
# ===========================================================================
def fig_capsule_bend_sweep(out_path):
    """Sweep beta.  beta = 0 is MPGS's Gaussian line.  beta > 0 bends it --
    no prior splatting work has curved 1D primitives."""
    betas = [0.0, 0.3, 0.6, 1.0, 1.5]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        r"Curved capsule  $\gamma(t) = (t, \beta t^2, 0)$  "
        r"with Gaussian kernel along spine and across cross-section.   "
        r"$\beta = 0$ is the straight Gaussian line (MPGS limit); "
        r"$\beta \ne 0$ is the novel curved 1-manifold primitive.",
        fontsize=12,
    )

    for k, beta in enumerate(betas):
        prim = CurvedCapsule(L=1.0, sigma_perp=0.10, beta=beta)
        label = rf"$\beta = {beta}$"
        if beta == 0.0:
            label += "  (MPGS line)"

        # Row 1: 3D density shells
        ax = fig.add_subplot(3, 5, k + 1, projection='3d')
        plot_capsule_shells(ax, prim, color='#c84d3c')
        style_ax(ax, title=label)

        # Row 2: top-down density slice (z = 0)
        ax = fig.add_subplot(3, 5, 5 + k + 1)
        n = 200
        bx = prim.L * 0.6 + 3 * prim.sigma_perp
        by_top = max(prim.beta * (prim.L * 0.5)**2 + 3 * prim.sigma_perp,
                     3 * prim.sigma_perp)
        by_bot = -3 * prim.sigma_perp
        xx = np.linspace(-bx, bx, n)
        yy = np.linspace(by_bot, by_top, n)
        XX, YY = np.meshgrid(xx, yy)
        P = np.stack([XX, YY, np.zeros_like(XX)], axis=-1)
        F2D = prim.density(P, n_t=120)
        im = ax.pcolormesh(XX, YY, F2D, cmap='Reds', vmin=0, vmax=1, shading='auto')
        # spine overlay
        ts = np.linspace(-prim.L * 0.5, prim.L * 0.5, 100)
        ax.plot(ts, prim.beta * ts**2, color='black', linewidth=1.8, alpha=0.9)
        ax.set_aspect('equal')
        ax.set_xlabel('u (along)', fontsize=9)
        ax.set_ylabel('v (bend)', fontsize=9)
        ax.set_title(r"density at $w=0$", fontsize=9)
        ax.tick_params(labelsize=7)

        # Row 3: density along spine
        ax = fig.add_subplot(3, 5, 10 + k + 1)
        n_t = 300
        ts = np.linspace(-prim.L * 0.6, prim.L * 0.6, n_t)
        # density at spine points (w = 0)
        spine_pts = np.stack([ts, prim.beta * ts**2, np.zeros_like(ts)], axis=-1)
        rho_on_spine = prim.density(spine_pts, n_t=200)
        ax.fill_between(ts, 0, rho_on_spine, color='#c84d3c', alpha=0.25)
        ax.plot(ts, rho_on_spine, color='#c84d3c', linewidth=2.5,
                label=r'$\rho$ on spine')
        ax.axvline(-prim.L * 0.5, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.axvline(+prim.L * 0.5, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.set_xlabel(r'$t$', fontsize=9)
        ax.set_ylabel(r'$\rho(\gamma(t))$', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure C : the 2 x 2 taxonomy table
# ===========================================================================
def fig_taxonomy(out_path):
    """Intrinsic dimension x intrinsic curvature.  CMPS fills (2D + 1D, curved)."""
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(
        r"Primitive taxonomy:  intrinsic dimension  $\times$  intrinsic curvature."
        "\n"
        r"3DGS = (3D, n/a).  2DGS / PGSR = (2D, flat).  MPGS = (2D + 1D, flat).  "
        r"QGS = (2D, curved).  "
        r"CMPS fills the (1D, curved) cell  +  unifies (2D, curved) + (1D, curved).",
        fontsize=12,
    )

    cells = [
        # row, col, label, primitive constructor, plotter
        (0, 0, "(2-manifold, flat)\nPGSR / 2DGS",
            lambda: CurvedDisk(s_u=0.7, s_v=0.7, a=0, b=0, c=0),
            'disk', '#1f5fb4'),
        (0, 1, "(2-manifold, curved)\nCMPS curved disk\n(also QGS)",
            lambda: CurvedDisk(s_u=0.7, s_v=0.7, a=0.5, b=0.0, c=-0.3),
            'disk', '#3aa860'),
        (1, 0, "(1-manifold, flat)\nMPGS Gaussian line",
            lambda: CurvedCapsule(L=1.0, sigma_perp=0.08, beta=0.0),
            'cap', '#a06030'),
        (1, 1, "(1-manifold, curved)\nCMPS curved capsule\nNOVEL",
            lambda: CurvedCapsule(L=1.0, sigma_perp=0.08, beta=1.0),
            'cap', '#c84d3c'),
    ]

    for r, c, label, ctor, kind, color in cells:
        idx = 2 * r + c + 1
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        prim = ctor()
        if kind == 'disk':
            plot_curved_disk_surface(ax, prim, color=color)
            cap_disk_zlim(ax, prim)
        else:
            plot_capsule_shells(ax, prim, color=color)
        style_ax(ax, title=label)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure D : aspect-ratio gating  (disk -> capsule)
# ===========================================================================
def fig_aspect_transition(out_path):
    """Stretched disk (s_u >> s_v) -- the model implicitly wants a 1D primitive.
    Show the disk's degeneracy at high aspect ratio and the capsule alternative.

    The conversion rule:  s_u / s_v > tau_hi  ->  disk to capsule
                           L = 2 s_u,  sigma_perp = s_v
    """
    aspect_ratios = [(0.7, 0.7), (0.9, 0.45), (1.0, 0.25), (1.1, 0.15), (1.2, 0.08)]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        r"Aspect-ratio gating:  as a curved disk gets stretched "
        r"($s_u/s_v \to \infty$), it tends to a 1D structure  "
        r"that the capsule represents more efficiently."
        "\n"
        r"Conversion rule (residual-driven, in train.py):  "
        r"$s_u/s_v > \tau_{hi}$  $\Rightarrow$  swap to capsule with  "
        r"$L = 2 s_u,\ \sigma_\perp = s_v,\ \beta = 0$.   "
        r"Reverse swap when in-plane gradient grows.",
        fontsize=12,
    )

    for k, (s_u, s_v) in enumerate(aspect_ratios):
        ratio = s_u / s_v
        col_disk = '#1f5fb4' if ratio < 4 else '#7e7e7e'  # grey out high-aspect disks
        col_cap = '#c84d3c'

        # Row 1: the stretched curved-disk version
        prim_d = CurvedDisk(s_u=s_u, s_v=s_v, a=0.4, b=0.0, c=0.0)
        ax = fig.add_subplot(2, 5, k + 1, projection='3d')
        plot_curved_disk_surface(ax, prim_d, color=col_disk)
        cap_disk_zlim(ax, prim_d)
        title = (rf"disk:  $s_u/s_v = {ratio:.1f}$")
        if ratio > 4:
            title += "    too stretched"
        style_ax(ax, title=title)

        # Row 2: the matched curved-capsule version (only when ratio is high enough)
        ax = fig.add_subplot(2, 5, 5 + k + 1, projection='3d')
        if ratio >= 2.0:
            prim_c = CurvedCapsule(L=2 * s_u, sigma_perp=s_v, beta=0.4 / s_u)
            plot_capsule_shells(ax, prim_c, color=col_cap)
            cap_title = rf"capsule:  $L={2*s_u:.2f},\ \sigma_\perp={s_v:.2f}$"
        else:
            # Render an empty 3D box with a centred annotation
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.text2D(0.5, 0.5, "(no swap)\nkeep as disk",
                      transform=ax.transAxes, ha='center', va='center',
                      fontsize=10, color='gray')
            cap_title = "no conversion"
        style_ax(ax, title=cap_title)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure E : comparison with 3DGS / PGSR / MGS / CMPS
# ===========================================================================
def fig_comparison(out_path):
    """Side-by-side: density along v=0 slice for the four formulations."""
    s = 0.7; s_n = 0.08

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        r"Four formulations, same scale.   "
        r"3DGS: point-centred 3D Gaussian.   "
        r"PGSR: planar disk, 2D Gaussian on the plane.   "
        r"MGS:  curved disk, 2D window $\times$ off-surface Gaussian thickness $\sigma_n$.   "
        r"CMPS: curved disk + curved capsule, 2D Gaussian on the surface, no $\sigma_n$.",
        fontsize=12,
    )

    # ----- Top row: 3D density / surface for each ---------------------------
    box_u = 1.05 * s; box_v = 1.05 * s; box_t = 0.8
    N = 60
    uu = np.linspace(-box_u, box_u, N)
    vv = np.linspace(-box_v, box_v, N)
    tt = np.linspace(-box_t * 0.5, box_t, N)
    U3, V3, T3 = np.meshgrid(uu, vv, tt, indexing='ij')

    # 3DGS volumetric
    F_3dgs = density_3dgs(U3, V3, T3, s_u=s, s_v=s, s_n=s_n)

    # MGS volumetric (compact-bump + sigma_n thickness)
    F_mgs = density_mgs_chart(U3, V3, T3, s_u=s, s_v=s, s_n=s_n,
                              kappa=1.0, eta=0.85)

    spacing = np.array([(uu[-1] - uu[0]) / (N - 1),
                        (vv[-1] - vv[0]) / (N - 1),
                        (tt[-1] - tt[0]) / (N - 1)])
    origin = np.array([uu[0], vv[0], tt[0]])

    def render_shells(ax, F, color):
        base_rgb = np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        for level, falpha in [(0.85, 0.95), (0.5, 0.40), (0.15, 0.15)]:
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
        ax.set_xlim(uu[0], uu[-1]); ax.set_ylim(vv[0], vv[-1]); ax.set_zlim(tt[0], tt[-1])

    # 3DGS panel
    ax = fig.add_subplot(2, 4, 1, projection='3d')
    render_shells(ax, F_3dgs, '#3a6db0')
    ax.scatter([0], [0], [0], color='#101a30', s=35, zorder=20)
    style_ax(ax, title="3DGS\npoint-centred")

    # PGSR panel (just the planar disk -- a flat surface, no volumetric thickness)
    ax = fig.add_subplot(2, 4, 2, projection='3d')
    prim_pgsr = CurvedDisk(s_u=s, s_v=s, a=0, b=0, c=0)
    plot_curved_disk_surface(ax, prim_pgsr, color='#1f5fb4')
    cap_disk_zlim(ax, prim_pgsr)
    style_ax(ax, title="PGSR\nplanar disk")

    # MGS panel (compact-bump x off-surface Gaussian)
    ax = fig.add_subplot(2, 4, 3, projection='3d')
    render_shells(ax, F_mgs, '#3aa860')
    style_ax(ax, title="MGS\ncurved + $\\sigma_n$ thickness")

    # CMPS panel (curved disk, surface-only kernel)
    ax = fig.add_subplot(2, 4, 4, projection='3d')
    prim_cmps = CurvedDisk(s_u=s, s_v=s, a=0.5, b=0, c=-0.3)
    plot_curved_disk_surface(ax, prim_cmps, color='#c84d3c')
    cap_disk_zlim(ax, prim_cmps)
    style_ax(ax, title="CMPS curved disk\nsurface-only kernel")

    # ----- Bottom row: 2D density slice at v=0 ------------------------------
    Ns = 300
    us = np.linspace(-1.4 * s, 1.4 * s, Ns)
    ts = np.linspace(-0.45, 0.85, Ns)
    Us2D, Ts2D = np.meshgrid(us, ts)
    Vs2D = np.zeros_like(Us2D)

    F3_slice = density_3dgs(Us2D, Vs2D, Ts2D, s_u=s, s_v=s, s_n=s_n)

    # PGSR slice: the kernel is 2D Gaussian on the t = 0 plane; in 3D it's
    # a delta along t.  For visualization, narrow Gaussian along t with
    # width = 1 voxel (purely for legibility).
    F_pgsr_slice = np.exp(-0.5 * (Us2D / s)**2) * np.exp(-0.5 * (Ts2D / (1.5 * s_n))**2)

    F_mgs_slice = density_mgs_chart(Us2D, Vs2D, Ts2D, s_u=s, s_v=s, s_n=s_n,
                                     kappa=1.0, eta=0.85)

    # CMPS slice: 2D Gaussian kernel evaluated on the curved disk at t = h(u, 0)
    # For visualization, narrow Gaussian along normal direction
    prim_cmps_for_slice = CurvedDisk(s_u=s, s_v=s, a=0.5, b=0, c=-0.3)
    H_slice = prim_cmps_for_slice.chart_h(Us2D, Vs2D)
    K_slice = prim_cmps_for_slice.kernel(Us2D, Vs2D)
    # narrow ridge along surface
    F_cmps_slice = K_slice * np.exp(-0.5 * ((Ts2D - H_slice) / (1.5 * s_n))**2)

    panels = [
        (F3_slice,    "3DGS  $v=0$ slice",           'Blues',  'point'),
        (F_pgsr_slice,"PGSR  $v=0$ slice",           'Blues',  'plane'),
        (F_mgs_slice, "MGS  $v=0$ slice",            'Greens', 'curve_mgs'),
        (F_cmps_slice,"CMPS curved-disk $v=0$ slice",'Reds',   'curve_cmps'),
    ]
    for col, (F, label, cmap, kind) in enumerate(panels):
        ax = fig.add_subplot(2, 4, 4 + col + 1)
        im = ax.pcolormesh(Us2D, Ts2D, F, cmap=cmap, vmin=0, vmax=1, shading='auto')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)

        for ub in [-s, s]:
            ax.axvline(ub, color='black', linestyle='--', linewidth=0.6, alpha=0.45)

        if kind == 'point':
            ax.scatter([0], [0], color='black', s=35, zorder=10)
        elif kind == 'plane':
            ax.axhline(0, color='black', linewidth=1.6, alpha=0.85)
        elif kind == 'curve_mgs':
            u_line = np.linspace(-s, s, 200)
            A = (0.85 * u_line)**2 + 1e-6
            h_line = 1.0 * (np.sqrt(A) - 1e-3)
            ax.plot(u_line, h_line, color='black', linewidth=1.6, alpha=0.85)
        elif kind == 'curve_cmps':
            u_line = np.linspace(-s, s, 200)
            h_line = 0.5 * u_line**2
            ax.plot(u_line, h_line, color='black', linewidth=1.6, alpha=0.85)

        ax.set_xlabel('u', fontsize=9); ax.set_ylabel('t', fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(us[0], us[-1]); ax.set_ylim(ts[0], ts[-1])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main():
    out_dir = "cmps_3d_viz"
    os.makedirs(out_dir, exist_ok=True)

    figs = [
        ("cmps_curved_disk_sweep.png",   fig_curved_disk_sweep),
        ("cmps_capsule_bend_sweep.png",  fig_capsule_bend_sweep),
        ("cmps_taxonomy.png",            fig_taxonomy),
        ("cmps_aspect_transition.png",   fig_aspect_transition),
        ("cmps_comparison.png",          fig_comparison),
    ]
    for fname, fn in figs:
        path = os.path.join(out_dir, fname)
        fn(path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()