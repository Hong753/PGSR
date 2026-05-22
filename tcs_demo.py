#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCS (Tangent Cone Splatting) demo, kappa-aware revision.

A TCS primitive carries a local scalar field

    h(u, v)        = kappa * (sqrt((eta^u u)^2 + (eta^v v)^2 + delta^2)
                              - delta)
    phi(u, v, t)   = t - h(u, v)
    rho(x)         = opacity * exp(-0.5 *
                       [phi^2 / sigma_n^2 + u^2/s_u^2 + v^2/s_v^2])

Parameters per primitive:
    mu, R         : center and local frame (columns t_u, t_v, n_0)
    s_u, s_v      : in-plane footprint scales
    sigma_n       : thickness along normal
    eta_u, eta_v >= 0 : anisotropic slope magnitudes
    delta > 0     : smoothing radius (apex rounding)
    kappa  in R   : signed curvature scale  <-- the NEW parameter

kappa = 0  : flat PGSR plane (eta, delta inactive)
kappa > 0  : cone opens in +n_0 direction (convex from +n_0)
kappa < 0  : cone opens in -n_0 direction
delta -> 0 : sharp anisotropic cone
delta large: paraboloid (QGS regime)
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


# ---------------------------------------------------------------------------
def make_camera(image_size=192, focal=190.0, cam_pos=(0.0, 0.0, 2.8),
                look_dir=(0, 0, -1)):
    H = W = image_size
    fx = fy = focal
    cx, cy = W / 2, H / 2

    z = -np.array(look_dir, dtype=np.float64); z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_cw = np.stack([x, y, z], axis=1)

    px, py = np.meshgrid(np.arange(W), np.arange(H))
    cam_dirs = np.stack([
        (px - cx) / fx,
        -(py - cy) / fy,
        -np.ones_like(px),
    ], axis=-1).astype(np.float64)
    cam_dirs /= np.linalg.norm(cam_dirs, axis=-1, keepdims=True)

    ray_dirs = cam_dirs @ R_cw.T
    ray_origins = np.broadcast_to(np.array(cam_pos), ray_dirs.shape).copy()
    return ray_origins, ray_dirs


# ---------------------------------------------------------------------------
class TCSPrimitive:
    """Tangent-cone splatting primitive (kappa-aware).

    h(u, v) = kappa * (sqrt((eta_u u)^2 + (eta_v v)^2 + delta^2) - delta)
    """
    def __init__(self, mu, R, s_u, s_v, sigma_n,
                 eta_u, eta_v, delta, kappa,
                 opacity, color):
        self.mu      = np.asarray(mu, dtype=np.float64)
        self.R       = np.asarray(R, dtype=np.float64)
        self.s_u     = float(s_u)
        self.s_v     = float(s_v)
        self.sigma_n = float(sigma_n)
        self.eta_u   = float(eta_u)
        self.eta_v   = float(eta_v)
        self.delta   = float(delta)
        self.kappa   = float(kappa)
        self.opacity = float(opacity)
        self.color   = np.asarray(color, dtype=np.float64)

    @property
    def eta_bar(self):
        """Dimensionless cone rise:
           |kappa| * sqrt((eta_u s_u)^2 + (eta_v s_v)^2).

        This is the actual quantity that bounds how much the surface chart
        bends across the footprint.  Densification triggers when it
        exceeds 1.
        """
        return abs(self.kappa) * np.sqrt((self.eta_u * self.s_u) ** 2
                                          + (self.eta_v * self.s_v) ** 2)


def world_to_local(prim, x):
    return (x - prim.mu) @ prim.R


def h_local(prim, uvt):
    """h(u, v) = kappa * (sqrt(A) - delta), with
       A = (eta_u u)^2 + (eta_v v)^2 + delta^2.
    """
    u, v = uvt[..., 0], uvt[..., 1]
    A = (prim.eta_u * u) ** 2 + (prim.eta_v * v) ** 2 + prim.delta ** 2
    return prim.kappa * (np.sqrt(A) - prim.delta)


def phi_local(prim, uvt):
    return uvt[..., 2] - h_local(prim, uvt)


def density_exponent_local(prim, uvt):
    u, v = uvt[..., 0], uvt[..., 1]
    phi = phi_local(prim, uvt)
    return (phi * phi) / (prim.sigma_n ** 2) \
           + (u * u) / (prim.s_u ** 2) \
           + (v * v) / (prim.s_v ** 2)


def density_world(prim, x):
    return prim.opacity * np.exp(
        -0.5 * density_exponent_local(prim, world_to_local(prim, x)))


# ---------------------------------------------------------------------------
def _exponent_along_ray(prim, o_local, d_local, tau):
    uvt = o_local + tau[..., None] * d_local
    return density_exponent_local(prim, uvt)


def ray_primitive_max(prim, o_world, d_world, tau_window=4.0, n_init=32,
                       newton_steps=6):
    shape = o_world.shape[:-1]
    o = o_world.reshape(-1, 3); d = d_world.reshape(-1, 3)
    Nrays = o.shape[0]
    o_loc = (o - prim.mu) @ prim.R
    d_loc = d @ prim.R
    tau_center = -np.einsum('ij,ij->i', o - prim.mu, d)

    grid = np.linspace(-tau_window, tau_window, n_init)
    taus = tau_center[:, None] + grid[None, :]
    uvt = o_loc[:, None, :] + taus[..., None] * d_loc[:, None, :]
    E = density_exponent_local(prim, uvt)
    best = np.argmin(E, axis=1)
    tau = taus[np.arange(Nrays), best]

    for _ in range(newton_steps):
        eps = 1e-3
        Ep = _exponent_along_ray(prim, o_loc, d_loc, tau + eps)
        Em = _exponent_along_ray(prim, o_loc, d_loc, tau - eps)
        E0 = _exponent_along_ray(prim, o_loc, d_loc, tau)
        first  = (Ep - Em) / (2 * eps)
        second = (Ep - 2 * E0 + Em) / (eps ** 2)
        step = first / np.where(np.abs(second) > 1e-8, second, 1e-8)
        step = np.clip(step, -0.3, 0.3)
        tau = tau - step

    uvt_star = o_loc + tau[:, None] * d_loc
    E_star = density_exponent_local(prim, uvt_star)
    return tau.reshape(shape), E_star.reshape(shape), \
           uvt_star.reshape(*shape, 3)


# ---------------------------------------------------------------------------
def make_tcs_primitive(eta_u=1.0, eta_v=1.0, delta=0.15, kappa=1.0,
                        scale=0.7, thin=0.12,
                        color=(0.86, 0.20, 0.25)):
    return TCSPrimitive(
        mu=np.zeros(3), R=np.eye(3),
        s_u=scale, s_v=scale, sigma_n=scale * thin,
        eta_u=eta_u, eta_v=eta_v, delta=delta, kappa=kappa,
        opacity=0.95, color=np.array(color),
    )


# ---------------------------------------------------------------------------
def _isosurface_local(prim, level_frac,
                      n_sigma_uv=2.5, N=48):
    iso_exp = -2.0 * np.log(level_frac)
    box_u = n_sigma_uv * prim.s_u
    box_v = n_sigma_uv * prim.s_v
    cone_h_pos = abs(prim.kappa) * (
        np.sqrt((prim.eta_u * box_u) ** 2 + (prim.eta_v * box_v) ** 2
                + prim.delta ** 2) - prim.delta)
    t_extent = 1.2 * cone_h_pos + 3 * prim.sigma_n
    if prim.kappa >= 0:
        t_min, t_max = -0.4 * t_extent, t_extent
    else:
        t_min, t_max = -t_extent, 0.4 * t_extent

    u = np.linspace(-box_u, box_u, N)
    v = np.linspace(-box_v, box_v, N)
    t = np.linspace(t_min, t_max, N)
    U, V, T = np.meshgrid(u, v, t, indexing='ij')
    uvt = np.stack([U, V, T], axis=-1)
    E = density_exponent_local(prim, uvt)
    try:
        verts, faces, _, _ = marching_cubes(E, level=iso_exp)
    except (ValueError, RuntimeError):
        return None, None, (box_u, box_v, (t_min, t_max))
    scale_vec = np.array([
        2 * box_u / (N - 1),
        2 * box_v / (N - 1),
        (t_max - t_min) / (N - 1),
    ])
    origin = np.array([-box_u, -box_v, t_min])
    verts = verts * scale_vec + origin
    return verts, faces, (box_u, box_v, (t_min, t_max))


# ---------------------------------------------------------------------------
def plot_cone_zero_set(ax, prim, n=80, box_factor=2.5,
                        color='#3aa860', alpha=0.85,
                        show_apex=True, show_footprint=True):
    box_u = box_factor * prim.s_u
    box_v = box_factor * prim.s_v
    u = np.linspace(-box_u, box_u, n)
    v = np.linspace(-box_v, box_v, n)
    U, V = np.meshgrid(u, v)
    A = (prim.eta_u * U) ** 2 + (prim.eta_v * V) ** 2 + prim.delta ** 2
    T = prim.kappa * (np.sqrt(A) - prim.delta)
    ax.plot_surface(U, V, T, color=color, alpha=alpha,
                    edgecolor='none', linewidth=0.0,
                    antialiased=True, rstride=1, cstride=1)
    ax.plot_surface(
        np.array([[-box_u, box_u], [-box_u, box_u]]),
        np.array([[-box_v, -box_v], [box_v, box_v]]),
        np.zeros((2, 2)),
        color='black', alpha=0.06, edgecolor='none',
    )
    if show_footprint:
        theta = np.linspace(0, 2 * np.pi, 80)
        ax.plot(box_factor * prim.s_u * np.cos(theta),
                box_factor * prim.s_v * np.sin(theta),
                np.zeros_like(theta),
                color='black', linewidth=1.4, zorder=18)
    if show_apex:
        ax.scatter([0], [0], [0], color='#1f6035', s=22, zorder=20)


def _set_zlim_for_cone(ax, prim, box_u=None, pad_frac=0.10):
    if box_u is None:
        box_u = 2.5 * prim.s_u
    cone_h_pos = abs(prim.kappa) * (
        np.sqrt((prim.eta_u * box_u) ** 2 + (prim.eta_v * box_u) ** 2
                + prim.delta ** 2) - prim.delta)
    cone_h_pos = max(cone_h_pos, 0.20)
    if prim.kappa >= 0:
        ax.set_zlim(-pad_frac * cone_h_pos, (1 + pad_frac) * cone_h_pos)
    else:
        ax.set_zlim(-(1 + pad_frac) * cone_h_pos, pad_frac * cone_h_pos)


def plot_cone_nested_shells(ax, prim,
                            level_fracs=(0.85, 0.5, 0.15),
                            face_alphas=(0.95, 0.55, 0.22),
                            N=50, color_base=(0.18, 0.62, 0.42)):
    box_u = 2.5 * prim.s_u
    box_v = 2.5 * prim.s_v
    cone_h_pos = abs(prim.kappa) * (
        np.sqrt((prim.eta_u * box_u) ** 2 + (prim.eta_v * box_v) ** 2
                + prim.delta ** 2) - prim.delta)

    sorted_layers = sorted(zip(level_fracs, face_alphas),
                            key=lambda x: x[0])
    for level_frac, falpha in sorted_layers:
        verts, faces, _ = _isosurface_local(prim, level_frac, N=N)
        if verts is None:
            continue
        mesh = Poly3DCollection(verts[faces], alpha=falpha,
                                linewidth=0.0, antialiased=True)
        tri = verts[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nlen = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
        normals /= nlen
        shade = 0.55 + 0.45 * normals[:, 2]
        face_colors = np.clip(np.array(color_base)[None, :] * shade[:, None],
                              0, 1)
        face_colors = np.concatenate(
            [face_colors, np.full((len(face_colors), 1), falpha)], axis=1)
        mesh.set_facecolor(face_colors)
        ax.add_collection3d(mesh)

    plot_cone_zero_set(ax, prim, n=44, alpha=0.30, color='#1f6035',
                        show_apex=False, show_footprint=False)

    if prim.kappa >= 0:
        z_min = -1.1 * prim.sigma_n
        z_max = 1.2 * cone_h_pos + 3 * prim.sigma_n
    else:
        z_min = -(1.2 * cone_h_pos + 3 * prim.sigma_n)
        z_max = 1.1 * prim.sigma_n
    ax.set_xlim(-box_u, box_u); ax.set_ylim(-box_v, box_v)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('u', fontsize=8, labelpad=-6)
    ax.set_ylabel('v', fontsize=8, labelpad=-6)
    ax.set_zlabel('t', fontsize=8, labelpad=-6)
    ax.tick_params(labelsize=6, pad=-2)
    ax.set_box_aspect((1.0, 1.0, 0.7))
    ax.view_init(elev=20, azim=-58)


# ===========================================================================
# Figure A : the new headline -- kappa sweep through PGSR
# ---------------------------------------------------------------------------
def figure_tcs_kappa_sweep(out_path):
    """Smooth transition through PGSR as kappa crosses zero.

    Top row: 3D zero-set surface.
    Bottom row: v = 0 slice.

    Five values: kappa in {-1, -1/2, 0, +1/2, +1}.  At kappa = 0 the surface
    is the flat PGSR plane exactly.  Sign(kappa) flips the cone direction.
    Optimisation can walk smoothly through kappa = 0 to change surface
    curvature direction without rotating R.
    """
    s = 0.7
    eta_u = eta_v = 0.85
    delta = 0.15
    kappas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    color_neg = '#c41e3a'
    color_pgsr = '#1f5fb4'
    color_pos = '#1f8a4a'
    colors = []
    for k in kappas:
        if k < 0:   colors.append(color_neg)
        elif k > 0: colors.append(color_pos)
        else:       colors.append(color_pgsr)

    fig = plt.figure(figsize=(17, 7.0))
    fig.suptitle(
        r"Smooth transition through PGSR as $\kappa$ crosses zero  "
        rf"($\eta^u=\eta^v={eta_u}$, $\delta={delta}$).    "
        r"Concave cone $\leftrightarrow$ flat plane $\leftrightarrow$ "
        r"convex cone in one differentiable parameter.",
        fontsize=11,
    )
    for k, (kappa, color) in enumerate(zip(kappas, colors)):
        ax = fig.add_subplot(2, 5, k + 1, projection='3d')
        prim = make_tcs_primitive(eta_u=eta_u, eta_v=eta_v,
                                   delta=delta, kappa=kappa,
                                   scale=s, thin=0.12)
        plot_cone_zero_set(ax, prim, color=color, alpha=0.88)
        box_u = 2.5 * s
        ax.set_xlim(-box_u, box_u); ax.set_ylim(-box_u, box_u)
        _set_zlim_for_cone(ax, prim, box_u=box_u, pad_frac=0.08)
        ax.set_box_aspect((1, 1, 0.55))
        ax.view_init(elev=18, azim=-58)
        title = rf"$\kappa={kappa:+.1f}$"
        if kappa == 0:
            title += "    (= PGSR)"
        ax.set_title(title, fontsize=11, color=color)
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_xlabel('u', fontsize=7, labelpad=-7)
        ax.set_ylabel('v', fontsize=7, labelpad=-7)
        ax.set_zlabel('t', fontsize=7, labelpad=-7)

    u_line = np.linspace(-2.0 * s, 2.0 * s, 320)
    for k, (kappa, color) in enumerate(zip(kappas, colors)):
        ax = fig.add_subplot(2, 5, 5 + k + 1)
        A_line = (eta_u * u_line) ** 2 + delta ** 2
        t_line = kappa * (np.sqrt(A_line) - delta)
        ax.plot(u_line, t_line, color=color, linewidth=2.6,
                label=rf'TCS  $\kappa={kappa:+.1f}$')
        ax.axhline(0, color='black', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.axvspan(-1.5 * s, 1.5 * s, alpha=0.07, color='black')
        ax.set_xlabel(r'$u$  (with $v=0$)', fontsize=9)
        ax.set_ylabel(r'$t = h(u,v)$', fontsize=9)
        max_height = abs(eta_u * 2.0 * s)
        ax.set_ylim(-1.1 * max_height, 1.1 * max_height)
        ax.set_xlim(-2.0 * s, 2.0 * s)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8, loc='upper center', framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure B : delta sweep (cone <-> paraboloid)
# ---------------------------------------------------------------------------
def figure_tcs_delta_sweep(out_path):
    s = 0.7
    eta_u = eta_v = 0.85
    kappa = 1.0
    deltas = [0.01, 0.06, 0.18, 0.50, 1.20]
    colors = cm.viridis(np.linspace(0.15, 0.85, len(deltas)))[..., :3]

    fig = plt.figure(figsize=(16, 7.0))
    fig.suptitle(
        rf"Learnable sharpness $\delta$ at fixed $\kappa={kappa}$, "
        rf"$\eta^u=\eta^v={eta_u}$.    "
        r"Small $\delta$ $\to$ sharp cone.    Large $\delta$ $\to$ paraboloid.",
        fontsize=11,
    )
    for k, (delta, color) in enumerate(zip(deltas, colors)):
        ax = fig.add_subplot(2, 5, k + 1, projection='3d')
        prim = make_tcs_primitive(eta_u=eta_u, eta_v=eta_v,
                                   delta=delta, kappa=kappa,
                                   scale=s, thin=0.12)
        plot_cone_zero_set(ax, prim, color=color, alpha=0.88)
        box_u = 2.5 * s
        ax.set_xlim(-box_u, box_u); ax.set_ylim(-box_u, box_u)
        _set_zlim_for_cone(ax, prim, box_u=box_u, pad_frac=0.08)
        ax.set_box_aspect((1, 1, 0.55))
        ax.view_init(elev=18, azim=-58)
        ax.set_title(rf"$\delta={delta:.2f}$", fontsize=10)
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_xlabel('u', fontsize=7, labelpad=-7)
        ax.set_ylabel('v', fontsize=7, labelpad=-7)
        ax.set_zlabel('t', fontsize=7, labelpad=-7)

    u_line = np.linspace(-2.0 * s, 2.0 * s, 320)
    for k, (delta, color) in enumerate(zip(deltas, colors)):
        ax = fig.add_subplot(2, 5, 5 + k + 1)
        A_line = (eta_u * u_line) ** 2 + delta ** 2
        t_tcs   = kappa * (np.sqrt(A_line) - delta)
        t_cone  = kappa * (eta_u * np.abs(u_line))
        t_parab = kappa * (eta_u ** 2) * u_line ** 2 / (2 * delta)

        ax.plot(u_line, t_cone, color='#888888', linestyle=':',
                linewidth=1.2, label='sharp cone (limit)')
        ax.plot(u_line, t_parab, color='#8a4ec0', linestyle='--',
                linewidth=1.2, alpha=0.65, label='paraboloid (limit)')
        ax.plot(u_line, t_tcs, color=color, linewidth=2.4,
                label=rf'TCS  $\delta={delta:.2f}$')
        ax.set_xlabel(r'$u$  (with $v=0$)', fontsize=9)
        ax.set_ylabel(r'$t$', fontsize=9)
        ax.set_ylim(-0.10, max(eta_u * 2.0 * s * 1.05, 0.3))
        ax.set_xlim(-2.0 * s, 2.0 * s)
        ax.axvspan(-1.5 * s, 1.5 * s, alpha=0.07, color='black')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper center', framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure C : panorama (PGSR -> sharp -> smooth -> paraboloid -> concave)
# ---------------------------------------------------------------------------
def figure_tcs_zero_set_panorama(out_path):
    s = 0.7
    cases = [
        dict(eta_u=0.0, eta_v=0.0, delta=0.10, kappa=0.0,
             title=r"PGSR limit ($\kappa=0$)",
             color='#3a6db0'),
        dict(eta_u=0.85, eta_v=0.85, delta=0.02, kappa=1.0,
             title=r"Sharp cone (small $\delta$, $\kappa>0$)",
             color='#ca6d34'),
        dict(eta_u=0.85, eta_v=0.85, delta=0.18, kappa=1.0,
             title=r"Smooth cone (moderate $\delta$, $\kappa>0$)",
             color='#3aa860'),
        dict(eta_u=0.85, eta_v=0.85, delta=0.80, kappa=1.0,
             title=r"Paraboloid (large $\delta$, $\kappa>0$)",
             color='#8a4ec0'),
        dict(eta_u=0.85, eta_v=0.85, delta=0.18, kappa=-1.0,
             title=r"Concave cone ($\kappa<0$)",
             color='#c41e3a'),
    ]
    fig = plt.figure(figsize=(20, 4.6))
    fig.suptitle(
        r"TCS spans a single primitive family across known surface primitives."
        r"    $h(u,v) = \kappa\,(\sqrt{(\eta^u u)^2+(\eta^v v)^2+\delta^2}-\delta)$",
        fontsize=12,
    )
    for k, case in enumerate(cases):
        ax = fig.add_subplot(1, 5, k + 1, projection='3d')
        prim = make_tcs_primitive(
            eta_u=case['eta_u'], eta_v=case['eta_v'],
            delta=case['delta'], kappa=case['kappa'],
            scale=s, thin=0.12,
        )
        plot_cone_zero_set(ax, prim, color=case['color'], alpha=0.88)
        box_u = 2.5 * s
        ax.set_xlim(-box_u, box_u); ax.set_ylim(-box_u, box_u)
        _set_zlim_for_cone(ax, prim, box_u=box_u, pad_frac=0.08)
        ax.set_box_aspect((1, 1, 0.55))
        ax.view_init(elev=18, azim=-58)
        ax.set_title(case['title'], fontsize=10)
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_xlabel('u', fontsize=8, labelpad=-6)
        ax.set_ylabel('v', fontsize=8, labelpad=-6)
        ax.set_zlabel('t', fontsize=8, labelpad=-6)
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure D : nested density shells in three regimes
# ---------------------------------------------------------------------------
def figure_tcs_nested_shells(out_path):
    s = 0.7
    cases = [
        ('Sharp cone   $\\delta=0.03$,  $\\kappa=+1$',  0.03,  1.0, '#ca6d34'),
        ('Smooth cone  $\\delta=0.18$,  $\\kappa=+1$',  0.18,  1.0, '#3aa860'),
        ('Concave cone $\\delta=0.18$,  $\\kappa=-1$',  0.18, -1.0, '#c41e3a'),
    ]
    fig = plt.figure(figsize=(15.5, 5.4))
    fig.suptitle(
        r"Nested density shells $\rho/o\in\{0.85,0.5,0.15\}$ around the cone "
        r"surface  ($\eta^u=\eta^v=0.85$).    "
        r"Outer shells trace the underlying zero set.",
        fontsize=11,
    )
    for k, (title, delta, kappa, color) in enumerate(cases):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        prim = make_tcs_primitive(eta_u=0.85, eta_v=0.85,
                                   delta=delta, kappa=kappa,
                                   scale=s, thin=0.10)
        c_rgb = np.array([int(color[1:3], 16), int(color[3:5], 16),
                          int(color[5:7], 16)]) / 255.0
        plot_cone_nested_shells(ax, prim,
                                level_fracs=(0.85, 0.5, 0.15),
                                face_alphas=(0.95, 0.55, 0.22),
                                N=50, color_base=c_rgb)
        ax.set_title(title, fontsize=10.5)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure E : rendered depth surface
# ---------------------------------------------------------------------------
def figure_tcs_rendered_depth(out_path):
    res = 64
    focal = res * 1.0
    cam_pos = (0.0, 0.0, 2.0)
    ro, rd = make_camera(image_size=res, focal=focal, cam_pos=cam_pos)

    cases = [
        ('PGSR limit  ($\\kappa=0$)',
         dict(eta_u=0.7, eta_v=0.4, delta=0.15, kappa=0.0), '#3a6db0'),
        ('Convex cone  ($\\kappa=+1$, sharp)',
         dict(eta_u=0.7, eta_v=0.4, delta=0.03, kappa=1.0), '#ca6d34'),
        ('Convex paraboloid  ($\\kappa=+1$, smooth)',
         dict(eta_u=0.7, eta_v=0.4, delta=0.50, kappa=1.0), '#3aa860'),
        ('Concave cone  ($\\kappa=-1$)',
         dict(eta_u=0.7, eta_v=0.4, delta=0.18, kappa=-1.0), '#c41e3a'),
    ]
    depths = []
    for title, kw, color in cases:
        prim = make_tcs_primitive(scale=0.7, thin=0.15, **kw)
        tau_star, E_star, _ = ray_primitive_max(prim, ro, rd)
        hit = ro + tau_star[..., None] * rd
        depth = -hit[..., 2]
        a_map = prim.opacity * np.exp(-0.5 * E_star)
        z_med = np.nanmedian(np.where(a_map > 0.05, depth, np.nan))
        Z = np.where((a_map > 0.05) & (np.abs(depth - z_med) < 0.50),
                     depth, np.nan)
        depths.append(Z)

    z_max_global = max(np.nanmax(Z) for Z in depths)
    z_min_global = min(np.nanmin(Z) for Z in depths)
    z_pad = 0.05 * (z_max_global - z_min_global + 1e-6)
    z_range = (z_min_global - z_pad, z_max_global + z_pad)

    fig = plt.figure(figsize=(17, 5.4))
    fig.suptitle(
        r"Rendered depth $\tau^*(p_x, p_y)$ from a single TCS primitive,"
        r" shared z-axis."
        "\n"
        r"$\kappa=0$ is flat; $\kappa>0$ curls one way, $\kappa<0$ the other;"
        r"  $\delta$ controls cone-vs-paraboloid character.",
        fontsize=11,
    )
    px = np.arange(res) - res / 2.0
    py = np.arange(res) - res / 2.0
    PX, PY = np.meshgrid(px, py)
    for k, ((title, kw, color), Z) in enumerate(zip(cases, depths)):
        ax = fig.add_subplot(1, 4, k + 1, projection='3d')
        ax.plot_surface(PX, PY, Z, cmap='viridis', edgecolor='none',
                        rstride=1, cstride=1, antialiased=True, alpha=0.95,
                        vmin=z_range[0], vmax=z_range[1])
        ax.plot_wireframe(PX, PY, Z, rstride=8, cstride=8,
                          color='#222222', linewidth=0.35, alpha=0.55)
        ax.set_xlabel('pixel x', fontsize=8, labelpad=-6)
        ax.set_ylabel('pixel y', fontsize=8, labelpad=-6)
        ax.set_zlabel(r'depth $\tau^*$', fontsize=8, labelpad=-4)
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_zlim(*z_range)
        ax.view_init(elev=22, azim=-65)
        ax.set_box_aspect((1, 1, 0.65))
        ax.set_title(title, fontsize=10, color=color)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure F : composite three-panel summary
# ---------------------------------------------------------------------------
def figure_tcs_composite(out_path, eta_u=0.7, eta_v=0.35, delta=0.18,
                          kappa=1.0, scale=0.7, thin=0.15):
    prim = make_tcs_primitive(eta_u=eta_u, eta_v=eta_v,
                               delta=delta, kappa=kappa,
                               scale=scale, thin=thin)
    eta_bar = prim.eta_bar

    fig = plt.figure(figsize=(15.5, 4.8))
    fig.suptitle(
        rf"A TCS primitive    $\kappa={kappa:+.2f},\ "
        rf"\eta^u={eta_u:.2f},\ \eta^v={eta_v:.2f},\ \delta={delta:.2f},\ "
        rf"\bar\eta={eta_bar:.2f};\ s={scale:.2f},\ \sigma_n={scale*thin:.3f}$",
        fontsize=12,
    )
    ax_a = fig.add_subplot(1, 3, 1, projection='3d')
    plot_cone_zero_set(ax_a, prim, color='#3aa860')
    box_u = 2.5 * scale
    ax_a.set_xlim(-box_u, box_u); ax_a.set_ylim(-box_u, box_u)
    _set_zlim_for_cone(ax_a, prim, box_u=box_u, pad_frac=0.10)
    ax_a.set_box_aspect((1, 1, 0.55))
    ax_a.view_init(elev=20, azim=-58)
    ax_a.set_xlabel('u', fontsize=8, labelpad=-6)
    ax_a.set_ylabel('v', fontsize=8, labelpad=-6)
    ax_a.set_zlabel('t', fontsize=8, labelpad=-6)
    ax_a.tick_params(labelsize=6, pad=-2)
    ax_a.set_title("(a) Zero set  $t = h(u,v)$\nSingle smooth surface",
                   fontsize=10)

    ax_b = fig.add_subplot(1, 3, 2, projection='3d')
    plot_cone_nested_shells(ax_b, prim,
                            level_fracs=(0.85, 0.5, 0.15),
                            face_alphas=(0.95, 0.55, 0.22), N=50)
    ax_b.set_title(r"(b) Density shells $\rho/o\in\{0.85,0.5,0.15\}$"
                   "\nlayered around the zero set",
                   fontsize=10)

    ax_c = fig.add_subplot(1, 3, 3, projection='3d')
    res = 64
    ro, rd = make_camera(image_size=res, focal=res * 1.0,
                         cam_pos=(0.0, 0.0, 2.0))
    tau_star, E_star, _ = ray_primitive_max(prim, ro, rd)
    hit = ro + tau_star[..., None] * rd
    depth = -hit[..., 2]
    a_map = prim.opacity * np.exp(-0.5 * E_star)
    z_med = np.nanmedian(np.where(a_map > 0.05, depth, np.nan))
    Z = np.where((a_map > 0.05) & (np.abs(depth - z_med) < 0.4),
                 depth, np.nan)
    px = np.arange(res) - res / 2.0
    py = np.arange(res) - res / 2.0
    PX, PY = np.meshgrid(px, py)
    ax_c.plot_surface(PX, PY, Z, cmap='viridis', edgecolor='none',
                       alpha=0.95, rstride=1, cstride=1, antialiased=True)
    ax_c.plot_wireframe(PX, PY, Z, rstride=8, cstride=8,
                         color='#222222', linewidth=0.35, alpha=0.55)
    ax_c.set_xlabel('pixel x', fontsize=8, labelpad=-6)
    ax_c.set_ylabel('pixel y', fontsize=8, labelpad=-6)
    ax_c.set_zlabel(r'depth $\tau^*$', fontsize=8, labelpad=-4)
    ax_c.tick_params(labelsize=6, pad=-2)
    ax_c.view_init(elev=22, azim=-65)
    ax_c.set_box_aspect((1, 1, 0.55))
    ax_c.set_title(r"(c) Rendered depth $\tau^*(p_x,p_y)$"
                   "\nfrom this single primitive",
                   fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
def main():
    out_dir = "demo_outputs"
    os.makedirs(out_dir, exist_ok=True)

    kappa_sweep_path = os.path.join(out_dir, "tcs_kappa_sweep.png")
    figure_tcs_kappa_sweep(kappa_sweep_path)
    print(f"Saved: {kappa_sweep_path}")

    panorama_path = os.path.join(out_dir, "tcs_zero_set_panorama.png")
    figure_tcs_zero_set_panorama(panorama_path)
    print(f"Saved: {panorama_path}")

    delta_sweep_path = os.path.join(out_dir, "tcs_delta_sweep.png")
    figure_tcs_delta_sweep(delta_sweep_path)
    print(f"Saved: {delta_sweep_path}")

    shells_path = os.path.join(out_dir, "tcs_nested_shells.png")
    figure_tcs_nested_shells(shells_path)
    print(f"Saved: {shells_path}")

    depth_path = os.path.join(out_dir, "tcs_rendered_depth.png")
    figure_tcs_rendered_depth(depth_path)
    print(f"Saved: {depth_path}")

    composite_path = os.path.join(out_dir, "tcs_composite.png")
    figure_tcs_composite(composite_path,
                         eta_u=0.7, eta_v=0.35, delta=0.18, kappa=1.0)
    print(f"Saved: {composite_path}")


if __name__ == "__main__":
    main()