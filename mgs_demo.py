#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MGS (Manifold Gaussian Splatting) demo.

Each MGS primitive carries an anisotropic L^2 cone chart and is rendered as
a *surface-centered* density kernel: density peaks on the entire footprint
disk of the chart, not at a single apex point.

    h(u, v)        = kappa * (sqrt((eta^u u)^2 + (eta^v v)^2 + delta_0^2)
                              - delta_0)
    phi(u, v, t)   = t - h(u, v)
    W(u, v)        = exp(-c r / (1 - r))   for r < 1,  else 0       (C^inf bump)
                     where r = u^2/s_u^2 + v^2/s_v^2  and  c = 0.10
    rho(x)         = opacity * W(u, v) * exp(-phi^2 / (2 sigma_n^2))

The compact-bump window is the canonical C^infinity partition-of-unity
building block: exactly compact support on the disk r < 1, all derivatives
smoothly matched to zero at the boundary, and a near-flat plateau through
most of the interior (at c = 0.10, density on the chart M is >= 0.50 * o on
roughly 87% of the disk, against 29% for the polynomial cap (1-r)^2).
The kernel value on the chart is W(u, v), not a Gaussian falloff from the
apex, so the high-density region is genuinely a 2-manifold patch.

Per-primitive parameters:
    mu, R                : center and local frame
    s_u, s_v             : in-plane footprint scales
    eta_u, eta_v >= 0    : anisotropic slope magnitudes
    kappa  in R          : signed curvature scale  (= 0  ->  PGSR-equivalent)
    sigma_n > 0          : normal-direction thickness   (PER-PRIMITIVE LEARNABLE,
                           analog of 3DGS's third scale s_3; initialised to
                           0.10 * sqrt(s_u * s_v) and regularised small)
    opacity              : scalar opacity in [0, 1]
    color, SH            : appearance

The compact-bump flatness c and the apex smoother delta_0 are GLOBAL fixed
constants (c = 0.10; delta_0 = 1e-3 * min(s_u, s_v)).  The thickness
sigma_n is per-primitive and learnable; gradient signal flows into
sigma_n through the line-integral alpha below.

Rendering uses the ray-chart intersection { phi(tau) = 0 }, the
maximum-density point inside the footprint.  This reduces to a CLOSED-FORM
QUADRATIC in tau -- a substantial simplification over quadric primitives'
quartic equation.  The per-pixel contribution is the line integral of the
kernel through the ray, evaluated in closed form:

    alpha = 1 - exp(- opacity * W(u*, v*)
                      * sqrt(2 pi) * sigma_n / |<grad phi, d_loc>|)

where d_loc is the ray direction in the primitive's local frame and
<grad phi, d_loc> is the cosine between the ray and the chart normal.
Both terms are computed from quantities the ray-chart solver already
produces.  sigma_n appears multiplicatively in alpha, so the photometric
loss can adapt per-primitive thickness directly.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


# ===========================================================================
# Global constants
# ===========================================================================
DELTA_0_FRAC     = 1e-3      # delta_0 = DELTA_0_FRAC * min(s_u, s_v)
SIGMA_N_INIT_FRAC = 0.10     # sigma_n initialisation: 0.10 * sqrt(s_u * s_v)
C_FLATNESS        = 0.10     # compact-bump flatness parameter (global, fixed)
GRAZING_EPS       = 1e-2     # |<grad phi, d_loc>| clamp for grazing rays


# ===========================================================================
# Camera
# ===========================================================================
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


# ===========================================================================
# Window function -- compact-bump
# ===========================================================================
def window_radial(r, c=None):
    """Compact-bump window W as a function of the squared radial coord r.

        W(r) = exp(-c r / (1 - r))   for r < 1,   else 0.

    C^infinity across the boundary r = 1.  All derivatives go to zero
    smoothly (because the exponential decay dominates any polynomial blow-up
    of 1/(1-r)^k as r -> 1^-).
    """
    if c is None:
        c = C_FLATNESS
    inside = r < 1.0
    safe = np.where(inside, r, 0.0)
    W_inside = np.exp(-c * safe / np.maximum(1.0 - safe, 1e-12))
    return np.where(inside, W_inside, 0.0)


# ===========================================================================
# Primitive
# ===========================================================================
class MGSPrimitive:
    """Manifold-Gaussian-Splatting primitive.

    Density (with the line-integral alpha computed at render time):
        rho(x) = opacity * W(u, v) * exp(-phi(u,v,t)^2 / (2 sigma_n^2))

    where the chart is an anisotropic L^2 cone:
        h(u, v) = kappa * (sqrt((eta_u u)^2 + (eta_v v)^2 + delta_0^2)
                           - delta_0)
        phi     = t - h(u, v)

    and the footprint window is the compact-bump function:
        W(u, v) = exp(-c r / (1 - r))     for r = u^2/s_u^2 + v^2/s_v^2 < 1,
                  0                       otherwise.

    sigma_n is a per-primitive learnable parameter (in the optimisation
    code it is parameterised through softplus); we initialise it to
    0.10 * sqrt(s_u * s_v).
    """

    def __init__(self, mu, R, s_u, s_v, eta_u, eta_v, kappa,
                 opacity, color,
                 sigma_n=None, delta_0=None):
        self.mu      = np.asarray(mu, dtype=np.float64)
        self.R       = np.asarray(R, dtype=np.float64)
        self.s_u     = float(s_u)
        self.s_v     = float(s_v)
        self.eta_u   = float(eta_u)
        self.eta_v   = float(eta_v)
        self.kappa   = float(kappa)
        self.opacity = float(opacity)
        self.color   = np.asarray(color, dtype=np.float64)
        s_min  = min(self.s_u, self.s_v)
        s_geom = np.sqrt(self.s_u * self.s_v)
        self.delta_0 = float(delta_0 if delta_0 is not None
                              else DELTA_0_FRAC * s_min)
        # Per-primitive learnable thickness (here just an initial value).
        self.sigma_n = float(sigma_n if sigma_n is not None
                              else SIGMA_N_INIT_FRAC * s_geom)

    @property
    def eta_bar(self):
        """Dimensionless cone rise: |kappa| * sqrt((eta_u s_u)^2 + (eta_v s_v)^2).
        Bounds the chart's vertical excursion across the footprint."""
        return abs(self.kappa) * np.sqrt((self.eta_u * self.s_u) ** 2
                                          + (self.eta_v * self.s_v) ** 2)

    @property
    def sigma_bar(self):
        """Dimensionless thickness ratio sigma_n / sqrt(s_u s_v)."""
        return self.sigma_n / np.sqrt(self.s_u * self.s_v)


def world_to_local(prim, x):
    return (x - prim.mu) @ prim.R


def h_local(prim, uvt):
    """Chart height field h(u, v) at the local coordinates."""
    u, v = uvt[..., 0], uvt[..., 1]
    A = (prim.eta_u * u) ** 2 + (prim.eta_v * v) ** 2 + prim.delta_0 ** 2
    return prim.kappa * (np.sqrt(A) - prim.delta_0)


def phi_local(prim, uvt):
    return uvt[..., 2] - h_local(prim, uvt)


def footprint_window(prim, uvt):
    """W(u, v) = exp(-c r / (1 - r))   for r < 1,  else 0.

    Compactly supported on the elliptical disk
        r(u, v) = u^2/s_u^2 + v^2/s_v^2 < 1
    and C^infinity at the support boundary.
    """
    u, v = uvt[..., 0], uvt[..., 1]
    r = (u * u) / (prim.s_u ** 2) + (v * v) / (prim.s_v ** 2)
    return window_radial(r, c=C_FLATNESS)


def density_local(prim, uvt):
    """rho/opacity in local coordinates.

    Returns W * exp(-phi^2 / 2 sigma_n^2).  Equals W(u, v) on the chart
    (phi = 0) inside the footprint, and drops to zero outside.  The window
    W is near-constant over most of the disk for our default c = 0.10, so
    the high-density region is a 2-manifold patch rather than a point.
    """
    W = footprint_window(prim, uvt)
    phi = phi_local(prim, uvt)
    return W * np.exp(-0.5 * phi ** 2 / prim.sigma_n ** 2)


def density_world(prim, x):
    return prim.opacity * density_local(prim, world_to_local(prim, x))


# ===========================================================================
# Closed-form quadratic ray-chart intersection
# ===========================================================================
def ray_chart_intersection(prim, o_world, d_world):
    """Solve phi(o + tau d) = 0 for the ray-chart intersection.

    Squaring `(t + kappa*delta_0) = kappa * sqrt(A)` gives a quadratic in
    tau:
        a tau^2 + b tau + c = 0,  with
        a = d_t^2  -  kappa^2 (eta_u^2 d_u^2 + eta_v^2 d_v^2)
        b = 2 (t_0 + kappa delta_0) d_t
            -  2 kappa^2 (eta_u^2 u_0 d_u + eta_v^2 v_0 d_v)
        c = (t_0 + kappa delta_0)^2  -  kappa^2 A(0)

    Returns:
        tau_hit  : (...,)        first valid root (NaN if no hit)
        valid    : (...,) bool   True iff a real, sign-consistent,
                                 footprint-interior root exists
    """
    shape = o_world.shape[:-1]
    o = o_world.reshape(-1, 3); d = d_world.reshape(-1, 3)
    o_loc = (o - prim.mu) @ prim.R         # (N, 3)
    d_loc = d @ prim.R                       # (N, 3)
    u0, v0, t0 = o_loc[:, 0], o_loc[:, 1], o_loc[:, 2]
    du, dv, dt = d_loc[:, 0], d_loc[:, 1], d_loc[:, 2]

    k  = prim.kappa
    eu = prim.eta_u; ev = prim.eta_v
    d0 = prim.delta_0

    A0 = (eu * u0) ** 2 + (ev * v0) ** 2 + d0 ** 2     # A at tau = 0

    a = dt ** 2 - k ** 2 * (eu ** 2 * du ** 2 + ev ** 2 * dv ** 2)
    b = 2 * (t0 + k * d0) * dt \
        - 2 * k ** 2 * (eu ** 2 * u0 * du + ev ** 2 * v0 * dv)
    c = (t0 + k * d0) ** 2 - k ** 2 * A0

    # Avoid divide-by-zero when kappa = 0 (a = dt^2, b = 2 t_0 d_t, c = t_0^2).
    # Then the quadratic is a perfect square and degenerates to tau = -t_0/d_t.
    near_zero = np.abs(a) < 1e-12
    a_safe = np.where(near_zero, 1.0, a)

    # Both roots
    disc = b ** 2 - 4 * a_safe * c
    has_real = disc >= 0
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    tau1 = (-b - sqrt_disc) / (2 * a_safe)
    tau2 = (-b + sqrt_disc) / (2 * a_safe)

    # Handle the a -> 0 degenerate case: linear equation b tau + c = 0.
    tau_linear = np.where(np.abs(b) > 1e-12, -c / np.where(b == 0, 1.0, b),
                          np.nan)
    tau1 = np.where(near_zero, tau_linear, tau1)
    tau2 = np.where(near_zero, tau_linear, tau2)
    has_real = np.where(near_zero, ~np.isnan(tau_linear), has_real)

    # Sign filter:  squaring may introduce extraneous roots.  The original
    # equation requires  sign(t(tau) + kappa delta_0) == sign(kappa).
    # If kappa == 0, both sides are zero on the plane t = 0, and any sign is
    # fine; we accept any root.
    def sign_ok(tau):
        if np.isclose(k, 0.0):
            return np.ones_like(tau, dtype=bool)
        lhs = (t0 + tau * dt) + k * d0
        return np.sign(lhs) == np.sign(k)

    # Footprint filter:  ray must hit inside the footprint disk W > 0.
    def in_footprint(tau):
        u = u0 + tau * du; v = v0 + tau * dv
        return (u ** 2 / prim.s_u ** 2 + v ** 2 / prim.s_v ** 2) < 1.0

    v1 = has_real & sign_ok(tau1) & in_footprint(tau1)
    v2 = has_real & sign_ok(tau2) & in_footprint(tau2)

    # Pick the front-face root: smaller tau among valid ones.
    tau_hit = np.full_like(tau1, np.nan)
    valid   = v1 | v2
    both = v1 & v2
    tau_hit[both] = np.minimum(tau1[both], tau2[both])
    only1 = v1 & ~v2
    only2 = v2 & ~v1
    tau_hit[only1] = tau1[only1]
    tau_hit[only2] = tau2[only2]

    return tau_hit.reshape(shape), valid.reshape(shape)


def grad_phi_local(prim, uvt):
    """Gradient of phi in local coordinates at the hit point:

        grad phi = ( -kappa eta_u^2 u / sqrt(A),
                     -kappa eta_v^2 v / sqrt(A),
                      1 )

    Returns an array of shape uvt.shape with the (u, v, t) components in
    the last axis.
    """
    u, v = uvt[..., 0], uvt[..., 1]
    A = (prim.eta_u * u) ** 2 + (prim.eta_v * v) ** 2 + prim.delta_0 ** 2
    sqrtA = np.sqrt(A)
    g_u = -prim.kappa * (prim.eta_u ** 2) * u / sqrtA
    g_v = -prim.kappa * (prim.eta_v ** 2) * v / sqrtA
    g_t = np.ones_like(u)
    return np.stack([g_u, g_v, g_t], axis=-1)


def ray_primitive_alpha(prim, o_world, d_world):
    """Closed-form ray-chart hit + line-integral alpha.

    alpha = 1 - exp( - opacity * W(u*, v*)
                       * sqrt(2 pi) * sigma_n / |<grad phi, d_loc>| ).

    Returns:
        tau_hit   : (...,)        hit distance (np.inf where invalid)
        alpha     : (...,)        line-integral alpha in [0, 1)
        hit_local : (..., 3)      hit point in primitive's local frame
    """
    tau_hit, valid = ray_chart_intersection(prim, o_world, d_world)
    tau_safe = np.where(valid, tau_hit, np.inf)

    hit_world = o_world + tau_safe[..., None] * d_world
    hit_local = (hit_world - prim.mu) @ prim.R

    # In-plane window value
    W_star = footprint_window(prim, hit_local)
    W_star = np.where(valid, W_star, 0.0)

    # Ray direction in local coords  (same shape as d_world)
    d_loc = d_world @ prim.R

    # Cosine factor  |<grad phi, d_loc>|
    g_loc = grad_phi_local(prim, hit_local)
    cos_factor = np.abs((g_loc * d_loc).sum(axis=-1))
    cos_factor = np.maximum(cos_factor, GRAZING_EPS)

    # Line-integral alpha
    integrand = (prim.opacity * W_star
                  * np.sqrt(2.0 * np.pi) * prim.sigma_n / cos_factor)
    alpha = 1.0 - np.exp(-integrand)
    alpha = np.where(valid, alpha, 0.0)

    return tau_safe, alpha, hit_local


# ===========================================================================
# Plot helpers
# ===========================================================================
def _set_zlim_for_cone(ax, prim, box_u=None, pad_frac=0.10):
    if box_u is None:
        box_u = 1.05 * prim.s_u
    cone_h_pos = abs(prim.kappa) * (
        np.sqrt((prim.eta_u * box_u) ** 2 + (prim.eta_v * box_u) ** 2
                + prim.delta_0 ** 2) - prim.delta_0)
    cone_h_pos = max(cone_h_pos, 0.20)
    if prim.kappa >= 0:
        ax.set_zlim(-pad_frac * cone_h_pos, (1 + pad_frac) * cone_h_pos)
    else:
        ax.set_zlim(-(1 + pad_frac) * cone_h_pos, pad_frac * cone_h_pos)


def plot_chart_with_window(ax, prim, n=100, alpha_max=0.95,
                            color_max=(0.18, 0.62, 0.42), show_boundary=True):
    """Draw the chart surface t = h(u, v) inside the footprint, with
    transparency proportional to the compact-bump window W.  Visualises
    'density peaks on the patch, weighted by W'."""
    u = np.linspace(-prim.s_u, prim.s_u, n)
    v = np.linspace(-prim.s_v, prim.s_v, n)
    U, V = np.meshgrid(u, v)
    r = U ** 2 / prim.s_u ** 2 + V ** 2 / prim.s_v ** 2
    in_disk = r < 1.0
    A = (prim.eta_u * U) ** 2 + (prim.eta_v * V) ** 2 + prim.delta_0 ** 2
    T = prim.kappa * (np.sqrt(A) - prim.delta_0)
    W = window_radial(r)

    facecolors = np.zeros((n - 1, n - 1, 4))
    Wmid = 0.25 * (W[:-1, :-1] + W[1:, :-1] + W[:-1, 1:] + W[1:, 1:])
    in_mid = 0.25 * (in_disk[:-1, :-1].astype(float)
                     + in_disk[1:, :-1] + in_disk[:-1, 1:] + in_disk[1:, 1:])
    facecolors[..., 0] = color_max[0]
    facecolors[..., 1] = color_max[1]
    facecolors[..., 2] = color_max[2]
    facecolors[..., 3] = alpha_max * Wmid * (in_mid > 0.99)

    Tplot = np.where(in_disk, T, np.nan)
    ax.plot_surface(U, V, Tplot, facecolors=facecolors,
                    rstride=1, cstride=1, linewidth=0, antialiased=True,
                    shade=False)

    if show_boundary:
        theta = np.linspace(0, 2 * np.pi, 120)
        bu = prim.s_u * np.cos(theta)
        bv = prim.s_v * np.sin(theta)
        bA = (prim.eta_u * bu) ** 2 + (prim.eta_v * bv) ** 2 + prim.delta_0 ** 2
        bt = prim.kappa * (np.sqrt(bA) - prim.delta_0)
        ax.plot(bu, bv, bt, color='black', linewidth=1.3, zorder=18,
                linestyle='--', alpha=0.7)

    ax.scatter([0], [0], [0], color='#1f6035', s=22, zorder=20)


def plot_density_shells(ax, prim, level_fracs=(0.85, 0.5, 0.15),
                         face_alphas=(0.95, 0.55, 0.22),
                         N=56, color_base=(0.18, 0.62, 0.42)):
    """Marching-cubes isosurfaces of rho/opacity at several levels."""
    box_u = 1.05 * prim.s_u
    box_v = 1.05 * prim.s_v
    cone_h_pos = abs(prim.kappa) * (
        np.sqrt((prim.eta_u * box_u) ** 2 + (prim.eta_v * box_v) ** 2
                + prim.delta_0 ** 2) - prim.delta_0)
    t_extent = max(1.2 * cone_h_pos + 3 * prim.sigma_n, 0.3)
    if prim.kappa >= 0:
        t_min, t_max = -3 * prim.sigma_n, t_extent
    else:
        t_min, t_max = -t_extent, 3 * prim.sigma_n

    u = np.linspace(-box_u, box_u, N)
    v = np.linspace(-box_v, box_v, N)
    t = np.linspace(t_min, t_max, N)
    U, V, T = np.meshgrid(u, v, t, indexing='ij')
    uvt = np.stack([U, V, T], axis=-1)
    F = density_local(prim, uvt)

    spacing = np.array([2 * box_u / (N - 1), 2 * box_v / (N - 1),
                        (t_max - t_min) / (N - 1)])
    origin  = np.array([-box_u, -box_v, t_min])

    for level, falpha in sorted(zip(level_fracs, face_alphas),
                                 key=lambda x: x[0]):
        try:
            verts, faces, _, _ = marching_cubes(F, level=level)
        except (ValueError, RuntimeError):
            continue
        verts = verts * spacing + origin
        mesh = Poly3DCollection(verts[faces], alpha=falpha,
                                linewidth=0.0, antialiased=True)
        tri = verts[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nl = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
        normals /= nl
        shade = 0.55 + 0.45 * normals[:, 2]
        rgba = np.concatenate([
            np.clip(np.array(color_base)[None, :] * shade[:, None], 0, 1),
            np.full((len(shade), 1), falpha),
        ], axis=1)
        mesh.set_facecolor(rgba)
        ax.add_collection3d(mesh)

    ax.set_xlim(-box_u, box_u); ax.set_ylim(-box_v, box_v)
    ax.set_zlim(t_min, t_max)


def style_3d_axes(ax, view=(20, -58), xl='u', yl='v', zl='t'):
    ax.set_xlabel(xl, fontsize=8, labelpad=-6)
    ax.set_ylabel(yl, fontsize=8, labelpad=-6)
    ax.set_zlabel(zl, fontsize=8, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_box_aspect((1, 1, 0.62))


# ===========================================================================
# Convenience constructor
# ===========================================================================
def make_mgs_primitive(eta_u=1.0, eta_v=1.0, kappa=1.0, scale=0.7,
                       sigma_n=None, color=(0.18, 0.62, 0.42)):
    return MGSPrimitive(
        mu=np.zeros(3), R=np.eye(3),
        s_u=scale, s_v=scale,
        eta_u=eta_u, eta_v=eta_v, kappa=kappa,
        opacity=0.95, color=np.array(color),
        sigma_n=sigma_n,
    )


# ===========================================================================
# Figure A : kappa sweep through PGSR
# ===========================================================================
def figure_kappa_sweep(out_path):
    s = 0.7
    eta_u = eta_v = 0.85
    kappas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    color_neg = '#c41e3a'; color_pos = '#3aa860'; color_pgsr = '#1f5fb4'

    fig = plt.figure(figsize=(17, 7.0))
    fig.suptitle(
        r"Smooth transition through PGSR as $\kappa$ crosses zero."
        rf"  ($\eta^u=\eta^v={eta_u}$, compact-bump $c={C_FLATNESS}$).    "
        r"Concave cone $\leftrightarrow$ flat plane $\leftrightarrow$ convex cone "
        r"in a single differentiable parameter.",
        fontsize=11,
    )

    for k, kappa in enumerate(kappas):
        color = color_neg if kappa < 0 else color_pos if kappa > 0 else color_pgsr
        ax = fig.add_subplot(2, 5, k + 1, projection='3d')
        prim = make_mgs_primitive(eta_u=eta_u, eta_v=eta_v,
                                   kappa=kappa, scale=s,
                                   color=tuple(int(color[i:i+2], 16)/255
                                               for i in (1, 3, 5)))
        plot_chart_with_window(ax, prim,
                                color_max=tuple(int(color[i:i+2], 16)/255
                                                for i in (1, 3, 5)))
        box_u = 1.1 * s
        ax.set_xlim(-box_u, box_u); ax.set_ylim(-box_u, box_u)
        _set_zlim_for_cone(ax, prim, box_u=box_u, pad_frac=0.08)
        style_3d_axes(ax)
        title = rf"$\kappa={kappa:+.1f}$"
        if kappa == 0:
            title += "    (= PGSR)"
        ax.set_title(title, fontsize=11, color=color)

    # Bottom row: v=0 slice, showing the chart's profile + the compact-bump W
    u_line = np.linspace(-1.05 * s, 1.05 * s, 320)
    delta_0 = DELTA_0_FRAC * s
    r_line = u_line ** 2 / s ** 2
    W_line = window_radial(r_line, c=C_FLATNESS)

    for k, kappa in enumerate(kappas):
        color = color_neg if kappa < 0 else color_pos if kappa > 0 else color_pgsr
        ax = fig.add_subplot(2, 5, 5 + k + 1)
        A_line = (eta_u * u_line) ** 2 + delta_0 ** 2
        t_line = kappa * (np.sqrt(A_line) - delta_0)

        ax2 = ax.twinx()
        ax2.fill_between(u_line, 0, W_line, color='gray', alpha=0.16)
        ax2.plot(u_line, W_line, color='gray', linewidth=1.3,
                 linestyle=':', alpha=0.65, label=r'$W(u,0)$')
        ax2.set_ylabel(r'$W$', fontsize=8, color='gray')
        ax2.tick_params(axis='y', labelsize=7, colors='gray')
        ax2.set_ylim(-0.05, 1.15)

        ax.plot(u_line, t_line, color=color, linewidth=2.6,
                label=rf'chart  $\kappa={kappa:+.1f}$')
        ax.axhline(0, color='black', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.set_xlabel(r'$u$  ($v=0$)', fontsize=9)
        ax.set_ylabel(r'$t = h(u, 0)$', fontsize=9, color=color)
        max_h = max(abs(eta_u * s), 0.3)
        ax.set_ylim(-1.1 * max_h, 1.1 * max_h)
        ax.set_xlim(-1.05 * s, 1.05 * s)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7, axis='y', colors=color)
        ax.tick_params(labelsize=7, axis='x')
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure B : 3DGS-vs-MGS structural comparison
# ===========================================================================
def figure_pointcentered_vs_surfacecentered(out_path):
    """Side-by-side: 3DGS point-centred density vs MGS surface-centred density.

    Top row -- 3D rho/opacity isosurfaces at four levels for both primitives.
    Bottom row -- (u, t) slices showing the density landscape.
    """
    s = 0.7
    sigma_n_viz = 0.10 * s

    fig = plt.figure(figsize=(15, 8.5))
    fig.suptitle(
        r"3DGS  (density peaks at the point $\mu_i$)"
        r"   vs.   "
        r"MGS  (density peaks on the entire footprint disk of the chart)."
        "\n"
        r"Same in-plane footprint scales $s_u = s_v$; same normal-direction "
        r"thickness $\sigma_n$.  MGS additionally curves its chart via $\kappa, \eta$ "
        rf"and uses the compact-bump window ($c={C_FLATNESS}$).",
        fontsize=11,
    )

    # 3D grid
    box_u = 1.05 * s; box_v = 1.05 * s; box_t = 4 * sigma_n_viz
    N = 64
    uu = np.linspace(-box_u, box_u, N)
    vv = np.linspace(-box_v, box_v, N)
    tt = np.linspace(-box_t, box_t, N)
    U, V, T = np.meshgrid(uu, vv, tt, indexing='ij')
    uvt = np.stack([U, V, T], axis=-1)

    # 3DGS: anisotropic Gaussian with thickness sigma_n along the third axis
    F_3dgs = np.exp(-0.5 * (U ** 2 / s ** 2 + V ** 2 / s ** 2
                              + T ** 2 / sigma_n_viz ** 2))
    # MGS primitive
    prim_mgs = MGSPrimitive(
        mu=np.zeros(3), R=np.eye(3),
        s_u=s, s_v=s, eta_u=0.85, eta_v=0.85, kappa=1.0,
        opacity=1.0, color=np.array([0.18, 0.62, 0.42]),
        sigma_n=sigma_n_viz,
    )
    F_mgs  = density_local(prim_mgs, uvt)

    spacing = np.array([2 * box_u / (N - 1), 2 * box_v / (N - 1),
                        2 * box_t / (N - 1)])
    origin = np.array([-box_u, -box_v, -box_t])

    for col, (F, name, color_base) in enumerate([
        (F_3dgs, '3DGS\n(point-centred)', '#3a6db0'),
        (F_mgs,  'MGS\n(surface-centred)',  '#3aa860'),
    ]):
        ax = fig.add_subplot(2, 2, col + 1, projection='3d')
        rgb = np.array([int(color_base[i:i+2], 16)/255 for i in (1, 3, 5)])
        for level, falpha in [(0.85, 0.92), (0.5, 0.42), (0.15, 0.16)]:
            try:
                verts, faces, _, _ = marching_cubes(F, level=level)
            except (ValueError, RuntimeError):
                continue
            verts = verts * spacing + origin
            mesh = Poly3DCollection(verts[faces], alpha=falpha,
                                    linewidth=0.0, antialiased=True)
            tri = verts[faces]
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nl = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
            normals /= nl
            shade = 0.55 + 0.45 * normals[:, 2]
            rgba = np.concatenate([
                np.clip(rgb[None, :] * shade[:, None], 0, 1),
                np.full((len(shade), 1), falpha),
            ], axis=1)
            mesh.set_facecolor(rgba)
            ax.add_collection3d(mesh)

        if col == 0:
            ax.scatter([0], [0], [0], color='#101a30', s=40, zorder=20)
            ax.text(0, 0, 0, r'  $\mu_i$', fontsize=10, color='#101a30',
                    zorder=21)
        else:
            theta = np.linspace(0, 2 * np.pi, 120)
            bu = s * np.cos(theta); bv = s * np.sin(theta)
            bA = (prim_mgs.eta_u * bu) ** 2 + (prim_mgs.eta_v * bv) ** 2 \
                 + prim_mgs.delta_0 ** 2
            bt = prim_mgs.kappa * (np.sqrt(bA) - prim_mgs.delta_0)
            ax.plot(bu, bv, bt, color='#101a30', linewidth=1.6,
                    linestyle='-', zorder=20, alpha=0.85,
                    label=r'peak locus  $\{W>0,\,\phi=0\}$')
            ax.legend(fontsize=8, loc='upper left')

        ax.set_xlim(-box_u, box_u); ax.set_ylim(-box_v, box_v)
        ax.set_zlim(-box_t, box_t)
        style_3d_axes(ax)
        ax.set_title(name, fontsize=11)

    # (u, t) slice of the density
    Ns = 240
    us = np.linspace(-1.1 * s, 1.1 * s, Ns)
    ts = np.linspace(-4 * sigma_n_viz, 4 * sigma_n_viz, Ns)
    Us, Ts = np.meshgrid(us, ts)
    Vs = np.zeros_like(Us)
    uvt_slice = np.stack([Us, Vs, Ts], axis=-1)

    F3 = np.exp(-0.5 * (Us ** 2 / s ** 2 + Ts ** 2 / sigma_n_viz ** 2))
    Fm = density_local(prim_mgs, uvt_slice)

    for col, (F, name, cmap) in enumerate([
        (F3, '3DGS  (slice $v=0$)', 'Blues'),
        (Fm, 'MGS  (slice $v=0$)', 'Greens'),
    ]):
        ax = fig.add_subplot(2, 2, 2 + col + 1)
        im = ax.pcolormesh(Us, Ts, F, cmap=cmap, vmin=0, vmax=1,
                           shading='auto')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(r'$\rho/o$', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        for u_bd in (-s, s):
            ax.axvline(u_bd, color='black', linestyle='--', linewidth=0.8,
                       alpha=0.5)
        if col == 1:
            u_line = np.linspace(-s, s, 200)
            A_line = (prim_mgs.eta_u * u_line) ** 2 + prim_mgs.delta_0 ** 2
            t_chart = prim_mgs.kappa * (np.sqrt(A_line) - prim_mgs.delta_0)
            ax.plot(u_line, t_chart, color='black', linewidth=1.6,
                    label='chart $\\{\\phi=0\\}$')
            ax.legend(fontsize=8, loc='upper right')
        else:
            ax.scatter([0], [0], color='black', s=40, zorder=10)
            ax.annotate(r'$\mu_i$', (0, 0), xytext=(0.07, 0.07),
                        fontsize=10)
        ax.set_xlabel(r'$u$', fontsize=9)
        ax.set_ylabel(r'$t$', fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_aspect('auto')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure C : footprint window W(u, v) -- compact-bump, with c sweep
# ===========================================================================
def figure_footprint_window(out_path):
    s_u = 0.7; s_v = 0.5
    n = 200
    u = np.linspace(-1.4 * s_u, 1.4 * s_u, n)
    v = np.linspace(-1.4 * s_v, 1.4 * s_v, n)
    U, V = np.meshgrid(u, v)
    r2d = U ** 2 / s_u ** 2 + V ** 2 / s_v ** 2
    W2d = window_radial(r2d, c=C_FLATNESS)

    fig = plt.figure(figsize=(15, 4.6))
    fig.suptitle(
        r"Footprint window  $W(u,v) = \exp(-c\, r/(1-r))$  for $r<1$,"
        r" else $0$  ($r=u^2/s_u^2+v^2/s_v^2$). "
        r"Compactly supported, $C^\infty$ at the boundary.    "
        rf"($s_u={s_u},\ s_v={s_v}$;  default $c={C_FLATNESS}$.)",
        fontsize=11,
    )

    # 2D heatmap with boundary
    ax = fig.add_subplot(1, 3, 1)
    im = ax.pcolormesh(U, V, W2d, cmap='Greens', vmin=0, vmax=1, shading='auto')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('W', fontsize=8); cb.ax.tick_params(labelsize=7)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(s_u * np.cos(theta), s_v * np.sin(theta),
            color='black', linewidth=1.4, linestyle='--')
    ax.set_xlabel('u'); ax.set_ylabel('v'); ax.set_aspect('equal')
    ax.set_title(rf'(a) $W(u,v)$, $c={C_FLATNESS}$  (top-down)')
    ax.tick_params(labelsize=8)

    # 3D surface
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot_surface(U, V, W2d, cmap='Greens', linewidth=0,
                    edgecolor='none', antialiased=True, alpha=0.95,
                    vmin=0, vmax=1)
    ax.set_xlabel('u', fontsize=8, labelpad=-6)
    ax.set_ylabel('v', fontsize=8, labelpad=-6)
    ax.set_zlabel('W', fontsize=8, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2)
    ax.set_title(rf'(b) $W(u,v)$, $c={C_FLATNESS}$  (3D surface)', fontsize=10)
    ax.view_init(elev=22, azim=-65)

    # (c) Radial profile: compact-bump c sweep + polynomial cap baseline
    ax = fig.add_subplot(1, 3, 3)
    r_line = np.linspace(0, 1.15, 600)
    c_values  = [0.02, 0.05, 0.10, 0.30, 1.00]
    c_colors  = ['#7d4a00', '#b56c00', '#3aa860', '#1f5fb4', '#a31f3a']
    for c_val, col in zip(c_values, c_colors):
        W_c = window_radial(r_line, c=c_val)
        lw = 2.6 if c_val == C_FLATNESS else 1.4
        ls = '-' if c_val == C_FLATNESS else (
                '--' if c_val > C_FLATNESS else ':')
        label = rf'bump  $c={c_val}$'
        if c_val == C_FLATNESS:
            label += "  (ours)"
        ax.plot(r_line, W_c, color=col, linewidth=lw, linestyle=ls,
                label=label)
    # Polynomial cap baseline
    W_poly = np.maximum(0.0, 1 - r_line) ** 2
    ax.plot(r_line, W_poly, color='#888888', linewidth=1.4, linestyle='-.',
            label=r'poly. cap $(1-r)_+^{2}$')
    ax.axvline(1.0, color='black', linewidth=0.8, alpha=0.5, linestyle='--')
    ax.set_xlabel(r'$r = u^2/s_u^2 + v^2/s_v^2$', fontsize=9)
    ax.set_ylabel('window value', fontsize=9)
    ax.set_title(r'(c) Radial profile  $W(r)$'
                 "\n"
                 r'smaller $c$ $\Rightarrow$ flatter plateau, sharper boundary',
                 fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.15); ax.set_ylim(-0.05, 1.1)
    ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure D : nested density shells across regimes
# ===========================================================================
def figure_nested_shells(out_path):
    s = 0.7
    cases = [
        ('PGSR limit\n$\\kappa=0$, $\\eta=0.85$',  0.85, 0.0,  '#3a6db0'),
        ('Convex cone\n$\\kappa=+1$, $\\eta=0.85$',  0.85, 1.0,  '#3aa860'),
        ('Concave cone\n$\\kappa=-1$, $\\eta=0.85$', 0.85, -1.0, '#c41e3a'),
        ('Anisotropic convex\n$\\kappa=+1$, '
         '$\\eta^u=1.0,\\eta^v=0.4$', None, 1.0, '#ca6d34'),
    ]
    fig = plt.figure(figsize=(17, 4.8))
    fig.suptitle(
        r"Nested density shells $\rho/o \in \{0.85,\,0.5,\,0.15\}$ around the "
        r"chart, with the compact-bump window $W$ shaping the in-plane extent."
        rf"  ($c={C_FLATNESS}$.)",
        fontsize=11,
    )
    for k, (title, eta, kappa, color) in enumerate(cases):
        ax = fig.add_subplot(1, 4, k + 1, projection='3d')
        if eta is None:
            prim = MGSPrimitive(
                mu=np.zeros(3), R=np.eye(3),
                s_u=s, s_v=s, eta_u=1.0, eta_v=0.4, kappa=kappa,
                opacity=1.0, color=np.array([0, 0, 0]),
            )
        else:
            prim = make_mgs_primitive(eta_u=eta, eta_v=eta,
                                       kappa=kappa, scale=s)
        rgb = np.array([int(color[i:i+2], 16)/255 for i in (1, 3, 5)])
        plot_density_shells(ax, prim, color_base=rgb)
        plot_chart_with_window(ax, prim, color_max=rgb,
                                show_boundary=True, alpha_max=0.30)
        style_3d_axes(ax)
        ax.set_title(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Figure E : rendered depth from a single primitive
# ===========================================================================
def figure_rendered_depth(out_path):
    res = 96
    focal = res * 1.0
    cam_pos = (0.0, 0.0, 2.0)
    ro, rd = make_camera(image_size=res, focal=focal, cam_pos=cam_pos)

    cases = [
        ('PGSR limit  $\\kappa=0$',
         dict(eta_u=0.7, eta_v=0.4, kappa=0.0),  '#3a6db0'),
        ('Mild convex  $\\kappa=+0.5$',
         dict(eta_u=0.7, eta_v=0.4, kappa=0.5),  '#3aa860'),
        ('Convex cone  $\\kappa=+1$',
         dict(eta_u=0.7, eta_v=0.4, kappa=1.0),  '#ca6d34'),
        ('Concave cone  $\\kappa=-1$',
         dict(eta_u=0.7, eta_v=0.4, kappa=-1.0), '#c41e3a'),
    ]
    depths = []
    for title, kw, _ in cases:
        prim = make_mgs_primitive(scale=0.7, **kw)
        tau, alpha, _ = ray_primitive_alpha(prim, ro, rd)
        hit = ro + tau[..., None] * rd
        depth = -hit[..., 2]
        Z = np.where(alpha > 1e-3, depth, np.nan)
        depths.append(Z)

    z_max_global = max(np.nanmax(Z) for Z in depths if np.any(np.isfinite(Z)))
    z_min_global = min(np.nanmin(Z) for Z in depths if np.any(np.isfinite(Z)))
    z_pad = 0.05 * (z_max_global - z_min_global + 1e-6)
    z_range = (z_min_global - z_pad, z_max_global + z_pad)

    fig = plt.figure(figsize=(17, 5.0))
    fig.suptitle(
        r"Rendered depth $\tau^*(p_x, p_y)$ from a single MGS primitive "
        r"(shared $z$ axis)."
        "\n"
        r"$\kappa=0$ is flat (PGSR);  $\kappa>0$ curves convex,  "
        r"$\kappa<0$ curves concave.    "
        r"Within-primitive depth is curved, not piecewise planar.",
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
        ax.plot_wireframe(PX, PY, Z, rstride=10, cstride=10,
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
# ===========================================================================
def figure_composite(out_path, eta_u=0.7, eta_v=0.35, kappa=1.0, scale=0.7):
    prim = make_mgs_primitive(eta_u=eta_u, eta_v=eta_v,
                              kappa=kappa, scale=scale)
    fig = plt.figure(figsize=(15.5, 4.8))
    fig.suptitle(
        rf"An MGS primitive    $\kappa={kappa:+.2f},\ "
        rf"\eta^u={eta_u:.2f},\ \eta^v={eta_v:.2f}; \ s={scale:.2f};\ "
        rf"\bar\eta={prim.eta_bar:.2f},\ "
        rf"\bar\sigma={prim.sigma_bar:.2f},\ c={C_FLATNESS}$",
        fontsize=12,
    )

    ax_a = fig.add_subplot(1, 3, 1, projection='3d')
    plot_chart_with_window(ax_a, prim, color_max=(0.18, 0.62, 0.42))
    box_u = 1.1 * scale
    ax_a.set_xlim(-box_u, box_u); ax_a.set_ylim(-box_u, box_u)
    _set_zlim_for_cone(ax_a, prim, box_u=box_u, pad_frac=0.10)
    style_3d_axes(ax_a)
    ax_a.set_title("(a) Chart with footprint window\n"
                   r"density peaks here, weighted by $W$",
                   fontsize=10)

    ax_b = fig.add_subplot(1, 3, 2, projection='3d')
    plot_density_shells(ax_b, prim,
                        level_fracs=(0.85, 0.5, 0.15),
                        face_alphas=(0.95, 0.55, 0.22),
                        color_base=(0.18, 0.62, 0.42))
    plot_chart_with_window(ax_b, prim, color_max=(0.18, 0.62, 0.42),
                            alpha_max=0.30, show_boundary=False)
    style_3d_axes(ax_b)
    ax_b.set_title(r"(b) Density shells $\rho/o \in \{0.85,\,0.5,\,0.15\}$",
                   fontsize=10)

    ax_c = fig.add_subplot(1, 3, 3, projection='3d')
    res = 96
    ro, rd = make_camera(image_size=res, focal=res * 1.0,
                          cam_pos=(0.0, 0.0, 2.0))
    tau, alpha, _ = ray_primitive_alpha(prim, ro, rd)
    hit = ro + tau[..., None] * rd
    depth = -hit[..., 2]
    Z = np.where(alpha > 1e-3, depth, np.nan)
    px = np.arange(res) - res / 2.0
    py = np.arange(res) - res / 2.0
    PX, PY = np.meshgrid(px, py)
    ax_c.plot_surface(PX, PY, Z, cmap='viridis', edgecolor='none',
                      alpha=0.95, rstride=1, cstride=1, antialiased=True)
    ax_c.plot_wireframe(PX, PY, Z, rstride=10, cstride=10,
                        color='#222222', linewidth=0.35, alpha=0.55)
    ax_c.set_xlabel('pixel x', fontsize=8, labelpad=-6)
    ax_c.set_ylabel('pixel y', fontsize=8, labelpad=-6)
    ax_c.set_zlabel(r'depth $\tau^*$', fontsize=8, labelpad=-4)
    ax_c.tick_params(labelsize=6, pad=-2)
    ax_c.view_init(elev=22, azim=-65)
    ax_c.set_box_aspect((1, 1, 0.55))
    ax_c.set_title("(c) Rendered depth from this single primitive",
                   fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main():
    out_dir = "mgs_demo"
    os.makedirs(out_dir, exist_ok=True)

    figs = [
        ("mgs_kappa_sweep.png",                  figure_kappa_sweep),
        ("mgs_pointcentered_vs_surface.png",     figure_pointcentered_vs_surfacecentered),
        ("mgs_footprint_window.png",             figure_footprint_window),
        ("mgs_nested_shells.png",                figure_nested_shells),
        ("mgs_rendered_depth.png",               figure_rendered_depth),
        ("mgs_composite.png",                    figure_composite),
    ]
    for fname, fn in figs:
        path = os.path.join(out_dir, fname)
        fn(path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()