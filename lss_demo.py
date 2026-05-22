#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:03:47 2026

@author: hong
"""

"""
LSS demo - QGS-style visualization for the Level-Set Splatting primitive.

Mirrors the spirit of `quadratic_demo.py` from QGS:

    * Place a small grid of primitives in world space.
    * Set up a perspective camera.
    * Render an image by ray-tracing each pixel against the primitives.
    * Sweep a primitive parameter and generate frames.

For LSS, the parameter being swept is the bending vector (alpha, beta) per
primitive.  Each frame shows three panels:

    (A) Camera-rendered RGB image of the grid of primitives.
    (B) 3D density isosurface for the centre primitive at this (alpha, beta).
    (C) Depth-along-ray profile through the centre primitive,
        with PGSR (alpha=beta=0) overlaid for reference.

Output: one PNG per frame, plus a montage of representative frames.

CPU-only.  Pure numpy / matplotlib.  ~60s for the full sweep.
"""

import os
from math import cos, sin, pi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


# ---------------------------------------------------------------------------
# Camera / projection helpers  (paralleling QGS demo's get_cameras)
# ---------------------------------------------------------------------------
def rotation_x(theta):
    c, s = cos(theta), sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(theta):
    c, s = cos(theta), sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(theta):
    c, s = cos(theta), sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_camera(image_size=192, focal=190.0, cam_pos=(0.0, 0.0, 2.8), look_dir=(0, 0, -1)):
    """Build a simple pinhole camera looking along look_dir from cam_pos.

    Returns:
        ray_origins : (H, W, 3)
        ray_dirs    : (H, W, 3)   unit vectors
    """
    H = W = image_size
    fx = fy = focal
    cx, cy = W / 2, H / 2

    z = -np.array(look_dir, dtype=np.float64)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    # If look_dir is parallel to up, choose a different up
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
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


def make_side_camera(image_size=192, focal=200.0, cam_pos=(0.0, -2.4, 0.4)):
    """Side-on camera looking from -y toward the origin.  Convenient for
    seeing bending in the (u, t) plane, which is face-perpendicular to
    a face-on camera."""
    cam_pos = np.array(cam_pos, dtype=np.float64)
    look_dir = -cam_pos / np.linalg.norm(cam_pos)
    return make_camera(image_size=image_size, focal=focal,
                       cam_pos=tuple(cam_pos), look_dir=tuple(look_dir))


# ---------------------------------------------------------------------------
# Primitive: an LSS primitive carries (mu, R, s_u, s_v, sigma_n, alpha, beta, opacity, color)
# ---------------------------------------------------------------------------
class LSSPrimitive:
    """Level-Set Splatting primitive.

    Local SDF:  phi(u, v, t) = t * (1 - alpha*u - beta*v)
    Density:    rho(x) = opacity * exp( -0.5 * [phi^2 / sigma_n^2 + u^2/s_u^2 + v^2/s_v^2] )
    """

    def __init__(self, mu, R, s_u, s_v, sigma_n, alpha, beta, opacity, color):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)          # local -> world; columns are (t_u, t_v, n0)
        self.s_u = float(s_u)
        self.s_v = float(s_v)
        self.sigma_n = float(sigma_n)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.opacity = float(opacity)
        self.color = np.asarray(color, dtype=np.float64)


def world_to_local(prim, x):
    """Transform world points x of shape (..., 3) to local (u, v, t)."""
    return (x - prim.mu) @ prim.R                    # because R is local->world, R^T world->local


def phi_local(prim, uvt):
    u, v, t = uvt[..., 0], uvt[..., 1], uvt[..., 2]
    return t * (1.0 - prim.alpha * u - prim.beta * v)


def density_exponent_local(prim, uvt):
    u, v, t = uvt[..., 0], uvt[..., 1], uvt[..., 2]
    phi = t * (1.0 - prim.alpha * u - prim.beta * v)
    return (phi * phi) / (prim.sigma_n ** 2) + (u * u) / (prim.s_u ** 2) + (v * v) / (prim.s_v ** 2)


def density_world(prim, x):
    return prim.opacity * np.exp(-0.5 * density_exponent_local(prim, world_to_local(prim, x)))


# ---------------------------------------------------------------------------
# Ray-density maximum: find tau* along ray, the maximum-density point.
#
# Along the ray  r(tau) = o + tau d,  the exponent E(tau) is a quartic in tau
# (because phi^2 is quartic in (u, v, t) when alpha/beta != 0).
# We minimize by 1D golden-section search inside a bounded window.
# Robust, fast enough for a 192x192 demo, no Cardano headaches at edge cases.
# ---------------------------------------------------------------------------
def _exponent_along_ray(prim, o_local, d_local, tau):
    """Vectorized: o_local, d_local shape (..., 3); tau shape matching leading dims or scalar."""
    uvt = o_local + tau[..., None] * d_local
    return density_exponent_local(prim, uvt)


def ray_primitive_max(prim, o_world, d_world, tau_window=4.0, n_init=24):
    """Find tau* minimizing the exponent along each ray (= maximum density).

    Strategy: coarse grid sample over [tau_center - tau_window, tau_center + tau_window],
    then a few Newton steps from the best grid point.  Vectorized over rays.

    Returns:
        tau_star : (...,)
        E_star   : (...,)    exponent at the maximum-density point
        uvt_star : (..., 3)  local hit coordinates
    """
    shape = o_world.shape[:-1]
    o = o_world.reshape(-1, 3)
    d = d_world.reshape(-1, 3)
    Nrays = o.shape[0]

    # Transform ray into the primitive's local frame
    o_loc = (o - prim.mu) @ prim.R                     # (N, 3)
    d_loc = d @ prim.R                                  # (N, 3); direction transforms with R only

    # Reasonable initial guess: the tau that places r(tau) at the closest world point to mu.
    tau_center = -np.einsum('ij,ij->i', o - prim.mu, d)  # = -(o - mu) . d / ||d||^2, with ||d||=1

    # Coarse grid
    grid = np.linspace(-tau_window, tau_window, n_init)       # (G,)
    taus = tau_center[:, None] + grid[None, :]                # (N, G)
    uvt = o_loc[:, None, :] + taus[..., None] * d_loc[:, None, :]   # (N, G, 3)
    E = density_exponent_local(prim, uvt)                     # (N, G)
    best = np.argmin(E, axis=1)
    tau = taus[np.arange(Nrays), best]                        # (N,)

    # 3 Newton refinements via finite differences (cheap, well-behaved here)
    for _ in range(4):
        eps = 1e-3
        Ep = _exponent_along_ray(prim, o_loc, d_loc, tau + eps)
        Em = _exponent_along_ray(prim, o_loc, d_loc, tau - eps)
        E0 = _exponent_along_ray(prim, o_loc, d_loc, tau)
        first = (Ep - Em) / (2 * eps)
        second = (Ep - 2 * E0 + Em) / (eps ** 2)
        # Damped Newton (avoid divide-by-zero)
        step = first / np.where(np.abs(second) > 1e-8, second, 1e-8)
        step = np.clip(step, -0.3, 0.3)
        tau = tau - step

    uvt_star = o_loc + tau[:, None] * d_loc
    E_star = density_exponent_local(prim, uvt_star)
    return tau.reshape(shape), E_star.reshape(shape), uvt_star.reshape(*shape, 3)


# ---------------------------------------------------------------------------
# Renderer: alpha-blend the primitives along each pixel ray.
# ---------------------------------------------------------------------------
def render_with_camera(primitives, ray_origins, ray_dirs, background=None):
    """Alpha-blend primitives along supplied rays (front-to-back by tau*)."""
    ro, rd = ray_origins, ray_dirs
    H, W = ro.shape[:2]

    taus = np.zeros((len(primitives), H, W))
    alphas = np.zeros((len(primitives), H, W))
    colors = np.zeros((len(primitives), H, W, 3))
    for i, p in enumerate(primitives):
        tau_star, E_star, _ = ray_primitive_max(p, ro, rd)
        alpha = p.opacity * np.exp(-0.5 * E_star)
        alpha = np.clip(alpha, 0.0, 0.999)
        taus[i] = tau_star
        alphas[i] = alpha
        colors[i] = p.color[None, None, :]

    order = np.argsort(taus, axis=0)
    sorted_alphas = np.take_along_axis(alphas, order, axis=0)
    sorted_colors = np.take_along_axis(
        colors,
        order[..., None].repeat(3, axis=-1),
        axis=0,
    )

    if background is None:
        background = np.array([1.0, 1.0, 1.0])
    img = np.broadcast_to(background, (H, W, 3)).astype(np.float64).copy()
    T = np.ones((H, W))
    for i in range(len(primitives)):
        a = sorted_alphas[i]
        c = sorted_colors[i]
        w = (T * a)[..., None]
        img = img * (1 - w) + c * w
        T = T * (1 - a)
    return img


def render_image(primitives, image_size=192, focal=190.0, cam_pos=(0.0, 0.0, 2.8), background=None):
    ro, rd = make_camera(image_size=image_size, focal=focal, cam_pos=cam_pos)
    return render_with_camera(primitives, ro, rd, background=background)


# ---------------------------------------------------------------------------
# Plot helpers for the 3D isosurface and the depth-profile comparison panel
# ---------------------------------------------------------------------------
def plot_density_isosurface(ax, prim, level_frac=0.5, n_sigma_uv=2.5, n_sigma_t=4.0, N=64):
    """Marching cubes of the density isosurface at `level_frac` of opacity, local frame.

    The box is sized in units of the primitive's own scales so the shape stays in view
    regardless of how thin sigma_n is or how big s_u/s_v are.  This was the bug in the
    earlier version - using a fixed box hid the bend whenever sigma_n was small.
    """
    iso_exponent = -2 * np.log(level_frac)
    box_u = n_sigma_uv * prim.s_u
    box_v = n_sigma_uv * prim.s_v
    box_t = n_sigma_t * prim.sigma_n
    u = np.linspace(-box_u, box_u, N)
    v = np.linspace(-box_v, box_v, N)
    t = np.linspace(-box_t, box_t, N)
    U, V, T = np.meshgrid(u, v, t, indexing='ij')
    uvt = np.stack([U, V, T], axis=-1)
    E = density_exponent_local(prim, uvt)
    try:
        verts, faces, _, _ = marching_cubes(E, level=iso_exponent)
    except (ValueError, RuntimeError):
        ax.text(0.5, 0.5, 0.5, "(empty)", transform=ax.transAxes, ha='center')
        return
    scale = np.array([2 * box_u / (N - 1), 2 * box_v / (N - 1), 2 * box_t / (N - 1)])
    origin = np.array([-box_u, -box_v, -box_t])
    verts = verts * scale + origin
    # color by surface normal z (which now corresponds nicely to the bending)
    tri_n = np.zeros((len(faces), 3))
    for i, f in enumerate(faces):
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.linalg.norm(n) + 1e-9
        tri_n[i] = n
    shade = 0.5 + 0.5 * tri_n[:, 2]
    mesh = Poly3DCollection(verts[faces], alpha=0.85, linewidth=0)
    mesh.set_facecolor(cm.plasma(shade))
    ax.add_collection3d(mesh)
    ax.set_xlim(-box_u, box_u)
    ax.set_ylim(-box_v, box_v)
    ax.set_zlim(-box_t, box_t)
    ax.set_xlabel('u', fontsize=8, labelpad=-8)
    ax.set_ylabel('v', fontsize=8, labelpad=-8)
    ax.set_zlabel('t', fontsize=8, labelpad=-8)
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=22, azim=-55)
    # Equal aspect makes the curvature easier to read
    ax.set_box_aspect((1, 1, 0.6))


def plot_tau_star_profile(ax, prim, prim_flat):
    """Plot tau*(pixel_offset) for a horizontal scanline through the primitive,
    comparing LSS (curved) with the flat PGSR-equivalent primitive.

    Also shades the *danger zone* where the auxiliary level set
    {1 - alpha*u - beta*v = 0} threatens to enter the primitive's support.
    The bending regularizer in the paper keeps |alpha|*s_u + |beta|*s_v < ~0.5
    so the auxiliary set stays outside the primitive footprint.
    """
    cam_pos = np.array([0.0, 0.0, 2.8])
    n_rays = 96
    sweep = 1.4 * prim.s_u
    offsets = np.linspace(-sweep, sweep, n_rays)
    targets = np.stack([offsets, np.zeros_like(offsets), np.zeros_like(offsets)], axis=-1)
    dirs = targets - cam_pos
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(cam_pos, dirs.shape)

    tau_lss, _, _ = ray_primitive_max(prim, origins, dirs)
    tau_flat, _, _ = ray_primitive_max(prim_flat, origins, dirs)

    hits_lss = origins + tau_lss[:, None] * dirs
    hits_flat = origins + tau_flat[:, None] * dirs
    depth_lss = -hits_lss[:, 2]
    depth_flat = -hits_flat[:, 2]

    # Shade the danger zone where 1 - alpha*u - beta*v approaches zero on the
    # primitive's surface y=0, i.e. u = 1/alpha (if alpha > 0).
    if abs(prim.alpha) > 1e-3:
        u_singular = 1.0 / prim.alpha
        if -sweep <= u_singular <= sweep:
            if u_singular > 0:
                ax.axvspan(u_singular - 0.05 * sweep, sweep, alpha=0.13,
                           color='red', label='aux-set danger zone')
            else:
                ax.axvspan(-sweep, u_singular + 0.05 * sweep, alpha=0.13,
                           color='red', label='aux-set danger zone')

    # The primitive's own footprint
    ax.axvspan(-prim.s_u, prim.s_u, alpha=0.06, color='blue', label='primitive footprint')

    ax.plot(offsets, depth_flat, 'k--', label=r'PGSR  ($\alpha=\beta=0$)', linewidth=1.5)
    ax.plot(offsets, depth_lss, color='#c41e3a', label='LSS', linewidth=2)
    ax.set_xlabel('horizontal pixel offset (world units)', fontsize=8)
    ax.set_ylabel(r'rendered depth $\tau^*$', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------
def make_scene(num_points=4, alpha=0.0, beta=0.0, length=0.5, thin=0.10):
    """Build a num_points x num_points grid of LSS primitives in the z=0 plane.

    Centre primitive is highlighted in red.  `thin` controls sigma_n as a fraction
    of the in-plane scale; smaller -> thinner shell.
    """
    xs = np.linspace(-1, 1, num_points) * length
    ys = np.linspace(-1, 1, num_points) * length
    X, Y = np.meshgrid(xs, ys)
    centers = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=-1)

    s = length / max(1, num_points - 1)
    s_u = s_v = s * 1.0
    sigma_n = s * thin

    primitives = []
    palette = cm.viridis(np.linspace(0.15, 0.85, len(centers)))[..., :3]
    R = np.eye(3)
    for i, c in enumerate(centers):
        col = palette[i]
        is_center = (i == len(centers) // 2)
        if is_center:
            col = np.array([0.92, 0.18, 0.18])
        primitives.append(LSSPrimitive(
            mu=c, R=R,
            s_u=s_u, s_v=s_v, sigma_n=sigma_n,
            alpha=alpha, beta=beta,
            opacity=0.95, color=col,
        ))
    return primitives


def make_single_primitive(alpha=0.0, beta=0.0, scale=0.6, thin=0.15):
    """A single big primitive at origin for clear close-up rendering."""
    return [LSSPrimitive(
        mu=np.zeros(3), R=np.eye(3),
        s_u=scale, s_v=scale, sigma_n=scale * thin,
        alpha=alpha, beta=beta,
        opacity=0.95, color=np.array([0.85, 0.20, 0.25]),
    )]


def center_primitive(primitives):
    return primitives[len(primitives) // 2]


# ---------------------------------------------------------------------------
# A single composite figure: RGB render + isosurface + depth profile
# ---------------------------------------------------------------------------
def composite_figure(alpha, beta, out_path, title_suffix=""):
    # Use a single big primitive for the close-up render; better contrast.
    prims_single = make_single_primitive(alpha=alpha, beta=beta, scale=0.7, thin=0.15)
    img = render_image(prims_single, image_size=240, focal=240.0, cam_pos=(0.6, 0.6, 2.0))
    centre = prims_single[0]
    centre_flat = LSSPrimitive(
        mu=centre.mu, R=centre.R, s_u=centre.s_u, s_v=centre.s_v,
        sigma_n=centre.sigma_n, alpha=0.0, beta=0.0,
        opacity=centre.opacity, color=centre.color,
    )

    fig = plt.figure(figsize=(14, 4.2))
    fig.suptitle(
        rf"LSS: $\alpha={alpha:+.2f}$, $\beta={beta:+.2f}$" + title_suffix,
        fontsize=12,
    )

    ax_a = fig.add_subplot(1, 3, 1)
    ax_a.imshow(img)
    ax_a.set_title("(a) Camera render (single primitive)", fontsize=10)
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    ax_b = fig.add_subplot(1, 3, 2, projection='3d')
    plot_density_isosurface(ax_b, centre, level_frac=0.5, n_sigma_uv=2.0, n_sigma_t=3.5)
    ax_b.set_title("(b) Density half-max (local frame)", fontsize=10)

    ax_c = fig.add_subplot(1, 3, 3)
    plot_tau_star_profile(ax_c, centre, centre_flat)
    ax_c.set_title("(c) Depth across pixel offset", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# 3D visualizations of the LSS primitive
# ---------------------------------------------------------------------------
# Three conceptual targets, each from §3.2-§3.3 of the paper:
#
#   (1) The density kernel is QUARTIC in (u, v, t), not Gaussian.  Showing
#       multiple nested isosurfaces of rho makes the curved-shell structure
#       visible (a single Gaussian would give nested ellipsoids).
#
#   (2) The zero set {phi = 0} is exactly the t=0 plane -- FLAT -- yet the
#       density isosurfaces at rho < o_max curl away from that plane wherever
#       (1 - eta^u u - eta^v v) approaches zero.  This is the conceptual
#       claim in §3.2 that LSS is "not a covariance reparameterization."
#
#   (3) For a single primitive, the alpha-blended depth tau*(pixel) is a
#       linear function of pixel coords under PGSR and a nonlinear (cubic-
#       root) function under LSS.  Plotting tau*(pixel_x, pixel_y) as a 3D
#       surface makes the curvature operational claim in §3.3 visible:
#       PGSR's rendered depth surface is FLAT; LSS's is curved.
# ---------------------------------------------------------------------------

def _isosurface_local(prim, level_frac, n_sigma_uv=2.5, n_sigma_t=4.5, N=48):
    """Marching-cubes isosurface of density at rho = level_frac * opacity,
    in the primitive's LOCAL frame.

    Returns:
        verts : (V, 3) in local (u, v, t) coords  -- or None if empty
        faces : (F, 3) int indices                -- or None if empty
        box   : (box_u, box_v, box_t) extents used
    """
    iso_exp = -2.0 * np.log(level_frac)
    box_u = n_sigma_uv * prim.s_u
    box_v = n_sigma_uv * prim.s_v
    box_t = n_sigma_t * prim.sigma_n
    u = np.linspace(-box_u, box_u, N)
    v = np.linspace(-box_v, box_v, N)
    t = np.linspace(-box_t, box_t, N)
    U, V, T = np.meshgrid(u, v, t, indexing='ij')
    uvt = np.stack([U, V, T], axis=-1)
    E = density_exponent_local(prim, uvt)
    try:
        verts, faces, _, _ = marching_cubes(E, level=iso_exp)
    except (ValueError, RuntimeError):
        return None, None, (box_u, box_v, box_t)
    scale = np.array([2 * box_u / (N - 1),
                      2 * box_v / (N - 1),
                      2 * box_t / (N - 1)])
    origin = np.array([-box_u, -box_v, -box_t])
    verts = verts * scale + origin
    return verts, faces, (box_u, box_v, box_t)


def _local_to_world_verts(prim, verts_local):
    """Transform (V, 3) local vertices to world coords using the primitive's
    local->world rotation R (columns are t_u, t_v, n_0)."""
    return verts_local @ prim.R.T + prim.mu


def plot_3d_nested_level_sets(ax, prim,
                              level_fracs=(0.85, 0.5, 0.15),
                              face_alphas=(0.95, 0.55, 0.22),
                              show_phi_zero=True,
                              show_aux_set=True,
                              show_aux_legend=True,
                              N=52):
    """Draw nested density isosurfaces of an LSS primitive in its local frame.

    Multiple level sets reveal the QUARTIC kernel shape.  When eta=0 the level
    sets are concentric ellipsoidal disks (PGSR); when eta != 0 the outer
    level sets bulge toward the auxiliary line {1 - eta^u u - eta^v v = 0}.

    Options:
        show_phi_zero : overlay the FLAT phi=0 mid-plane (t=0) -- the kernel's
                        local surface chart.  Its flatness contrasted with the
                        curved level sets is the visual point of the figure.
        show_aux_set  : draw the auxiliary line {1 - eta.u = 0} in the t=0
                        plane.  The bending regularizer keeps it outside the
                        primitive's footprint (paper §3.5).
    """
    box_u = 2.5 * prim.s_u
    box_v = 2.5 * prim.s_v
    box_t = 4.5 * prim.sigma_n

    # Three colors -- outer (lightest), mid, inner (darkest) -- from a warm map
    base_colors = [
        np.array([0.96, 0.82, 0.45]),   # outer, pale gold
        np.array([0.92, 0.45, 0.18]),   # mid, orange
        np.array([0.72, 0.10, 0.18]),   # inner, deep red
    ]
    # Reorder: skimage gives a single surface per call.  Render outer first so
    # inner shells layer on top.
    levels_sorted = sorted(
        zip(level_fracs, face_alphas, base_colors[:len(level_fracs)]),
        key=lambda x: x[0],
    )
    for level_frac, falpha, col in levels_sorted:
        verts, faces, _ = _isosurface_local(prim, level_frac, N=N)
        if verts is None:
            continue
        mesh = Poly3DCollection(verts[faces], alpha=falpha,
                                linewidth=0.0, antialiased=True)
        # Light shading from the surface normal's t-component to bring out
        # the bend (otherwise solid color reads as a featureless blob).
        tri = verts[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nlen = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
        normals = normals / nlen
        shade = 0.55 + 0.45 * normals[:, 2]      # 0.1 .. 1.0 ish
        face_colors = np.clip(col[None, :] * shade[:, None], 0, 1)
        face_colors = np.concatenate(
            [face_colors, np.full((len(face_colors), 1), falpha)], axis=1)
        mesh.set_facecolor(face_colors)
        ax.add_collection3d(mesh)

    if show_phi_zero:
        # Flat mid-plane t=0 (= the kernel's local surface chart {phi=0})
        pu = np.linspace(-box_u, box_u, 2)
        pv = np.linspace(-box_v, box_v, 2)
        PU, PV = np.meshgrid(pu, pv)
        PT = np.zeros_like(PU)
        ax.plot_surface(PU, PV, PT, color='#202020', alpha=0.16,
                        edgecolor='#404040', linewidth=0.7, zorder=0)

    if show_aux_set:
        # Auxiliary line  1 - alpha u - beta v = 0  inside the t=0 plane,
        # if it actually crosses the plotting window.
        a, b = prim.alpha, prim.beta
        # Parameterize a line in the (u, v) plane; clip to window.
        if abs(a) > 1e-6 or abs(b) > 1e-6:
            if abs(b) >= abs(a):
                u_line = np.linspace(-box_u, box_u, 64)
                v_line = (1.0 - a * u_line) / (b if abs(b) > 1e-6 else 1e6)
            else:
                v_line = np.linspace(-box_v, box_v, 64)
                u_line = (1.0 - b * v_line) / (a if abs(a) > 1e-6 else 1e6)
            mask = (np.abs(u_line) <= box_u) & (np.abs(v_line) <= box_v)
            if mask.any():
                ax.plot(u_line[mask], v_line[mask], np.zeros(mask.sum()),
                        color='#c41e3a', linestyle='--', linewidth=1.8,
                        zorder=10, label=r'aux. line $1-\eta\cdot(u,v)=0$')
                if show_aux_legend:
                    ax.legend(loc='upper right', fontsize=7, framealpha=0.85)

    ax.set_xlim(-box_u, box_u)
    ax.set_ylim(-box_v, box_v)
    ax.set_zlim(-box_t, box_t)
    ax.set_xlabel('u', fontsize=8, labelpad=-6)
    ax.set_ylabel('v', fontsize=8, labelpad=-6)
    ax.set_zlabel('t  (normal direction)', fontsize=8, labelpad=-6)
    ax.tick_params(labelsize=6, pad=-2)
    # Aspect ratio that doesn't lie about the bend.  Honest equal-axis would
    # crush the t direction since sigma_n << s_u; use box ratios instead.
    ax.set_box_aspect((1.0, 1.0, 0.65))
    ax.view_init(elev=18, azim=-58)


def figure_pgsr_vs_lss_isosurface(out_path):
    """Two side-by-side 3D plots: PGSR-style flat primitive vs LSS bent
    primitive.  Identical (s_u, s_v, sigma_n); only the tilt vector differs.

    This is the central "before / after" figure.  Both panels show the same
    three nested isosurfaces rho/o = 0.85, 0.5, 0.15 and the same flat phi=0
    mid-plane.  The PGSR panel shows nested ellipsoidal disks; the LSS panel
    shows them curling toward the auxiliary line.
    """
    s_u = s_v = 0.7
    sigma_n = 0.12
    prim_pgsr = LSSPrimitive(
        mu=np.zeros(3), R=np.eye(3),
        s_u=s_u, s_v=s_v, sigma_n=sigma_n,
        alpha=0.0, beta=0.0,
        opacity=1.0, color=np.array([0.85, 0.25, 0.25]),
    )
    # Choose eta so dimensionless bending eta_bar = sqrt((eta_u s_u)^2 +
    # (eta_v s_v)^2) is just under the regularizer's 0.5 cap.
    eta_u, eta_v = 0.65, 0.25
    prim_lss = LSSPrimitive(
        mu=np.zeros(3), R=np.eye(3),
        s_u=s_u, s_v=s_v, sigma_n=sigma_n,
        alpha=eta_u, beta=eta_v,
        opacity=1.0, color=np.array([0.85, 0.25, 0.25]),
    )
    eta_bar = np.sqrt((eta_u * s_u) ** 2 + (eta_v * s_v) ** 2)

    fig = plt.figure(figsize=(12, 5.5))
    fig.suptitle(
        "Same in-plane scales, same thickness, same flat zero set "
        r"$\{\phi_i=0\}$ (grey square)." "\n"
        r"PGSR has ellipsoidal level sets;  LSS has CURVED level sets pulled "
        r"toward the auxiliary line $1-\eta\!\cdot\!(u,v)=0$.",
        fontsize=11,
    )
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_3d_nested_level_sets(ax1, prim_pgsr, show_aux_set=False)
    ax1.set_title(r"PGSR  $(\eta=0)$    $\phi(u,v,t)=t$", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_3d_nested_level_sets(ax2, prim_lss, show_aux_set=True)
    ax2.set_title(
        rf"LSS  $(\eta^u={eta_u:.2f},\ \eta^v={eta_v:.2f},\ \bar\eta={eta_bar:.2f})$"
        + "\n" + r"$\phi(u,v,t) = t\,(1 - \eta^u u - \eta^v v)$",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def figure_eta_3d_sweep(out_path):
    """3x3 grid of 3D isosurface plots, sweeping (eta^u, eta^v).

    Center cell (eta=0) is exactly PGSR; off-axis cells show the level set
    curling.  Each cell is a single mid-level isosurface (rho/o = 0.5) so
    the curvature is read cleanly without occlusion."""
    etas_u = [-0.7, 0.0, 0.7]
    etas_v = [-0.7, 0.0, 0.7]
    s_u = s_v = 0.6
    sigma_n = 0.13

    fig = plt.figure(figsize=(11.5, 11.5))
    fig.suptitle(
        r"Density isosurface $\rho/o = 0.3$ swept over $(\eta^u,\,\eta^v)$.  "
        r"Centre ($\eta=0$) is PGSR (flat ellipsoidal disk);  off-axis cells bend." "\n"
        r"Grey square: the flat zero set $\{\phi=0\}$ (identical across all nine cells).  "
        r"Red dashes: the auxiliary line $1-\eta\!\cdot\!(u,v)=0$.",
        fontsize=11,
    )
    for i, eu in enumerate(etas_u):
        for j, ev in enumerate(etas_v):
            prim = LSSPrimitive(
                mu=np.zeros(3), R=np.eye(3),
                s_u=s_u, s_v=s_v, sigma_n=sigma_n,
                alpha=eu, beta=ev, opacity=1.0,
                color=np.array([0.85, 0.25, 0.25]),
            )
            ax = fig.add_subplot(3, 3, i * 3 + j + 1, projection='3d')
            plot_3d_nested_level_sets(
                ax, prim,
                level_fracs=(0.3,),
                face_alphas=(0.88,),
                show_aux_set=(eu != 0 or ev != 0),
                show_aux_legend=False,
                N=46,
            )
            label = rf"$\eta^u={eu:+.1f},\ \eta^v={ev:+.1f}$"
            color = 'k'
            if eu == 0 and ev == 0:
                label += "    (PGSR)"
                color = '#1f5fb4'
            ax.set_title(label, fontsize=9.5, color=color)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def figure_rendered_depth_surface(out_path):
    """The operational claim of §3.3 made visible.

    For a single primitive directly in front of the camera, we ray-cast a
    dense grid of camera rays and record the per-pixel hit depth tau*.  The
    resulting field depth(pixel_x, pixel_y) is plotted as a 3D surface.

      * PGSR (eta=0):  depth is a rational-linear function of pixel coords
                       -> the surface is a tilted plane.
      * LSS (eta!=0):  depth is a nonlinear function of pixel coords
                       -> the surface CURLS by O(eta_bar * s_u) within ONE
                       primitive's footprint.

    Two side-by-side 3D panels.  Same primitive geometry except the tilt.
    """
    s = 0.7
    thin = 0.15
    eta_u_lss, eta_v_lss = 0.65, 0.35
    prims_pgsr = make_single_primitive(alpha=0.0, beta=0.0, scale=s, thin=thin)
    prims_lss = make_single_primitive(alpha=eta_u_lss, beta=eta_v_lss,
                                       scale=s, thin=thin)
    eta_bar = np.sqrt((eta_u_lss * s) ** 2 + (eta_v_lss * s) ** 2)

    # Render rays.  Modest resolution -- the surface is smooth, no aliasing.
    res = 72
    focal = res * 1.0
    cam_pos = (0.0, 0.0, 2.0)
    ro, rd = make_camera(image_size=res, focal=focal, cam_pos=cam_pos)

    tau_pgsr, E_pgsr, _ = ray_primitive_max(prims_pgsr[0], ro, rd)
    tau_lss, E_lss, _ = ray_primitive_max(prims_lss[0], ro, rd)
    hit_pgsr = ro + tau_pgsr[..., None] * rd
    hit_lss = ro + tau_lss[..., None] * rd
    depth_pgsr = -hit_pgsr[..., 2]                  # camera looks along -z
    depth_lss = -hit_lss[..., 2]

    # Mask to only show pixels where this primitive contributes meaningfully.
    a_pgsr = prims_pgsr[0].opacity * np.exp(-0.5 * E_pgsr)
    a_lss = prims_lss[0].opacity * np.exp(-0.5 * E_lss)
    mask_thresh = 0.04
    Z_pgsr = np.where(a_pgsr > mask_thresh, depth_pgsr, np.nan)
    Z_lss = np.where(a_lss > mask_thresh, depth_lss, np.nan)

    # Pixel grid in pixel units, centered on principal point.
    px = np.arange(res) - res / 2.0
    py = np.arange(res) - res / 2.0
    PX, PY = np.meshgrid(px, py)

    # Use a common z-axis range so the bend reads as a direct comparison.
    z_min = np.nanmin([np.nanmin(Z_pgsr), np.nanmin(Z_lss)])
    z_max = np.nanmax([np.nanmax(Z_pgsr), np.nanmax(Z_lss)])
    z_pad = 0.08 * (z_max - z_min + 1e-6)
    z_lim = (z_min - z_pad, z_max + z_pad)

    fig = plt.figure(figsize=(13, 5.7))
    fig.suptitle(
        r"Rendered depth $\tau^*(\mathrm{pixel}_x,\mathrm{pixel}_y)$ from a "
        r"SINGLE primitive in front of the camera." "\n"
        r"PGSR's depth is a tilted plane (rational-linear in pixel coords);  "
        r"LSS's depth is curved (nonlinear root of a cubic).",
        fontsize=11,
    )
    for k, (Z, label) in enumerate([
        (Z_pgsr, r"PGSR  $(\eta=0)$ — flat depth"),
        (Z_lss,
         rf"LSS  $(\eta^u={eta_u_lss:.2f},\eta^v={eta_v_lss:.2f},"
         rf"\bar\eta={eta_bar:.2f})$ — curved depth"),
    ]):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        surf = ax.plot_surface(PX, PY, Z, cmap='viridis',
                               edgecolor='none', alpha=0.95,
                               rstride=1, cstride=1, antialiased=True,
                               vmin=z_lim[0], vmax=z_lim[1])
        # Wireframe overlay every few pixels so the curvature reads even
        # at small print sizes.
        ax.plot_wireframe(PX, PY, Z, rstride=8, cstride=8,
                          color='#222222', linewidth=0.35, alpha=0.55)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('pixel x', fontsize=8, labelpad=-6)
        ax.set_ylabel('pixel y', fontsize=8, labelpad=-6)
        ax.set_zlabel(r'depth $\tau^*$', fontsize=8, labelpad=-4)
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_zlim(*z_lim)
        ax.view_init(elev=22, azim=-65)
        ax.set_box_aspect((1, 1, 0.55))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def figure_3d_composite(out_path, eta_u=0.60, eta_v=0.30, scale=0.7, thin=0.15):
    """Three-panel 3D summary of a single LSS primitive.

    (a) Nested density isosurfaces (with flat phi=0 plane overlaid).
    (b) Single 0.3-level isosurface viewed face-on from above the (u, v) plane,
        making the asymmetry in the level set easy to read.
    (c) The rendered depth surface tau*(pixel_x, pixel_y) for the same
        primitive.

    Default (eta_u, eta_v) = (0.60, 0.30) puts eta_bar safely under the
    regularizer's 0.5 cap (eta_bar ~= 0.47 at scale 0.7), so the cubic
    stationarity solve has a clean unique root for every visible ray.

    Useful as the main "what is an LSS primitive" figure for the paper.
    """
    s_u = s_v = scale
    sigma_n = scale * thin
    prim = LSSPrimitive(
        mu=np.zeros(3), R=np.eye(3),
        s_u=s_u, s_v=s_v, sigma_n=sigma_n,
        alpha=eta_u, beta=eta_v, opacity=1.0,
        color=np.array([0.85, 0.25, 0.25]),
    )
    eta_bar = np.sqrt((eta_u * s_u) ** 2 + (eta_v * s_v) ** 2)

    fig = plt.figure(figsize=(15.5, 4.7))
    fig.suptitle(
        rf"An LSS primitive  ($\eta^u={eta_u:.2f},\ \eta^v={eta_v:.2f},\ "
        rf"\bar\eta={eta_bar:.2f};\ s={scale:.2f},\ \sigma_n={sigma_n:.3f}$)",
        fontsize=12,
    )

    # (a) Nested level sets, perspective view
    ax_a = fig.add_subplot(1, 3, 1, projection='3d')
    plot_3d_nested_level_sets(ax_a, prim,
                              level_fracs=(0.85, 0.5, 0.15),
                              face_alphas=(0.95, 0.55, 0.22),
                              show_phi_zero=True,
                              show_aux_set=True,
                              show_aux_legend=True,
                              N=52)
    ax_a.set_title("(a) Nested density level sets\n"
                   r"$\rho/o\in\{0.85,\,0.5,\,0.15\}$", fontsize=10)

    # (b) Single low level set, near top-down view to show in-plane asymmetry
    ax_b = fig.add_subplot(1, 3, 2, projection='3d')
    plot_3d_nested_level_sets(ax_b, prim,
                              level_fracs=(0.3,),
                              face_alphas=(0.88,),
                              show_phi_zero=True,
                              show_aux_set=True,
                              show_aux_legend=False,
                              N=52)
    ax_b.view_init(elev=68, azim=-55)
    ax_b.set_title(r"(b) Level set $\rho/o=0.3$, viewed"
                   "\nfrom above to show in-plane asymmetry",
                   fontsize=10)

    # (c) Rendered depth surface for this primitive (LSS)
    ax_c = fig.add_subplot(1, 3, 3, projection='3d')
    res = 64
    ro, rd = make_camera(image_size=res, focal=res * 1.0,
                         cam_pos=(0.0, 0.0, 2.0))
    tau_lss, E_lss, _ = ray_primitive_max(prim, ro, rd)
    hit_lss = ro + tau_lss[..., None] * rd
    depth_lss = -hit_lss[..., 2]
    alpha_map = prim.opacity * np.exp(-0.5 * E_lss)
    # Stricter alpha mask + clip to within s_u/2 of the primitive's t=0 plane,
    # which is well within the unique-root regime of the cubic.
    mask = (alpha_map > 0.06) & (np.abs(depth_lss) < 0.35)
    Z = np.where(mask, depth_lss, np.nan)
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
    ax_c.set_title(r"(c) Rendered depth surface $\tau^*(p_x,p_y)$"
                   "\nfrom this single primitive", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def figure_3d_zero_set(out_path):
    """Visualize what the zero level set {phi_i = 0} actually looks like.

    From phi(u, v, t) = t * (1 - eta^u u - eta^v v), the zero set factors:

        {phi = 0}  =  {t = 0}                            (the mid-plane)
                  UNION
                      {1 - eta^u u - eta^v v = 0}        (the auxiliary plane,
                                                         vertical, all t)

    These two planes meet along the auxiliary LINE in t = 0.

    Three panels, increasing eta:
        (a) PGSR (eta = 0)        : the aux plane is at infinity; zero set
                                    is just the flat mid-plane.
        (b) LSS, eta_bar < 0.5    : two planes meet along the aux line, which
                                    sits OUTSIDE the primitive's footprint.
                                    This is the "well-behaved" regime the
                                    regularizer keeps us in.
        (c) LSS, eta_bar > 0.5    : the aux line cuts INTO the footprint.
                                    The cubic for tau* can have multiple
                                    roots here -- exactly what L_eta prevents.
    """
    s_u = s_v = 0.8
    foot_u = 2.5 * s_u             # ~3-sigma footprint, matches typical alpha cutoff
    foot_v = 2.5 * s_v
    plot_range = 2.6
    t_range = 1.3

    cases = [
        (0.0, 0.0,  "(a) PGSR  ($\\eta=0$)\nZero set = flat mid-plane only",      '#1f5fb4'),
        (0.5, 0.1,  "(b) LSS  ($\\bar\\eta\\approx 0.41 < 0.5$)\nAux line OUTSIDE footprint",  'k'),
        (1.2, 0.4,  "(c) LSS  ($\\bar\\eta\\approx 1.01 > 0.5$)\nAux line intrudes -- regularized away",  '#c41e3a'),
    ]

    fig = plt.figure(figsize=(16.5, 5.6))
    fig.suptitle(
        r"The zero set $\{\phi_i=0\}$ is the union of two planes:  "
        r"$\{t=0\}\,\cup\,\{1-\eta^u u-\eta^v v=0\}$.    "
        r"They meet along the auxiliary line." "\n"
        r"The bending regularizer $\bar\eta<0.5$ is exactly the constraint that "
        r"keeps the aux line outside the primitive's footprint.",
        fontsize=11,
    )

    for k, (eta_u, eta_v, title, title_color) in enumerate(cases):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        eta_bar = np.sqrt((eta_u * s_u) ** 2 + (eta_v * s_v) ** 2)

        # ----- The mid-plane t = 0 (always part of the zero set)
        uu = np.array([[-plot_range, plot_range], [-plot_range, plot_range]])
        vv = np.array([[-plot_range, -plot_range], [plot_range, plot_range]])
        tt = np.zeros_like(uu)
        ax.plot_surface(uu, vv, tt, color='#3a6db0', alpha=0.28,
                        edgecolor='#1a3a70', linewidth=0.8)

        # ----- The auxiliary plane 1 - eta_u u - eta_v v = 0 (only when eta != 0)
        if abs(eta_u) > 1e-4 or abs(eta_v) > 1e-4:
            eta_sq = eta_u ** 2 + eta_v ** 2
            eta_norm = np.sqrt(eta_sq)
            # In-plane direction of the aux line (perpendicular to (eta_u, eta_v))
            line_dir = np.array([-eta_v, eta_u]) / eta_norm
            # Point on the aux line closest to the origin in (u, v):
            line_pt = np.array([eta_u, eta_v]) / eta_sq
            # Build the aux plane surface
            s_param = np.linspace(-plot_range * 1.4, plot_range * 1.4, 2)
            t_param = np.linspace(-t_range, t_range, 2)
            S, TT = np.meshgrid(s_param, t_param)
            U_aux = line_pt[0] + line_dir[0] * S
            V_aux = line_pt[1] + line_dir[1] * S
            ax.plot_surface(U_aux, V_aux, TT, color='#c8404a', alpha=0.30,
                            edgecolor='#7a181f', linewidth=0.8)

            # The aux LINE (intersection of the two planes), drawn in t = 0
            s_line = np.linspace(-plot_range * 1.4, plot_range * 1.4, 200)
            U_line = line_pt[0] + line_dir[0] * s_line
            V_line = line_pt[1] + line_dir[1] * s_line
            T_line = np.zeros_like(U_line)
            mask = (np.abs(U_line) <= plot_range) & (np.abs(V_line) <= plot_range)
            if mask.any():
                ax.plot(U_line[mask], V_line[mask], T_line[mask],
                        color='#c41e3a', linestyle='-', linewidth=2.8, zorder=20,
                        label=r'aux. line  $\{\phi=0\}\cap\{t=0\}$')

        # ----- The primitive's footprint at t = 0 (ellipse of half-axes ~3 sigma)
        theta = np.linspace(0, 2 * np.pi, 120)
        u_circ = foot_u * np.cos(theta)
        v_circ = foot_v * np.sin(theta)
        t_circ = np.zeros_like(theta)
        ax.plot(u_circ, v_circ, t_circ, color='black', linestyle='-',
                linewidth=2.0, zorder=18, label=r'~3$\sigma$ footprint')
        # Light fill on the disc
        n_disc = 20
        r_disc = np.linspace(0, 1, n_disc)
        th_disc = np.linspace(0, 2 * np.pi, n_disc)
        R_disc, TH_disc = np.meshgrid(r_disc, th_disc)
        Xd = foot_u * R_disc * np.cos(TH_disc)
        Yd = foot_v * R_disc * np.sin(TH_disc)
        Zd = np.zeros_like(Xd)
        ax.plot_surface(Xd, Yd, Zd, color='#202020', alpha=0.10, edgecolor='none')

        ax.set_xlabel('u', fontsize=9, labelpad=-4)
        ax.set_ylabel('v', fontsize=9, labelpad=-4)
        ax.set_zlabel('t', fontsize=9, labelpad=-4)
        ax.tick_params(labelsize=7, pad=-1)
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.set_zlim(-t_range, t_range)
        ax.set_title(title, fontsize=10.5, color=title_color)
        ax.set_box_aspect((1.0, 1.0, 0.55))
        ax.view_init(elev=18, azim=-58)
        ax.legend(loc='upper left', fontsize=7, framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep: produce a montage of representative (alpha, beta) settings
# ---------------------------------------------------------------------------
def main():
    out_dir = "demo_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Sweep 1: 3x3 grid of (alpha, beta), single big primitive viewed from the side.
    # A side-on camera (looking along +y) is the right view to see the bending,
    # because bending happens in the (u, t) plane.
    alphas = [-1.0, -0.4, 0.0, 0.4, 1.0]
    betas = [0.0]              # Fix beta=0 so the sweep is purely on alpha
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    fig.suptitle(
        r"LSS primitive swept over $\alpha$  ($\beta=0$, side view).  "
        r"At $\alpha=0$ the primitive is a flat PGSR disk; "
        r"nonzero $\alpha$ bends the density.",
        fontsize=11,
    )
    for j, a in enumerate(alphas):
        prims = make_single_primitive(alpha=a, beta=0.0, scale=0.7, thin=0.15)
        # Side-on camera: looking along +y axis at origin
        img = render_image(prims, image_size=200, focal=220.0,
                           cam_pos=(0.0, -2.4, 0.0))
        # Re-render: the make_camera fn aims along -z, so rotate the scene.
        # Easier: use a camera at (0, -2.4, 0.6) looking toward origin.
        # Override: build a custom side camera.
        ro, rd = make_side_camera(image_size=200, focal=220.0,
                                  cam_pos=(0.0, -2.4, 0.4))
        img = render_with_camera(prims, ro, rd)
        ax = axes[j]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        title = rf"$\alpha={a:+.1f}$"
        if a == 0:
            title += "  (= PGSR)"
            for spine in ax.spines.values():
                spine.set_edgecolor('#1f77b4')
                spine.set_linewidth(2.0)
            ax.set_title(title, fontsize=11, color='#1f77b4')
        else:
            ax.set_title(title, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    sweep_path = os.path.join(out_dir, "lss_alpha_sweep_sideview.png")
    plt.savefig(sweep_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {sweep_path}")

    # ---- Sweep 2: 3x3 grid for full (alpha, beta) coverage, single primitive close-up
    alphas2 = [-0.8, 0.0, 0.8]
    betas2 = [-0.8, 0.0, 0.8]
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5))
    fig.suptitle(
        r"LSS primitive: bending sweep over $(\alpha,\beta)$.  "
        r"Camera looks at the primitive face-on; bending shows as a curved silhouette.",
        fontsize=11,
    )
    for i, a in enumerate(alphas2):
        for j, b in enumerate(betas2):
            prims = make_single_primitive(alpha=a, beta=b, scale=0.7, thin=0.15)
            img = render_image(prims, image_size=200, focal=240.0, cam_pos=(0.4, 0.4, 2.0))
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            title = rf"$\alpha={a:+.1f},\ \beta={b:+.1f}$"
            if a == 0 and b == 0:
                title += "  (= PGSR)"
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1f77b4')
                    spine.set_linewidth(2.0)
                ax.set_title(title, fontsize=10, color='#1f77b4')
            else:
                ax.set_title(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    grid_path = os.path.join(out_dir, "lss_alphabeta_grid.png")
    plt.savefig(grid_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {grid_path}")

    # ---- Composite figures (3-panel)
    cases = [
        (0.0, 0.0, "lss_composite_flat.png", " (PGSR-equivalent)"),
        (0.5, 0.3, "lss_composite_mild.png", " (mild bending)"),
        (1.0, 0.6, "lss_composite_strong.png", " (strong bending; aux-set encroaching)"),
    ]
    for a, b, fname, suffix in cases:
        path = os.path.join(out_dir, fname)
        composite_figure(a, b, path, title_suffix=suffix)
        print(f"Saved: {path}")

    # =======================================================================
    # 3D visualizations
    # -----------------------------------------------------------------------
    # These four figures collectively make the central claims of §3.2-§3.3
    # of the paper visually inspectable, without any training run.
    # =======================================================================

    pgsr_lss_path = os.path.join(out_dir, "lss_3d_pgsr_vs_lss.png")
    figure_pgsr_vs_lss_isosurface(pgsr_lss_path)
    print(f"Saved: {pgsr_lss_path}")

    zero_set_path = os.path.join(out_dir, "lss_3d_zero_set.png")
    figure_3d_zero_set(zero_set_path)
    print(f"Saved: {zero_set_path}")

    eta_sweep_path = os.path.join(out_dir, "lss_3d_eta_sweep.png")
    figure_eta_3d_sweep(eta_sweep_path)
    print(f"Saved: {eta_sweep_path}")

    depth_surf_path = os.path.join(out_dir, "lss_3d_rendered_depth_surface.png")
    figure_rendered_depth_surface(depth_surf_path)
    print(f"Saved: {depth_surf_path}")

    composite_3d_path = os.path.join(out_dir, "lss_3d_composite.png")
    figure_3d_composite(composite_3d_path, eta_u=0.60, eta_v=0.30,
                        scale=0.7, thin=0.15)
    print(f"Saved: {composite_3d_path}")


if __name__ == "__main__":
    main()