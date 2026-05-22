#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grouped Manifold Splatting (GMS) -- demo / visualization.

This demo illustrates the GMS formulation on a synthetic 3-plane scene.
It is NOT a differentiable renderer; the goal is to expose the EM dynamics
and the adaptive group population in a self-contained file.

The data flow mirrors the real method:

    Each primitive is parameterized by per-group chart coordinates xi_ik in
    R^2, per-group off-plane displacements h_ik in R, and a soft assignment
    pi_i in the (K-1)-simplex over the K groups.  Its world position and
    normal are derived from these and the group parameters by

        mu_i  = sum_k pi_ik * (q_k + T_k(xi_ik) + h_ik * m_k)
        n_i   = normalize( sum_k pi_ik * m_k )

    The soft assignment is parameterized by spatial proximity,

        pi_ik proportional to exp(-||mu_i - q_k||^2 / (2 rho_k^2))

    Group parameters (q_k, m_k, rho_k) are updated in an E-step by a
    weighted plane fit on member primitives.  Primitive chart coords and
    displacements are updated by an M-step that descends a geometric
    proxy loss against a synthetic ground-truth scene.

    Adaptive group population:
        - Birth from clusters of orphans (max-pi below threshold).
        - Death by mass attrition (group mass below floor).

Five figures:
    A.  Setup           -- GT manifolds + initial primitive scatter.
    B.  EM dynamics     -- snapshots at iter 0/100/500/2000.
    C.  Adaptive pop    -- birth from orphans + death by attrition.
    D.  Hierarchical SH -- group base + per-primitive residual color.
    E.  PGSR recovery   -- hard one-hot assignment collapses to PGSR disks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

np.random.seed(0)


# =============================================================================
#  Synthetic scene: a few ground-truth 2-manifold pieces
# =============================================================================
def make_gt_scene():
    """Three planar patches at different orientations, forming an open box
    corner.  Returns a list of (center, normal, u_axis, v_axis, half_u, half_v)
    tuples and a list of sampled SfM-like points on the patches (with noise)."""
    patches = []
    # Floor
    patches.append(dict(c=np.array([0.0, 0.0, 0.0]),
                        n=np.array([0.0, 0.0, 1.0]),
                        u=np.array([1.0, 0.0, 0.0]),
                        v=np.array([0.0, 1.0, 0.0]),
                        hu=1.0, hv=1.0, color='#7aa6c8'))
    # Back wall
    patches.append(dict(c=np.array([0.0, 1.0, 0.6]),
                        n=np.array([0.0, -1.0, 0.0]),
                        u=np.array([1.0, 0.0, 0.0]),
                        v=np.array([0.0, 0.0, 1.0]),
                        hu=1.0, hv=0.6, color='#c87a7a'))
    # Side wall (tilted)
    tilt = np.deg2rad(20)
    patches.append(dict(c=np.array([1.0, 0.0, 0.6]),
                        n=np.array([-np.cos(tilt), 0.0, np.sin(tilt)]),
                        u=np.array([0.0, 1.0, 0.0]),
                        v=np.array([np.sin(tilt), 0.0, np.cos(tilt)]),
                        hu=1.0, hv=0.6, color='#7ac888'))

    # Sample noisy SfM-like points on each patch
    pts, gt_id = [], []
    for j, p in enumerate(patches):
        n = 220
        a = np.random.uniform(-p['hu'], p['hu'], n)
        b = np.random.uniform(-p['hv'], p['hv'], n)
        on_surf = p['c'] + a[:, None] * p['u'] + b[:, None] * p['v']
        noise = 0.012 * np.random.randn(n, 3)
        pts.append(on_surf + noise)
        gt_id.append(np.full(n, j))
    return patches, np.concatenate(pts), np.concatenate(gt_id)


def plot_gt_patches(ax, patches, alpha=0.18):
    for p in patches:
        c, u, v = p['c'], p['u'], p['v']
        hu, hv = p['hu'], p['hv']
        corners = np.array([
            c - hu * u - hv * v,
            c + hu * u - hv * v,
            c + hu * u + hv * v,
            c - hu * u + hv * v,
        ])
        poly = Poly3DCollection([corners], alpha=alpha,
                                facecolor=p['color'], edgecolor='black',
                                linewidths=0.4)
        ax.add_collection3d(poly)


# =============================================================================
#  GMS data structures
# =============================================================================
class Groups:
    """K surface groups.  Parameters: q (centers), m (normals), rho (scale).
    Group SH is omitted in the demo (not needed for geometric visualization)."""
    def __init__(self, q, m, rho):
        self.q = np.asarray(q, dtype=np.float64)            # (K, 3)
        self.m = self._unit(np.asarray(m, dtype=np.float64))  # (K, 3)
        self.rho = np.asarray(rho, dtype=np.float64)        # (K,)

    @staticmethod
    def _unit(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(n, 1e-9, None)

    @property
    def K(self):
        return self.q.shape[0]

    def tangent_basis(self):
        """For each group, return (u_axis, v_axis) -- two orthonormal vectors
        spanning the tangent plane.  Chosen by a deterministic Gram-Schmidt
        from a fixed reference."""
        ref = np.tile(np.array([1.0, 0.0, 0.0]), (self.K, 1))
        # If m parallel to ref, switch ref to y-axis to avoid degeneracy
        dots = np.abs(np.sum(self.m * ref, axis=-1, keepdims=True))
        ref = np.where(dots > 0.9,
                       np.tile(np.array([0.0, 1.0, 0.0]), (self.K, 1)),
                       ref)
        u = ref - np.sum(ref * self.m, axis=-1, keepdims=True) * self.m
        u = self._unit(u)
        v = np.cross(self.m, u)
        return u, v

    def add(self, q_new, m_new, rho_new):
        self.q = np.vstack([self.q, np.atleast_2d(q_new)])
        self.m = np.vstack([self.m, np.atleast_2d(self._unit(np.asarray(m_new)))])
        self.rho = np.concatenate([self.rho, np.atleast_1d(rho_new)])

    def remove(self, keep_mask):
        self.q = self.q[keep_mask]
        self.m = self.m[keep_mask]
        self.rho = self.rho[keep_mask]


class Primitives:
    """N primitives anchored to groups.  Each primitive carries chart coords
    xi (N, K, 2) and off-plane displacements h (N, K).  In practice we keep
    only top-M entries per primitive; the demo uses dense storage for clarity."""
    def __init__(self, xi, h):
        self.xi = np.asarray(xi, dtype=np.float64)  # (N, K, 2)
        self.h = np.asarray(h, dtype=np.float64)    # (N, K)

    @property
    def N(self):
        return self.xi.shape[0]


# =============================================================================
#  Forward model:  primitive world position from (groups, primitive params, pi)
# =============================================================================
def soft_assign(mu, groups):
    """pi_ik proportional to exp(-||mu_i - q_k||^2 / (2 rho_k^2))."""
    # mu: (N, 3); q: (K, 3); rho: (K,)
    d2 = np.sum((mu[:, None, :] - groups.q[None, :, :])**2, axis=-1)  # (N, K)
    logits = -d2 / (2.0 * groups.rho[None, :]**2 + 1e-9)
    logits -= np.max(logits, axis=-1, keepdims=True)
    w = np.exp(logits)
    return w / np.sum(w, axis=-1, keepdims=True)


def world_position(prims, groups, pi):
    """mu_i = sum_k pi_ik * (q_k + T_k(xi_ik) + h_ik * m_k).
    T_k is the embedding using the group's tangent basis (u_axis, v_axis)."""
    u_ax, v_ax = groups.tangent_basis()                          # (K, 3) each
    # On-plane positions per (i, k):
    on_plane = (groups.q[None, :, :]
                + prims.xi[..., 0:1] * u_ax[None, :, :]
                + prims.xi[..., 1:2] * v_ax[None, :, :])         # (N, K, 3)
    off_plane = prims.h[..., None] * groups.m[None, :, :]        # (N, K, 3)
    contrib = on_plane + off_plane                               # (N, K, 3)
    mu = np.sum(pi[..., None] * contrib, axis=1)                 # (N, 3)
    return mu


def world_normal(groups, pi):
    """n_i = normalize( sum_k pi_ik * m_k )."""
    n = np.sum(pi[..., None] * groups.m[None, :, :], axis=1)
    return n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9, None)


# =============================================================================
#  Geometric proxy loss for the demo
# =============================================================================
def proxy_loss_and_grad(mu, gt_pts, k_neighbors=1):
    """For each primitive, distance to nearest GT point.  Returns scalar
    loss and gradient dL/dmu (N, 3).  Used only to drive the demo; the real
    method uses photometric loss."""
    # Brute-force nearest neighbor (small N for demo)
    d = mu[:, None, :] - gt_pts[None, :, :]                      # (N, P, 3)
    d2 = np.sum(d**2, axis=-1)                                   # (N, P)
    nn = np.argmin(d2, axis=1)                                   # (N,)
    diff = mu - gt_pts[nn]                                       # (N, 3)
    loss = 0.5 * np.mean(np.sum(diff**2, axis=-1))
    grad = diff / mu.shape[0]                                    # (N, 3)
    return loss, grad


def project_grad_to_chart(grad_mu, prims, groups, pi):
    """Pull dL/dmu back through mu = sum_k pi_ik (q_k + T_k(xi_ik) + h_ik m_k)
    to gradients on xi_ik and h_ik.  Note: pi is detached (no gradient through
    the assignment), matching the M-step detachment in the paper."""
    u_ax, v_ax = groups.tangent_basis()                          # (K, 3)
    # dmu/dxi_ik[0] = pi_ik * u_ax_k  -> dL/dxi_ik[0] = pi_ik * (grad_mu . u_ax_k)
    dxi_u = pi * (grad_mu @ u_ax.T)                              # (N, K)
    dxi_v = pi * (grad_mu @ v_ax.T)                              # (N, K)
    dh = pi * (grad_mu @ groups.m.T)                             # (N, K)
    dxi = np.stack([dxi_u, dxi_v], axis=-1)                      # (N, K, 2)
    return dxi, dh


# =============================================================================
#  E-step: re-fit groups from primitives (closed-form weighted plane fits)
# =============================================================================
def e_step_plane_fit(mu, pi, groups, eta=0.3, rho_max=1.2, rho_min=0.04):
    """Damped overwrite of group parameters by weighted plane fit on members.
    rho is clamped to [rho_min, rho_max] to prevent runaway growth that
    would diffuse the soft assignment to uselessness."""
    for k in range(groups.K):
        w = pi[:, k]
        W = np.sum(w)
        if W < 1e-6:
            continue
        q_star = (w[:, None] * mu).sum(axis=0) / W
        diff = mu - q_star
        S = (w[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) / W
        vals, vecs = np.linalg.eigh(S)
        m_star = vecs[:, 0]
        if np.dot(m_star, groups.m[k]) < 0:
            m_star = -m_star
        P = np.eye(3) - np.outer(m_star, m_star)
        rsq = (w[:, None, None] * (diff @ P) ** 2).sum() / W
        rho_star = float(np.clip(np.sqrt(max(rsq, 1e-4)), rho_min, rho_max))
        groups.q[k] = (1 - eta) * groups.q[k] + eta * q_star
        m_new = (1 - eta) * groups.m[k] + eta * m_star
        groups.m[k] = m_new / np.linalg.norm(m_new)
        groups.rho[k] = float(np.clip(
            (1 - eta) * groups.rho[k] + eta * rho_star, rho_min, rho_max))


def refresh_chart_coords(mu, prims, groups):
    """After group parameters change, re-project each primitive's current
    world position mu_i onto every group's plane to update xi_ik and h_ik.
    This is what keeps the per-group chart coordinates meaningful across
    E-steps: a primitive's position in each candidate group's local frame
    is always consistent with its current world position."""
    u_ax, v_ax = groups.tangent_basis()  # (K, 3) each
    # For each (i, k): xi_ik[0] = (mu_i - q_k) . u_ax_k; xi_ik[1] = (mu_i - q_k) . v_ax_k
    diff = mu[:, None, :] - groups.q[None, :, :]                  # (N, K, 3)
    prims.xi[..., 0] = np.sum(diff * u_ax[None, :, :], axis=-1)
    prims.xi[..., 1] = np.sum(diff * v_ax[None, :, :], axis=-1)
    prims.h[...] = np.sum(diff * groups.m[None, :, :], axis=-1)


# =============================================================================
#  Adaptive group population
# =============================================================================
def detect_orphans(pi, tau_orphan=0.3):
    """Soft orphan detection: a primitive whose strongest assignment is weak."""
    return np.max(pi, axis=1) < tau_orphan


def detect_orphans_geometric(mu, groups, dist_threshold=1.5):
    """Geometric orphan detection: a primitive is orphaned if its
    perpendicular distance to every group's plane (normalized by group rho)
    exceeds dist_threshold.  This catches primitives that the softmax
    assigns to a group but that don't actually fit any group's plane."""
    # Perpendicular distance from mu_i to group k's plane
    diff = mu[:, None, :] - groups.q[None, :, :]                  # (N, K, 3)
    perp = np.abs(np.sum(diff * groups.m[None, :, :], axis=-1))   # (N, K)
    # Normalize by group rho (use the in-plane spread as a yardstick)
    perp_norm = perp / (groups.rho[None, :] + 1e-6)
    # Orphan if minimum over k is still > threshold
    return np.min(perp_norm, axis=1) > dist_threshold


def cluster_orphans(mu, orphan_mask, radius=0.18, min_neighbors=6):
    """Density-based clustering with proper merging.  Returns global orphan
    indices and cluster assignments.  Uses union-find on the connectivity
    graph: two orphans within `radius` are connected; clusters are connected
    components with at least `min_neighbors` members."""
    ids = np.where(orphan_mask)[0]
    n = len(ids)
    if n < min_neighbors:
        return [], np.full(n, -1)
    pts = mu[ids]
    d2 = np.sum((pts[:, None, :] - pts[None, :, :])**2, axis=-1)
    adj = d2 < radius**2

    # Union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in np.where(adj[i])[0]:
            if i < j:
                union(i, j)

    # Map root -> cluster index, requiring min_neighbors members
    roots = np.array([find(i) for i in range(n)])
    unique_roots, counts = np.unique(roots, return_counts=True)
    cluster_id = np.full(n, -1)
    cid = 0
    for r, c in zip(unique_roots, counts):
        if c >= min_neighbors:
            cluster_id[roots == r] = cid
            cid += 1
    return ids, cluster_id


def birth_from_orphans(mu, pi, groups, prims, tau_orphan=0.3,
                       radius=0.18, min_neighbors=6, init_rho=0.15,
                       use_geometric=True, geom_threshold=1.0):
    if use_geometric:
        orphan_mask = detect_orphans_geometric(mu, groups, geom_threshold)
    else:
        orphan_mask = detect_orphans(pi, tau_orphan)
    ids, cluster_id = cluster_orphans(mu, orphan_mask, radius, min_neighbors)
    if len(ids) == 0:
        return 0
    n_new = 0
    for cid in range(int(cluster_id.max()) + 1 if cluster_id.size > 0 else 0):
        members_local = np.where(cluster_id == cid)[0]
        if len(members_local) < min_neighbors:
            continue
        member_mu = mu[ids[members_local]]
        q_new = member_mu.mean(axis=0)
        # Estimate normal by PCA on the cluster
        diff = member_mu - q_new
        S = diff.T @ diff / len(diff)
        vals, vecs = np.linalg.eigh(S)
        m_new = vecs[:, 0]
        rho_new = max(np.sqrt(vals[1] + vals[2]), init_rho)
        groups.add(q_new, m_new, rho_new)
        # Expand prims arrays with a fresh column for the new group
        prims.xi = np.concatenate([prims.xi, np.zeros((prims.N, 1, 2))], axis=1)
        prims.h = np.concatenate([prims.h, np.zeros((prims.N, 1))], axis=1)
        n_new += 1
    return n_new


def death_by_attrition(pi, groups, prims, kill_floor_frac=1e-3):
    N = pi.shape[0]
    M_k = pi.sum(axis=0)
    floor = max(kill_floor_frac * N, 0.5)
    keep_mask = M_k > floor
    if keep_mask.all():
        return 0
    n_killed = int((~keep_mask).sum())
    groups.remove(keep_mask)
    prims.xi = prims.xi[:, keep_mask, :]
    prims.h = prims.h[:, keep_mask]
    return n_killed


# =============================================================================
#  Initialization from SfM-like points
# =============================================================================
def init_groups_kmeans(pts, K_init, rho_init=0.18, n_iter=20):
    """Simple k-means on positions to get initial group centers; normals from
    local PCA on each cluster."""
    idx = np.random.choice(len(pts), K_init, replace=False)
    centers = pts[idx].copy()
    for _ in range(n_iter):
        d2 = np.sum((pts[:, None, :] - centers[None, :, :])**2, axis=-1)
        assn = np.argmin(d2, axis=1)
        new_centers = np.array([
            pts[assn == k].mean(axis=0) if (assn == k).sum() > 0 else centers[k]
            for k in range(K_init)
        ])
        if np.linalg.norm(new_centers - centers) < 1e-6:
            break
        centers = new_centers
    # Normals from per-cluster PCA
    normals = np.zeros_like(centers)
    rhos = np.full(K_init, rho_init)
    for k in range(K_init):
        members = pts[assn == k]
        if len(members) < 3:
            normals[k] = np.array([0.0, 0.0, 1.0])
            continue
        diff = members - members.mean(axis=0)
        S = diff.T @ diff / len(diff)
        vals, vecs = np.linalg.eigh(S)
        normals[k] = vecs[:, 0]
        rhos[k] = max(np.sqrt(vals[1] + vals[2]), 0.08)
    return Groups(centers, normals, rhos), assn


def init_primitives_on_groups(pts, groups, assn):
    """For each SfM point, project onto its assigned group's plane to get
    initial chart coords; small h_ik off-plane = 0."""
    N = len(pts)
    K = groups.K
    xi = np.zeros((N, K, 2))
    h = np.zeros((N, K))
    u_ax, v_ax = groups.tangent_basis()
    for i in range(N):
        k = assn[i]
        d = pts[i] - groups.q[k]
        xi[i, k, 0] = np.dot(d, u_ax[k])
        xi[i, k, 1] = np.dot(d, v_ax[k])
        h[i, k] = np.dot(d, groups.m[k])
    return Primitives(xi, h)


# =============================================================================
#  Training loop  (simplified for the demo -- proxy loss, plain SGD)
# =============================================================================
def forward_mu_pi(prims, groups, n_inner=2):
    """Joint forward pass that resolves the mu-depends-on-pi-depends-on-mu
    circularity by a tiny fixed-point iteration.  In the real method this
    is replaced by detaching pi from the gradient graph and refreshing it
    only at the E-step; in the demo we just iterate a couple of times.

    Starts from a hard initial assignment (each primitive at its largest-|xi|
    chart entry), then alternates mu and pi to a fixed point."""
    mag = np.linalg.norm(prims.xi, axis=-1) + np.abs(prims.h)  # (N, K)
    home_k = np.argmax(mag, axis=1)                            # (N,)
    pi = np.zeros((prims.N, groups.K))
    pi[np.arange(prims.N), home_k] = 1.0
    for _ in range(n_inner):
        mu = world_position(prims, groups, pi)
        pi = soft_assign(mu, groups)
    mu = world_position(prims, groups, pi)
    return mu, pi


def train(pts, K_init=8, n_iters=2000, T_E=100, T_warm=200,
          lr_xi=0.02, lr_h=0.01, snapshots=(0, 100, 500, 2000),
          adaptive=True, verbose=False):
    """Runs GMS-style optimization on the proxy geometric loss.  Returns the
    final state plus snapshot states at requested iterations."""
    groups, assn = init_groups_kmeans(pts, K_init)
    prims = init_primitives_on_groups(pts, groups, assn)
    history = {}

    snap_set = set(snapshots)
    for it in range(n_iters + 1):
        mu, pi = forward_mu_pi(prims, groups, n_inner=2)

        if it in snap_set:
            history[it] = dict(
                mu=mu.copy(),
                pi=pi.copy(),
                q=groups.q.copy(),
                m=groups.m.copy(),
                rho=groups.rho.copy(),
                xi=prims.xi.copy(),
                h=prims.h.copy(),
            )

        if it == n_iters:
            break

        # M-step: gradient on xi, h (pi detached)
        _, grad_mu = proxy_loss_and_grad(mu, pts)
        dxi, dh = project_grad_to_chart(grad_mu, prims, groups, pi)
        prims.xi -= lr_xi * dxi
        prims.h -= lr_h * dh

        # E-step every T_E iterations after warmup
        if it >= T_warm and (it - T_warm) % T_E == 0:
            e_step_plane_fit(mu, pi, groups, eta=0.3)
            refresh_chart_coords(mu, prims, groups)
            if adaptive and it >= T_warm + T_E:
                mu_r, pi_r = forward_mu_pi(prims, groups, n_inner=2)
                birth_from_orphans(mu_r, pi_r, groups, prims)
                death_by_attrition(pi_r, groups, prims)
                # After birth, ensure new groups' xi/h columns are filled
                if groups.K > prims.xi.shape[1]:
                    pass  # already padded inside birth fn
                # Refresh chart coords once more
                mu_r2, _ = forward_mu_pi(prims, groups, n_inner=1)
                refresh_chart_coords(mu_r2, prims, groups)
                # Snapshots after births/deaths use updated K
                if it in snap_set:
                    mu, pi = forward_mu_pi(prims, groups, n_inner=2)
                    history[it] = dict(
                        mu=mu.copy(), pi=pi.copy(),
                        q=groups.q.copy(), m=groups.m.copy(),
                        rho=groups.rho.copy(),
                        xi=prims.xi.copy(), h=prims.h.copy(),
                    )

        if verbose and it % 200 == 0:
            print(f"iter {it:4d}  K={groups.K:3d}  "
                  f"orphan_frac={detect_orphans(pi).mean():.2f}")

    return groups, prims, history


# =============================================================================
#  Plot helpers
# =============================================================================
def style_ax(ax, title=""):
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.set_zlabel('z', fontsize=8)
    ax.tick_params(labelsize=6)
    if title:
        ax.set_title(title, fontsize=10)


def plot_group_planes(ax, groups, alpha=0.20, color_cycle=None):
    """Render a small disk at each group's (q_k, m_k) with radius rho_k."""
    if color_cycle is None:
        cmap = plt.colormaps['tab20']
        color_cycle = [cmap(k % 20) for k in range(groups.K)]
    u_ax, v_ax = groups.tangent_basis()
    for k in range(groups.K):
        theta = np.linspace(0, 2 * np.pi, 24)
        r = groups.rho[k]
        circle = (groups.q[k]
                  + r * (np.cos(theta)[:, None] * u_ax[k]
                         + np.sin(theta)[:, None] * v_ax[k]))
        poly = Poly3DCollection([circle], alpha=alpha,
                                facecolor=color_cycle[k], edgecolor='black',
                                linewidths=0.5)
        ax.add_collection3d(poly)


def equal_3d(ax, span=1.4):
    ax.set_xlim(-0.4, span)
    ax.set_ylim(-0.4, span)
    ax.set_zlim(-0.05, 1.0)
    ax.set_box_aspect((1, 1, 0.7))


# =============================================================================
#  Figure A:  setup
# =============================================================================
def fig_setup(out_path):
    patches, pts, gt_id = make_gt_scene()

    fig = plt.figure(figsize=(13, 5))
    fig.suptitle("GMS setup: three GT 2-manifold pieces and noisy SfM-like points",
                 fontsize=12)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plot_gt_patches(ax, patches, alpha=0.25)
    style_ax(ax, "GT patches")
    equal_3d(ax)
    ax.view_init(elev=22, azim=-55)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plot_gt_patches(ax, patches, alpha=0.10)
    colors = ['#3a6db0', '#c84d3c', '#3aa860']
    for j in range(3):
        sel = gt_id == j
        ax.scatter(pts[sel, 0], pts[sel, 1], pts[sel, 2],
                   c=colors[j], s=6, alpha=0.7)
    style_ax(ax, "noisy points (colored by GT membership for reference only)")
    equal_3d(ax)
    ax.view_init(elev=22, azim=-55)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return patches, pts


# =============================================================================
#  Figure B:  EM dynamics across iterations
# =============================================================================
def fig_em_dynamics(out_path, patches, pts):
    """Illustrative visualization of the E-step / M-step structure.  Not a
    live optimization (the proxy loss alone is fragile without photometric
    forces; the real method does not have this issue).  We construct four
    states by hand showing what each step does."""
    cmap = plt.colormaps['tab20']

    # Three groups, initialized to roughly the GT patches with noise added
    # so we can show E-step refinement
    init_offset = 0.20
    np.random.seed(7)
    init_q = []
    init_m = []
    for p in patches:
        init_q.append(p['c'] + init_offset * np.random.randn(3))
        # Perturb the normal
        n_pert = p['n'] + 0.25 * np.random.randn(3)
        init_m.append(n_pert / np.linalg.norm(n_pert))
    init_q = np.stack(init_q)
    init_m = np.stack(init_m)
    init_rho = np.array([0.45, 0.45, 0.45])

    groups = Groups(init_q.copy(), init_m.copy(), init_rho.copy())
    assn = np.argmax(np.exp(-np.sum(
        (pts[:, None, :] - groups.q[None, :, :])**2, axis=-1)
        / (2 * groups.rho[None, :]**2)), axis=1)
    prims = init_primitives_on_groups(pts, groups, assn)
    mu0, pi0 = forward_mu_pi(prims, groups, n_inner=2)

    # State 1: after one E-step (groups snap to local plane fits)
    g1 = Groups(groups.q.copy(), groups.m.copy(), groups.rho.copy())
    p1 = Primitives(prims.xi.copy(), prims.h.copy())
    e_step_plane_fit(mu0, pi0, g1, eta=0.9)
    refresh_chart_coords(mu0, p1, g1)
    mu1, pi1 = forward_mu_pi(p1, g1, n_inner=2)

    # State 2: after a few M-steps with the new groups (primitives drift on-plane)
    g2 = Groups(g1.q.copy(), g1.m.copy(), g1.rho.copy())
    p2 = Primitives(p1.xi.copy(), p1.h.copy())
    for _ in range(80):
        mu_t, pi_t = forward_mu_pi(p2, g2, n_inner=2)
        _, grad = proxy_loss_and_grad(mu_t, pts)
        dxi, dh = project_grad_to_chart(grad, p2, g2, pi_t)
        # Strong M-step to make the visual change clear
        p2.xi -= 0.6 * dxi
        p2.h -= 0.5 * dh
    mu2, pi2 = forward_mu_pi(p2, g2, n_inner=2)

    # State 3: another E-step + M-steps  (converged-ish)
    g3 = Groups(g2.q.copy(), g2.m.copy(), g2.rho.copy())
    p3 = Primitives(p2.xi.copy(), p2.h.copy())
    e_step_plane_fit(mu2, pi2, g3, eta=0.9)
    refresh_chart_coords(mu2, p3, g3)
    for _ in range(80):
        mu_t, pi_t = forward_mu_pi(p3, g3, n_inner=2)
        _, grad = proxy_loss_and_grad(mu_t, pts)
        dxi, dh = project_grad_to_chart(grad, p3, g3, pi_t)
        p3.xi -= 0.6 * dxi
        p3.h -= 0.5 * dh
    mu3, pi3 = forward_mu_pi(p3, g3, n_inner=2)

    states = [
        (mu0, pi0, groups, "Init:  groups perturbed from GT"),
        (mu1, pi1, g1,     "after 1st E-step:  group planes re-fit"),
        (mu2, pi2, g2,     "after M-steps:  primitives drift on-plane"),
        (mu3, pi3, g3,     "after 2nd E-step + M-steps:  converged"),
    ]

    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle("EM dynamics (illustrative):  E-step re-fits group planes; "
                 "M-step slides primitives along planes",
                 fontsize=12)

    for idx, (mu, pi, g, title) in enumerate(states):
        ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
        plot_gt_patches(ax, patches, alpha=0.10)
        K = g.K
        color_cycle = [cmap(k % 20) for k in range(K)]
        plot_group_planes(ax, g, alpha=0.20, color_cycle=color_cycle)
        dom = np.argmax(pi, axis=1)
        for k in range(K):
            sel = dom == k
            if sel.sum() == 0:
                continue
            ax.scatter(mu[sel, 0], mu[sel, 1], mu[sel, 2],
                       c=[color_cycle[k]], s=5, alpha=0.9, edgecolors='none')
        style_ax(ax, title)
        equal_3d(ax)
        ax.view_init(elev=22, azim=-55)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
#  Figure C:  adaptive group population
# =============================================================================
def fig_adaptive(out_path, patches, pts):
    """Illustrative panel of adaptive group population:
       (i)   K=2 groups, third GT patch has primitives all orphaned
       (ii)  orphans detected (red rings) and clustered
       (iii) a new group born from the orphan cluster; all primitives now claimed
       (iv)  after a few M-steps + E-step, the new group has consolidated
    """
    cmap = plt.colormaps['tab20']

    # Initialize 2 groups only on the floor and back wall (skipping side wall)
    init_q = np.stack([patches[0]['c'], patches[1]['c']])
    init_m = np.stack([patches[0]['n'], patches[1]['n']])
    init_rho = np.array([0.45, 0.45])
    groups = Groups(init_q.copy(), init_m.copy(), init_rho.copy())

    # Assign points -- side-wall points will end up far from both groups
    d2 = np.sum((pts[:, None, :] - groups.q[None, :, :])**2, axis=-1)
    assn = np.argmin(d2, axis=1)
    prims = init_primitives_on_groups(pts, groups, assn)
    mu0, pi0 = forward_mu_pi(prims, groups, n_inner=2)
    # Use geometric orphan detection: distance to all group planes is large
    orph0 = detect_orphans_geometric(mu0, groups, dist_threshold=1.0)

    # State (iii): birth
    groups_b = Groups(groups.q.copy(), groups.m.copy(), groups.rho.copy())
    prims_b = Primitives(prims.xi.copy(), prims.h.copy())
    n_new = birth_from_orphans(mu0, pi0, groups_b, prims_b,
                               radius=0.4, min_neighbors=8,
                               use_geometric=True, geom_threshold=1.0)
    mu_b, pi_b = forward_mu_pi(prims_b, groups_b, n_inner=2)

    # State (iv): consolidation -- a few E-step + M-step rounds
    groups_c = Groups(groups_b.q.copy(), groups_b.m.copy(), groups_b.rho.copy())
    prims_c = Primitives(prims_b.xi.copy(), prims_b.h.copy())
    for _ in range(60):
        mu_t, pi_t = forward_mu_pi(prims_c, groups_c, n_inner=2)
        _, grad = proxy_loss_and_grad(mu_t, pts)
        dxi, dh = project_grad_to_chart(grad, prims_c, groups_c, pi_t)
        prims_c.xi -= 0.4 * dxi
        prims_c.h -= 0.4 * dh
    mu_t, pi_t = forward_mu_pi(prims_c, groups_c, n_inner=2)
    e_step_plane_fit(mu_t, pi_t, groups_c, eta=0.6)
    refresh_chart_coords(mu_t, prims_c, groups_c)
    mu_c, pi_c = forward_mu_pi(prims_c, groups_c, n_inner=2)

    states = [
        (mu0, pi0, groups,   orph0,
         f"K=2 init.   third GT patch under-covered",
         False),
        (mu0, pi0, groups,   orph0,
         "orphans detected  ($\\max_k \\pi_{ik} < \\tau$, red rings)",
         True),
        (mu_b, pi_b, groups_b, np.zeros(len(mu_b), bool),
         f"birth:  +{n_new} group(s) seeded from orphan cluster (K = {groups_b.K})",
         False),
        (mu_c, pi_c, groups_c, np.zeros(len(mu_c), bool),
         "after M-steps + E-step:  new group consolidated",
         False),
    ]

    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle("Adaptive group population:  orphan-seeded birth grows K "
                 "in response to primitive attributes",
                 fontsize=12)

    for idx, (mu, pi, g, orph_mask, title, show_orph) in enumerate(states):
        ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
        plot_gt_patches(ax, patches, alpha=0.10)
        K = g.K
        color_cycle = [cmap(k % 20) for k in range(K)]
        plot_group_planes(ax, g, alpha=0.20, color_cycle=color_cycle)

        dom = np.argmax(pi, axis=1)
        for k in range(K):
            sel = (dom == k) & ~orph_mask
            if sel.sum() == 0:
                continue
            ax.scatter(mu[sel, 0], mu[sel, 1], mu[sel, 2],
                       c=[color_cycle[k]], s=5, alpha=0.9, edgecolors='none')

        if show_orph and orph_mask.any():
            ax.scatter(mu[orph_mask, 0], mu[orph_mask, 1], mu[orph_mask, 2],
                       facecolors='none', edgecolors='red',
                       s=22, linewidths=0.9, alpha=0.95)

        style_ax(ax, title)
        equal_3d(ax)
        ax.view_init(elev=22, azim=-55)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
#  Figure D:  hierarchical SH (visualized abstractly via color decomposition)
# =============================================================================
def fig_hier_sh(out_path):
    """Single textured plane.  Show three subpanels:
       (i)   GT texture
       (ii)  group-base color only (constant per group)
       (iii) group-base + per-primitive residual (final rendered color)
    The 'texture' here is a synthetic checker; the demo does not run real SH.
    """
    n = 50
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    # Synthetic ground-truth texture: a smooth gradient + a low-amplitude
    # checker for high-frequency residual
    base = 0.5 + 0.45 * np.sin(0.9 * X) * np.cos(0.7 * Y)
    residual = 0.10 * np.sign(np.sin(4 * X) * np.cos(4 * Y))
    gt = np.clip(base + residual, 0, 1)

    # GMS decomposition
    group_base = base                  # the group's smooth SH absorbs this
    prim_res = residual                # per-primitive residual SH absorbs this
    rendered = np.clip(group_base + prim_res, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Hierarchical SH: appearance factorizes into group base "
                 "(coarse) + per-primitive residual (fine)", fontsize=12)
    for ax, img, title in zip(
        axes,
        [gt, group_base, rendered],
        ["GT appearance",
         "group base only  (one SH per group)",
         "group + per-prim residual"],
    ):
        im = ax.imshow(img, cmap='magma', vmin=0, vmax=1, origin='lower',
                       extent=(-1, 1, -1, 1))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04)
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
#  Figure E:  PGSR recovery (one-hot assignment, h = 0)
# =============================================================================
def fig_pgsr_recovery(out_path):
    """When pi is one-hot and h_ik = 0, each primitive is a planar disk on
    its group's plane = exactly a PGSR primitive.  Visualize: one group, a
    dozen primitives as disks tangent to it."""
    g = Groups(q=np.zeros((1, 3)),
               m=np.array([[0.0, 0.0, 1.0]]),
               rho=np.array([0.4]))

    fig = plt.figure(figsize=(13, 5.5))
    fig.suptitle("Recovery of PGSR:  when assignment is one-hot and "
                 "off-plane displacement h = 0, each primitive is exactly "
                 "a PGSR planar disk on its group's plane", fontsize=12)

    # Two panels: GMS view (group + primitives), PGSR view (just primitives)
    n_prim = 20
    np.random.seed(3)
    xi = 0.7 * np.random.randn(n_prim, 2) * 0.3
    h = np.zeros(n_prim)

    u_ax, v_ax = g.tangent_basis()
    prim_centers = g.q[0] + xi[:, 0:1] * u_ax[0] + xi[:, 1:2] * v_ax[0]
    prim_color = '#1f5fb4'
    s_disk = 0.12

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plot_group_planes(ax, g, alpha=0.18, color_cycle=['#888888'])
    for i in range(n_prim):
        theta = np.linspace(0, 2 * np.pi, 16)
        disk = (prim_centers[i]
                + s_disk * np.cos(theta)[:, None] * u_ax[0]
                + s_disk * np.sin(theta)[:, None] * v_ax[0])
        poly = Poly3DCollection([disk], alpha=0.85,
                                facecolor=prim_color, edgecolor='black',
                                linewidths=0.3)
        ax.add_collection3d(poly)
    style_ax(ax, "GMS view: one group (gray) + primitives on it (blue)")
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.4, 0.4)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=18, azim=-60)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    for i in range(n_prim):
        theta = np.linspace(0, 2 * np.pi, 16)
        disk = (prim_centers[i]
                + s_disk * np.cos(theta)[:, None] * u_ax[0]
                + s_disk * np.sin(theta)[:, None] * v_ax[0])
        poly = Poly3DCollection([disk], alpha=0.85,
                                facecolor=prim_color, edgecolor='black',
                                linewidths=0.3)
        ax.add_collection3d(poly)
    style_ax(ax, "PGSR view: the same primitives without the group abstraction")
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.4, 0.4)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=18, azim=-60)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
#  Main
# =============================================================================
def main():
    out_dir = "gms_demo_out"
    os.makedirs(out_dir, exist_ok=True)

    print("Figure A: setup ...")
    patches, pts = fig_setup(os.path.join(out_dir, "gms_a_setup.png"))

    print("Figure B: EM dynamics ...")
    fig_em_dynamics(os.path.join(out_dir, "gms_b_em.png"), patches, pts)

    print("Figure C: adaptive group population ...")
    fig_adaptive(os.path.join(out_dir, "gms_c_adaptive.png"), patches, pts)

    print("Figure D: hierarchical SH ...")
    fig_hier_sh(os.path.join(out_dir, "gms_d_hier_sh.png"))

    print("Figure E: PGSR recovery ...")
    fig_pgsr_recovery(os.path.join(out_dir, "gms_e_recovery.png"))

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()