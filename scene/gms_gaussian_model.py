# scene/gms_gaussian_model.py
#
# Grouped Manifold Splatting -- Stage 2: hook GroupModel into the
# PGSR GaussianModel as a subclass.
#
# DESIGN: primitives are anchored to groups in position, normal, AND
# appearance.  A primitive's world position is NOT a free parameter; it
# is derived from per-(primitive, top-M-group) chart coordinates
# xi (N, M, 2), off-plane displacements h (N, M), top-M group indices
# (N, M), and the current group parameters.  See Eq. 6 of the method.
#
# We expose `get_xyz` as a derived buffer recomputed on demand.  This
# preserves the public API the rasterizer and TSDF extractor depend on,
# while making the underlying parameters chart-space rather than
# world-space.
#
# The PGSR densifier creates new primitives at new world positions.  We
# intercept that: when new world positions are produced, we project them
# onto the current groups to derive fresh chart coords for the new
# primitives.  The densifier doesn't need to know it's been swapped --
# `_xyz` looks like it's still being read and written, but underneath
# the chart-coord parameters are what actually carry the geometry.
#
# Assignment kernel: anisotropic Gaussian in perpendicular-distance and
# in-plane-distance.  A primitive belongs to a group iff it is BOTH on
# the group's plane (perpendicular distance < sigma_n) AND within the
# group's lateral extent (in-plane distance < rho_k).

import torch
import torch.nn as nn
import torch.nn.functional as F

from scene.gaussian_model import GaussianModel
from scene.group_model import GroupModel, _safe_unit, _tangent_basis


# =============================================================================
#  Anisotropic perpendicular + in-plane assignment kernel
# =============================================================================
@torch.no_grad()
def soft_assign_anisotropic(
    mu: torch.Tensor,
    group_model: GroupModel,
    top_m: int = 4,
    sigma_n: float = 0.05,
    chunk_size: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-M soft assignment using an anisotropic Gaussian kernel on
    (perpendicular distance to plane, in-plane distance to center).

    pi_ik proportional to exp(- (m_k . d_ik)^2 / (2 sigma_n^2)
                              - (u_k . d_ik)^2 / (2 rho_u_k^2)
                              - (v_k . d_ik)^2 / (2 rho_v_k^2))
    where d_ik = mu_i - q_k, and (u_k, v_k) are the Gram-Schmidt tangent
    basis built from m_k (cached in the group model).  rho_u, rho_v are
    per-group, per-axis in-plane scales.

    Memory-bounded by chunking over primitives.  Peak intermediate tensor
    is (chunk_size, K, 3), not (N, K, 3).

    Parameters
    ----------
    mu : (N, 3) tensor
    group_model : GroupModel
    top_m : int
    sigma_n : float
    chunk_size : int
        Primitives per chunk.

    Returns
    -------
    pi : (N, M) tensor
    idx : (N, M) long tensor
    """
    N = mu.shape[0]
    K = group_model.K
    assert K > 0, "Cannot compute soft assignment when K = 0"
    M = min(top_m, K)
    device = mu.device
    dtype = mu.dtype

    q = group_model.q                                  # (K, 3)
    m_k = group_model.m                                # (K, 3) unit
    rho_uv = group_model.rho                           # (K, 2): (rho_u, rho_v)
    # Ensure tangent basis is cached and use it directly to avoid
    # recomputing Gram-Schmidt per chunk
    if (getattr(group_model, '_u_cache', None) is None
            or group_model._u_cache.shape[0] != K):
        group_model._refresh_tangent_basis()
    u_k = group_model._u_cache                         # (K, 3)
    v_k = group_model._v_cache                         # (K, 3)
    inv_2sn2 = 1.0 / (2.0 * sigma_n ** 2 + 1e-12)
    inv_2rho_u2 = 1.0 / (2.0 * rho_uv[:, 0].pow(2) + 1e-12)  # (K,)
    inv_2rho_v2 = 1.0 / (2.0 * rho_uv[:, 1].pow(2) + 1e-12)  # (K,)

    pi_out = torch.empty(N, M, device=device, dtype=dtype)
    idx_out = torch.empty(N, M, device=device, dtype=torch.long)

    # Process in chunks to bound peak memory.
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        mu_chunk = mu[start:end]                                  # (C, 3)
        # (C, K, 3) -- this is the big tensor; chunked
        diff = mu_chunk.unsqueeze(1) - q.unsqueeze(0)
        # Project diff onto each group's (m, u, v) axes
        perp = (diff * m_k.unsqueeze(0)).sum(dim=-1)              # (C, K)
        d_u  = (diff * u_k.unsqueeze(0)).sum(dim=-1)              # (C, K)
        d_v  = (diff * v_k.unsqueeze(0)).sum(dim=-1)              # (C, K)
        del diff
        logits = -(perp.pow(2) * inv_2sn2
                   + d_u.pow(2) * inv_2rho_u2.unsqueeze(0)
                   + d_v.pow(2) * inv_2rho_v2.unsqueeze(0))
        del perp, d_u, d_v

        top_vals, top_idx = torch.topk(logits, k=M, dim=1)        # (C, M)
        del logits
        top_vals = top_vals - top_vals.max(dim=1, keepdim=True).values
        pi_chunk = top_vals.exp()
        pi_chunk = pi_chunk / pi_chunk.sum(dim=1, keepdim=True).clamp_min(1e-12)

        pi_out[start:end] = pi_chunk
        idx_out[start:end] = top_idx

    return pi_out, idx_out


# =============================================================================
#  GMSGaussianModel
# =============================================================================
class GMSGaussianModel(GaussianModel):
    """PGSR GaussianModel with group-anchored primitives.

    The primitive's position is no longer free.  Each primitive carries
    chart coordinates per top-M group; world position is derived via
    Eq. 6.  `_xyz` is retained as a non-Parameter buffer (computed and
    cached) so the rasterizer and densifier can read it through the
    inherited get_xyz property.
    """

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)

        # Group state
        self.use_groups: bool = False
        self.group_model: GroupModel | None = None
        self.top_m: int = 4
        self.sigma_n: float = 0.05            # perpendicular bandwidth
        self.normal_mode: str = 'rotation'    # 'group' or 'rotation'
        self.position_mode: str = 'group'     # 'group' or 'free'
        self.appearance_mode: str = 'group'   # 'group' or 'free'

        # Primary geometric parameters (set in `enable_groups`):
        #   _xi (N, top_m, 2)   chart coords per (primitive, candidate group)
        #   _h  (N, top_m)      off-plane displacements
        # These replace _xyz as the source of truth for primitive positions.
        self._xi: nn.Parameter | None = None
        self._h: nn.Parameter | None = None

        # Top-M indices (NOT a parameter - integer tensor, refreshed at E-step)
        self._topk_idx: torch.Tensor | None = None      # (N, top_m) long

        # Cached current assignment (recomputed at iteration boundaries)
        self._cached_pi: torch.Tensor | None = None     # (N, top_m)
        self._cached_iter: int = -1

        # Cached derived _xyz (recomputed each forward pass / on demand)
        # We override get_xyz to use this.  Note: parent's _xyz still exists
        # as a placeholder for densifier compatibility but is overwritten.
        self._xyz_cache: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    #  Initialization                                                     #
    # ------------------------------------------------------------------ #
    def enable_groups(
        self,
        K_init: int,
        top_m: int = 4,
        sigma_n: float = 0.05,
        rho_init_scale: float = 1.5,
        normal_mode: str = 'rotation',
        position_mode: str = 'group',
        appearance_mode: str = 'group',
    ) -> None:
        """Initialize the GroupModel from the current SfM-derived primitives
        and switch on group-anchored geometry.

        Call AFTER `create_from_pcd` and AFTER `training_setup`, because we
        read `_xyz` and `_features_dc` and we register new params with the
        existing optimizer.

        Parameters
        ----------
        K_init : int
            Number of initial groups.
        top_m : int
            Each primitive softly assigned to its top-M nearest groups.
        sigma_n : float
            Perpendicular-distance bandwidth for the assignment kernel.
        rho_init_scale : float
            Initial group in-plane scale = rho_init_scale * sqrt(PCA spread).
        normal_mode : {'group', 'rotation'}
            'group':    override get_normal with pi-weighted group normals.
            'rotation': PGSR's smallest-axis normal + soft alignment loss
                        (recommended; avoids rasterizer-frame decoupling).
        position_mode : {'group', 'free'}
            'group':    primitive position is derived from group chart coords
                        (Eq. 6 of the paper).  _xyz is non-trainable; _xi, _h
                        are the geometric parameters.
            'free':     position stays a free per-primitive parameter (vanilla
                        PGSR behavior).  Groups still inform normal and SH.
                        This ablation isolates the effect of position re-
                        parameterization.
        appearance_mode : {'group', 'free'}
            'group':    c_i = sum_k pi_ik c^grp_k + c^res_i (Eq. 8).
            'free':     c_i = c^res_i only (i.e., group SH disabled).  This
                        ablation isolates the effect of hierarchical SH.
        """
        assert self._xyz.numel() > 0, "Call create_from_pcd before enable_groups."
        assert normal_mode in ('group', 'rotation'), normal_mode
        assert position_mode in ('group', 'free'), position_mode
        assert appearance_mode in ('group', 'free'), appearance_mode
        device = self._xyz.device
        self.use_groups = True
        self.top_m = top_m
        self.sigma_n = sigma_n
        self.normal_mode = normal_mode
        self.position_mode = position_mode
        self.appearance_mode = appearance_mode

        # Build group model
        self.group_model = GroupModel(sh_degree=self.max_sh_degree, device=device)

        # K-means + PCA on the current primitive cloud.
        # Pass sh_init=None so group SH starts at ZERO (not the cluster mean
        # of SfM colors).  Combined with leaving _features_dc/rest untouched,
        # this means at iter 0 the rendered color is exactly the PGSR
        # SfM-derived per-primitive color (sh_grp_mix=0, residual=PGSR init).
        # Group SH then learns over training to absorb the coarse shared
        # appearance; the per-primitive residual handles position-dependent
        # high-frequency detail.
        self.group_model.init_from_kmeans(
            self._xyz.detach(),
            K_init=K_init,
            rho_init_scale=rho_init_scale,
            sh_init=None,
        )

        # Initialize per-primitive chart coords by projecting current _xyz
        # onto its top-M groups.
        self._refresh_topk_and_project(initial=True)

        # DO NOT zero _features_dc / _features_rest -- they remain the
        # PGSR-initialized per-primitive SH, which is our color at iter 0.

        # Register group params with optimizer.  Each gets its own param
        # group so we can scale LR appropriately (group params see gradient
        # contributions from many primitives, so LR is scaled down).
        position_lr = next(
            g['lr'] for g in self.optimizer.param_groups if g['name'] == 'xyz'
        )
        feature_lr = next(
            g['lr'] for g in self.optimizer.param_groups if g['name'] == 'f_dc'
        )

        gm = self.group_model
        self.optimizer.add_param_group(
            {'params': [gm._q], 'lr': 0.05 * position_lr, 'name': 'grp_q'}
        )
        self.optimizer.add_param_group(
            {'params': [gm._m], 'lr': 0.05 * position_lr, 'name': 'grp_m'}
        )
        self.optimizer.add_param_group(
            {'params': [gm._log_rho], 'lr': 0.05 * position_lr, 'name': 'grp_log_rho'}
        )
        self.optimizer.add_param_group(
            {'params': [gm._sh_grp], 'lr': 0.10 * feature_lr, 'name': 'grp_sh'}
        )

        # In 'group' position mode, the primitive's _xyz is derived from
        # group state, so we freeze it.  In 'free' mode, _xyz remains the
        # primary geometric parameter (PGSR behavior).
        if self.position_mode == 'group':
            self._xyz.requires_grad_(False)

        # Register _xi and _h as their own param groups, inheriting the
        # position_lr scale (they are the new geometric parameters in
        # 'group' position mode; harmless extras in 'free' mode).
        self.optimizer.add_param_group(
            {'params': [self._xi], 'lr': position_lr, 'name': 'xi'}
        )
        self.optimizer.add_param_group(
            {'params': [self._h], 'lr': position_lr, 'name': 'h'}
        )

        self._invalidate_cache()

    # ------------------------------------------------------------------ #
    #  Top-M + projection refresh                                         #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _refresh_topk_and_project(self, initial: bool = False) -> None:
        """Recompute top-M group indices for each primitive, and project
        the primitive's current world position onto each candidate group's
        frame to populate xi, h.

        Called:
          * once at `enable_groups`
          * after each E-step (group params changed)
          * after each densification event (primitive set changed)
        """
        N = self._xyz.shape[0]
        device = self._xyz.device

        # Use current world position (cached _xyz_cache if available, else _xyz)
        if self._xyz_cache is not None and self._xyz_cache.shape[0] == N:
            mu = self._xyz_cache.detach()
        else:
            mu = self._xyz.detach()

        pi, idx = soft_assign_anisotropic(
            mu, self.group_model, top_m=self.top_m, sigma_n=self.sigma_n
        )
        xi, h = self.group_model.project(mu, idx)       # (N, M, 2), (N, M)

        if initial:
            # Fresh allocation
            self._xi = nn.Parameter(xi.contiguous().requires_grad_(True))
            self._h = nn.Parameter(h.contiguous().requires_grad_(True))
            self._topk_idx = idx.contiguous()
        else:
            # Update buffers in place, preserving optimizer state shape.
            # Note: top-M index changes can shuffle which group each chart
            # entry refers to, so the Adam momentum on _xi / _h is no
            # longer meaningful for entries whose index changed.  We
            # selectively reset Adam state for those entries.
            self._reset_optimizer_state_at_changed_indices(idx)
            with torch.no_grad():
                self._xi.copy_(xi)
                self._h.copy_(h)
                self._topk_idx = idx.contiguous()

    @torch.no_grad()
    def _reset_optimizer_state_at_changed_indices(self, new_idx: torch.Tensor) -> None:
        """When top-M indices change for a primitive, the Adam momentum on
        its (xi, h) entries for the changed groups becomes meaningless.
        Zero out those entries' momentum.  This is the analog of how the
        E-step damped overwrite resets group Adam momentum.
        """
        if self._topk_idx is None:
            return
        # Per-(primitive, slot) mask of which slots changed
        changed = (self._topk_idx != new_idx)            # (N, M)
        for group in self.optimizer.param_groups:
            if group['name'] not in ('xi', 'h'):
                continue
            p = group['params'][0]
            state = self.optimizer.state.get(p, None)
            if state is None:
                continue
            # xi state has shape (N, M, 2); h state has shape (N, M)
            if 'exp_avg' in state and 'exp_avg_sq' in state:
                if state['exp_avg'].dim() == 3:
                    mask3 = changed.unsqueeze(-1)
                    state['exp_avg'][mask3.expand_as(state['exp_avg'])] = 0
                    state['exp_avg_sq'][mask3.expand_as(state['exp_avg_sq'])] = 0
                else:
                    state['exp_avg'][changed] = 0
                    state['exp_avg_sq'][changed] = 0

    # ------------------------------------------------------------------ #
    #  Cache invalidation                                                 #
    # ------------------------------------------------------------------ #
    def _invalidate_cache(self) -> None:
        self._cached_pi = None
        self._cached_iter = -1
        self._xyz_cache = None

    def _get_assignment(self) -> torch.Tensor:
        """Return current top-M assignment pi (N, M), detached.  Uses the
        cached _topk_idx (refreshed at E-step / densification only).

        For speed, uses the previous iteration's mu (cached in _xyz_cache)
        to compute pi.  Adam updates between iterations are small, so
        previous mu is a good estimate of current mu for assignment purposes.
        The first call after enable_groups / after invalidation falls back
        to a uniform-pi reconstruction of mu.
        """
        assert self.use_groups
        if self._cached_pi is not None and self._cached_pi.shape[0] == self._xi.shape[0]:
            return self._cached_pi

        with torch.no_grad():
            if (self._xyz_cache is not None
                    and self._xyz_cache.shape[0] == self._xi.shape[0]):
                # Use cached mu from previous iteration: free, accurate enough
                mu = self._xyz_cache
            else:
                # First call or post-densification: bootstrap with uniform-pi
                # reconstruction.  This is the expensive path, taken once.
                mu = self._reconstruct_mu_with_uniform_pi()
            pi, _ = soft_assign_anisotropic(
                mu, self.group_model, top_m=self.top_m, sigma_n=self.sigma_n
            )
        self._cached_pi = pi
        return pi

    @torch.no_grad()
    def _reconstruct_mu_with_uniform_pi(self) -> torch.Tensor:
        """Helper: compute mu using uniform pi over top-M.  Used only to
        bootstrap the pi cache on first call after invalidation; pi will
        be refined on subsequent calls."""
        N, M, _ = self._xi.shape
        uniform_pi = torch.full(
            (N, M), 1.0 / M, device=self._xi.device, dtype=self._xi.dtype
        )
        return self._reconstruct_mu(uniform_pi)

    def _reconstruct_mu(self, pi: torch.Tensor) -> torch.Tensor:
        """Compute mu from current (group state, xi, h, top-M idx, pi) via Eq. 6.
        This is differentiable in xi, h, and the group params.
        """
        attrs = self.group_model.gather_at(self._topk_idx)
        on_plane = (
            attrs['q']
            + self._xi[..., 0:1] * attrs['u']
            + self._xi[..., 1:2] * attrs['v']
        )                                                       # (N, M, 3)
        off_plane = self._h.unsqueeze(-1) * attrs['m']          # (N, M, 3)
        contrib = on_plane + off_plane
        return (pi.unsqueeze(-1) * contrib).sum(dim=1)          # (N, 3)

    # ------------------------------------------------------------------ #
    #  Overridden getters                                                 #
    # ------------------------------------------------------------------ #
    @property
    def get_xyz(self) -> torch.Tensor:
        """Derived world position from group state + chart coords + h + pi.

        Differentiable in xi, h, and group params.  pi is detached (treated
        as a fixed weight, refreshed only when _topk_idx is refreshed).

        Note: callers that only need the count should read self._xi.shape[0]
        directly to avoid triggering a full assignment+forward computation.
        """
        if not self.use_groups:
            return self._xyz

        # 'free' position mode: bypass group derivation, keep _xyz as the
        # primary parameter (vanilla PGSR).  Used for ablation.
        if self.position_mode == 'free':
            return self._xyz

        # Reuse the cache for non-grad callers (densifier, mesh extraction,
        # tensorboard).  In grad mode the cache is stale w.r.t. autograd's
        # leaves -- but it's still useful as the seed for next-iter's
        # _get_assignment, so we update it after computing mu.
        if (self._xyz_cache is not None
                and self._xyz_cache.shape[0] == self._xi.shape[0]
                and not torch.is_grad_enabled()):
            return self._xyz_cache

        pi = self._get_assignment()                       # (N, M)  detached
        mu = self._reconstruct_mu(pi)                     # (N, 3)
        # Always cache the detached value (cheap; reused next iteration as
        # the seed for the assignment bootstrap).
        self._xyz_cache = mu.detach()
        return mu

    def get_normal(self, view_cam):
        """Normal vector for each primitive.

        Two modes (controlled by self.normal_mode):
          'group'   -- group-mixed normal (default).  Each primitive's normal
                       is sum_k pi_ik m_k, normalized.  Multiple primitives
                       in the same group share the same normal direction.
                       This is the architectural choice from the paper.
          'rotation'-- PGSR's convention: smallest-axis of self._rotation,
                       with a soft alignment loss pulling rotation's smallest
                       axis toward the group-mixed direction.  Useful when
                       'group' mode causes rendered-normal noise (e.g., from
                       decoupling between rasterizer's frame and reported
                       normal).
        """
        if not self.use_groups:
            return super().get_normal(view_cam)

        if self.normal_mode == 'rotation':
            # Defer to PGSR's rotation-derived normal.  Group structure
            # influences this only via the soft alignment loss (see
            # get_group_aux_losses).
            return super().get_normal(view_cam)

        # 'group' mode: per-primitive normal is the pi-weighted group normal
        pi = self._get_assignment()
        attrs = self.group_model.gather_at(self._topk_idx)
        n = (pi.unsqueeze(-1) * attrs['m']).sum(dim=1)
        n = _safe_unit(n)
        # Camera-facing flip (PGSR convention)
        mu = self.get_xyz
        cam_dir = view_cam.camera_center - mu
        flip = (n * cam_dir).sum(dim=-1) < 0
        n = torch.where(flip.unsqueeze(-1), -n, n)
        return n

    @property
    def get_features(self) -> torch.Tensor:
        """c_i = sum_k pi_ik c^grp_k + c^res_i  (Eq. 8).

        Maintains the (N, num_basis, 3) layout PGSR expects.

        appearance_mode='free' bypasses the group SH contribution entirely,
        making c_i = c^res_i (which is just the standard PGSR per-primitive
        SH).  Used for ablation.
        """
        if not self.use_groups:
            return super().get_features

        # The residual SH (with PGSR-initialized values; never zeroed)
        residual_dc = self._features_dc                             # (N, 1, 3)
        residual_rest = self._features_rest                         # (N, B-1, 3)
        residual = torch.cat([residual_dc, residual_rest], dim=1)   # (N, B, 3)

        if self.appearance_mode == 'free':
            return residual

        pi = self._get_assignment()                       # (N, M)
        K = self.group_model.K
        num_basis = (self.max_sh_degree + 1) ** 2
        sh_grp = self.group_model._sh_grp.reshape(K, num_basis, 3)  # (K, B, 3)
        sh_top = sh_grp[self._topk_idx]                             # (N, M, B, 3)
        sh_mix = (pi.unsqueeze(-1).unsqueeze(-1) * sh_top).sum(dim=1)  # (N, B, 3)
        return sh_mix + residual

    # ------------------------------------------------------------------ #
    #  E-step                                                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def e_step_update(
        self,
        eta_geom: float = 0.3,
        eta_sh: float = 0.1,
        rho_min: float = 1e-3,
        rho_max: float = 10.0,
    ) -> None:
        """Closed-form damped weighted plane fit on currently-anchored
        primitives, then refresh per-primitive top-M indices and chart
        coords.

        Called every T_E iterations during training.
        """
        assert self.use_groups
        mu = self.get_xyz.detach()
        pi = self._get_assignment()

        # SH target: the rendered SH of each primitive (group mix + residual)
        with torch.no_grad():
            primitive_sh = self.get_features.detach()
            primitive_sh_flat = primitive_sh.reshape(primitive_sh.shape[0], -1)
            residual_sh = torch.cat([self._features_dc, self._features_rest], dim=1)
            residual_sh_flat = residual_sh.reshape(residual_sh.shape[0], -1)

        self.group_model.e_step(
            mu=mu,
            pi=pi,
            idx=self._topk_idx,
            eta_geom=eta_geom,
            eta_sh=eta_sh,
            rho_min=rho_min,
            rho_max=rho_max,
            sh_residual=residual_sh_flat,
            primitive_sh=primitive_sh_flat,
        )

        # Reset Adam momentum on group params (the damped overwrite makes
        # the previous momentum meaningless).
        self._reset_group_adam_state()

        # Refresh top-M and chart coords with updated group params.
        self._refresh_topk_and_project(initial=False)

        # Cache invalidated by the refresh; pi will be recomputed lazily.
        self._invalidate_cache()

    @torch.no_grad()
    def _reset_group_adam_state(self) -> None:
        """Zero the Adam exp_avg / exp_avg_sq for all group parameters."""
        for group in self.optimizer.param_groups:
            if not group['name'].startswith('grp_'):
                continue
            p = group['params'][0]
            state = self.optimizer.state.get(p, None)
            if state is not None:
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()

    # ------------------------------------------------------------------ #
    #  Densifier rewiring                                                 #
    # ------------------------------------------------------------------ #
    # We intercept the parent class's `densification_postfix` and `prune_points`
    # so that whenever new primitives are created at new world positions, we
    # populate their xi, h, _topk_idx by projecting onto current groups.

    def densification_postfix(self, new_xyz, new_knn_f, new_features_dc,
                              new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        """Override to avoid calling self.get_xyz six times (which would
        trigger six full soft-assignment computations).  Reads the primitive
        count from _xi after extension, which is O(1).
        """
        if not self.use_groups:
            return super().densification_postfix(
                new_xyz, new_knn_f, new_features_dc, new_features_rest,
                new_opacities, new_scaling, new_rotation,
            )
        d = {"xyz": new_xyz, "knn_f": new_knn_f, "f_dc": new_features_dc,
             "f_rest": new_features_rest, "opacity": new_opacities,
             "scaling": new_scaling, "rotation": new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Use _xi.shape[0] for the count -- it's already been extended by
        # cat_tensors_to_optimizer.  No assignment computation needed.
        N = self._xi.shape[0]
        device = self._xi.device
        self.xyz_gradient_accum = torch.zeros((N, 1), device=device)
        self.xyz_gradient_accum_abs = torch.zeros((N, 1), device=device)
        self.denom = torch.zeros((N, 1), device=device)
        self.denom_abs = torch.zeros((N, 1), device=device)
        self.max_radii2D = torch.zeros((N,), device=device)
        self.max_weight = torch.zeros((N,), device=device)

    def cat_tensors_to_optimizer(self, tensors_dict):
        """Override to handle xi and h alongside the PGSR-native params.

        The parent iterates over ALL optimizer param groups; when groups
        are enabled we additionally have xi, h, grp_q, grp_m, etc.  The
        PGSR-native densifier only knows about its own keys, so we need
        to:
          (a) extend only the PGSR-native params via the parent's logic
          (b) synthesize xi/h extensions from the new positions
          (c) skip the grp_* params (they are per-group, not per-primitive)
        """
        if not self.use_groups:
            return super().cat_tensors_to_optimizer(tensors_dict)

        # The parent's implementation assumes every param group's name is in
        # tensors_dict.  We bypass that by reimplementing here with a guard.
        PGSR_NATIVE = {'xyz', 'knn_f', 'f_dc', 'f_rest',
                       'opacity', 'scaling', 'rotation'}

        # Synthesize new xi, h from the new positions
        new_xyz = tensors_dict['xyz']                     # (M_new, 3)
        with torch.no_grad():
            new_pi, new_idx = soft_assign_anisotropic(
                new_xyz, self.group_model,
                top_m=self.top_m, sigma_n=self.sigma_n,
            )
            new_xi, new_h = self.group_model.project(new_xyz, new_idx)

        ext_full = dict(tensors_dict)
        ext_full['xi'] = new_xi
        ext_full['h'] = new_h

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group['name']
            # Skip group-level parameters (their count is K, not N)
            if name.startswith('grp_'):
                continue
            if name not in ext_full:
                # Shouldn't happen, but be defensive
                continue
            extension_tensor = ext_full[name]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat(
                    (stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state['exp_avg_sq'] = torch.cat(
                    (stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(
                    torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group['params'][0] = nn.Parameter(
                    torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True)
                )
            optimizable_tensors[name] = group['params'][0]

        # Update handles for xi, h, and the rest of the PGSR params is done
        # by parent's densification_postfix which we don't override; but it
        # only writes back to _xyz, _knn_f, etc.  We need to also update
        # _xi, _h, _topk_idx ourselves.
        self._xi = optimizable_tensors['xi']
        self._h = optimizable_tensors['h']
        self._topk_idx = torch.cat([self._topk_idx, new_idx], dim=0).contiguous()
        # Turn off requires_grad on _xyz only in 'group' position mode
        # (in 'free' mode, _xyz is still the primary geometric parameter).
        if self.position_mode == 'group':
            optimizable_tensors['xyz'].requires_grad_(False)

        self._invalidate_cache()
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """Override to handle xi, h alongside PGSR-native params, and to
        skip the grp_* per-group params (their count is K, not N).
        """
        if not self.use_groups:
            return super()._prune_optimizer(mask)

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group['name']
            if name.startswith('grp_'):
                # Group-level params -- not affected by primitive pruning
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(
                    group['params'][0][mask].requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group['params'][0] = nn.Parameter(
                    group['params'][0][mask].requires_grad_(True)
                )
            optimizable_tensors[name] = group['params'][0]

        # Update GMS-specific handles
        self._xi = optimizable_tensors['xi']
        self._h = optimizable_tensors['h']
        # Prune top-M indices buffer
        self._topk_idx = self._topk_idx[mask].contiguous()
        # Position derived only in 'group' mode; keep _xyz trainable in 'free'.
        if self.position_mode == 'group':
            optimizable_tensors['xyz'].requires_grad_(False)

        self._invalidate_cache()
        return optimizable_tensors

    # ------------------------------------------------------------------ #
    #  Adaptive group population                                          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def adaptive_group_population(
        self,
        current_iter: int,
        grad_threshold_percentile: float = 0.90,
        orphan_pi_threshold: float = 0.3,
        seed_radius_frac: float = 0.05,
        min_seed_count: int = 8,
        death_score_threshold: float = 0.9,
        T_grace: int = 2000,
    ) -> tuple[int, int]:
        """Run one round of adaptive group control.  Should be called at
        the same cadence as densify_and_prune.

        Returns
        -------
        (n_born, n_died) : int, int
        """
        if not self.use_groups:
            return 0, 0

        gm = self.group_model
        N = self._xi.shape[0]

        # ----- BIRTH: image-evidence-driven seed clustering ------------
        # Identify orphan primitives:
        #   (a) high positional gradient (image is pushing them)
        #   (b) low max(pi) (no group strongly claims them)
        g_i = self.xyz_gradient_accum.squeeze(-1) / self.denom.squeeze(-1).clamp_min(1)
        g_i = torch.nan_to_num(g_i, nan=0.0)
        if g_i.numel() == 0:
            return 0, 0

        # Percentile threshold for high gradient
        grad_thr = torch.quantile(g_i, grad_threshold_percentile)
        high_grad_mask = g_i >= grad_thr

        # Low max(pi)
        pi = self._get_assignment()
        max_pi = pi.max(dim=1).values
        weak_assn_mask = max_pi < orphan_pi_threshold

        orphan_mask = high_grad_mask & weak_assn_mask
        orphan_idx = torch.where(orphan_mask)[0]

        n_born = 0
        if orphan_idx.numel() >= min_seed_count:
            mu = self.get_xyz.detach()
            orphan_pts = mu[orphan_idx]

            # Spatial clustering: union-find with radius = seed_radius_frac
            # times the median group rho (scene-scale heuristic)
            # Use overall median of rho (across groups AND both u,v axes)
            # for a scene-scale heuristic.
            radius = seed_radius_frac * gm.rho.median().item()
            n_born = self._birth_from_orphan_positions(
                orphan_pts, radius=radius,
                min_count=min_seed_count, current_iter=current_iter,
            )

        # ----- DEATH: joint mass + gradient evidence -------------------
        death_scores = gm.death_score()
        in_grace = (current_iter - gm.birth_iter.to(self._xi.device)) < T_grace
        kill_mask = (death_scores > death_score_threshold) & ~in_grace

        # Refuse to kill below K = top_m + 1 so soft assignment stays
        # meaningful (each primitive needs at least top_m + 1 candidates).
        # Also keep at least 3 groups regardless.
        min_K = max(self.top_m + 1, 3)
        n_kill_max = max(gm.K - min_K, 0)
        if int(kill_mask.sum().item()) > n_kill_max:
            scores_sorted, sort_idx = death_scores.sort(descending=True)
            kill_mask = torch.zeros_like(kill_mask)
            if n_kill_max > 0:
                kill_idx = sort_idx[:n_kill_max]
                grace_ok = ~in_grace[kill_idx]
                kill_mask[kill_idx[grace_ok]] = True

        n_died = 0
        if kill_mask.any():
            keep_mask = ~kill_mask
            n_died = self._death_with_topk_repair(keep_mask)

        # Reset accumulators after using them
        gm.reset_stats()

        # If anything changed, refresh top-M (group K changed)
        if n_born > 0 or n_died > 0:
            self._refresh_topk_and_project(initial=False)
            self._invalidate_cache()

        return n_born, n_died

    @torch.no_grad()
    def _birth_from_orphan_positions(
        self,
        orphan_pts: torch.Tensor,
        radius: float,
        min_count: int,
        current_iter: int,
    ) -> int:
        """Cluster orphan positions by union-find within `radius`; each
        cluster of size >= min_count spawns one new group, parameters
        fitted by PCA on the cluster points.  Returns number of new groups.
        """
        n = orphan_pts.shape[0]
        if n < min_count:
            return 0

        # Pairwise distances (n^2, fine for typical n ~ a few hundred)
        if n > 5000:
            # Heuristic: subsample if too many orphans
            perm = torch.randperm(n, device=orphan_pts.device)[:5000]
            orphan_pts = orphan_pts[perm]
            n = orphan_pts.shape[0]

        d2 = torch.cdist(orphan_pts, orphan_pts, p=2).pow(2)
        adj = d2 < (radius * radius)
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
        adj_cpu = adj.cpu().numpy()
        for i in range(n):
            for j in range(i + 1, n):
                if adj_cpu[i, j]:
                    union(i, j)
        # Group by root
        roots = [find(i) for i in range(n)]
        cluster_map: dict[int, list[int]] = {}
        for i, r in enumerate(roots):
            cluster_map.setdefault(r, []).append(i)

        n_born = 0
        for members in cluster_map.values():
            if len(members) < min_count:
                continue
            cluster_pts = orphan_pts[torch.tensor(members, device=orphan_pts.device)]
            # Spawn a new group via the GroupModel's birth helper
            added = self.group_model.birth(
                seed_positions=cluster_pts,
                seed_normals=None,
                seed_sh=None,
                min_seeds=min_count,
                current_iter=current_iter,
            )
            if added > 0:
                # Newly-added group parameter is NOT yet in the optimizer.
                # Register it.
                self._register_new_group_params_with_optimizer(added)
                n_born += added
        return n_born

    @torch.no_grad()
    def _register_new_group_params_with_optimizer(self, n_added: int) -> None:
        """After group_model.birth() rebuilds the underlying parameters
        (it had to, since nn.Parameters are immutable in size), the
        optimizer's references are stale.  We rebuild each grp_* param
        group in place, preserving Adam state for the existing entries
        and initializing zeros for the new ones.
        """
        gm = self.group_model
        param_to_attr = {
            'grp_q': gm._q,
            'grp_m': gm._m,
            'grp_log_rho': gm._log_rho,
            'grp_sh': gm._sh_grp,
        }
        for group in self.optimizer.param_groups:
            if group['name'] not in param_to_attr:
                continue
            new_param = param_to_attr[group['name']]
            old_param = group['params'][0]
            old_state = self.optimizer.state.get(old_param, None)
            if old_state is not None:
                # Pad Adam state with zeros for the new entries
                K_old = old_param.shape[0]
                K_new = new_param.shape[0]
                pad_shape = list(old_state['exp_avg'].shape)
                pad_shape[0] = K_new - K_old
                pad_zero = torch.zeros(*pad_shape, device=new_param.device,
                                       dtype=new_param.dtype)
                old_state['exp_avg'] = torch.cat([old_state['exp_avg'], pad_zero], dim=0)
                old_state['exp_avg_sq'] = torch.cat([old_state['exp_avg_sq'], pad_zero], dim=0)
                del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = old_state
            group['params'][0] = new_param

    @torch.no_grad()
    def _death_with_topk_repair(self, keep_mask: torch.Tensor) -> int:
        """Remove groups for which keep_mask is False.  Updates the
        optimizer's references to grp_* parameters and removes their
        Adam state for the killed entries.

        After this call, primitive _topk_idx may point at deleted groups;
        the caller is responsible for invoking _refresh_topk_and_project.
        """
        n_killed = int((~keep_mask).sum().item())
        if n_killed == 0:
            return 0
        gm = self.group_model

        # Prune optimizer state for each grp_* param group BEFORE the
        # underlying parameters are recreated by gm.death()
        param_to_attr_old = {
            'grp_q': gm._q,
            'grp_m': gm._m,
            'grp_log_rho': gm._log_rho,
            'grp_sh': gm._sh_grp,
        }
        old_state = {}
        for group in self.optimizer.param_groups:
            if group['name'] not in param_to_attr_old:
                continue
            p = group['params'][0]
            state = self.optimizer.state.get(p, None)
            if state is not None:
                old_state[group['name']] = {
                    'exp_avg': state['exp_avg'][keep_mask],
                    'exp_avg_sq': state['exp_avg_sq'][keep_mask],
                    'step': state.get('step', None),
                }
                del self.optimizer.state[p]

        # Now actually delete the groups (rebuilds gm._q, gm._m, etc.)
        gm.death(keep_mask)

        # Re-attach new parameter references to the optimizer
        new_params = {
            'grp_q': gm._q,
            'grp_m': gm._m,
            'grp_log_rho': gm._log_rho,
            'grp_sh': gm._sh_grp,
        }
        for group in self.optimizer.param_groups:
            if group['name'] not in new_params:
                continue
            new_p = new_params[group['name']]
            group['params'][0] = new_p
            if group['name'] in old_state:
                st = old_state[group['name']]
                if st['step'] is not None:
                    st_dict = {'exp_avg': st['exp_avg'],
                               'exp_avg_sq': st['exp_avg_sq'],
                               'step': st['step']}
                else:
                    st_dict = {'exp_avg': st['exp_avg'],
                               'exp_avg_sq': st['exp_avg_sq']}
                self.optimizer.state[new_p] = st_dict

        return n_killed

    # ------------------------------------------------------------------ #
    #  Loss terms                                                         #
    # ------------------------------------------------------------------ #
    def get_group_aux_losses(
        self,
        lambda_h: float = 1e-3,
        lambda_r: float = 1e-3,
        lambda_d: float = 1e-2,
        lambda_rho: float = 1e-4,
        lambda_align: float = 0.0,
        lambda_plane: float = 0.0,
    ) -> torch.Tensor:
        """Compute the auxiliary GMS loss terms.

        Args:
          lambda_h     : weight on per-primitive off-plane displacement h
                         (only active when position_mode='group')
          lambda_r     : weight on per-primitive residual SH magnitude
          lambda_d     : weight on the joint mass+evidence death prior
          lambda_rho   : weight on the rho floor
          lambda_align : weight on the normal-alignment loss (only meaningful
                         when normal_mode='rotation')
          lambda_plane : weight on the position-plane soft prior:
                         sum_i sum_k pi_ik * (m_k . (mu_i - q_k))^2 / N
                         Pulls each primitive toward its group's plane in
                         position.  Active in any position mode; in 'free'
                         mode it gives groups a direct geometric influence
                         on primitive positions that the alignment loss
                         alone does not provide.

        Returns a scalar loss to add to L_PGSR.
        """
        if not self.use_groups:
            return torch.zeros((), device=self._xyz.device)

        pi = self._get_assignment()
        # Off-plane displacement prior
        loss_h = (pi * self._h.pow(2)).sum() / max(1, self._xi.shape[0])

        # Residual SH prior
        loss_r = (self._features_dc.pow(2).sum()
                  + self._features_rest.pow(2).sum()) / max(1, self._features_dc.shape[0])

        # Joint mass+evidence death prior
        gm = self.group_model
        G, M = gm.get_stats()
        N_est = M.sum().clamp_min(1.0)
        M0 = (N_est / (2.0 * gm.K)).clamp_min(1.0)
        G0 = G.median().clamp_min(1e-6)
        loss_d = torch.exp(-(M / M0) - (G / G0)).sum() / gm.K

        # Rho floor
        loss_rho = (1.0 / gm.rho.pow(2)).sum() / gm.K

        total = (lambda_h * loss_h
                 + lambda_r * loss_r
                 + lambda_d * loss_d
                 + lambda_rho * loss_rho)

        # Optional alignment loss (used when normal_mode='rotation')
        if lambda_align > 0:
            # Smallest-scale axis of the per-primitive rotation
            rotmat = self.get_rotation_matrix()                  # (N, 3, 3)
            smallest_axis_idx = self.get_scaling.min(dim=-1)[1]  # (N,)
            # Gather the column of rotmat corresponding to the smallest scale
            sa_idx = smallest_axis_idx[..., None, None].expand(-1, 3, -1)
            smallest_axis = rotmat.gather(2, sa_idx).squeeze(-1)  # (N, 3)
            # Group-mixed normal (no camera flip; alignment is signless)
            attrs = self.group_model.gather_at(self._topk_idx)
            n_grp = (pi.unsqueeze(-1) * attrs['m']).sum(dim=1)
            n_grp = _safe_unit(n_grp)
            # Signless alignment: 1 - (a . n)^2
            cos_sq = (smallest_axis * n_grp).sum(dim=-1).pow(2)
            loss_align = (1.0 - cos_sq).mean()
            total = total + lambda_align * loss_align

        # Optional plane-position prior:
        # Pull each primitive toward its assigned group's plane.
        # L_plane = (1/N) * sum_i sum_k pi_ik * (m_k . (mu_i - q_k))^2
        # This gives groups a direct geometric influence on primitive
        # positions (not just normals).  Differentiable in mu via the
        # gradient path back to the underlying position parameters.
        if lambda_plane > 0:
            attrs = self.group_model.gather_at(self._topk_idx)   # q (N,M,3), m (N,M,3)
            mu = self.get_xyz                                    # (N, 3)
            # Perpendicular displacement of each primitive from each top-M group
            diff = mu.unsqueeze(1) - attrs['q']                  # (N, M, 3)
            perp = (diff * attrs['m']).sum(dim=-1)               # (N, M)
            loss_plane = (pi * perp.pow(2)).sum() / max(1, mu.shape[0])
            total = total + lambda_plane * loss_plane

        return total

    # ------------------------------------------------------------------ #
    #  Save_ply bake -- write GMS-derived values into PGSR-native slots   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def save_ply(self, path, mask=None):
        """Save a PGSR-compatible .ply with GMS-derived values baked in.

        At training time, get_xyz / get_normal / get_features depend on the
        active modes:
          * position_mode='group': _xyz is derived from chart coords.  Bake
              _xyz <- get_xyz (and bake _rotation so the smallest axis is the
              group normal, since rotation was not trained against image
              loss in this mode).
          * position_mode='free': _xyz is the primary parameter; nothing to
              bake for geometry.  _rotation was trained against image loss
              (via PGSR's get_normal) -- baking it would corrupt the mesh.
          * appearance_mode='group': features are (group_SH_mix + residual);
              bake the sum into _features_dc/rest.
          * appearance_mode='free': features are already the per-primitive
              SH -- no appearance bake needed.

        After this bake, vanilla PGSR's render.py / TSDF / DTU eval all
        operate correctly without knowing about groups.
        """
        if not self.use_groups:
            return super().save_ply(path, mask)

        # Snapshot the originals
        orig_xyz = self._xyz
        orig_fdc = self._features_dc
        orig_frest = self._features_rest
        orig_rot = self._rotation
        orig_scale = self._scaling

        # ---- Geometry bake (only when position_mode='group') ----
        if self.position_mode == 'group':
            mu = self.get_xyz                                  # (N, 3)
            # Build a rotation whose 3rd column = group-mixed normal.
            # Only do this when position is also group-derived; otherwise
            # _rotation was trained directly by the image loss and overwriting
            # it would corrupt the rendered depth maps.
            pi = self._get_assignment()
            attrs = self.group_model.gather_at(self._topk_idx)
            n_world = (pi.unsqueeze(-1) * attrs['m']).sum(dim=1)
            n_world = _safe_unit(n_world)
            N = n_world.shape[0]
            ref = torch.zeros(N, 3, device=n_world.device, dtype=n_world.dtype)
            ref[:, 0] = 1.0
            too_parallel = (n_world[:, 0].abs() > 0.9).unsqueeze(-1)
            ref_alt = torch.zeros_like(ref); ref_alt[:, 1] = 1.0
            ref = torch.where(too_parallel, ref_alt, ref)
            u = ref - (ref * n_world).sum(dim=-1, keepdim=True) * n_world
            u = _safe_unit(u)
            v = torch.linalg.cross(n_world, u, dim=-1)
            R = torch.stack([u, v, n_world], dim=-1)
            q_baked = _rotmat_to_quaternion(R)
            scale_baked = orig_scale.detach().clone()
            min_s01 = torch.minimum(scale_baked[:, 0], scale_baked[:, 1]) - 1.5
            scale_baked[:, 2] = min_s01
            self._xyz = nn.Parameter(mu.contiguous(), requires_grad=False)
            self._rotation = nn.Parameter(q_baked.contiguous(), requires_grad=False)
            self._scaling = nn.Parameter(scale_baked.contiguous(), requires_grad=False)

        # ---- Appearance bake (only when appearance_mode='group') ----
        if self.appearance_mode == 'group':
            feats = self.get_features                          # (N, B, 3)
            feats_dc = feats[:, 0:1, :]                        # (N, 1, 3)
            feats_rest = feats[:, 1:, :]                       # (N, B-1, 3)
            self._features_dc = nn.Parameter(
                feats_dc.contiguous(), requires_grad=False
            )
            self._features_rest = nn.Parameter(
                feats_rest.contiguous(), requires_grad=False
            )

        try:
            super().save_ply(path, mask)
        finally:
            # ALWAYS restore originals so training can continue if not at end
            self._xyz = orig_xyz
            self._features_dc = orig_fdc
            self._features_rest = orig_frest
            self._rotation = orig_rot
            self._scaling = orig_scale
            self._invalidate_cache()


def _rotmat_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert (N, 3, 3) rotation matrices to (N, 4) quaternions in
    real-first (r, i, j, k) order.  Standard Shepperd's method.
    """
    N = R.shape[0]
    device = R.device
    dtype = R.dtype

    m00 = R[:, 0, 0]; m01 = R[:, 0, 1]; m02 = R[:, 0, 2]
    m10 = R[:, 1, 0]; m11 = R[:, 1, 1]; m12 = R[:, 1, 2]
    m20 = R[:, 2, 0]; m21 = R[:, 2, 1]; m22 = R[:, 2, 2]

    tr = m00 + m11 + m22
    q = torch.zeros(N, 4, device=device, dtype=dtype)

    # Case 1: tr > 0
    mask1 = tr > 0
    S = torch.sqrt(tr[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * S
    q[mask1, 1] = (m21[mask1] - m12[mask1]) / S
    q[mask1, 2] = (m02[mask1] - m20[mask1]) / S
    q[mask1, 3] = (m10[mask1] - m01[mask1]) / S

    # Case 2: m00 is largest diagonal
    rest = ~mask1
    mask2 = rest & (m00 > m11) & (m00 > m22)
    S = torch.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2]) * 2
    q[mask2, 0] = (m21[mask2] - m12[mask2]) / S
    q[mask2, 1] = 0.25 * S
    q[mask2, 2] = (m01[mask2] + m10[mask2]) / S
    q[mask2, 3] = (m02[mask2] + m20[mask2]) / S

    # Case 3: m11 is largest diagonal
    mask3 = rest & ~mask2 & (m11 > m22)
    S = torch.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3]) * 2
    q[mask3, 0] = (m02[mask3] - m20[mask3]) / S
    q[mask3, 1] = (m01[mask3] + m10[mask3]) / S
    q[mask3, 2] = 0.25 * S
    q[mask3, 3] = (m12[mask3] + m21[mask3]) / S

    # Case 4: m22 is largest diagonal
    mask4 = rest & ~mask2 & ~mask3
    S = torch.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4]) * 2
    q[mask4, 0] = (m10[mask4] - m01[mask4]) / S
    q[mask4, 1] = (m02[mask4] + m20[mask4]) / S
    q[mask4, 2] = (m12[mask4] + m21[mask4]) / S
    q[mask4, 3] = 0.25 * S

    # Normalize (numerical safety)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


# =============================================================================
#  Quick smoke test  (run with:  python -m scene.gms_gaussian_model)
# =============================================================================
def _smoke_test():
    """End-to-end sanity check.  Doesn't run the full PGSR pipeline (no
    rasterizer here), just exercises the model creation, enable_groups,
    E-step, densification interception, and the loss API.
    """
    import sys
    # Provide a stub `simple_knn._C.distCUDA2` if not available
    try:
        from simple_knn._C import distCUDA2
    except ImportError:
        import simple_knn._C
        def distCUDA2(pts):
            # crude per-point nearest-neighbor sq dist
            d = torch.cdist(pts, pts)
            d.fill_diagonal_(float('inf'))
            return d.min(dim=1).values.pow(2)
        simple_knn._C.distCUDA2 = distCUDA2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[gms_gaussian_model smoke test] device = {device}')

    # Synthesize an SfM-like point cloud on 3 planes
    torch.manual_seed(0)
    pts_list = []
    centers = [(0., 0., 0.), (0., 1., 0.5), (1., 0., 0.5)]
    for c in centers:
        c_t = torch.tensor(c)
        a = (torch.rand(150, 1) - 0.5) * 2
        b = (torch.rand(150, 1) - 0.5) * 2
        u = torch.tensor([1., 0., 0.])
        v = torch.tensor([0., 1., 0.]) if c == centers[0] else torch.tensor([0., 0., 1.])
        pts_list.append(c_t + a * u + b * v + 0.01 * torch.randn(150, 3))
    pts = torch.cat(pts_list).to(device)
    colors = torch.rand(pts.shape[0], 3).to(device)

    # Build a fake BasicPointCloud-like object
    class _BasicPointCloud:
        def __init__(self, points, colors):
            self.points = points.cpu().numpy()
            self.colors = colors.cpu().numpy()
            self.normals = None
    pcd = _BasicPointCloud(pts, colors)

    # Fake training args -- a minimal subset of what create_from_pcd +
    # training_setup access.
    class TrainingArgs:
        position_lr_init = 1e-4
        position_lr_final = 1e-6
        position_lr_delay_mult = 1.0
        position_lr_max_steps = 30000
        feature_lr = 2.5e-3
        opacity_lr = 5e-2
        scaling_lr = 5e-3
        rotation_lr = 1e-3
        percent_dense = 0.01
        abs_split_radii2D_threshold = 20
        max_abs_split_points = 50000
        max_all_points = 1000000

    model = GMSGaussianModel(sh_degree=3)
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.training_setup(TrainingArgs())
    print(f'  [1/4] PGSR model created with {model._xyz.shape[0]} primitives')

    # Enable groups
    model.enable_groups(K_init=5, top_m=4, sigma_n=0.05)
    print(f'  [2/4] groups enabled: K = {model.group_model.K}, '
          f'xi shape = {tuple(model._xi.shape)}, h shape = {tuple(model._h.shape)}')

    # Check derived position is sensible (should be very close to the
    # original PGSR _xyz at initialization, since we projected onto the
    # nearest group and used h = (perpendicular distance) which exactly
    # reconstructs mu)
    mu_derived = model.get_xyz
    mu_init = pts
    err = (mu_derived - mu_init).norm(dim=-1).mean()
    print(f'  [3/4] derived mu vs SfM points: mean err = {err.item():.2e}')
    assert err.item() < 1e-3, f'reconstruction error too high: {err.item()}'

    # Run an E-step
    model.e_step_update()
    mu_after = model.get_xyz
    err_after = (mu_after - mu_init).norm(dim=-1).mean()
    print(f'  [4/4] after E-step: mean err vs SfM = {err_after.item():.2e}')

    # Aux loss should be a scalar
    loss = model.get_group_aux_losses()
    print(f'        aux loss = {loss.item():.4e}')

    print('[gms_gaussian_model smoke test] OK')


if __name__ == '__main__':
    _smoke_test()