#
# VPSS (Variational Per-primitive Surface Splatting)
# Extension of PGSR's GaussianModel with per-primitive variational posterior
# over surface offset along the normal axis.
#
# Each primitive carries q_i(h) = N(h; mu_i, sigma_i^2) — a 1D Gaussian
# over the signed offset h along its normal direction.  Training uses
# the reparameterisation trick to sample candidate surface positions
#     epsilon_i = mu_i + sigma_i * z_i,  z_i ~ N(0, 1)
# and renders the primitive at c_i + epsilon_i * n_i.
#
# Together with the standard image / depth-normal losses, an ELBO-style
# KL regulariser keeps sigma_i meaningful:
#     KL(q_i || p)  with  p = N(0, sigma_0^2)  (closed form for Gaussians)
#
# Two new optimisable parameters per primitive:
#     _mu        : (N,) float — posterior mean along normal axis
#     _log_sigma : (N,) float — log of posterior std along normal axis
#
# Decoupling note: c_i and (mu_i, sigma_i) have a residual gauge freedom
# along the normal direction.  The KL term anchors mu_i toward zero
# (penalty mu_i^2 / (2 sigma_0^2)), so long-range position drift is
# absorbed by c_i and only short-range refinement plus uncertainty live
# in (mu_i, sigma_i).  This is approach (A) — soft decoupling via KL.

import os
import numpy as np
import torch
import torch.nn as nn

from plyfile import PlyData, PlyElement

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class VPSSGaussianModel(GaussianModel):
    """PGSR GaussianModel + per-primitive variational posterior over surface
    offset along the primitive normal axis.

        q_i(h) = N(h; mu_i, sigma_i^2)
        candidate position: p_i = c_i + (mu_i + sigma_i * z_i) * n_i

    The normal n_i is taken to be the smallest-scale eigenvector of the
    primitive (PGSR convention, inherited from base class via
    get_smallest_axis).

    Two new parameters per primitive:
        _mu        : (N, 1)   posterior mean
        _log_sigma : (N, 1)   log posterior std (positivity via exp)

    The stochastic offset is applied through a wrapper method
    get_offset_xyz(training=True/False) — callers (renderer, depth-
    normal losses, etc.) should use this instead of get_xyz when VPSS
    is active.
    """

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)

        # VPSS state — populated in enable_vpss()
        self.use_vpss: bool = False
        self.sigma_0: float = 0.002    # prior std on h (units of scene length)
        self.beta_kl: float = 0.0      # current KL weight (annealed in training)
        self.training_mode: bool = True   # toggle stochastic vs deterministic
                                          # offset; set False during eval / save_ply

        # New parameters (None until enable_vpss is called)
        self._mu: nn.Parameter | None = None
        self._log_sigma: nn.Parameter | None = None

        # Cached sample of z per iteration — set by sample_z(), read by
        # get_offset_xyz() within the same iteration's forward pass.
        # None during evaluation (deterministic, eps = mu).
        self._z_cached: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    #  Enable / initialise VPSS                                           #
    # ------------------------------------------------------------------ #
    def enable_vpss(
        self,
        sigma_0: float = 0.002,
        mu_init_zero: bool = True,
    ) -> None:
        """Initialise mu_i and sigma_i for all existing primitives.

        Call AFTER create_from_pcd / load_ply (so we know N) and BEFORE
        training_setup (so the new parameters get into the optimiser).
        """
        if self.use_vpss:
            return
        self.use_vpss = True
        self.sigma_0 = float(sigma_0)
        N = self._xyz.shape[0]
        device = self._xyz.device

        if mu_init_zero:
            mu_init = torch.zeros((N, 1), device=device)
        else:
            mu_init = torch.zeros((N, 1), device=device)  # always zero init
        # log(sigma_0) so the initial posterior matches the prior (KL = 0)
        log_sigma_init = torch.full((N, 1), float(np.log(self.sigma_0)),
                                    device=device)

        self._mu = nn.Parameter(mu_init.requires_grad_(True))
        self._log_sigma = nn.Parameter(log_sigma_init.requires_grad_(True))

    # ------------------------------------------------------------------ #
    #  Training / eval mode                                                #
    # ------------------------------------------------------------------ #
    # The renderer calls gaussians.get_xyz to read positions.  When VPSS is
    # active, we want get_xyz to return the stochastic offset position
    # during training and the deterministic posterior-mean position during
    # eval / mesh extraction.
    #
    # train.py should set this flag explicitly:
    #     gaussians.training_mode = True  during the training loop
    #     gaussians.training_mode = False during eval / save_ply
    # If unset, default to True (stochastic).

    def train(self) -> None:
        """nn.Module-style train()."""
        self.training_mode = True

    def eval(self) -> None:
        """nn.Module-style eval()."""
        self.training_mode = False
        self._z_cached = None

    # ------------------------------------------------------------------ #
    #  get_xyz override — returns stochastic / deterministic offset       #
    # ------------------------------------------------------------------ #
    @property
    def get_xyz(self) -> torch.Tensor:
        """Returns world positions (with VPSS offset applied if active).

        Behaviour:
            VPSS disabled:                 return _xyz unchanged
            VPSS enabled, training_mode:   return _xyz + (mu + sigma*z) * n
            VPSS enabled, eval mode:       return _xyz + mu * n

        Gradients flow naturally to _xyz, _mu, _log_sigma through this.
        """
        if not self.use_vpss or self._mu is None:
            return self._xyz
        return self.get_offset_xyz(training=getattr(self, 'training_mode', True))

    # ------------------------------------------------------------------ #
    #  Accessors                                                          #
    # ------------------------------------------------------------------ #
    @property
    def get_mu(self) -> torch.Tensor:
        """Per-primitive posterior mean along normal axis. Shape (N, 1)."""
        if self._mu is None:
            return torch.zeros((self._xyz.shape[0], 1), device=self._xyz.device)
        return self._mu

    @property
    def get_log_sigma(self) -> torch.Tensor:
        if self._log_sigma is None:
            return torch.full((self._xyz.shape[0], 1),
                              float(np.log(self.sigma_0)),
                              device=self._xyz.device)
        return self._log_sigma

    @property
    def get_sigma(self) -> torch.Tensor:
        """Per-primitive posterior std. Shape (N, 1).  Always > 0."""
        return torch.exp(self.get_log_sigma)

    def get_normal_world(self) -> torch.Tensor:
        """Per-primitive world-space normal (smallest-axis convention).
        Shape (N, 3).  Differentiable through the rotation.
        """
        # Reuse PGSR's smallest-axis logic (defined in base class).
        # get_smallest_axis() returns the local-frame axis aligned with
        # smallest scale, applied through the rotation matrix.
        # That gives the world-frame normal.
        n = self.get_smallest_axis()
        # Ensure unit-norm in case
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
        return n

    # ------------------------------------------------------------------ #
    #  Stochastic offset sampling                                         #
    # ------------------------------------------------------------------ #
    def sample_z(self) -> None:
        """Sample the standard-normal noise for the reparameterisation
        trick. Called once per training iteration's forward pass; the
        sample is then read by get_offset_xyz() throughout the iteration."""
        if not self.use_vpss:
            return
        with torch.no_grad():
            self._z_cached = torch.randn_like(self._mu)

    def get_offset_xyz(self, training: bool = True) -> torch.Tensor:
        """Return per-primitive positions WITH the stochastic offset
        applied along the normal axis.

        Training:    p_i = c_i + (mu_i + sigma_i * z_i) * n_i,  z ~ N(0,1)
        Inference:   p_i = c_i + mu_i * n_i  (posterior mean, deterministic)

        Falls back to the raw base position when VPSS is disabled.
        Gradients flow through _xyz, mu (directly), sigma (via z) when training.

        IMPORTANT: uses self._xyz directly, NOT self.get_xyz (the property
        is overridden and would cause infinite recursion).
        """
        if not self.use_vpss or self._mu is None:
            return self._xyz                            # raw base position

        n = self.get_normal_world()                    # (N, 3), differentiable
        mu = self.get_mu                                # (N, 1)
        sigma = self.get_sigma                          # (N, 1)

        if training:
            # Resample z if absent OR if shape went stale (densification / prune
            # change N between the previous sample_z() and this call).
            if (self._z_cached is None
                    or self._z_cached.shape[0] != self._mu.shape[0]):
                self.sample_z()
            eps = mu + sigma * self._z_cached           # reparam trick
        else:
            eps = mu                                    # deterministic

        return self._xyz + eps * n                      # (N, 3)

    # ------------------------------------------------------------------ #
    #  ELBO regulariser                                                   #
    # ------------------------------------------------------------------ #
    def compute_kl_loss(self) -> torch.Tensor:
        """KL(q_i || p) summed over primitives.  Closed form for Gaussian
        q = N(mu, sigma^2) and Gaussian p = N(0, sigma_0^2):

            KL = (mu^2 + sigma^2) / (2 sigma_0^2)
                  - log(sigma / sigma_0)  -  1/2
        """
        if not self.use_vpss or self._mu is None:
            return torch.tensor(0.0, device='cuda')

        mu = self.get_mu.squeeze(-1)                       # (N,)
        log_sigma = self.get_log_sigma.squeeze(-1)         # (N,)
        sigma = torch.exp(log_sigma)                       # (N,)
        log_sigma_0 = float(np.log(self.sigma_0))
        sigma_0_sq = self.sigma_0 ** 2

        kl = (mu * mu + sigma * sigma) / (2.0 * sigma_0_sq) \
             - log_sigma + log_sigma_0 - 0.5
        return kl.sum()

    # ------------------------------------------------------------------ #
    #  Optimiser registration                                             #
    # ------------------------------------------------------------------ #
    def training_setup(self, training_args):
        """Override base to register _mu and _log_sigma."""
        # Base sets up _xyz / _knn_f / _features_dc / _features_rest /
        # _opacity / _scaling / _rotation in the Adam optimiser.
        super().training_setup(training_args)

        if not self.use_vpss or self._mu is None:
            return

        # Add VPSS parameters to the optimiser.
        # Learning rates:
        #   mu        ~ same scale as position_lr_init * spatial_lr_scale
        #               (mu is in scene-length units, like _xyz)
        #   log_sigma ~ slower (log-scale parameter, large updates cause issues)
        mu_lr = getattr(training_args, 'mu_lr',
                        training_args.position_lr_init * self.spatial_lr_scale * 0.5)
        log_sigma_lr = getattr(training_args, 'log_sigma_lr', 0.01)

        self.optimizer.add_param_group({
            'params': [self._mu], 'lr': mu_lr, 'name': 'mu',
        })
        self.optimizer.add_param_group({
            'params': [self._log_sigma], 'lr': log_sigma_lr, 'name': 'log_sigma',
        })

    # ------------------------------------------------------------------ #
    #  Densification — children inherit (mu, log_sigma) from parent       #
    # ------------------------------------------------------------------ #
    def densification_postfix(self, new_xyz, new_knn_f, new_features_dc,
                              new_features_rest, new_opacities, new_scaling,
                              new_rotation,
                              new_mu: torch.Tensor | None = None,
                              new_log_sigma: torch.Tensor | None = None):
        """Override base to also concatenate _mu and _log_sigma.

        The parent (base) method calls cat_tensors_to_optimizer with a dict
        that doesn't know about 'mu' / 'log_sigma'.  We extend it.

        Args:
            new_mu (Tensor | None):  if VPSS is on AND new_mu is None,
                children inherit parent's mu (passed in by densify_and_*).
            new_log_sigma similar.
        """
        if not self.use_vpss or self._mu is None:
            return super().densification_postfix(
                new_xyz, new_knn_f, new_features_dc, new_features_rest,
                new_opacities, new_scaling, new_rotation)

        # Standard tensors plus VPSS tensors
        d = {
            "xyz":      new_xyz,
            "knn_f":    new_knn_f,
            "f_dc":     new_features_dc,
            "f_rest":   new_features_rest,
            "opacity":  new_opacities,
            "scaling":  new_scaling,
            "rotation": new_rotation,
            "mu":       new_mu,
            "log_sigma": new_log_sigma,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz           = optimizable_tensors["xyz"]
        self._knn_f         = optimizable_tensors["knn_f"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]
        self._mu            = optimizable_tensors["mu"]
        self._log_sigma     = optimizable_tensors["log_sigma"]

        # Reset gradient accumulators
        N = self._xyz.shape[0]
        self.xyz_gradient_accum     = torch.zeros((N, 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((N, 1), device="cuda")
        self.denom                  = torch.zeros((N, 1), device="cuda")
        self.denom_abs              = torch.zeros((N, 1), device="cuda")
        self.max_radii2D            = torch.zeros((N,),    device="cuda")
        self.max_weight             = torch.zeros((N,),    device="cuda")

        # Invalidate any cached z (shapes changed)
        self._z_cached = None

    def densify_and_split(self, grads, grad_threshold, grads_abs,
                          grad_abs_threshold, scene_extent, max_radii2D, N=2):
        """When a primitive is split into N children, children are spawned
        at the parent's POSTERIOR MEAN position (c_i + mu_i * n_i), with
        mu reset to 0 and sigma inherited.  This is a clean implementation
        of "mu absorption at densification" — the parent's mu is folded
        into the children's _xyz, preventing accumulation.
        """
        if not self.use_vpss or self._mu is None:
            return super().densify_and_split(grads, grad_threshold, grads_abs,
                                             grad_abs_threshold, scene_extent,
                                             max_radii2D, N)

        # Selection logic mirrors base.
        n_init_points = self._xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grads_abs = torch.zeros((n_init_points), device="cuda")
        padded_grads_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points), device="cuda")
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold,
                                        True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            padded_grad[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(padded_grad, (1.0 - ratio))
            selected_pts_mask = torch.where(padded_grad > threshold,
                                            True, False)
        else:
            padded_grads_abs[selected_pts_mask] = 0
            mask = (torch.max(self.get_scaling, dim=1).values
                    > self.percent_dense * scene_extent) \
                   & (padded_max_radii2D > self.abs_split_radii2D_threshold)
            padded_grads_abs[~mask] = 0
            selected_pts_mask_abs = torch.where(
                padded_grads_abs >= grad_abs_threshold, True, False)
            limited_num = min(
                self.max_all_points - n_init_points - selected_pts_mask.sum(),
                self.max_abs_split_points)
            if selected_pts_mask_abs.sum() > limited_num:
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(padded_grads_abs, (1.0 - ratio))
                selected_pts_mask_abs = torch.where(
                    padded_grads_abs > threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask,
                                                 selected_pts_mask_abs)

        from utils.general_utils import build_rotation

        # Parent's posterior-mean world position = _xyz + mu * normal
        # (use _xyz directly, not get_xyz — we want the base position,
        # not the stochastic offset).
        with torch.no_grad():
            normals_world = self.get_normal_world().detach()  # (N, 3)
        parent_mean_pos = self._xyz[selected_pts_mask] \
            + self._mu[selected_pts_mask] * normals_world[selected_pts_mask]
        # (M, 3) where M = selected_pts_mask.sum()

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        # Children's _xyz = parent's posterior-mean position + sample noise.
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) \
                   + parent_mean_pos.repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation      = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc   = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity       = self._opacity[selected_pts_mask].repeat(N, 1)
        new_knn_f         = self._knn_f[selected_pts_mask].repeat(N, 1)

        # Children: mu reset to 0 (parent's mu absorbed into _xyz above);
        # log_sigma inherited from parent (uncertainty is intrinsic).
        new_mu        = torch.zeros((selected_pts_mask.sum() * N, 1),
                                     device='cuda')
        new_log_sigma = self._log_sigma[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_knn_f, new_features_dc,
                                    new_features_rest, new_opacity,
                                    new_scaling, new_rotation,
                                    new_mu=new_mu, new_log_sigma=new_log_sigma)

        prune_filter = torch.cat((selected_pts_mask,
                                  torch.zeros(N * selected_pts_mask.sum(),
                                              device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Children spawn at parent's posterior-mean position (c_i + mu_i n_i),
        with mu reset to 0 (absorbed) and log_sigma inherited."""
        if not self.use_vpss or self._mu is None:
            return super().densify_and_clone(grads, grad_threshold,
                                              scene_extent)

        n_init_points = self._xyz.shape[0]
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            grads_tmp = grads.squeeze().clone()
            grads_tmp[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(grads_tmp, (1.0 - ratio))
            selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

        if selected_pts_mask.sum() > 0:
            from utils.general_utils import build_rotation
            # Parent's posterior-mean position
            with torch.no_grad():
                normals_world = self.get_normal_world().detach()
            parent_mean_pos = self._xyz[selected_pts_mask] \
                + self._mu[selected_pts_mask] * normals_world[selected_pts_mask]

            stds = self.get_scaling[selected_pts_mask]
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) \
                       + parent_mean_pos
            new_features_dc   = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities     = self._opacity[selected_pts_mask]
            new_scaling       = self._scaling[selected_pts_mask]
            new_rotation      = self._rotation[selected_pts_mask]
            new_knn_f         = self._knn_f[selected_pts_mask]
            # mu reset to 0; log_sigma inherited
            new_mu        = torch.zeros((selected_pts_mask.sum(), 1),
                                         device='cuda')
            new_log_sigma = self._log_sigma[selected_pts_mask]

            self.densification_postfix(new_xyz, new_knn_f, new_features_dc,
                                        new_features_rest, new_opacities,
                                        new_scaling, new_rotation,
                                        new_mu=new_mu,
                                        new_log_sigma=new_log_sigma)

    # ------------------------------------------------------------------ #
    #  Pruning                                                             #
    # ------------------------------------------------------------------ #
    def prune_points(self, mask):
        """Override base to also reassign _mu and _log_sigma after pruning.

        The base prune_points calls _prune_optimizer (which DOES prune
        every param group, including 'mu' / 'log_sigma' once registered)
        and then hardcodes assignments to self._xyz / etc.  We extend
        the assignment list.
        """
        if not self.use_vpss or self._mu is None:
            return super().prune_points(mask)

        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz           = optimizable_tensors["xyz"]
        self._knn_f         = optimizable_tensors["knn_f"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity       = optimizable_tensors["opacity"]
        self._scaling       = optimizable_tensors["scaling"]
        self._rotation      = optimizable_tensors["rotation"]
        self._mu            = optimizable_tensors["mu"]
        self._log_sigma     = optimizable_tensors["log_sigma"]

        self.xyz_gradient_accum     = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.denom                  = self.denom[valid_points_mask]
        self.denom_abs              = self.denom_abs[valid_points_mask]
        self.max_radii2D            = self.max_radii2D[valid_points_mask]
        self.max_weight             = self.max_weight[valid_points_mask]
        self._z_cached              = None     # shapes changed

    # ------------------------------------------------------------------ #
    #  Save / load                                                        #
    # ------------------------------------------------------------------ #
    def construct_list_of_attributes(self):
        """Extend base attribute list with mu and log_sigma."""
        attrs = super().construct_list_of_attributes()
        if self.use_vpss and self._mu is not None:
            attrs.append('mu')
            attrs.append('log_sigma')
        return attrs

    def save_ply(self, path, mask=None):
        """Save with VPSS attributes appended to each vertex.

        IMPORTANT: when writing, mu_i is ABSORBED into xyz:
            saved_xyz   = c_i + mu_i * n_i      (= posterior-mean position)
            saved_mu    = 0                       (absorbed)
            saved_sigma = current sigma           (preserved)

        This makes the saved PLY self-contained for downstream tools that
        don't know about VPSS (e.g. render.py loading via base GaussianModel
        will read the correct surface positions directly).  Re-loading via
        VPSSGaussianModel and resuming training also works correctly: mu
        is reset to 0 but log_sigma is preserved, so the posterior over
        future offsets continues from the right uncertainty.

        In-memory state is NOT modified by this method.
        """
        if not self.use_vpss or self._mu is None:
            return super().save_ply(path, mask)

        prev_mode = self.training_mode
        self.training_mode = False                  # deterministic for save

        try:
            mkdir_p(os.path.dirname(path))

            # Compute absorbed positions (does not modify in-memory state).
            with torch.no_grad():
                n_world = self.get_normal_world()                # (N, 3)
                absorbed_xyz = self._xyz + self._mu * n_world    # (N, 3)
                # mu after absorption = 0
                absorbed_mu = torch.zeros_like(self._mu)         # (N, 1)

            xyz = absorbed_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(
                start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(
                start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()
            mu = absorbed_mu.detach().cpu().numpy()                 # zeros
            log_sigma = self._log_sigma.detach().cpu().numpy()      # preserved

            dtype_full = [(attribute, 'f4')
                          for attribute in self.construct_list_of_attributes()]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation,
                 mu, log_sigma), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)
        finally:
            self.training_mode = prev_mode             # restore

    def load_ply(self, path):
        """Load PGSR-format .ply, and VPSS params if present."""
        super().load_ply(path)

        # Try to read mu / log_sigma from the .ply.  If absent (loading a
        # vanilla PGSR file as the start of VPSS training), enable_vpss()
        # should be called by the user afterwards to initialise these.
        try:
            plydata = PlyData.read(path)
            v = plydata.elements[0]
            if 'mu' in v.data.dtype.names and 'log_sigma' in v.data.dtype.names:
                mu = np.asarray(v['mu'])[..., np.newaxis]
                log_sigma = np.asarray(v['log_sigma'])[..., np.newaxis]
                self._mu = nn.Parameter(
                    torch.tensor(mu, dtype=torch.float, device='cuda')
                    .requires_grad_(True))
                self._log_sigma = nn.Parameter(
                    torch.tensor(log_sigma, dtype=torch.float, device='cuda')
                    .requires_grad_(True))
                self.use_vpss = True
                # Note: sigma_0 is not in the .ply — must be re-passed via
                # enable_vpss() or set externally.
        except Exception as e:
            print(f"  VPSS load_ply: no mu/log_sigma in file (vanilla PGSR), "
                  f"call enable_vpss() to initialise.")

    # ------------------------------------------------------------------ #
    #  Capture / restore (checkpointing)                                  #
    # ------------------------------------------------------------------ #
    def capture(self):
        base = super().capture()
        # Append VPSS state at the end of the tuple
        return base + (self.use_vpss, self.sigma_0,
                       self._mu, self._log_sigma)

    def restore(self, model_args, training_args):
        # Split off VPSS state from the tail of the tuple
        n_base = 16  # number of fields the base captures (count of fields in base.capture())
        base_args = model_args[:n_base]
        if len(model_args) > n_base:
            self.use_vpss = model_args[n_base]
            self.sigma_0 = model_args[n_base + 1]
            self._mu = model_args[n_base + 2]
            self._log_sigma = model_args[n_base + 3]
        super().restore(base_args, training_args)