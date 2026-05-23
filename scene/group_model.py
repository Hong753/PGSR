# scene/group_model.py
#
# Grouped Manifold Splatting -- Stage 1: the GroupModel class.
#
# A standalone module that owns the latent surface-group state:
#     q   in R^{K x 3}     -- group centers
#     m   in S^2 (R^{K x 3})  -- group normals (unit length, renormalized each step)
#     rho in R^K_+         -- group spatial scales
#     sh_grp in R^{K x C}  -- group spherical-harmonic coefficients (DC + rest)
#
# Provides:
#     forward      : given primitive positions mu, return top-M soft assignment
#                    pi (N, M) and the top-M group indices (N, M)
#     e_step       : closed-form weighted plane fit per group (damped overwrite)
#     refresh_top_m: recompute top-M indices for each primitive
#     birth        : add new groups initialized from a cluster of orphan
#                    primitive positions + normals
#     death        : remove groups whose joint mass+gradient floor is too low
#     project      : given mu, recover per-(primitive, top-M-group) chart
#                    coords xi (N, M, 2) and off-plane displacements h (N, M)
#
# This module is intentionally independent of the GaussianModel.  Stage 2
# will hook it into GaussianModel's get_xyz / get_normal / get_features.
#
# All tensors live on CUDA by default.  Float32 throughout.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_unit(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """L2-normalize along last dim with a numerical floor."""
    n = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    return v / n


def _tangent_basis(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row m_k in (K, 3), return (u_k, v_k) -- two orthonormal vectors
    spanning the tangent plane.  Uses a deterministic Gram-Schmidt against a
    fixed reference, switching the reference axis if m is too close to it."""
    K = m.shape[0]
    ref = torch.zeros(K, 3, device=m.device, dtype=m.dtype)
    ref[:, 0] = 1.0
    too_parallel = (m[:, 0].abs() > 0.9).unsqueeze(-1)
    ref_alt = torch.zeros_like(ref)
    ref_alt[:, 1] = 1.0
    ref = torch.where(too_parallel, ref_alt, ref)

    proj = (ref * m).sum(dim=-1, keepdim=True) * m
    u = _safe_unit(ref - proj)
    v = torch.linalg.cross(m, u, dim=-1)
    return u, v


class GroupModel:
    """Latent surface-group state for GMS.

    Stores group parameters as nn.Parameters so they can be optimized by
    an Adam optimizer (alongside primitive parameters in GaussianModel),
    but the E-step overwrites these closed-form (with damping) rather than
    leaving them purely to gradient descent.

    Parameters
    ----------
    sh_degree : int
        Maximum SH degree (matches GaussianModel.max_sh_degree).
    device : torch.device or str
        Device for all tensors.  Defaults to CUDA if available.
    """

    def __init__(self, sh_degree: int = 3, device: str = "cuda"):
        self.sh_degree = sh_degree
        self.sh_channels = 3 * (sh_degree + 1) ** 2     # RGB x (L+1)^2
        self.device = device

        # Parameters are created in `init_from_kmeans`.  Placeholder until then.
        self._q: nn.Parameter | None = None             # (K, 3)
        self._m: nn.Parameter | None = None             # (K, 3) -- unit
        self._log_rho: nn.Parameter | None = None       # (K,)   -- log for positivity
        self._sh_grp: nn.Parameter | None = None        # (K, sh_channels)

        # Running statistics for adaptive control (updated by user via accumulate())
        self.grad_accum: torch.Tensor | None = None     # (K,)
        self.mass_accum: torch.Tensor | None = None     # (K,)
        self._accum_count: int = 0

        # Per-group birth iteration (for grace-period suppression of death prior)
        self.birth_iter: torch.Tensor | None = None     # (K,) int

        # Cached tangent basis (refreshed when group normals change).
        # Reading u, v at every forward pass is a major cost; caching is a
        # large speedup.
        self._u_cache: torch.Tensor | None = None       # (K, 3)
        self._v_cache: torch.Tensor | None = None       # (K, 3)

    # ------------------------------------------------------------------ #
    #  Accessors                                                          #
    # ------------------------------------------------------------------ #
    @property
    def K(self) -> int:
        return 0 if self._q is None else self._q.shape[0]

    @property
    def q(self) -> torch.Tensor:
        return self._q

    @property
    def m(self) -> torch.Tensor:
        """Always returns a unit-normalized normal."""
        return _safe_unit(self._m)

    @property
    def rho(self) -> torch.Tensor:
        """In-plane scales per group, shape (K, 2): (rho_u, rho_v)
        in the deterministic tangent basis built from m_k."""
        return torch.exp(self._log_rho)

    @property
    def H(self) -> torch.Tensor:
        """Curvature tensor per group as a (K, 2, 2) symmetric matrix.
        Packed storage is (K, 3) = (Huu, Huv, Hvv).
        Initialized to zero (flat groups); updated by E-step."""
        H_packed = self._H                                # (K, 3)
        # Build full (K, 2, 2) symmetric matrix
        H_mat = torch.zeros(self.K, 2, 2, device=H_packed.device, dtype=H_packed.dtype)
        H_mat[:, 0, 0] = H_packed[:, 0]
        H_mat[:, 1, 1] = H_packed[:, 2]
        H_mat[:, 0, 1] = H_packed[:, 1]
        H_mat[:, 1, 0] = H_packed[:, 1]
        return H_mat

    @property
    def sh_grp(self) -> torch.Tensor:
        return self._sh_grp

    def parameters(self) -> list[nn.Parameter]:
        """Return the list of parameters for optimizer registration."""
        return [self._q, self._m, self._log_rho, self._sh_grp]

    # ------------------------------------------------------------------ #
    #  Initialization                                                     #
    # ------------------------------------------------------------------ #
    def init_from_kmeans(
        self,
        pts: torch.Tensor,
        K_init: int,
        rho_init_scale: float = 1.5,
        n_iter: int = 25,
        sh_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """K-means on SfM-like points, then per-cluster PCA for normals.

        Parameters
        ----------
        pts : (N, 3) tensor
            SfM-like point cloud (must be on `self.device`).
        K_init : int
            Number of initial groups.
        rho_init_scale : float
            rho_k initialized to rho_init_scale * sqrt(in-plane PCA variance).
        n_iter : int
            K-means iterations.
        sh_init : (N, sh_channels) tensor or None
            Optional per-point SH coefficients (RGB2SH(SfM colors), already in
            the same flattened layout as group SH).  If provided, group SH is
            initialized to the cluster-mean SH.

        Returns
        -------
        assn : (N,) long tensor
            Hard k-means assignment of each input point to a group (for the
            caller to use as initial primitive→group association).
        """
        assert pts.dim() == 2 and pts.shape[1] == 3
        device = pts.device
        N = pts.shape[0]
        K = min(K_init, N)

        # Random init via k-means++ for stability
        first = torch.randint(0, N, (1,), device=device).item()
        centers = [pts[first]]
        # Coarse k-means++ seeding (one pass is sufficient at this scale)
        for _ in range(K - 1):
            d2 = torch.stack([(pts - c).pow(2).sum(-1) for c in centers], dim=1).min(dim=1).values
            probs = d2 / d2.sum().clamp_min(1e-12)
            idx = torch.multinomial(probs, 1).item()
            centers.append(pts[idx])
        centers = torch.stack(centers, dim=0)            # (K, 3)

        assn = torch.zeros(N, dtype=torch.long, device=device)
        for _ in range(n_iter):
            d2 = torch.cdist(pts, centers, p=2).pow(2)   # (N, K)
            assn = d2.argmin(dim=1)
            for k in range(K):
                mask = assn == k
                if mask.sum() > 0:
                    centers[k] = pts[mask].mean(dim=0)

        # Per-cluster normals via PCA + anisotropic rho from per-axis spread
        normals = torch.zeros(K, 3, device=device)
        rhos_uv = torch.zeros(K, 2, device=device)
        for k in range(K):
            mask = assn == k
            cnt = int(mask.sum().item())
            if cnt < 3:
                normals[k] = torch.tensor([0.0, 0.0, 1.0], device=device)
                rhos_uv[k] = torch.tensor([0.1, 0.1], device=device)
                continue
            diff = pts[mask] - centers[k]
            S = diff.T @ diff / cnt
            vals, vecs = torch.linalg.eigh(S)
            normals[k] = vecs[:, 0]                      # smallest eigenvalue dir
            # Compute tangent basis from this normal (matches what
            # gather_at will use at query time -- deterministic Gram-Schmidt
            # from a fixed reference) so the rho values are in the right basis.
            u_k, v_k = _tangent_basis(normals[k:k+1])
            u_k = u_k[0]; v_k = v_k[0]
            # Project scatter onto u_k and v_k to get per-axis variances
            var_u = (u_k @ S @ u_k).clamp_min(1e-6)
            var_v = (v_k @ S @ v_k).clamp_min(1e-6)
            rhos_uv[k, 0] = rho_init_scale * var_u.sqrt()
            rhos_uv[k, 1] = rho_init_scale * var_v.sqrt()

        # Group SH init: mean of cluster-member SH if provided, else zeros
        if sh_init is None:
            sh_grp = torch.zeros(K, self.sh_channels, device=device)
        else:
            assert sh_init.shape == (N, self.sh_channels), (
                f"sh_init shape {tuple(sh_init.shape)} != ({N}, {self.sh_channels})"
            )
            sh_grp = torch.zeros(K, self.sh_channels, device=device)
            for k in range(K):
                mask = assn == k
                if mask.sum() > 0:
                    sh_grp[k] = sh_init[mask].mean(dim=0)

        self._q = nn.Parameter(centers.contiguous())
        self._m = nn.Parameter(_safe_unit(normals).contiguous())
        self._log_rho = nn.Parameter(
            torch.log(rhos_uv.clamp_min(1e-3)).contiguous()
        )                                                # (K, 2)
        # Curvature tensor H_k in {Huu, Huv, Hvv} packed form (K, 3).
        # Initialized to zero (flat groups).  Updated by E-step via
        # least-squares quadric fit to assigned primitives.
        self._H = nn.Parameter(
            torch.zeros(K, 3, device=device).contiguous()
        )
        self._sh_grp = nn.Parameter(sh_grp.contiguous())

        self.grad_accum = torch.zeros(K, device=device)
        self.mass_accum = torch.zeros(K, device=device)
        self._accum_count = 0
        self.birth_iter = torch.zeros(K, dtype=torch.long, device=device)

        return assn

    # ------------------------------------------------------------------ #
    #  Soft assignment and top-M selection                                #
    # ------------------------------------------------------------------ #
    def soft_assign(
        self,
        mu: torch.Tensor,
        top_m: int = 4,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the top-M soft assignment for primitive positions mu.

        pi_ik proportional to exp(-||mu_i - q_k||^2 / (2 rho_k^2)),
        truncated to the top-M nearest groups per primitive.

        Parameters
        ----------
        mu : (N, 3) tensor
        top_m : int
            How many groups each primitive keeps non-zero weight on.

        Returns
        -------
        pi : (N, M) tensor
            Soft assignment over the top-M groups, summing to 1 along last dim.
        idx : (N, M) long tensor
            Indices into the global group array.
        """
        assert self.K > 0, "Initialize the GroupModel before calling soft_assign."
        N = mu.shape[0]
        M = min(top_m, self.K)

        # Pairwise squared distances (N, K).  This simple isotropic version
        # is used for tests and quick clustering checks; production uses
        # the anisotropic kernel in gms_gaussian_model.py.
        d2 = torch.cdist(mu, self._q, p=2).pow(2)        # (N, K)
        # rho is (K, 2); take per-group mean of u,v scales for the
        # isotropic-equivalent radius here.
        rho_iso = self.rho.mean(dim=-1)                  # (K,)
        rho2 = (rho_iso.pow(2) * 2.0).unsqueeze(0)       # (1, K)
        logits = -d2 / rho2.clamp_min(1e-9)

        # Top-M selection along K
        top_vals, top_idx = torch.topk(logits, k=M, dim=1)        # (N, M)
        # Numerical-safe softmax
        top_vals = top_vals - top_vals.max(dim=1, keepdim=True).values
        pi = top_vals.exp()
        pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-12)

        if return_logits:
            return pi, top_idx, logits
        return pi, top_idx

    # ------------------------------------------------------------------ #
    #  Gather group attributes for the top-M assignment                   #
    # ------------------------------------------------------------------ #
    def _refresh_tangent_basis(self) -> None:
        """Compute and cache the tangent basis (u_k, v_k) for each group.

        These depend only on m_k.  Call after any update that changes m
        (E-step, birth, death).  Holds (K, 3) tensors; cheap to keep.
        """
        with torch.no_grad():
            m_unit = self.m                                 # (K, 3) unit
            u, v = _tangent_basis(m_unit)
            self._u_cache = u.contiguous()
            self._v_cache = v.contiguous()

    def gather_at(self, idx: torch.Tensor) -> dict:
        """Gather group parameters at indices idx (N, M).

        Returns a dict with:
            q_top   : (N, M, 3)
            m_top   : (N, M, 3)   -- unit
            rho_top : (N, M)
            sh_top  : (N, M, sh_channels)
            u_top, v_top : (N, M, 3) -- tangent basis per (i, k)

        Uses cached per-group (u_k, v_k) when available -- much cheaper
        than recomputing Gram-Schmidt on (N*M, 3) every forward pass.
        """
        N, M = idx.shape
        q_top = self._q[idx]                              # (N, M, 3)
        m_top = self.m[idx]                               # (N, M, 3)  unit
        rho_top = self.rho[idx]                           # (N, M)
        sh_top = self._sh_grp[idx]                        # (N, M, C)

        # Tangent basis: use per-group cache when available
        if getattr(self, '_u_cache', None) is None or self._u_cache.shape[0] != self.K:
            self._refresh_tangent_basis()
        u_top = self._u_cache[idx]                        # (N, M, 3)
        v_top = self._v_cache[idx]                        # (N, M, 3)
        H_top = self._H[idx]                              # (N, M, 3) packed

        return dict(q=q_top, m=m_top, rho=rho_top, sh=sh_top,
                    u=u_top, v=v_top, H=H_top)

    # ------------------------------------------------------------------ #
    #  Chart-coordinate projection                                        #
    # ------------------------------------------------------------------ #
    def project(
        self, mu: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project primitive world positions onto each of their top-M
        groups' frames, returning chart coords and off-plane displacements.

        Parameters
        ----------
        mu : (N, 3) tensor
        idx : (N, M) long tensor

        Returns
        -------
        xi : (N, M, 2) tensor
            (u, v) coordinates of mu_i in group idx[i, k]'s tangent plane.
        h  : (N, M) tensor
            Perpendicular displacement along group idx[i, k]'s normal.
        """
        attrs = self.gather_at(idx)
        diff = mu.unsqueeze(1) - attrs["q"]               # (N, M, 3)
        xi_u = (diff * attrs["u"]).sum(dim=-1)            # (N, M)
        xi_v = (diff * attrs["v"]).sum(dim=-1)            # (N, M)
        h = (diff * attrs["m"]).sum(dim=-1)               # (N, M)
        return torch.stack([xi_u, xi_v], dim=-1), h

    # ------------------------------------------------------------------ #
    #  Forward: build primitive world positions and normals from group +  #
    #  per-primitive (xi, h, idx) state                                   #
    # ------------------------------------------------------------------ #
    def reconstruct_mu_n(
        self,
        xi: torch.Tensor,
        h: torch.Tensor,
        idx: torch.Tensor,
        pi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implements Eq. 6 of the paper:
            mu_i = sum_k pi_ik (q_k + T_k(xi_ik) + h_ik m_k)
            n_i  = normalize( sum_k pi_ik m_k )

        Parameters
        ----------
        xi  : (N, M, 2)
        h   : (N, M)
        idx : (N, M) long
        pi  : (N, M)

        Returns
        -------
        mu : (N, 3)
        n  : (N, 3) unit
        """
        attrs = self.gather_at(idx)
        on_plane = (
            attrs["q"]
            + xi[..., 0:1] * attrs["u"]
            + xi[..., 1:2] * attrs["v"]
        )                                                  # (N, M, 3)
        off_plane = h.unsqueeze(-1) * attrs["m"]           # (N, M, 3)
        contrib = on_plane + off_plane                     # (N, M, 3)
        mu = (pi.unsqueeze(-1) * contrib).sum(dim=1)       # (N, 3)
        n_unnorm = (pi.unsqueeze(-1) * attrs["m"]).sum(dim=1)
        n = _safe_unit(n_unnorm)
        return mu, n

    def reconstruct_sh(
        self,
        idx: torch.Tensor,
        pi: torch.Tensor,
        sh_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Implements Eq. 8 of the paper:
            c_i(d) = sum_k pi_ik c^grp_k(d) + c^res_i(d)

        For SH coefficients this is just a weighted sum:
            sh_i = sum_k pi_ik sh^grp_k + sh^res_i

        Parameters
        ----------
        idx : (N, M) long
        pi : (N, M)
        sh_residual : (N, sh_channels)

        Returns
        -------
        sh : (N, sh_channels)
        """
        sh_grp_top = self._sh_grp[idx]                     # (N, M, C)
        sh_grp_mix = (pi.unsqueeze(-1) * sh_grp_top).sum(dim=1)
        return sh_grp_mix + sh_residual

    # ------------------------------------------------------------------ #
    #  E-step: closed-form damped weighted plane fit                      #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def e_step(
        self,
        mu: torch.Tensor,
        pi: torch.Tensor,
        idx: torch.Tensor,
        eta_geom: float = 0.3,
        eta_sh: float = 0.1,
        rho_min: float = 1e-3,
        rho_max: float = 10.0,
        sh_residual: torch.Tensor | None = None,
        primitive_sh: torch.Tensor | None = None,
    ) -> None:
        """Update group params (q, m, rho, sh_grp) by weighted ML fit on
        currently-assigned primitive positions, with damped overwrite.

        Parameters
        ----------
        mu : (N, 3)
        pi : (N, M)
        idx : (N, M) long
        eta_geom : damping for q, m, rho
        eta_sh : damping for sh_grp
        rho_min, rho_max : clamp for rho (scene-scale dependent)
        sh_residual : (N, sh_channels) or None
            Per-primitive residual SH.  If provided, group SH target is the
            assignment-weighted (primitive_sh - residual) so that
            sum_k pi_ik c^grp_k = c^prim_i - c^res_i in mean.
        primitive_sh : (N, sh_channels) or None
            The primitives' fitted total SH.  Required if updating sh_grp.

        Notes
        -----
        Scatters per-primitive contributions to group accumulators via
        index_add_, which is O(N M) memory and exact.
        """
        K = self.K
        device = mu.device

        # Scatter the soft mass to group slots.  Each (i, k) contributes
        # weight pi[i, k] to group idx[i, k].
        w_flat = pi.reshape(-1)                            # (N*M,)
        idx_flat = idx.reshape(-1)                         # (N*M,)
        N, M = pi.shape

        # mass: M_k = sum_i sum_k pi_ik [k] = sum where idx=k
        mass = torch.zeros(K, device=device).index_add_(0, idx_flat, w_flat)
        mass_safe = mass.clamp_min(1e-6)

        # Weighted centroid q_star_k = sum (mu_i * w) / sum w
        mu_rep = mu.unsqueeze(1).expand(-1, M, -1).reshape(-1, 3)   # (N*M, 3)
        w_mu = w_flat.unsqueeze(-1) * mu_rep
        q_star = torch.zeros(K, 3, device=device).index_add_(0, idx_flat, w_mu)
        q_star = q_star / mass_safe.unsqueeze(-1)

        # Weighted scatter matrix per group
        # S_k = sum_i (w_ik * (mu_i - q*_k)(mu_i - q*_k)^T) / sum w_ik
        diff = mu_rep - q_star[idx_flat]                                # (N*M, 3)
        outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)                  # (N*M, 3, 3)
        outer_w = w_flat.view(-1, 1, 1) * outer
        S = torch.zeros(K, 3, 3, device=device).index_add_(0, idx_flat, outer_w)
        S = S / mass_safe.view(-1, 1, 1)

        # Smallest-eigenvalue eigenvector = new normal
        # eigh returns vals ascending; take vecs[:, 0]
        # Skip groups with no mass (their S is zero)
        eligible = mass > 1e-4
        m_star = self._m.detach().clone()
        rho_star = self.rho.detach().clone()                            # (K, 2)
        if eligible.any():
            S_eli = S[eligible]
            vals, vecs = torch.linalg.eigh(S_eli)
            m_eli = vecs[:, :, 0]                                       # smallest
            # Orientation continuity: align with current m
            cur_m = self.m[eligible]
            sign = (m_eli * cur_m).sum(dim=-1).sign()
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            m_eli = m_eli * sign.unsqueeze(-1)
            m_star[eligible] = m_eli

            # Anisotropic in-plane rho: project S onto the tangent basis
            # built from m_star (the same deterministic Gram-Schmidt basis
            # that gather_at will use).  rho_u = sqrt(u^T S u), similarly v.
            # IMPORTANT: use m_star, not the current m, because the basis
            # rotates with the new normal.
            u_eli, v_eli = _tangent_basis(m_eli)                        # (Ke, 3) each
            # var_u_k = u_k^T S_k u_k    (Ke,)
            S_u = (S_eli @ u_eli.unsqueeze(-1)).squeeze(-1)             # (Ke, 3)
            var_u = (S_u * u_eli).sum(dim=-1).clamp_min(1e-6)
            S_v = (S_eli @ v_eli.unsqueeze(-1)).squeeze(-1)
            var_v = (S_v * v_eli).sum(dim=-1).clamp_min(1e-6)
            rho_star[eligible, 0] = var_u.sqrt()
            rho_star[eligible, 1] = var_v.sqrt()

        # Damped overwrite of geometry
        with torch.no_grad():
            new_q = (1.0 - eta_geom) * self._q + eta_geom * q_star
            self._q.copy_(new_q)
            new_m_raw = (1.0 - eta_geom) * self._m + eta_geom * m_star
            self._m.copy_(_safe_unit(new_m_raw))
            new_rho = (1.0 - eta_geom) * self.rho + eta_geom * rho_star  # (K, 2)
            new_rho = new_rho.clamp(min=rho_min, max=rho_max)
            self._log_rho.copy_(torch.log(new_rho))

        # Curvature update: least-squares quadric fit to assigned primitives.
        # We fit h_i = 1/2 xi^T H xi + c, where c is a per-group constant
        # that absorbs the centroid offset caused by q_star being a
        # *position* centroid (whereas the surface-parameterization origin
        # should be the point of zero quadric height).  Without the c
        # term, the offset would systematically bias H toward zero for
        # convex surfaces (E[z] > 0 if H positive definite).
        #
        # Pack as h_i = [0.5 xi_u^2, xi_u*xi_v, 0.5 xi_v^2, 1] . [Huu, Huv, Hvv, c]
        # Solve 4x4 normal equations per group, then discard c.
        H_star = self._H.detach().clone()                              # (K, 3)
        if eligible.any():
            # Compute new tangent basis from m_star for the eligible groups
            u_full = self._u_cache if self._u_cache is not None else None
            if u_full is None or u_full.shape[0] != self.K:
                self._refresh_tangent_basis()
                u_full = self._u_cache
            v_full = self._v_cache
            # Use NEW basis (built from m_star), since H is in that basis
            u_eli2, v_eli2 = _tangent_basis(m_star[eligible])           # (Ke, 3)
            # Re-gather per-primitive (xi, h) using new basis
            mu_rep_eli = mu_rep                                         # (N*M, 3)
            idx_flat_eli = idx_flat                                     # (N*M,)
            elig_mask = eligible[idx_flat_eli]                          # (N*M,)
            if elig_mask.any():
                # Build full-K u/v tensors using the new basis for eligible groups
                u_at = torch.zeros_like(self._u_cache)                  # (K, 3)
                v_at = torch.zeros_like(self._v_cache)
                u_at[eligible] = u_eli2
                v_at[eligible] = v_eli2
                m_at = m_star                                            # (K, 3)
                # Relative position from new q_star, in new tangent frame
                diff_to_new_q = mu_rep - q_star[idx_flat_eli]            # (N*M, 3)
                u_per = u_at[idx_flat_eli]
                v_per = v_at[idx_flat_eli]
                m_per = m_at[idx_flat_eli]
                xi_u = (diff_to_new_q * u_per).sum(dim=-1)               # (N*M,)
                xi_v = (diff_to_new_q * v_per).sum(dim=-1)
                h_off = (diff_to_new_q * m_per).sum(dim=-1)              # (N*M,)
                # Build design row: a = (0.5 xi_u^2, xi_u*xi_v, 0.5 xi_v^2, 1)
                ones = torch.ones_like(xi_u)
                a = torch.stack([0.5 * xi_u.pow(2),
                                 xi_u * xi_v,
                                 0.5 * xi_v.pow(2),
                                 ones], dim=-1)                          # (N*M, 4)
                # Weighted outer product accumulator per group  (K, 4, 4)
                aa = a.unsqueeze(-1) * a.unsqueeze(-2)
                aa_w = w_flat.view(-1, 1, 1) * aa
                ATA = torch.zeros(K, 4, 4, device=device).index_add_(0, idx_flat_eli, aa_w)
                # Weighted RHS: a * h    (K, 4)
                ah_w = (w_flat * h_off).unsqueeze(-1) * a
                ATb = torch.zeros(K, 4, device=device).index_add_(0, idx_flat_eli, ah_w)
                # Solve only for eligible groups; ridge-regularize to avoid singularity
                lam_reg = 1e-4
                ATA_eli = ATA[eligible] + lam_reg * torch.eye(4, device=device).unsqueeze(0)
                ATb_eli = ATb[eligible]
                try:
                    sol = torch.linalg.solve(ATA_eli, ATb_eli)            # (Ke, 4)
                    # Take only the first 3 entries: (Huu, Huv, Hvv).  The
                    # 4th entry (constant offset c) is discarded.
                    H_star[eligible] = sol[:, :3]
                except Exception:
                    # Singular system -- leave H unchanged for these groups
                    pass

        # Damped overwrite of curvature (separately from geometry; safer to
        # use larger damping on H since it can grow large quickly)
        with torch.no_grad():
            eta_curv = 0.2  # smaller damping for stability
            new_H = (1.0 - eta_curv) * self._H + eta_curv * H_star
            # Clamp curvature magnitude to avoid runaway
            H_max = 50.0
            new_H = new_H.clamp(min=-H_max, max=H_max)
            self._H.copy_(new_H)

        # SH update (optional)
        if primitive_sh is not None:
            assert primitive_sh.shape == (N, self.sh_channels), (
                f"primitive_sh shape mismatch: {tuple(primitive_sh.shape)}"
            )
            # Target: group SH that, when mixed by pi and added to residual,
            # matches primitive_sh on average.
            # Simplest: weighted mean of (primitive_sh - residual) per group.
            target_sh = primitive_sh.clone()
            if sh_residual is not None:
                target_sh = target_sh - sh_residual
            # Scatter to groups
            sh_rep = target_sh.unsqueeze(1).expand(-1, M, -1).reshape(
                -1, self.sh_channels
            )
            w_sh = w_flat.unsqueeze(-1) * sh_rep
            sh_star = torch.zeros(K, self.sh_channels, device=device).index_add_(
                0, idx_flat, w_sh
            )
            sh_star = sh_star / mass_safe.unsqueeze(-1)
            with torch.no_grad():
                new_sh = (1.0 - eta_sh) * self._sh_grp + eta_sh * sh_star
                self._sh_grp.copy_(new_sh)

        # Group normals changed -- invalidate cached tangent basis
        self._u_cache = None
        self._v_cache = None

    # ------------------------------------------------------------------ #
    #  Adaptive control: birth and death                                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def accumulate_stats(self, pi: torch.Tensor, idx: torch.Tensor, g_i: torch.Tensor) -> None:
        """Accumulate the per-primitive gradient magnitude g_i (and assignment
        mass) into the running per-group statistics G_k and M_k.

        Called every iteration during training, in addition to whatever
        the rasterizer already accumulates."""
        K = self.K
        N, M = pi.shape
        g_rep = g_i.unsqueeze(1).expand(-1, M).reshape(-1)
        w = pi.reshape(-1)
        idx_flat = idx.reshape(-1)
        self.grad_accum.index_add_(0, idx_flat, w * g_rep)
        self.mass_accum.index_add_(0, idx_flat, w)
        self._accum_count += 1

    def get_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the running per-group (G_k, M_k) and reset accumulators."""
        if self._accum_count == 0:
            G = torch.zeros_like(self.grad_accum)
            M = torch.zeros_like(self.mass_accum)
        else:
            # Average-of-running-totals: G_k = sum(pi g) / sum(pi)
            M_safe = self.mass_accum.clamp_min(1e-6)
            G = self.grad_accum / M_safe
            M = self.mass_accum / self._accum_count
        return G, M

    def reset_stats(self) -> None:
        self.grad_accum.zero_()
        self.mass_accum.zero_()
        self._accum_count = 0

    @torch.no_grad()
    def birth(
        self,
        seed_positions: torch.Tensor,
        seed_normals: torch.Tensor | None = None,
        seed_sh: torch.Tensor | None = None,
        min_seeds: int = 8,
        rho_init: float = 0.1,
        current_iter: int = 0,
    ) -> int:
        """Spawn new groups from a set of seed positions (typically orphan
        primitives identified by image-evidence triggers).  Each connected
        component of seeds (computed by the caller and passed as a flat list,
        with one new group per call) becomes one new group.

        Here we accept a flat batch and let the caller do the clustering.

        Parameters
        ----------
        seed_positions : (S, 3) tensor
            Positions of the seed primitives constituting ONE new group.
        seed_normals : (S, 3) tensor or None
            Optional normals; if provided, group normal is the mean.
            If None, we PCA-fit the positions.
        seed_sh : (S, sh_channels) tensor or None
            Optional per-seed SH coefficients; group SH = mean.
        min_seeds : int
            If S < min_seeds, do nothing and return 0.
        rho_init : float
            Floor for rho on the newborn group.
        current_iter : int
            Stored as birth_iter for grace-period accounting.

        Returns
        -------
        n_added : int
            1 if a new group was added, else 0.
        """
        S = seed_positions.shape[0]
        if S < min_seeds:
            return 0
        device = self._q.device
        q_new = seed_positions.mean(dim=0)                              # (3,)
        diff = seed_positions - q_new
        scatter = diff.T @ diff / S
        vals, vecs = torch.linalg.eigh(scatter)
        if seed_normals is not None and seed_normals.shape[0] >= 3:
            m_new = _safe_unit(seed_normals.mean(dim=0))
        else:
            m_new = vecs[:, 0]                                           # smallest
        # Anisotropic rho: project scatter onto Gram-Schmidt tangent basis
        u_new, v_new = _tangent_basis(m_new.unsqueeze(0))
        u_new = u_new[0]; v_new = v_new[0]
        var_u = (u_new @ scatter @ u_new).clamp_min(1e-6)
        var_v = (v_new @ scatter @ v_new).clamp_min(1e-6)
        rho_u_new = max(float(var_u.sqrt().item()), rho_init)
        rho_v_new = max(float(var_v.sqrt().item()), rho_init)
        sh_new = (
            seed_sh.mean(dim=0)
            if seed_sh is not None
            else torch.zeros(self.sh_channels, device=device)
        )

        # Append to parameters.  We have to rebuild the nn.Parameters since
        # Parameter sizes are not mutable.  The caller is responsible for
        # also updating the optimizer (handled in Stage 2).
        self._q = nn.Parameter(torch.cat([self._q.detach(), q_new.unsqueeze(0)], dim=0))
        self._m = nn.Parameter(torch.cat([self._m.detach(), m_new.unsqueeze(0)], dim=0))
        log_rho_new = torch.log(
            torch.tensor([[rho_u_new, rho_v_new]], device=device)
        )                                                                # (1, 2)
        self._log_rho = nn.Parameter(
            torch.cat([self._log_rho.detach(), log_rho_new], dim=0)
        )
        self._sh_grp = nn.Parameter(
            torch.cat([self._sh_grp.detach(), sh_new.unsqueeze(0)], dim=0)
        )
        # New group starts flat (H = 0)
        self._H = nn.Parameter(
            torch.cat([self._H.detach(), torch.zeros(1, 3, device=device)], dim=0)
        )
        self.grad_accum = torch.cat([self.grad_accum, torch.zeros(1, device=device)])
        self.mass_accum = torch.cat([self.mass_accum, torch.zeros(1, device=device)])
        self.birth_iter = torch.cat(
            [self.birth_iter, torch.tensor([current_iter], device=device, dtype=torch.long)]
        )
        self._u_cache = None
        self._v_cache = None
        return 1

    @torch.no_grad()
    def death(
        self,
        keep_mask: torch.Tensor,
    ) -> int:
        """Remove groups for which keep_mask[k] is False.

        Index remapping: returns the index-mapping table old->new so the
        caller (Stage 2) can repair primitive top-M indices that point at
        deleted groups.  Caller is also responsible for updating the
        optimizer state.

        Parameters
        ----------
        keep_mask : (K,) bool

        Returns
        -------
        n_killed : int
        """
        keep = keep_mask.bool()
        n_killed = int((~keep).sum().item())
        if n_killed == 0:
            return 0
        self._q = nn.Parameter(self._q.detach()[keep])
        self._m = nn.Parameter(self._m.detach()[keep])
        self._log_rho = nn.Parameter(self._log_rho.detach()[keep])
        self._sh_grp = nn.Parameter(self._sh_grp.detach()[keep])
        self._H = nn.Parameter(self._H.detach()[keep])
        self.grad_accum = self.grad_accum[keep]
        self.mass_accum = self.mass_accum[keep]
        self.birth_iter = self.birth_iter[keep]
        self._u_cache = None
        self._v_cache = None
        return n_killed

    def death_score(
        self,
        M0: float | None = None,
        G0: float | None = None,
    ) -> torch.Tensor:
        """Compute the death-prior score per group.  Groups with low mass
        AND low gradient pressure score high (i.e., should die).

        Score = exp(-(M_k / M0) - (G_k / G0))    (Eq. 11 of the paper)

        Returns a (K,) tensor in [0, 1]; close to 1 = strong death pressure.
        """
        G, M = self.get_stats()
        if M0 is None:
            # Default: half the expected uniform mass
            N_est = M.sum().clamp_min(1.0)
            M0 = max(float((N_est / (2.0 * self.K)).item()), 1.0)
        if G0 is None:
            G0 = max(float(G.median().item()), 1e-6)
        score = torch.exp(-(M / M0) - (G / G0))
        return score


# =============================================================================
#  Unit tests (run standalone:  python -m scene.group_model)
# =============================================================================
def _run_unit_tests():
    """Standalone tests with synthetic primitive points on three known
    planes.  Tests:
        1. K-means init recovers approximately the right cluster centers
        2. soft_assign + top-M returns sensible probability mass
        3. project+reconstruct_mu_n is identity when xi/h are projections
        4. e_step pulls a perturbed group back toward the data centroid
        5. birth adds a group; death removes one
    """
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[group_model tests] device = {device}")
    torch.manual_seed(0)
    sh_degree = 0   # DC only, channels = 3
    G = GroupModel(sh_degree=sh_degree, device=device)

    # Synthetic three-plane scene
    def make_plane(c, u, v, n_pts):
        a = (torch.rand(n_pts, 1) - 0.5) * 2
        b = (torch.rand(n_pts, 1) - 0.5) * 2
        pts = c + a * u + b * v + 0.01 * torch.randn(n_pts, 3)
        return pts

    p0 = make_plane(torch.tensor([0.0, 0.0, 0.0]),
                     torch.tensor([1.0, 0.0, 0.0]),
                     torch.tensor([0.0, 1.0, 0.0]), 200)
    p1 = make_plane(torch.tensor([0.0, 1.0, 0.5]),
                     torch.tensor([1.0, 0.0, 0.0]),
                     torch.tensor([0.0, 0.0, 1.0]), 200)
    p2 = make_plane(torch.tensor([1.0, 0.0, 0.5]),
                     torch.tensor([0.0, 1.0, 0.0]),
                     torch.tensor([0.0, 0.0, 1.0]), 200)
    pts = torch.cat([p0, p1, p2], dim=0).to(device)

    # 1. Init
    assn = G.init_from_kmeans(pts, K_init=3, n_iter=30)
    assert G.K == 3, f"K = {G.K}"
    assert assn.shape == (600,)
    print("[test 1] init_from_kmeans OK")

    # 2. Soft assign with top-M
    pi, idx = G.soft_assign(pts, top_m=2)
    assert pi.shape == (600, 2)
    assert idx.shape == (600, 2)
    # Each row sums to 1
    assert torch.allclose(pi.sum(dim=1), torch.ones(600, device=device), atol=1e-5)
    # The top assignment should match the k-means assn for most points
    top_match = (idx[:, 0] == assn).float().mean()
    assert top_match > 0.9, f"only {top_match.item():.2f} match k-means"
    print(f"[test 2] soft_assign OK  (top-1 matches k-means at {top_match.item():.3f})")

    # 3. Project then reconstruct is identity
    pi, idx = G.soft_assign(pts, top_m=4)
    xi, h = G.project(pts, idx)
    mu_rec, _ = G.reconstruct_mu_n(xi, h, idx, pi)
    err = (mu_rec - pts).norm(dim=-1).mean()
    assert err < 1e-4, f"reconstruction error {err.item():.6f}"
    print(f"[test 3] project+reconstruct identity OK  (mean err {err.item():.2e})")

    # 4. E-step pulls a perturbed group back
    perturb = 0.3 * torch.randn(3, device=device)
    with torch.no_grad():
        G._q[0].add_(perturb)
    q0_before = G._q[0].clone()
    pi, idx = G.soft_assign(pts, top_m=4)
    G.e_step(pts, pi, idx, eta_geom=0.9)
    q0_after = G._q[0].clone()
    movement_toward_truth = (q0_before - q0_after).norm().item()
    assert movement_toward_truth > 0.1, (
        f"E-step moved group 0 only {movement_toward_truth:.3f}; expected > 0.1"
    )
    print(f"[test 4] e_step pulls perturbed group  (Δq = {movement_toward_truth:.3f})")

    # 5. Birth and death
    K0 = G.K
    # Birth: pretend p2 (third plane) is an orphan cluster
    G.birth(p2.to(device), min_seeds=50, current_iter=42)
    assert G.K == K0 + 1, f"expected K = {K0 + 1}, got {G.K}"
    print(f"[test 5a] birth added a group  (K: {K0} -> {G.K})")

    keep = torch.ones(G.K, dtype=torch.bool, device=device)
    keep[1] = False
    G.death(keep)
    assert G.K == K0, f"after death K should be {K0}, got {G.K}"
    print(f"[test 5b] death removed a group  (K -> {G.K})")

    # 6. Sanity: gather_at and reconstruct still consistent after birth/death
    pi, idx = G.soft_assign(pts, top_m=4)
    xi, h = G.project(pts, idx)
    mu_rec, _ = G.reconstruct_mu_n(xi, h, idx, pi)
    err = (mu_rec - pts).norm(dim=-1).mean()
    assert err < 1e-4, f"post-death reconstruction error {err.item():.6f}"
    print(f"[test 6] post-birth/death reconstruct identity OK  (mean err {err.item():.2e})")

    print("\n[group_model tests] all passed.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_run_unit_tests())