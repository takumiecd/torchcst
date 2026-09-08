"""Prepared first-order actions with bounded atom-overlap scratch storage.

Only jvp/vjp and the explicit reference backend materialize visible matrices.
Fast Gram actions contract the source direction before contracting atom pairs.
"""

from __future__ import annotations

import hashlib

import torch

from torchcst._runtime.validation import require


def _configuration(modules):
    return tuple(
        (
            root,
            name,
            type(m).__module__,
            type(m).__qualname__,
            getattr(m, "tangent_layout_version", 1),
            repr(m.tangent_config()) if hasattr(m, "tangent_config") else "()",
        )
        for root, module in enumerate(modules)
        for name, m in module.named_modules()
    )


class TangentOps:
    """Bind a fixed kernel/chart interpretation, then prepare parameter snapshots.

    Custom fixed non-tensor settings must be declared by tangent_config().
    Changing fixed modules/buffers requires a new operator and optimizer history.
    Tensor .data writes bypass PyTorch version checks and are unsupported.
    """

    def __init__(
        self,
        derivatives,
        *,
        backend="auto",
        atom_tile=32,
        specialized=None,
        modules=(),
        execution="eager",
        gram_action="pair",
    ):
        if backend not in ("auto", "specialized", "factor_autograd", "reference"):
            raise ValueError("unknown tangent backend")
        if (
            isinstance(atom_tile, bool)
            or not isinstance(atom_tile, int)
            or atom_tile < 1
        ):
            raise ValueError("atom_tile must be a positive integer")
        if backend == "auto":
            backend = (
                "specialized"
                if specialized is not None
                else "factor_autograd"
                if derivatives.factor_atoms is not None
                else "reference"
            )
        if backend == "specialized" and specialized is None:
            raise ValueError("kernel has no specialized tangent backend")
        if backend == "factor_autograd" and derivatives.factor_atoms is None:
            raise ValueError("kernel has no factorized backend")
        if execution not in ("eager", "triton"):
            raise ValueError("execution must be eager or triton")
        if execution == "triton" and (
            backend == "reference"
            or not derivatives.atoms.p.is_cuda
            or derivatives.atoms.p.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError(
                "Triton execution requires factorized CUDA float32/float64 parameters"
            )
        if gram_action not in ("pair", "jvp_vjp"):
            raise ValueError("gram_action must be pair or jvp_vjp")
        if gram_action == "jvp_vjp" and backend == "reference":
            raise ValueError("jvp_vjp requires factorized tangents")
        self.gram_action = gram_action
        self.execution = execution
        self.backend = backend
        self.atom_tile = atom_tile
        self.derivatives = derivatives
        self._specialized = specialized
        self._modules = modules
        self._token = self._live_token()
        self._config = _configuration(modules)
        self._fixed = tuple(
            (root, name, t.detach().clone())
            for root, module in enumerate(modules)
            for name, t in self._tensors(module)
        )
        self._descriptor = None

    @staticmethod
    def _tensors(module):
        return (*module.named_buffers(), *module.named_parameters())

    def _live_token(self):
        return (
            _configuration(self._modules),
            tuple(
                (root, name, id(t), t._version)
                for root, module in enumerate(self._modules)
                for name, t in self._tensors(module)
            ),
            tuple(id(m) for root in self._modules for m in root.modules()),
        )

    def check_configuration(self):
        if self._live_token() != self._token:
            raise ValueError(
                "fixed kernel/chart configuration changed; rebuild optimizer"
            )

    @property
    def descriptor(self):
        """Serializable cold-path identity, independent of backend and tile size."""
        if self._descriptor is None:
            tensors = tuple(
                (
                    root,
                    name,
                    tuple(t.shape),
                    str(t.dtype),
                    hashlib.sha256(
                        t.cpu()
                        .contiguous()
                        .reshape(-1)
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    ).hexdigest(),
                )
                for root, name, t in self._fixed
            )
            # Unbound callbacks cannot assert semantic compatibility with others.
            self._descriptor = (self._config, tensors)
        return self._descriptor

    @torch.no_grad()
    def prepare(self, p):
        self.check_configuration()
        if tuple(p.shape) != self.derivatives.point_shape:
            raise ValueError("parameter shape does not match tangent operator")
        parameter = self.derivatives.atoms.p
        if p.dtype != parameter.dtype or p.device != parameter.device:
            raise ValueError("parameter dtype/device must match the bound site")
        if not p.is_floating_point():
            raise ValueError("parameters must be finite floating-point values")
        require(
            torch.isfinite(p).all(), "parameters must be finite floating-point values"
        )
        point = p.detach().clone()
        if self.backend == "reference":
            jacobian = torch.func.jacfwd(self.derivatives.represented)(point)
            return PreparedReference(self, point, jacobian.detach())
        if self.backend == "specialized":
            parts = self._specialized(point)
        else:

            def one(row):
                u, v = self.derivatives.factor_atoms(row[None])
                return u[:, 0], v[:, 0]

            u, v = torch.vmap(one)(point)
            du, dv = torch.vmap(torch.func.jacfwd(one))(point)
            parts = u, v, du, dv
        u, v, du, dv = parts
        k, q = point.shape
        if (
            u.ndim != 2
            or v.ndim != 2
            or u.shape[0] != k
            or v.shape[0] != k
            or du.shape != (*u.shape, q)
            or dv.shape != (*v.shape, q)
        ):
            raise ValueError("invalid kernel tangent factor layout")
        return PreparedFactors(
            self,
            point,
            tuple(
                t.detach().clone().contiguous()
                if self.execution == "triton"
                else t.detach().clone()
                for t in parts
            ),
        )


class _Prepared:
    def __init__(self, ops, point):
        self._ops, self._point = ops, point
        self.backend = ops.backend

    @property
    def point(self):
        return self._point.clone()

    def _vector(self, x):
        if x.shape != self._point.shape:
            raise ValueError("tangent vector has the wrong shape")
        if x.dtype != self._point.dtype or x.device != self._point.device:
            raise ValueError(
                "tangent vector dtype/device must match prepared parameters"
            )

    def _compatible(self, source):
        if not isinstance(source, _Prepared):
            raise TypeError("source must be a prepared tangent")
        if self._ops is not source._ops and (
            not self._ops._modules
            or not source._ops._modules
            or self._ops.descriptor != source._ops.descriptor
        ):
            raise ValueError("incompatible kernel/chart tangent descriptors")
        self._vector(source._point)

    def gram_matvec(self, x):
        return self.cross_gram_matvec(self, x)

    def weighted_gram_matvec(self, metric, x):
        return self.cross_gram_matvec(self, x, metric=metric)


class PreparedFactors(_Prepared):
    def __init__(self, ops, point, parts):
        super().__init__(ops, point)
        self._u, self._v, self._du, self._dv = parts

    def factor_jvp(self, x):
        self._vector(x)
        return (
            torch.einsum("kiq,kq->ki", self._du, x),
            torch.einsum("koq,kq->ko", self._dv, x),
        )

    def jvp(self, x):
        du, dv = self.factor_jvp(x)
        return dv.T @ self._u + self._v.T @ du

    def vjp(self, force):
        if force.shape != (self._v.shape[1], self._u.shape[1]):
            raise ValueError("visible cotangent has the wrong shape")
        return torch.einsum("kiq,ki->kq", self._du, self._v @ force) + torch.einsum(
            "koq,ko->kq", self._dv, self._u @ force.T
        )

    def cross_gram_matvec(self, source, x, *, metric=None):
        self._compatible(source)
        source._vector(x)
        if not isinstance(source, PreparedFactors):
            raise TypeError("fast cross action requires two factorized preparations")
        if getattr(self._ops, "gram_action", "pair") == "jvp_vjp":
            return self._stream_cross(source, x, metric=metric)
        if self._ops.execution == "triton":
            from .tangent_triton import cross

            if metric is None:
                return cross(self, source, x)
            row, column, eps = metric.separable_weights()
            return cross(self, source, x, row=row, column=column) + eps * cross(
                self, source, x
            )
        ds_u, ds_v = source.factor_jvp(x)
        terms = [(None, None, 1.0)]
        if metric is not None:
            if not hasattr(metric, "separable_weights"):
                raise TypeError("fast weighted action requires a separable metric")
            row, column, eps = metric.separable_weights()
            terms = [(row, column, 1.0), (None, None, eps)]
        result = torch.zeros_like(self._point)
        tile = self._ops.atom_tile
        for row, column, scale in terms:
            if scale == 0:
                continue
            for t in range(0, self._point.shape[0], tile):
                sl = slice(t, t + tile)
                ut, vt = self._u[sl], self._v[sl]
                if row is not None:
                    ut, vt = ut * column, vt * row
                cu, cv = torch.zeros_like(ut), torch.zeros_like(vt)
                for s in range(0, source._point.shape[0], tile):
                    ss = slice(s, s + tile)
                    us, vs = source._u[ss], source._v[ss]
                    a, b = ds_u[ss], ds_v[ss]
                    cv.add_((ut @ us.T) @ b + (ut @ a.T) @ vs)
                    cu.add_((vt @ b.T) @ us + (vt @ vs.T) @ a)
                if row is not None:
                    cu, cv = cu * column, cv * row
                value = torch.einsum("kiq,ki->kq", self._du[sl], cu) + torch.einsum(
                    "koq,ko->kq", self._dv[sl], cv
                )
                result[sl].add_(value, alpha=scale)
        return result

    def _stream_cross(self, source, x, *, metric=None, active=None):
        """J_target.T D J_source x with at most 16 visible rows in scratch."""
        row = self._v.new_ones(self._v.shape[1])
        column = self._u.new_ones(self._u.shape[1])
        eps = 0.0
        if metric is not None:
            row, column, eps = metric.separable_weights()
        if self._ops.execution == "triton":
            from .tangent_triton import metric_action

            return metric_action(self, x, row, column, eps, 1.0, active, source=source)
        a, b = source.factor_jvp(x)
        result = torch.zeros_like(x)
        for start in range(0, self._v.shape[1], 16):
            sl = slice(start, start + 16)
            force = b[:, sl].T @ source._u + source._v[:, sl].T @ a
            force = force * (row[sl, None] * column[None, :] + eps)
            result += torch.einsum("kiq,ki->kq", self._du, self._v[:, sl] @ force)
            result += torch.einsum("koq,ko->kq", self._dv[:, sl], self._u @ force.T)
        return result if active is None else torch.where(active, result, 0)

    def _device_gram(self, x, *, damping=0.0, active=None):
        if getattr(self._ops, "gram_action", "pair") == "jvp_vjp":
            result = self._stream_cross(self, x, active=active) + damping * x
            return result if active is None else torch.where(active, result, 0)
        if self._ops.execution == "triton":
            from .tangent_triton import cross

            return cross(self, self, x, damping=damping, active=active)
        result = self.gram_matvec(x) + damping * x
        return result if active is None else torch.where(active, result, 0)

    def gram_blocks(self):
        """Exact matching-atom diagonal blocks; cross-atom terms remain in actions."""
        u, v, du, dv = self._u, self._v, self._du, self._dv
        uv = torch.einsum("kiq,ki->kq", du, u)
        vv = torch.einsum("koq,ko->kq", dv, v)
        return (
            torch.einsum("kiq,kir->kqr", du, du) * v.square().sum(-1)[:, None, None]
            + torch.einsum("koq,kor->kqr", dv, dv) * u.square().sum(-1)[:, None, None]
            + uv[:, :, None] * vv[:, None, :]
            + vv[:, :, None] * uv[:, None, :]
        )


class PreparedReference(_Prepared):
    def __init__(self, ops, point, jacobian):
        super().__init__(ops, point)
        self._jacobian = jacobian

    def jvp(self, x):
        self._vector(x)
        return torch.einsum("oikq,kq->oi", self._jacobian, x)

    def vjp(self, force):
        return torch.einsum("oikq,oi->kq", self._jacobian, force)

    def cross_gram_matvec(self, source, x, *, metric=None):
        self._compatible(source)
        force = source.jvp(x)
        if metric is not None:
            force = force * metric.diagonal()
        return self.vjp(force)

    def gram_blocks(self):
        return torch.einsum("oikq,oikr->kqr", self._jacobian, self._jacobian)
