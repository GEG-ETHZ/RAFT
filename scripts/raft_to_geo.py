# Copyright 2026 The RAFT Community
#
# This file is part of RAFT.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a Gmsh .geo (and optionally .msh) file from a RAFT config.

The mesh conforms to the config's regions, fractures and point/line BCs, so region
interfaces are sharp, BC points land on nodes and fractures follow element faces.

Write config/<mesh>.geo and .msh:
    just geo config_example_2D

The outer boundary comes from `mesh.domain` (same shape syntax as `mesh.regions`,
plus "box"); element sizes come from `mesh.sizing`, which the CLI flags override:

    mesh:
      domain: {shape: "box", values: [[xmin, ymin, zmin], [xmax, ymax, zmax]]}
      sizing:
        outer: 10                                    # size far from any feature
        frac: [[0.05, 0.05, 1, 10], [0.2, 5, 5, 5]]  # profile off the fractures
        bc: 0.5                                      # one stage over `transition`
        well: [[5, 10], [5, 50]]                     # profile off the wells

A profile [[s1, ..., sn], [L1, ..., Ln]] sets the size to s1 at the feature, then
ramps s1 -> s2 over L1, ..., and sn -> outer over Ln. Run with --help for the size
overrides and the region-boundary refinement they control.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from scipy.spatial import ConvexHull

TOL = 1.0e-9
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@dataclass
class Profile:
    """Piecewise-linear size vs. distance: s[k] -> s[k+1] over lengths[k], last stage -> outer."""

    sizes: list[float]
    lengths: list[float]

    def expr(self, dist: str, outer: float, scale: float = 1.0) -> str:
        """MathEval size at distance `dist` (optionally scaled), as s1 + sum of clamped ramps."""
        d = dist if scale == 1.0 else f"{scale:.10g}*({dist})"
        terms = [f"{self.sizes[0]:.10g}"]
        start = 0.0
        for k, length in enumerate(self.lengths):
            nxt = self.sizes[k + 1] if k + 1 < len(self.sizes) else outer
            if abs(nxt - self.sizes[k]) > TOL:
                ramp = f"Max(0, Min(1, (({d}) - {start:.10g})/{length:.10g}))"
                terms.append(f"({nxt - self.sizes[k]:.10g})*{ramp}")
            start += length
        return " + ".join(terms)

    @property
    def finest(self) -> float:
        """Smallest size in the profile."""
        return min(self.sizes)

    def last_slope(self, outer: float) -> float:
        """Slope of the final stage, used to relax region-boundary fields."""
        return (outer - self.sizes[-1]) / self.lengths[-1]


@dataclass
class Sizes:
    """Mesh size controls (see module docstring)."""

    size_outer: float
    frac: Profile
    bc: Profile
    well: Profile
    size_region_max: float
    boundary_growth: float = 0.25


class GeoWriter:
    """Accumulates .geo statements, using gmsh `new*` tags to avoid collisions with OCC auto-tags."""

    def __init__(self, nsd: int) -> None:
        """Start an empty .geo for `nsd` spatial dimensions."""
        self.nsd = nsd
        self.out: list[str] = []
        self._n = 0

    def emit(self, line: str = "") -> None:
        """Append one raw .geo line."""
        self.out.append(line)

    def _var(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def point(self, xyz: npt.ArrayLike) -> str:
        """Emit a point; return its variable name."""
        x, y, z = _pad3(xyz)
        v = self._var("p")
        self.emit(f"{v} = newp; Point({v}) = {{{x:.10g}, {y:.10g}, {z:.10g}}};")
        return v

    def line(self, p1: str, p2: str) -> str:
        """Emit a straight line between two point variables."""
        v = self._var("l")
        self.emit(f"{v} = newc; Line({v}) = {{{p1}, {p2}}};")
        return v

    def polyline(self, pts: npt.ArrayLike) -> list[str]:
        """Emit consecutive line segments through `pts`."""
        ptags = [self.point(p) for p in np.atleast_2d(pts)]
        return [self.line(a, b) for a, b in zip(ptags[:-1], ptags[1:], strict=True)]

    def polygon(self, pts: npt.NDArray[np.float64]) -> str:
        """Plane surface through ordered vertices."""
        ptags = [self.point(p) for p in pts]
        ltags = [self.line(ptags[i], ptags[(i + 1) % len(ptags)]) for i in range(len(ptags))]
        cl, s = self._var("cl"), self._var("s")
        self.emit(f"{cl} = newcl; Curve Loop({cl}) = {{{', '.join(ltags)}}};")
        self.emit(f"{s} = news; Plane Surface({s}) = {{{cl}}};")
        return s

    def convex_polyhedron(self, pts: npt.NDArray[np.float64]) -> str:
        """Volume bounded by the convex hull of `pts` (coplanar hull triangles merged into facets)."""
        facets = _hull_facets(pts)
        ptags = {i: self.point(pts[i]) for i in sorted({i for _, _, vids in facets for i in vids})}
        edges: dict[tuple[int, int], str] = {}
        surfs = []
        for normal, _, vids in facets:
            ordered = _order_planar(pts, vids, normal)
            loop = []
            for a, b in zip(ordered, ordered[1:] + ordered[:1], strict=True):
                if (a, b) in edges:
                    loop.append(edges[(a, b)])
                elif (b, a) in edges:
                    loop.append("-" + edges[(b, a)])
                else:
                    edges[(a, b)] = self.line(ptags[a], ptags[b])
                    loop.append(edges[(a, b)])
            cl, s = self._var("cl"), self._var("s")
            self.emit(f"{cl} = newcl; Curve Loop({cl}) = {{{', '.join(loop)}}};")
            self.emit(f"{s} = news; Plane Surface({s}) = {{{cl}}};")
            surfs.append(s)
        sl, v = self._var("sl"), self._var("v")
        self.emit(f"{sl} = newsl; Surface Loop({sl}) = {{{', '.join(surfs)}}};")
        self.emit(f"{v} = newv; Volume({v}) = {{{sl}}};")
        return v

    def rotate(self, entity: str, kind: str, axis: npt.ArrayLike, origin: npt.ArrayLike, deg: float) -> None:
        """Rotate an entity in place about `axis` through `origin`."""
        if abs(deg) < TOL:
            return
        ax, oy = _pad3(axis), _pad3(origin)
        self.emit(
            f"Rotate {{{{{ax[0]:.10g}, {ax[1]:.10g}, {ax[2]:.10g}}}, "
            f"{{{oy[0]:.10g}, {oy[1]:.10g}, {oy[2]:.10g}}}, {np.deg2rad(deg):.10g}}} "
            f"{{ {kind}{{{entity}}}; }}"
        )

    def extrude(self, surface: str, vec: npt.ArrayLike) -> str:
        """Extrude a surface into a volume; return the volume's .geo expression."""
        d = _pad3(vec)
        v = self._var("ex")
        self.emit(f"{v}[] = Extrude {{{d[0]:.10g}, {d[1]:.10g}, {d[2]:.10g}}} {{ Surface{{{surface}}}; }};")
        return f"{v}[1]"  # Extrude returns [top surface, volume, lateral surfaces...]

    def occ(self, kind: str, args: list[float]) -> str:
        """OCC primitive (Disk, Rectangle, Sphere, Box, Cylinder)."""
        prefix = "s" if kind in ("Disk", "Rectangle") else "v"
        v = self._var(prefix)
        new = "news" if prefix == "s" else "newv"
        self.emit(f"{v} = {new}; {kind}({v}) = {{{', '.join(f'{a:.10g}' for a in args)}}};")
        return v


def _pad3(xyz: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(xyz, dtype=float).ravel()
    return np.concatenate([a, np.zeros(3 - a.size)])


def _order_planar(
    pts: npt.NDArray[np.float64], vids: list[int], normal: npt.NDArray[np.float64]
) -> list[int]:
    """Order coplanar vertices counter-clockwise about `normal`."""
    c = pts[vids].mean(axis=0)
    e1 = pts[vids[0]] - c
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    ang = [np.arctan2((pts[i] - c) @ e2, (pts[i] - c) @ e1) for i in vids]
    return [vids[i] for i in np.argsort(ang)]


# ---- Shapes


def build_shape(geo: GeoWriter, spec: dict[str, Any], diag: float) -> str:
    """Emit a region/domain shape; return its surface (2D) or volume (3D) variable."""
    shape, vals, nsd = spec.get("shape"), spec["values"], geo.nsd

    if nsd == 2 and (poly := _polygon_2d(spec, diag)) is not None:
        return geo.polygon(poly)

    if shape == "box":
        lo, hi = _pad3(vals[0]), _pad3(vals[1])
        return geo.occ("Box", [*lo, *(hi - lo)])

    if shape == "polyhedron":
        return geo.convex_polyhedron(np.asarray(vals, dtype=float))

    if shape == "sphere":
        c, r = _pad3(vals[0]), float(vals[1])
        return geo.occ("Disk", [c[0], c[1], 0.0, r]) if nsd == 2 else geo.occ("Sphere", [*c, r])

    if shape == "cylinder":
        c, r = _pad3(vals[0]), float(vals[1])
        u = _pad3(vals[2])
        u /= np.linalg.norm(u)
        # RAFT: |axial distance| <= height, so `height` is a half-length; omitted means unbounded
        h = float(vals[3]) if len(vals) > 3 else diag
        if nsd == 3:
            return geo.occ("Cylinder", [*(c - h * u), *(2 * h * u), r])
        if abs(abs(u[2]) - 1.0) < TOL:  # axis out of plane -> disk
            return geo.occ("Disk", [c[0], c[1], 0.0, r])
        raise NotImplementedError(f"2D cylinder with an oblique axis is not supported: {spec}")

    raise ValueError(f"Unknown shape '{shape}' (supported: box, polyhedron, sphere, cylinder)")


def _polygon_2d(spec: dict[str, Any], diag: float) -> Array | None:
    """CCW vertices for 2D shapes with straight edges (box, polyhedron, in-plane cylinder), else None."""
    shape, vals = spec.get("shape"), spec["values"]
    if shape == "box":
        (x0, y0), (x1, y1) = _pad3(vals[0])[:2], _pad3(vals[1])[:2]
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    if shape == "polyhedron":
        pts = np.asarray(vals, dtype=float)[:, :2]
        return pts[ConvexHull(pts).vertices]  # 2D hull vertices are already CCW
    if shape == "cylinder":
        c, r, u = _pad3(vals[0]), float(vals[1]), _pad3(vals[2])
        u /= np.linalg.norm(u)
        if abs(u[2]) > TOL:
            return None
        h = float(vals[3]) if len(vals) > 3 else diag
        n = np.array([-u[1], u[0], 0.0])
        corners = [c - h * u - r * n, c + h * u - r * n, c + h * u + r * n, c - h * u + r * n]
        return np.array(corners)[:, :2]
    return None


def contains(spec: dict[str, Any], xyz: npt.ArrayLike, nsd: int) -> bool:
    """Point-in-shape test matching RAFT's region logic."""
    shape, vals, x = spec.get("shape"), spec["values"], _pad3(xyz)
    if shape == "box":
        lo, hi = _pad3(vals[0]), _pad3(vals[1])
        return bool(np.all(x[:nsd] >= lo[:nsd] - TOL) and np.all(x[:nsd] <= hi[:nsd] + TOL))
    if shape == "polyhedron":
        pts = np.asarray(vals, dtype=float)
        k = pts.shape[1]
        hull = ConvexHull(pts)
        return bool(np.all(hull.equations[:, :-1] @ x[:k] + hull.equations[:, -1] < 1e-6))
    if shape == "sphere":
        return bool(np.linalg.norm(x - _pad3(vals[0])) <= float(vals[1]) + TOL)
    if shape == "cylinder":
        c, u = _pad3(vals[0]), _pad3(vals[2])
        u /= np.linalg.norm(u)
        a = (x - c) @ u
        h = float(vals[3]) if len(vals) > 3 else np.inf
        return bool(abs(a) <= h + TOL and np.linalg.norm(x - c - a * u) <= float(vals[1]) + TOL)
    raise ValueError(f"Unknown domain shape '{shape}'")


def _hull_facets(pts: Array) -> list[tuple[Array, float, list[int]]]:
    """Convex hull facets (coplanar triangles merged) as (unit normal, offset, vertex ids)."""
    hull = ConvexHull(pts)
    scale = max(np.ptp(pts, axis=0).max(), 1.0)
    facets: dict[tuple[float, ...], set[int]] = {}
    for simplex, eq in zip(hull.simplices, hull.equations, strict=True):
        key = tuple(np.round(np.append(eq[:3], eq[3] / scale), 6))
        facets.setdefault(key, set()).update(int(i) for i in simplex)
    return [(np.array(k[:3]), k[3] * scale, sorted(v)) for k, v in facets.items()]


def domain_diag(spec: dict[str, Any]) -> float:
    """Characteristic domain size, used to cap unbounded cylinders."""
    if spec["shape"] == "box":
        return float(np.linalg.norm(_pad3(spec["values"][1]) - _pad3(spec["values"][0])))
    if spec["shape"] == "polyhedron":
        return float(np.linalg.norm(np.ptp(np.asarray(spec["values"], dtype=float), axis=0)))
    vals = spec["values"]
    return 2.0 * float(vals[1]) + (2.0 * float(vals[3]) if len(vals) > 3 else 0.0)


# ---- Features


@dataclass
class FracPlane:
    """Bounded fracture plane: centre, in-plane half-extents a (along u) and b (along v)."""

    shape: str
    centre: Array
    a: float
    b: float
    theta1: float
    theta2: float
    theta3: float

    @property
    def frame(self) -> tuple[Array, Array, Array]:
        """In-plane axes (u, v) and normal, from R = Rx(theta1) @ Ry(theta2) @ Rz(theta3)."""
        t1, t2, t3 = np.deg2rad([self.theta1, self.theta2, self.theta3])
        rx = np.array([[1, 0, 0], [0, np.cos(t1), -np.sin(t1)], [0, np.sin(t1), np.cos(t1)]])
        ry = np.array([[np.cos(t2), 0, np.sin(t2)], [0, 1, 0], [-np.sin(t2), 0, np.cos(t2)]])
        rz = np.array([[np.cos(t3), -np.sin(t3), 0], [np.sin(t3), np.cos(t3), 0], [0, 0, 1]])
        r = rx @ ry @ rz
        return r[:, 0], r[:, 1], r[:, 2]

    def corners(self, grow: float = 0.0, offset: float = 0.0) -> Array:
        """Rectangle corners, optionally grown in-plane and shifted along the normal."""
        u, v, n = self.frame
        a, b = self.a + grow, self.b + grow
        c = self.centre + offset * n
        return np.array([c - a * u - b * v, c + a * u - b * v, c + a * u + b * v, c - a * u + b * v])


def fracture_planes(config_mesh: dict[str, Any], nsd: int) -> list[FracPlane]:
    """3D fracture specs [[xc, yc, zc], [a, b], [theta1, theta2, theta3]] -> bounded planes.

    `a`/`b` are half-extents along the plane's local x/y, which at zero angles lie in
    the xy-plane with normal +z; R = Rx(theta1) @ Ry(theta2) @ Rz(theta3) as elsewhere
    in RAFT. theta3 is required because the first two angles only orient the normal.
    """
    if nsd != 3:
        return []
    planes: list[FracPlane] = []
    for spec in config_mesh.get("fractures") or []:
        shape = spec.get("shape", "rectangle")
        if shape not in ("rectangle", "ellipse"):
            raise ValueError(f"Fracture shape must be 'rectangle' or 'ellipse', got {shape!r}")
        centre, extents, angles = spec["values"][0], spec["values"][1], spec["values"][2]
        if len(angles) != 3:
            raise ValueError(
                f"3D fracture {len(planes)}: expected three angles [theta1, theta2, theta3], got "
                f"{list(angles)!r}. Two angles only orient the normal; theta3 spins the plane about "
                "it and sets where the a/b axes point (e.g. [0, 0, 90] puts a along y)."
            )
        planes.append(
            FracPlane(shape, _pad3(centre), float(extents[0]), float(extents[1]), *(float(t) for t in angles))
        )
    return planes


def fracture_segments(config_mesh: dict[str, Any], nsd: int) -> list[npt.NDArray[np.float64]]:
    """2D fracture specs [[xc, yc], [half_length], [angle_deg]] -> segment endpoints."""
    specs = config_mesh.get("fractures") or []
    if nsd != 2:
        return []
    segs = []
    for spec in specs:
        c = _pad3(spec["values"][0])
        half = float(spec["values"][1][0])
        th = np.deg2rad(float(spec["values"][2][0]))
        u = np.array([np.cos(th), np.sin(th), 0.0])
        segs.append(np.array([c - half * u, c + half * u]))
    return segs


Array = npt.NDArray[np.float64]


def warn_steep_stages(profiles: dict[str, Profile], outer: float, nsd: int) -> None:  # noqa: D417
    """Warn about stages whose size grows faster than the mesher can follow.

    A field growing by much more than ~30% per element is stepped over and the
    refinement missed entirely; 2D tolerates far more than 3D.
    """
    limit = 0.3 if nsd == 3 else 1.0
    if not profiles:
        return
    for name, profile in profiles.items():
        start = 0.0
        for k, length in enumerate(profile.lengths):
            nxt = profile.sizes[k + 1] if k + 1 < len(profile.sizes) else outer
            slope = (nxt - profile.sizes[k]) / length
            if slope > limit:
                needed = (nxt - profile.sizes[k]) / limit
                warnings.warn(
                    f"{name} stage {k + 1} ({profile.sizes[k]:g} -> {nxt:g} over {length:g}) grows at "
                    f"{slope:.2g} per unit distance; {nsd}D meshing may ignore it. Lengthen the stage to "
                    f"~{needed:.3g} or lower outer.",
                    stacklevel=3,
                )
            start += length


def point_shells(pts: list[Array], profile: Profile, nsd: int, diag: float) -> list[tuple[Array, float]]:
    """Boxes at each 3D point BC's stage radii, so the stages exist as geometry.

    The 3D mesher samples the size field too coarsely to find a small refined zone
    around a point, but it has to mesh a box. Straight-sided, since curved volumes
    clipping into each other leave sliver facets that the mesher rejects.
    """
    if nsd != 3:
        return []
    radii = np.cumsum(profile.lengths)
    return [(p, float(r)) for p in pts for r in radii if r < 0.5 * diag]


def line_shells(
    lines: list[Array], profile: Profile, nsd: int, diag: float
) -> list[tuple[Array, Array, float]]:
    """Square prisms along each 3D line BC segment at the profile's stage radii.

    The line is meshed in 1D, but the zone around it is missed like a point's.
    Straight-sided for the same reason as the point-BC boxes.
    """
    if nsd != 3:
        return []
    radii = [float(r) for r in np.cumsum(profile.lengths) if r < 0.5 * diag]
    return [
        (a, b, r)
        for pl in lines
        for a, b in zip(pl[:-1], pl[1:], strict=True)
        for r in radii
        if np.linalg.norm(b - a) > TOL
    ]


def emit_frac_plane(geo: GeoWriter, fp: FracPlane, grow: float = 0.0, offset: float = 0.0) -> str:
    """Emit the fracture surface (optionally grown/offset, for slabs); return its variable."""
    if fp.shape == "rectangle":
        return geo.polygon(fp.corners(grow, offset))
    c = fp.centre + offset * fp.frame[2]
    disk = geo.occ("Disk", [*fp.centre, fp.a + grow, fp.b + grow])
    for axis, angle in (([0, 0, 1], fp.theta3), ([0, 1, 0], fp.theta2), ([1, 0, 0], fp.theta1)):
        geo.rotate(disk, "Surface", axis, fp.centre, angle)  # Rz, then Ry, then Rx
    if offset:
        d = c - fp.centre
        geo.emit(f"Translate {{{d[0]:.10g}, {d[1]:.10g}, {d[2]:.10g}}} {{ Surface{{{disk}}}; }}")
    return disk


def frac_slabs(planes: list[FracPlane], profile: Profile, diag: float) -> list[tuple[FracPlane, float]]:
    """Stage radii for the slab volumes wrapped around each 3D fracture plane."""
    radii = [float(r) for r in np.cumsum(profile.lengths) if r < 0.5 * diag]
    return [(fp, r) for fp in planes for r in radii]


def _plane_dist_expr(fp: FracPlane) -> str:
    """Distance to a bounded plane: exact for a rectangle, radial approximation for an ellipse."""
    u, v, n = fp.frame
    c = fp.centre

    def proj(vec: Array) -> str:
        terms = (f"({x} - ({c[i]:.10g}))*({vec[i]:.10g})" for i, x in enumerate("xyz"))
        return "(" + " + ".join(terms) + ")"

    dn, ds, dt = proj(n), proj(u), proj(v)
    if fp.shape == "rectangle":
        out = f"Max(0, Abs({ds}) - {fp.a:.10g})"
        in_plane = f"({out})^2 + (Max(0, Abs({dt}) - {fp.b:.10g}))^2"
    else:
        # Radial excess outside the ellipse: r - r_edge, with r_edge = a*b*r / sqrt(b^2 s^2 + a^2 t^2)
        rad = f"Sqrt(({ds})^2 + ({dt})^2)"
        edge = f"{fp.a * fp.b:.10g}*{rad}/Sqrt({fp.b**2:.10g}*({ds})^2 + {fp.a**2:.10g}*({dt})^2 + 1e-12)"
        in_plane = f"(Max(0, {rad} - ({edge})))^2"
    return f"Sqrt(({dn})^2 + {in_plane})"


def warn_outside_domain(name: str, polylines: list[Array], domain: dict[str, Any], nsd: int) -> None:
    """Warn about geometry the domain clip will trim or drop entirely."""
    for i, pl in enumerate(polylines):
        samples = [pl[0]] + [
            a + (b - a) * f for a, b in zip(pl[:-1], pl[1:], strict=True) for f in (0.5, 1.0)
        ]
        inside = [contains(domain, p, nsd) for p in samples]
        if not any(inside):
            warnings.warn(f"{name} {i} lies entirely outside the domain; it will not appear.", stacklevel=3)
        elif not all(inside):
            warnings.warn(f"{name} {i} extends outside the domain and will be clipped.", stacklevel=3)


def point_seeds(pts: list[Array], size: float, domain: dict[str, Any], nsd: int) -> list[Array]:
    """Short segments running inward from each 3D point BC, already inside the domain.

    Meshed in 1D, their nodes anchor refinement that a lone embedded vertex never gets.
    Inward because a segment lying in a boundary face is left out of the tets, which
    would orphan its nodes.
    """
    if nsd != 3:
        return []
    half = 4.0 * size
    seeds = []
    for p in pts:
        for e in np.eye(3):
            for direction in (e, -e):
                tip = p + half * direction
                if contains(domain, tip, nsd) and not _on_boundary(domain, tip, nsd):
                    seeds.append(np.array([p, tip]))
    return seeds


def _on_boundary(domain: dict[str, Any], xyz: Array, nsd: int, rtol: float = 1e-9) -> bool:
    """Is the point on the domain's outer boundary (box domains only)?"""
    if domain.get("shape") != "box":
        return False
    lo, hi = _pad3(domain["values"][0]), _pad3(domain["values"][1])
    tol = rtol * max(float(np.max(hi - lo)), 1.0)
    return bool(np.any(np.abs(xyz[:nsd] - lo[:nsd]) < tol) or np.any(np.abs(xyz[:nsd] - hi[:nsd]) < tol))


def well_alignments(config_mesh: dict[str, Any], domain: dict[str, Any], nsd: int) -> list[Array]:
    """Well `alignment` segments, clipped to the domain.

    Each well is a list of [start, end] pairs that may run outside the reservoir; only
    the part inside is meshed, so `dL_max` (RAFT's spacing outside) does not apply here.
    """
    specs = config_mesh.get("wells") or []
    if specs and nsd != 3:
        warnings.warn("Wells are only supported in 3D; skipping.", stacklevel=2)
        return []
    segments = []
    for i, spec in enumerate(specs):
        alignment = spec.get("alignment", spec.get("alignments"))
        if alignment is None:
            raise ValueError(f"Well {i}: missing `alignment` (a list of [start, end] point pairs)")
        trimmed = dropped = 0
        for seg in alignment:
            if len(seg) != 2 or not all(isinstance(q, (list, tuple)) for q in seg):
                raise ValueError(
                    f"Well {i}: each `alignment` entry must be a [start, end] pair of points, got {seg!r}"
                )
            a = _pad3(_check_point(seg[0], f"Well {i}: alignment"))
            b = _pad3(_check_point(seg[1], f"Well {i}: alignment"))
            inside = _clip_segment(domain, a, b, nsd)
            if inside is None:
                dropped += 1
                continue
            segments.append(np.array(inside))
            trimmed += int(not (np.allclose(inside[0], a) and np.allclose(inside[1], b)))
        if trimmed or dropped:
            detail = ", ".join(
                part
                for part in (
                    f"{trimmed} segment(s) clipped to the domain boundary" if trimmed else "",
                    f"{dropped} segment(s) dropped as entirely outside the domain" if dropped else "",
                )
                if part
            )
            warnings.warn(f"Well {i}: {detail}.", stacklevel=2)
    return segments


def _clip_segment(domain: dict[str, Any], a: Array, b: Array, nsd: int) -> tuple[Array, Array] | None:
    """Trim a segment to the (convex) domain; None if it lies entirely outside."""
    ts = np.linspace(0.0, 1.0, 101)
    inside = [i for i, t in enumerate(ts) if contains(domain, a + t * (b - a), nsd)]
    if not inside:
        return None
    lo, hi = ts[inside[0]], ts[inside[-1]]
    if inside[0] > 0:
        lo = _bisect_boundary(domain, a, b, ts[inside[0] - 1], lo, nsd)
    if inside[-1] < len(ts) - 1:
        hi = _bisect_boundary(domain, a, b, ts[inside[-1] + 1], hi, nsd)
    return (a + lo * (b - a), a + hi * (b - a))


def _bisect_boundary(
    domain: dict[str, Any], a: Array, b: Array, outside: float, inside: float, nsd: int
) -> float:
    """Parameter of the domain boundary between an outside and an inside point."""
    for _ in range(40):
        mid = 0.5 * (outside + inside)
        if contains(domain, a + mid * (b - a), nsd):
            inside = mid
        else:
            outside = mid
    return inside


def bc_features(config: dict[str, Any]) -> tuple[list[Array], list[Array]]:
    """Collect point (`coords`) and polyline (`lines`) BC geometry. `faces` BCs need no geometry."""
    points, lines = [], []
    for group in config.get("boundary_conditions") or []:
        label = group.get("label", "unlabeled")
        points += [_pad3(_check_point(xi, f"BC {label}: coords")) for xi in group.get("coords", [])]
        for line in group.get("lines", []):
            pts = [_pad3(_check_point(p, f"BC {label}: lines")) for p in _check_line(line, label)]
            lines.append(np.array(pts))
    return points, lines


def _check_point(value: Any, where: str) -> Any:
    """A point must be a list of 2-3 numbers, not a bare coordinate."""
    if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= 3:
        raise ValueError(
            f"{where}: expected a point like [x, y, z], got {value!r}. "
            "`coords` takes a list of points and `lines` a list of lines, each a list of points, "
            "e.g. lines: [[[0, 152, 340], [10, 152, 340]]]"
        )
    return value


def _check_line(line: Any, label: str) -> list[Any]:
    """A line must be a list of points; one bracket too few silently makes nonsense geometry."""
    if not isinstance(line, (list, tuple)) or len(line) < 2 or not isinstance(line[0], (list, tuple)):
        raise ValueError(
            f"BC {label}: each entry of `lines` must be a list of points, got {line!r}. "
            "Note the extra brackets: lines: [[[0, 152, 340], [10, 152, 340]]]"
        )
    return list(line)


# ---- Size fields


def _seg_dist_expr(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], nsd: int) -> str:
    """MathEval expression for distance to segment a-b (clamp written with Abs only)."""
    d = b - a
    comps = "xyz"[:nsd]
    l2 = float(d[:nsd] @ d[:nsd])
    if l2 < TOL:
        return _pt_dist_expr(a, nsd)
    t_raw = " + ".join(f"({c} - ({a[i]:.10g}))*({d[i]:.10g})" for i, c in enumerate(comps))
    t = f"((Abs(({t_raw})/{l2:.10g}) - Abs(({t_raw})/{l2:.10g} - 1) + 1)/2)"
    terms = (f"({c} - ({a[i]:.10g}) - {t}*({d[i]:.10g}))^2" for i, c in enumerate(comps))
    return "Sqrt(" + " + ".join(terms) + ")"


def _pt_dist_expr(p: npt.NDArray[np.float64], nsd: int) -> str:
    return "Sqrt(" + " + ".join(f"({c} - ({p[i]:.10g}))^2" for i, c in enumerate("xyz"[:nsd])) + ")"


def _nested(fn: str, exprs: list[str]) -> str:
    """Fold with binary Min/Max (MathEval's multi-argument behaviour is not relied on)."""
    out = exprs[0]
    for e in exprs[1:]:
        out = f"{fn}({out}, {e})"
    return out


def _box_planes(domain: dict[str, Any], nsd: int) -> list[tuple[Array, float]]:
    """Face planes (n, b) with n.x + b = 0 of a box domain; empty for other domain shapes."""
    if domain.get("shape") != "box":
        return []
    lo, hi = _pad3(domain["values"][0]), _pad3(domain["values"][1])
    planes = []
    for k in range(nsd):
        e = np.eye(3)[k]
        planes += [(-e, float(lo[k])), (e, -float(hi[k]))]
    return planes


def _on_domain_boundary(pts: Array, planes: list[tuple[Array, float]], tol: float) -> bool:
    return any(np.all(np.abs(_pad3_rows(pts) @ n + b) < tol) for n, b in planes)


def _pad3_rows(pts: Array) -> Array:
    return np.array([_pad3(p) for p in np.atleast_2d(pts)])


def region_boundary_dist(
    spec: dict[str, Any], nsd: int, diag: float, planes: list[tuple[Array, float]]
) -> str | None:
    """MathEval distance to a region's boundary, skipping parts on the (box) domain boundary.

    Exact for circles/spheres/polygons; for 3D polyhedra and finite cylinders it is the
    usual max-of-planes estimate (exact inside, slightly low near outside edges/corners).
    """
    shape, vals = spec.get("shape"), spec["values"]
    tol = 1e-6 * diag
    if shape == "sphere" or (shape == "cylinder" and nsd == 2 and abs(_pad3(vals[2])[2]) > TOL):
        return f"Abs({_pt_dist_expr(_pad3(vals[0]), nsd)} - {float(vals[1]):.10g})"
    if nsd == 2:
        poly = _polygon_2d(spec, diag)
        if poly is None:
            return None
        edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
        exprs = [
            _seg_dist_expr(_pad3(a), _pad3(b), 2)
            for a, b in edges
            if not _on_domain_boundary(np.array([a, b]), planes, tol)
        ]
        return _nested("Min", exprs) if exprs else None
    if shape in ("box", "polyhedron"):
        pts = np.asarray(vals, dtype=float)
        if shape == "box":
            lo, hi = _pad3(vals[0]), _pad3(vals[1])
            pts = np.array(
                [[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]
            )
        exprs = [
            f"(({n[0]:.10g})*x + ({n[1]:.10g})*y + ({n[2]:.10g})*z + ({b:.10g}))"
            for n, b, vids in _hull_facets(pts)
            if not _on_domain_boundary(pts[vids], planes, tol)
        ]
        return f"Abs({_nested('Max', exprs)})" if exprs else None
    if shape == "cylinder":
        c, r, u = _pad3(vals[0]), float(vals[1]), _pad3(vals[2])
        u /= np.linalg.norm(u)
        axial = " + ".join(f"({v} - ({c[i]:.10g}))*({u[i]:.10g})" for i, v in enumerate("xyz"))
        radial = f"Sqrt(Max(({_pt_dist_expr(c, 3)})^2 - ({axial})^2, 0))"
        if len(vals) > 3:
            return f"Abs(Max({radial} - {r:.10g}, Abs({axial}) - {float(vals[3]):.10g}))"
        return f"Abs({radial} - {r:.10g})"
    return None


def _geo_string(s: str, chunk: int = 800) -> str:
    """Quoted .geo string; long ones are split with StrCat since the parser caps literal length."""
    if len(s) <= chunk:
        return f'"{s}"'
    return "StrCat(" + ", ".join(f'"{s[i : i + chunk]}"' for i in range(0, len(s), chunk)) + ")"


def write_size_fields(
    geo: GeoWriter, features: list[tuple[str, Profile]], sizes: Sizes, d_b: str | None
) -> None:
    """Background field = Min of the per-feature profiles and the region-boundary fields.

    Analytic distances are used because BooleanFragments renumbers the curves/points.
    """
    outer = sizes.size_outer
    geo.emit("// ---- Mesh size fields")
    geo.emit(f"Mesh.MeshSizeMax = {outer:.10g};")
    geo.emit("Mesh.MeshSizeFromPoints = 0;")
    geo.emit("Mesh.MeshSizeFromCurvature = 0;")
    geo.emit("Mesh.MeshSizeExtendFromBoundary = 0;")
    fid = 0
    size_fields: list[int] = []

    def math_eval(expr: str) -> int:
        nonlocal fid
        fid += 1
        geo.emit(f"Field[{fid}] = MathEval; Field[{fid}].F = {_geo_string(expr)};")
        return fid

    for dist, profile in features:
        size_fields.append(math_eval(profile.expr(dist, outer)))

    if d_b is not None:
        geo.emit("// Region boundaries: feature profile at a fraction of the distance, then relaxing")
        cap = sizes.size_region_max
        slopes = [p.last_slope(outer) for _, p in features] or [0.0]
        size_fields.append(math_eval(f"{cap:.10g} + {min(slopes):.10g}*({d_b})"))
        for dist, profile in features:
            near = profile.expr(dist, outer, scale=sizes.boundary_growth)
            relax = f"{profile.last_slope(outer):.10g}*({d_b})"
            size_fields.append(math_eval(f"Min({cap:.10g}, {near}) + {relax}"))

    if size_fields:
        fid += 1
        geo.emit(f"Field[{fid}] = Min; Field[{fid}].FieldsList = {{{', '.join(map(str, size_fields))}}};")
        geo.emit(f"Background Field = {fid};")


# ---- Driver


def generate_geo(
    config: dict[str, Any],
    sizes: Sizes,
    quads: bool = False,
) -> str:
    """Return .geo text for the mesh described by a RAFT config dict."""
    cm = config["mesh"]
    nsd = int(cm["dimensions"])
    if "domain" not in cm:
        raise KeyError("Config is missing `mesh.domain` (outer boundary); see script docstring.")
    domain = cm["domain"]
    diag = domain_diag(domain)
    ent = "Surface" if nsd == 2 else "Volume"

    geo = GeoWriter(nsd)
    _emit_header(geo, quads)

    geo.emit("// ---- Domain")
    dom = build_shape(geo, domain, diag)
    geo.emit()

    # Embedded curves (fractures, line BCs), clipped to the domain
    fracs = fracture_segments(cm, nsd)
    bc_pts, bc_lines = bc_features(config)
    bc_pts = _inside_domain(bc_pts, domain, nsd)
    warn_outside_domain("Fracture", fracs, domain, nsd)
    warn_outside_domain("Line BC", bc_lines, domain, nsd)

    wells = well_alignments(cm, domain, nsd)
    frac_planes_3d = fracture_planes(cm, nsd)
    regions = _region_specs(cm)
    shells = point_shells(bc_pts, sizes.bc, nsd, diag)
    tubes = line_shells(bc_lines, sizes.bc, nsd, diag) + line_shells(wells, sizes.well, nsd, diag)
    slabs = frac_slabs(frac_planes_3d, sizes.frac, diag)
    _emit_regions(geo, regions, shells, tubes, slabs, ent, dom, diag)

    geo.emit("fsurf[] = {};")
    for i, fp in enumerate(frac_planes_3d):
        geo.emit(f"// ---- Fracture {i} ({fp.shape})")
        _emit_clipped(geo, "fsurf", "Surface", emit_frac_plane(geo, fp), dom)
    geo.emit()

    geo.emit("curves[] = {};")
    for kind, polylines in (("Fracture", fracs), ("Line BC", bc_lines), ("Well", wells)):
        for i, pl in enumerate(polylines):
            geo.emit(f"// ---- {kind} {i}")
            for c in geo.polyline(pl[:, :nsd] if nsd == 2 else pl):
                _emit_clipped(geo, "curves", "Curve", c, dom)
    geo.emit()

    seeds = point_seeds(bc_pts, sizes.bc.sizes[0], domain, nsd)
    _emit_seeds_and_points(geo, seeds, bc_pts)

    geo.emit("// ---- Make everything conforming")
    tools = [f"{ent}{{regs[]}};"] if (regions or shells or tubes or frac_planes_3d) else []
    tools += ["Surface{fsurf[]};"] if frac_planes_3d else []
    tools += ["Curve{curves[]};"] if (fracs or bc_lines or wells or seeds) else []
    tools += ["Point{pts[]};"] if bc_pts else []
    if tools:
        geo.emit(f"BooleanFragments {{ {ent}{{{dom}}}; Delete; }}{{ {' '.join(tools)} Delete; }}")
    geo.emit()

    features = [(_seg_dist_expr(s[0], s[1], nsd), sizes.frac) for s in fracs]
    features += [(_plane_dist_expr(fp), sizes.frac) for fp in frac_planes_3d]
    for pl in bc_lines:
        pairs = zip(pl[:-1], pl[1:], strict=True)
        features += [(_seg_dist_expr(a, b, nsd), sizes.bc) for a, b in pairs]
    features += [(_pt_dist_expr(p, nsd), sizes.bc) for p in bc_pts]
    features += [(_seg_dist_expr(w[0], w[1], nsd), sizes.well) for w in wells]

    field_only = {"frac": sizes.frac} if fracs else {}  # 3D planes get slabs, like points/lines
    if (bc_lines or bc_pts) and nsd == 2:
        field_only["bc"] = sizes.bc
    warn_steep_stages(field_only, sizes.size_outer, nsd)
    planes = _box_planes(domain, nsd)
    dists = [d for spec in regions if (d := region_boundary_dist(spec, nsd, diag, planes))]
    write_size_fields(geo, features, sizes, _nested("Min", dists) if dists else None)

    return "\n".join(geo.out) + "\n"


def _region_specs(config_mesh: dict[str, Any]) -> list[dict[str, Any]]:
    """Region shapes from the config; a premade element list yields no geometry."""
    regions = config_mesh.get("regions") or []
    if isinstance(regions, str):
        warnings.warn("`mesh.regions` is a file list; no region geometry generated.", stacklevel=3)
        return []
    return list(regions)


def _emit_regions(  # noqa: PLR0913
    geo: GeoWriter,
    regions: list[dict[str, Any]],
    shells: list[tuple[Array, float]],
    tubes: list[tuple[Array, Array, float]],
    slabs: list[tuple[FracPlane, float]],
    ent: str,
    dom: str,
    diag: float,
) -> None:
    """Region shapes plus BC/fracture refinement volumes, each clipped to the domain."""
    geo.emit("regs[] = {};")
    for i, spec in enumerate(regions):
        geo.emit(f"// ---- Region {i} (region_number {spec['region_number']}, {spec['shape']})")
        _emit_clipped(geo, "regs", ent, build_shape(geo, spec, diag), dom)
    for i, (centre, radius) in enumerate(shells):
        geo.emit(f"// ---- Point BC refinement box {i} (r = {radius:g})")
        corner = centre - radius
        _emit_clipped(geo, "regs", ent, geo.occ("Box", [*corner, *([2 * radius] * 3)]), dom)
    for i, (a, b, radius) in enumerate(tubes):
        geo.emit(f"// ---- Line BC refinement prism {i} (r = {radius:g})")
        e1, e2 = _perp_frame(b - a)
        face = geo.polygon(
            np.array(
                [
                    a - radius * e1 - radius * e2,
                    a + radius * e1 - radius * e2,
                    a + radius * e1 + radius * e2,
                    a - radius * e1 + radius * e2,
                ]
            )
        )
        _emit_clipped(geo, "regs", ent, geo.extrude(face, b - a), dom)
    for i, (fp, radius) in enumerate(slabs):
        geo.emit(f"// ---- Fracture refinement slab {i} (r = {radius:g})")
        face = geo.polygon(fp.corners(radius, -radius))
        _emit_clipped(geo, "regs", ent, geo.extrude(face, 2 * radius * fp.frame[2]), dom)
    geo.emit()


def _perp_frame(direction: Array) -> tuple[Array, Array]:
    """Two unit vectors spanning the plane perpendicular to `direction`."""
    u = direction / np.linalg.norm(direction)
    seed = np.eye(3)[int(np.argmin(np.abs(u)))]  # least-aligned axis, so the cross product is stable
    e1 = np.cross(u, seed)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def _emit_seeds_and_points(geo: GeoWriter, seeds: list[Array], bc_pts: list[Array]) -> None:
    """Seed segments (already inside the domain, so no clipping) and the point BCs themselves."""
    for i, seg in enumerate(seeds):
        geo.emit(f"// ---- Point BC seed {i}")
        for c in geo.polyline(seg):
            geo.emit(f"curves[] += {c};")
    geo.emit()

    # Points outside the domain were dropped earlier; they would leave orphan nodes
    geo.emit("pts[] = {};")
    for i, p in enumerate(bc_pts):
        geo.emit(f"// ---- Point BC {i}")
        geo.emit(f"pts[] += {geo.point(p)};")
    geo.emit()


def _emit_header(geo: GeoWriter, quads: bool) -> None:
    geo.emit("// Generated by scripts/raft_to_geo.py -- edit the RAFT config and regenerate instead.")
    geo.emit('SetFactory("OpenCASCADE");')
    geo.emit("Mesh.MshFileVersion = 2.2;  // RAFT reads msh v2")
    # Set explicitly so user gmsh option files can't recombine 3D surfaces
    if quads and geo.nsd == 2:
        geo.emit("Mesh.RecombineAll = 1; Mesh.RecombinationAlgorithm = 2;  // full-quad Q4")
    else:
        geo.emit("Mesh.RecombineAll = 0;")
    geo.emit()


def _inside_domain(pts: list[Array], domain: dict[str, Any], nsd: int) -> list[Array]:
    """Drop points outside the domain (they would become orphan nodes)."""
    kept = []
    for p in pts:
        if contains(domain, p, nsd):
            kept.append(p)
        else:
            warnings.warn(f"BC point {p[:nsd].tolist()} lies outside the domain; skipping.", stacklevel=3)
    return kept


def _emit_clipped(geo: GeoWriter, target: str, kind: str, tag: str, dom: str) -> None:
    """Intersect entity `tag` with a copy of the domain, appending the result to `target[]`."""
    ent = "Surface" if geo.nsd == 2 else "Volume"
    geo.emit(f"dup[] = Translate {{0, 0, 0}} {{ Duplicata {{ {ent}{{{dom}}}; }} }};")
    geo.emit(
        f"{target}[] += BooleanIntersection {{ {kind}{{{tag}}}; Delete; }}{{ {ent}{{dup[0]}}; Delete; }};"
    )


def _resolve(name: Path, suffix: str) -> Path:
    """Paths with a directory are used as given; bare names go in config/ (suffix optional)."""
    path = name if name.suffix else name.with_suffix(suffix)
    return path if name.parent != Path() else CONFIG_DIR / path


def _floats(text: str | None) -> list[float] | None:
    """Parse a comma-separated CLI list."""
    return [float(v) for v in text.replace(",", " ").split()] if text else None


def _profile(
    name: str, spec: Any, sizes_cli: list[float] | None, dists_cli: list[float] | None, ctx: _SizeCtx
) -> Profile:
    """Build one feature's profile from CLI lists over a config value (number or [[sizes], [lengths]])."""
    sizes, lengths = sizes_cli, dists_cli
    if sizes is None or lengths is None:
        if isinstance(spec, (int, float)):
            sizes = sizes or [float(spec)]
        elif isinstance(spec, list) and len(spec) == 2:
            sizes = sizes or [float(v) for v in spec[0]]
            lengths = lengths or [float(v) for v in spec[1]]
        elif spec is not None:
            ctx.fail(f"{name}: expected a number or [[s1, ..., sn], [L1, ..., Ln]], got {spec!r}")
    sizes = sizes or [ctx.outer]
    lengths = lengths or [ctx.transition] * len(sizes)

    if len(sizes) != len(lengths):
        ctx.fail(f"{name}: {len(sizes)} sizes but {len(lengths)} lengths; they must match")
    if any(v <= 0 for v in sizes + lengths):
        ctx.fail(f"{name}: sizes and lengths must be positive")
    if max(sizes) > ctx.outer:
        ctx.fail(f"{name}: sizes must not exceed outer ({ctx.outer:g})")
    return Profile(sizes, lengths)


@dataclass
class _SizeCtx:
    """Shared values and error reporting while parsing size options."""

    outer: float
    transition: float
    fail: Callable[[str], None]


def _sizes_from(args: argparse.Namespace, sizing: dict[str, Any], ap: argparse.ArgumentParser) -> Sizes:
    """Merge CLI flags over the config's `mesh.sizing` block and validate."""

    def pick(flag: str, key: str) -> float | None:
        val = getattr(args, flag)
        return float(val) if val is not None else (float(sizing[key]) if key in sizing else None)

    outer = pick("size_outer", "outer")
    if outer is None:
        ap.error("no outer size: set mesh.sizing.outer in the config or pass --size-outer")
    ctx = _SizeCtx(outer, pick("transition", "transition") or 5.0 * outer, ap.error)
    frac = _profile("frac", sizing.get("frac"), _floats(args.size_frac), _floats(args.dist_frac), ctx)
    bc = _profile("bc", sizing.get("bc"), _floats(args.size_bc), _floats(args.dist_bc), ctx)
    well = _profile("well", sizing.get("well"), _floats(args.size_well), _floats(args.dist_well), ctx)

    finest = min(frac.finest, bc.finest, well.finest)
    region_max = pick("size_region_max", "region_max") or 0.5 * (finest + outer)
    growth = pick("boundary_growth", "boundary_growth")
    growth = 0.25 if growth is None else growth
    if not 0 < region_max <= outer or growth < 0:
        ap.error("expected 0 < region_max <= outer and boundary_growth >= 0")
    return Sizes(outer, frac, bc, well, region_max, growth)


def write_msh(geo_path: Path, nsd: int) -> Path:
    """Mesh the .geo with the gmsh Python API (e.g. `uv run --with gmsh`)."""
    try:
        import gmsh  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError:
        sys.exit("--mesh needs the gmsh Python package: run via `uv run --with gmsh ...` or `just geo`")
    msh = geo_path.with_suffix(".msh")
    gmsh.initialize()
    try:
        gmsh.open(str(geo_path))
        gmsh.model.mesh.generate(nsd)
        gmsh.write(str(msh))
    finally:
        gmsh.finalize()
    return msh


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="RAFT YAML config (bare name -> config/<name>.yaml)")
    ap.add_argument(
        "-o", "--output", type=Path, help="output .geo (bare name -> config/; default: mesh.file stem)"
    )
    sz = ap.add_argument_group("size overrides (default: config mesh.sizing)")
    sz.add_argument("--size-outer", type=float, help="element size at the outer boundary")
    sz.add_argument("--size-frac", help="fracture sizes, comma-separated for a multi-stage profile")
    sz.add_argument("--dist-frac", help="fracture stage lengths, comma-separated")
    sz.add_argument("--size-bc", "--size-BC", help="point/line BC sizes, comma-separated")
    sz.add_argument("--dist-bc", "--dist-BC", help="BC stage lengths, comma-separated")
    sz.add_argument("--size-well", help="well sizes, comma-separated")
    sz.add_argument("--dist-well", help="well stage lengths, comma-separated")
    sz.add_argument("--transition", type=float, help="stage length when only sizes are given")
    sz.add_argument("--size-region-max", type=float, help="coarsest size on region boundaries")
    sz.add_argument("--boundary-growth", type=float, help="boundary grading as fraction of normal slope")
    sz.add_argument("--quads", action="store_true", default=None, help="Q4 elements instead of T3 (2D only)")
    ap.add_argument("--mesh", action="store_true", help="also write the .msh (needs the gmsh package)")
    ap.add_argument("-f", "--force", action="store_true", help="overwrite existing output files")
    args = ap.parse_args(argv)

    args.config = _resolve(args.config, ".yaml")
    if not args.config.exists():
        ap.error(f"config not found: {args.config}")
    config = yaml.safe_load(args.config.read_text())
    sizing = config["mesh"].get("sizing") or {}
    sizes = _sizes_from(args, sizing, ap)
    quads = bool(sizing.get("quads", False)) if args.quads is None else True

    if "domain" not in config["mesh"]:
        ap.error(f"{args.config} has no `mesh.domain` (outer boundary); see --help")

    out = (
        _resolve(args.output, ".geo")
        if args.output
        else args.config.parent / Path(config["mesh"]["file"]).with_suffix(".geo")
    )
    targets = [out, out.with_suffix(".msh")] if args.mesh else [out]
    if (existing := [str(t) for t in targets if t.exists()]) and not args.force:
        ap.error(f"already exists: {', '.join(existing)} (use --force to overwrite or -o for another name)")

    out.write_text(generate_geo(config, sizes, quads))
    print(f"Wrote {out}")
    if args.mesh:
        print(f"Wrote {write_msh(out, int(config['mesh']['dimensions']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
