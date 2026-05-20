"""3D (multi-layer) electrical bundle routing for the gpdk PDK.

Wraps `doroutes.multilayer` so it can be invoked as a gdsfactory
``routing_strategy`` from schematic-driven layout YAMLs, and persists
per-net waypoint corners to disk so re-runs only recompute the wires
whose endpoints or configuration actually changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import doroutes
import gdsfactory as gf
import yaml
from doroutes.multilayer import (
    MetalLayerSpec,
    PortGeometry,
    RouteNetSpec,
    RoutingConfig,
    route_batch_multilayer,
)
from doroutes.multilayer.draw import _draw_dynamic_geometry_for_corners
from doroutes.multilayer.engine_multinet import _precompute_port_geometries


# Older doroutes builds don't accept stats_sink on route_batch_multilayer.
# Feature-test once so route_bundle_3d stays compatible with both.
def _route_batch_supports_stats_sink() -> bool:
    import inspect

    try:
        sig = inspect.signature(route_batch_multilayer)
    except (TypeError, ValueError):
        return False
    return "stats_sink" in sig.parameters


_ROUTE_BATCH_STATS_SINK = _route_batch_supports_stats_sink()

__all__ = [
    "GPDK_CONFIG",
    "route_bundle_3d",
    "via_m2_m3",
    "WaypointCache",
    "default_cache_dir",
    "active_stats_sink",
]

# Benchmark/profiling hook. When a caller sets this ContextVar (e.g. via
# ``active_stats_sink.set({})`` in a benchmark harness), ``route_bundle_3d``
# also writes stage timings into that dict — even when invoked indirectly
# through a ``@gf.cell``-wrapped component factory that can't forward the
# ``stats_sink`` kwarg without busting its cache key.
active_stats_sink: ContextVar[dict[str, Any] | None] = ContextVar(
    "active_stats_sink", default=None
)

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1


@gf.cell
def via_m2_m3(width: float = 11.0, length: float = 11.0) -> gf.Component:
    """Via transition between M2 and M3 for 3D electrical routing."""
    return gf.components.via_stack(
        size=(width, length), layers=("M2", "MTOP"), vias=("via2", None)
    )


GPDK_CONFIG = RoutingConfig(
    metal_layers=(
        MetalLayerSpec(
            layer_tuple=(45, 0),  # M2
            preferred_direction="h",
            min_width=10.0,
            min_via_pad=11.0,
            below_cut_layer=None,
            pitch=20.0,
            spacing=5.0,
        ),
        MetalLayerSpec(
            layer_tuple=(49, 0),  # M3
            preferred_direction="v",
            min_width=10.0,
            min_via_pad=11.0,
            below_cut_layer=(43, 0),  # VIA2
            pitch=20.0,
            spacing=5.0,
        ),
    ),
    via_factory=via_m2_m3,
    via_metal_enclosure_add=0.5,
)


def default_cache_dir() -> Path:
    """Default location where per-net waypoint YAMLs are persisted."""
    return Path(__file__).parent / "samples" / "_waypoints_cache"


# --------------------------------------------------------------------------- #
# Waypoint cache
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _NetSignatureInputs:
    """Inputs that determine whether a cached net is still valid."""

    start_x_dbu: int
    start_y_dbu: int
    start_orientation: int
    start_layer: tuple[int, int]
    stop_x_dbu: int
    stop_y_dbu: int
    stop_orientation: int
    stop_layer: tuple[int, int]
    width: float
    clearance: float
    grid_unit: float
    layers_to_avoid: tuple[tuple[int, int], ...]
    config_digest: str
    doroutes_version: str
    schema_version: int = CACHE_SCHEMA_VERSION


def _via_factory_id(via_factory: Any) -> str:
    mod = getattr(via_factory, "__module__", "?")
    qn = getattr(via_factory, "__qualname__", repr(via_factory))
    return f"{mod}.{qn}"


def _config_digest(config: RoutingConfig) -> str:
    parts: list[str] = []
    for spec in config.metal_layers:
        parts.append(
            f"{spec.layer_tuple}:{spec.preferred_direction}:{spec.min_width}:"
            f"{spec.min_via_pad}:{spec.below_cut_layer}:{spec.pitch}:{spec.spacing}"
        )
    parts.append(f"enc+{config.via_metal_enclosure_add}")
    parts.append(f"via_factory={_via_factory_id(config.via_factory)}")
    parts.append(f"via_straighten={config.via_straighten_threshold_um}")
    parts.append(f"min_jog={config.min_jog_coalesce_factor}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _port_layer_tuple(port: gf.Port) -> tuple[int, int]:
    info = port.layer_info
    return (int(info.layer), int(info.datatype))


def _net_signature(
    net: RouteNetSpec,
    *,
    width: float,
    clearance: float,
    grid_unit: float,
    layers_to_avoid: tuple[tuple[int, int], ...],
    config: RoutingConfig,
    dbu: float,
) -> str:
    sig = _NetSignatureInputs(
        start_x_dbu=int(round(net.start.dcenter[0] / dbu)),
        start_y_dbu=int(round(net.start.dcenter[1] / dbu)),
        start_orientation=int(round(net.start.orientation)) % 360,
        start_layer=_port_layer_tuple(net.start),
        stop_x_dbu=int(round(net.stop.dcenter[0] / dbu)),
        stop_y_dbu=int(round(net.stop.dcenter[1] / dbu)),
        stop_orientation=int(round(net.stop.orientation)) % 360,
        stop_layer=_port_layer_tuple(net.stop),
        width=round(float(width), 6),
        clearance=round(float(clearance), 6),
        grid_unit=round(float(grid_unit), 6),
        layers_to_avoid=tuple(sorted((int(a), int(b)) for a, b in layers_to_avoid)),
        config_digest=_config_digest(config),
        doroutes_version=getattr(doroutes, "__version__", "unknown"),
    )
    return hashlib.sha1(repr(sig).encode()).hexdigest()[:24]


class WaypointCache:
    """On-disk cache of per-net routing corners, keyed by net signature."""

    # Uses .json so it doesn't show up in the `*.yml` globs that tests
    # use to count per-net cache files.
    MANIFEST_NAME = "_manifest.json"

    def __init__(self, cache_dir: Path | str | None):
        """Initialize the cache, falling back to ``default_cache_dir()`` when unset."""
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def path_for(self, signature: str) -> Path:
        return self.cache_dir / f"{signature}.yml"

    def manifest_path(self) -> Path:
        return self.cache_dir / self.MANIFEST_NAME

    def load_manifest(self) -> dict[str, Any]:
        p = self.manifest_path()
        if not p.is_file():
            return {}
        try:
            data = json.loads(p.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path().write_text(json.dumps(manifest, indent=2))

    def load(self, signature: str) -> list[tuple[int, int, int]] | None:
        path = self.path_for(signature)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            data = yaml.safe_load(path.read_text())
        except (yaml.YAMLError, ValueError, OSError) as exc:
            logger.warning("waypoint cache load failed for %s: %s", path, exc)
            self.misses += 1
            return None
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != CACHE_SCHEMA_VERSION
        ):
            self.misses += 1
            return None
        corners = data.get("corners", [])
        if not isinstance(corners, list):
            self.misses += 1
            return None
        self.hits += 1
        return [tuple(int(v) for v in c) for c in corners]  # type: ignore[misc]

    def save(
        self,
        signature: str,
        *,
        net_name: str,
        corners: list[tuple[int, int, int]],
        meta: dict[str, Any],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(signature)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "doroutes_version": getattr(doroutes, "__version__", "unknown"),
            "gdsfactory_version": getattr(gf, "__version__", "unknown"),
            "net_name": net_name,
            "signature": signature,
            "corners": [list(c) for c in corners],
            **meta,
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        self.writes += 1


# --------------------------------------------------------------------------- #
# route_bundle_3d
# --------------------------------------------------------------------------- #


def _derive_bundle_width(
    geom_cache: dict[tuple[int, int], PortGeometry],
    nets: list[RouteNetSpec],
    dbu: float,
    fallback: float,
) -> float:
    """Derive a bundle routing width from port geometry.

    Takes the per-net min longest-extent, then the min across nets. Matches the
    width-selection logic used in the existing DoRoutes 1x16 switch sample.
    """
    per_net: list[float] = []
    for net in nets:
        longest_pair: list[float] = []
        for port in (net.start, net.stop):
            key = (int(port.dcenter[0] / dbu), int(port.dcenter[1] / dbu))
            geom = geom_cache.get(key)
            if geom:
                longest_pair.append(max(geom.orig_extent_x_um, geom.orig_extent_y_um))
        if len(longest_pair) == 2:
            per_net.append(min(longest_pair))
    return min(per_net) if per_net else fallback


def _replay_cached_corners(
    c: gf.Component,
    *,
    net: RouteNetSpec,
    corners: list[tuple[int, int, int]],
    config: RoutingConfig,
    width: float,
    clearance: float,
    dbu: float,
    geom_cache: dict[tuple[int, int], PortGeometry],
) -> bool:
    """Draw a single cached net's metal + vias from persisted corners.

    Returns True if the draw succeeded.
    """
    start_key = (
        int(net.start.dcenter[0] / dbu),
        int(net.start.dcenter[1] / dbu),
    )
    stop_key = (int(net.stop.dcenter[0] / dbu), int(net.stop.dcenter[1] / dbu))
    start_geom = geom_cache.get(start_key)
    stop_geom = geom_cache.get(stop_key)
    assert start_geom is not None, (
        f"missing geom_cache entry for start port at {start_key}"
    )
    assert stop_geom is not None, (
        f"missing geom_cache entry for stop port at {stop_key}"
    )
    body_width = max(config.min_width(config.layer_m1), width)
    try:
        out = _draw_dynamic_geometry_for_corners(
            c=c,
            corners_3d=list(corners),
            start_geom=start_geom,
            stop_geom=stop_geom,
            body_width=body_width,
            width=width,
            dynamic_width=False,
            m1_polys=[],
            m2_polys=[],
            start_xy_dbu=start_key,
            stop_xy_dbu=stop_key,
            dbu=dbu,
            clearance=clearance,
            add_segment_ports=False,
            port_name_prefix=net.port_name_prefix,
            config=config,
            debug=False,
        )
    except Exception as exc:
        logger.warning("waypoint replay failed for net %r: %s", net.name, exc)
        return False
    return out is not None


def route_bundle_3d(
    component: gf.Component,
    ports1: list[gf.Port],
    ports2: list[gf.Port],
    *,
    config: RoutingConfig | None = None,
    grid_unit: float = 10.0,
    width: float | None = None,
    clearance: float = 5.0,
    layers_to_avoid: tuple[tuple[int, int], ...] = ((45, 0), (49, 0)),
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force: bool = False,
    net_prefix: str = "e",
    require_all: bool = False,
    debug: bool = False,
    stats_sink: dict[str, Any] | None = None,
    **_ignored: Any,  # gdsfactory's routing_strategy may pass extra kwargs (e.g. cross_section); accept and drop.
) -> list[Any]:
    """Route a bundle of electrical nets with 3D (multi-layer) A* routing.

    Signature matches gdsfactory's routing_strategy contract
    (component + ports1 + ports2) so it can be used directly from
    ``.pic.yml`` schematic files registered in ``PDK.routing_strategies``.

    Per-net waypoint corners are persisted to ``cache_dir`` (default:
    ``gpdk/samples/_waypoints_cache``) keyed by a signature of the net
    endpoints and routing configuration. Nets whose cache hits are re-drawn
    from corners without re-running A*; only unmatched nets go through
    ``doroutes.multilayer.route_batch_multilayer``.

    Args:
        component: The Component to draw into.
        ports1: Start ports.
        ports2: Stop ports.
        config: RoutingConfig. Defaults to GPDK_CONFIG (M2/M3 + VIA2).
        grid_unit: A* grid resolution in microns.
        width: Routing width. If None, derived from port geometry.
        clearance: Clearance from obstacles in microns.
        layers_to_avoid: Layers treated as full obstacles.
        cache_dir: Where to persist per-net waypoint YAMLs. None ⇒ default.
        use_cache: If False, ignore existing cache files.
        force: If True, re-route every net and overwrite cache files.
        net_prefix: Prefix used for generated RouteNetSpec names.
        require_all: If True, raise when a net cannot be routed.
        debug: Pass through to the router for verbose logging.
        stats_sink: Optional dict populated with per-phase timings and counts.

    Returns:
        Empty list. The routing is drawn directly on ``component``; this
        matches the return shape of ``add_bundle_astar`` used by the
        existing ``route_astar`` strategy.
    """
    if len(ports1) != len(ports2):
        raise ValueError(
            f"ports1 ({len(ports1)}) and ports2 ({len(ports2)}) must have equal length"
        )
    if not ports1:
        return []

    cfg = config or GPDK_CONFIG
    cache = WaypointCache(cache_dir)
    dbu = component.kcl.dbu

    _ambient_sink = active_stats_sink.get()
    _sinks = [s for s in (stats_sink, _ambient_sink) if s is not None]

    def _record(key: str, value: Any) -> None:
        for s in _sinks:
            s[key] = value

    nets: list[RouteNetSpec] = [
        RouteNetSpec(
            name=f"{net_prefix}{i}",
            start=p1,
            stop=p2,
            port_name_prefix=f"{net_prefix}{i}",
        )
        for i, (p1, p2) in enumerate(zip(ports1, ports2))
    ]

    def _compute_sigs(w: float) -> dict[str, str]:
        return {
            net.name: _net_signature(
                net,
                width=w,
                clearance=clearance,
                grid_unit=grid_unit,
                layers_to_avoid=tuple(layers_to_avoid),
                config=cfg,
                dbu=dbu,
            )
            for net in nets
        }

    # Fast-path: if a manifest width exists and every net has a cache file for
    # the signatures computed from that width, skip the expensive port-geometry
    # precompute and the width derivation. This is the pure-warm hot path.
    manifest = cache.load_manifest() if use_cache and not force else {}
    manifest_width = manifest.get("last_effective_width_um") if width is None else width
    signatures: dict[str, str] = {}
    cached_corners: dict[str, list[tuple[int, int, int]]] = {}
    cached_nets: list[RouteNetSpec] = []
    missing_nets: list[RouteNetSpec] = []
    geom_cache: dict[tuple[int, int], PortGeometry] = {}
    effective_width: float
    skipped_precompute = False

    t_sigs = time.perf_counter()
    if manifest_width is not None and use_cache and not force:
        probe_sigs = _compute_sigs(float(manifest_width))
        t_probe_load = time.perf_counter()
        probe_corners = {n.name: cache.load(probe_sigs[n.name]) for n in nets}
        t_cache_load = time.perf_counter() - t_probe_load
        if all(v is not None for v in probe_corners.values()):
            signatures = probe_sigs
            cached_corners = {k: v for k, v in probe_corners.items() if v is not None}  # type: ignore[misc]
            cached_nets = list(nets)
            effective_width = float(manifest_width)
            skipped_precompute = True
            # _replay_cached_corners asserts that every port has a geom_cache
            # entry. Build synthetic entries so we can skip the expensive
            # _precompute_port_geometries walk.
            synthetic_w = float(effective_width)
            for net in nets:
                for port in (net.start, net.stop):
                    key = (
                        int(port.dcenter[0] / dbu),
                        int(port.dcenter[1] / dbu),
                    )
                    if key not in geom_cache:
                        geom_cache[key] = PortGeometry(
                            synthetic_w,
                            synthetic_w,
                            None,
                            None,
                            None,
                            0,
                            orig_min_extent_um=synthetic_w,
                            orig_extent_x_um=synthetic_w,
                            orig_extent_y_um=synthetic_w,
                        )
        else:
            t_cache_load = 0.0  # discard probe timing; we'll re-probe below
    else:
        t_cache_load = 0.0

    if not skipped_precompute:
        t_pre = time.perf_counter()
        geom_cache = _precompute_port_geometries(
            component, nets, cfg, width=width or 1.0
        )
        _record("t_precompute_geom", time.perf_counter() - t_pre)
        effective_width = (
            width
            if width is not None
            else _derive_bundle_width(geom_cache, nets, dbu, fallback=10.0)
        )
        signatures = _compute_sigs(effective_width)
        t_probe_load = time.perf_counter()
        for net in nets:
            corners = (
                None if (force or not use_cache) else cache.load(signatures[net.name])
            )
            if corners is not None:
                cached_nets.append(net)
                cached_corners[net.name] = corners
            else:
                missing_nets.append(net)
        t_cache_load = time.perf_counter() - t_probe_load
    else:
        _record("t_precompute_geom", 0.0)

    _record("t_signatures", time.perf_counter() - t_sigs - t_cache_load)
    _record("t_cache_load", t_cache_load)

    t_replay_start = time.perf_counter()
    replayed_ok: list[str] = []
    replay_failed: list[RouteNetSpec] = []
    for net in cached_nets:
        if _replay_cached_corners(
            component,
            net=net,
            corners=cached_corners[net.name],
            config=cfg,
            width=effective_width,
            clearance=clearance,
            dbu=dbu,
            geom_cache=geom_cache,
        ):
            replayed_ok.append(net.name)
        else:
            replay_failed.append(net)
    _record("t_replay", time.perf_counter() - t_replay_start)

    nets_to_route = missing_nets + replay_failed
    failed_names: list[str] = []
    t_save = 0.0
    if nets_to_route:
        rust_stats: dict[str, Any] = {}
        # If we took the fast path but a replay failed, we don't have geom_cache.
        # Route-batch needs it; compute now.
        if not geom_cache:
            t_pre = time.perf_counter()
            geom_cache = _precompute_port_geometries(
                component, nets, cfg, width=width or 1.0
            )
            _record("t_precompute_geom_late", time.perf_counter() - t_pre)
        t0 = time.perf_counter()
        batch_kwargs: dict[str, Any] = dict(
            nets=nets_to_route,
            config=cfg,
            grid_unit=grid_unit,
            width=effective_width,
            dynamic_width=False,
            layers_to_avoid=list(layers_to_avoid),
            clearance=clearance,
            geom_cache=geom_cache,
            debug=debug,
        )
        if _ROUTE_BATCH_STATS_SINK:
            batch_kwargs["stats_sink"] = rust_stats
        _, routed_failed, corners_by_net = route_batch_multilayer(
            component, **batch_kwargs
        )
        elapsed = time.perf_counter() - t0
        failed_names = list(routed_failed)
        _record("t_rust_batch", elapsed)
        for k in ("t_rust_grid_s", "t_rust_grt_s", "t_rust_drt_s", "t_rust_total_s"):
            if k in rust_stats:
                _record(k, rust_stats[k])
        logger.info(
            "routed %d/%d missing/replay-failed nets in %.2fs (cache hits=%d)",
            len(nets_to_route) - len(failed_names),
            len(nets_to_route),
            elapsed,
            len(replayed_ok),
        )

        t_save_start = time.perf_counter()
        for net in nets_to_route:
            if net.name in failed_names:
                continue
            corners = corners_by_net.get(net.name)
            if not corners:
                continue
            sig = signatures[net.name]
            cache.save(
                sig,
                net_name=net.name,
                corners=corners,
                meta={
                    "width_um": effective_width,
                    "clearance_um": clearance,
                    "grid_unit_um": grid_unit,
                    "layers_to_avoid": [list(layer) for layer in layers_to_avoid],
                },
            )
        t_save = time.perf_counter() - t_save_start

        cache.save_manifest({"last_effective_width_um": float(effective_width)})
    else:
        _record("t_rust_batch", 0.0)
        logger.info(
            "all %d nets served from cache (hits=%d, misses=%d, skipped_precompute=%s)",
            len(nets),
            cache.hits,
            cache.misses,
            skipped_precompute,
        )

    _record("t_cache_save", t_save)
    _record("n_hits", cache.hits)
    _record("n_misses", cache.misses)
    _record("n_routed", len(nets_to_route) - len(failed_names))
    _record("n_nets", len(nets))
    _record("skipped_precompute", skipped_precompute)

    if failed_names and require_all:
        raise RuntimeError(f"3D routing failed for nets: {failed_names}")

    return []
