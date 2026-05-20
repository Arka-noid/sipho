"""1x16 optical switch tree using cascaded MZI 2x2 switches."""

__all__ = ["switch_nxn", "switch_nxn_with_fiber_array"]

import hashlib
import json
import os
from functools import partial
from pathlib import Path

import gdsfactory as gf
from gdsfactory.components.containers.splitter_tree import splitter_tree
from gdsfactory.components.mzis import mzi1x2_2x2
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Spacing

from gpdk.routing_3d import GPDK_CONFIG, route_bundle_3d, via_m2_m3  # noqa: F401

_CACHE_VERSION = "v1"


def _stable_key(value):
    """Return a process-stable representation of ``value`` for hashing.

    The default ``repr`` of ``functools.partial`` (and bare functions) embeds
    ``<function ... at 0x...>`` — the address changes every process, so the
    hash misses on every fresh run. Unwrap callables to their qualname +
    bound args instead.
    """
    from functools import partial

    if isinstance(value, partial):
        return {
            "__partial__": _stable_key(value.func),
            "args": [_stable_key(a) for a in value.args],
            "kwargs": {k: _stable_key(v) for k, v in value.keywords.items()},
        }
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _cache_path(fn_name: str, kwargs: dict) -> Path:
    root = Path(os.environ.get("GFP_CACHE_DIR", Path.home() / ".cache" / "gfp"))
    payload = {"_v": _CACHE_VERSION, **{k: _stable_key(v) for k, v in kwargs.items()}}
    key = hashlib.md5(
        json.dumps(payload, sort_keys=True, default=repr).encode()
    ).hexdigest()[:10]
    return root / f"{fn_name}_{key}.gds"


_mzi1x2_2x2 = partial(
    mzi1x2_2x2,
    combiner="mmi2x2",
    delta_length=0,
    straight_x_top="straight_heater_metal",
    length_x=None,
)


@gf.cell
def switch_nxn(
    coupler: ComponentSpec = _mzi1x2_2x2,
    spacing: Spacing = (500, 100),
    bend_s: ComponentSpec | None = "bend_s",
    bend_s_xsize: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    noutputs: int = 16,
) -> gf.Component:
    """1x16 optical switch tree using cascaded MZI 2x2 switches.

    Args:
        coupler: coupler factory (default: MZI 1x2 to 2x2 switch element).
        spacing: x, y spacing between couplers.
        bend_s: S-bend function for termination.
        bend_s_xsize: xsize for the S-bend.
        cross_section: cross_section spec.
        noutputs: number of outputs.
    """
    return splitter_tree(
        coupler=coupler,
        noutputs=noutputs,
        spacing=spacing,
        bend_s=bend_s,
        bend_s_xsize=bend_s_xsize,
        cross_section=cross_section,
    )


@gf.cell
def switch_nxn_with_fiber_array(
    coupler: ComponentSpec = _mzi1x2_2x2,
    spacing: Spacing = (500, 100),
    bend_s: ComponentSpec | None = "bend_s",
    bend_s_xsize: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    pad: ComponentSpec = "pad",
    pad_pitch: float = 120.0,
    cross_section_metal: CrossSectionSpec = "metal_routing",
    noutputs: int = 2**2,
    use_cache: bool = True,
) -> gf.Component:
    """1x16 switch tree with fiber array grating couplers and electrical pads.

    Args:
        coupler: coupler factory (default: MZI 1x2 to 2x2 switch element).
        spacing: x, y spacing between couplers.
        bend_s: S-bend function for termination.
        bend_s_xsize: xsize for the S-bend.
        cross_section: cross_section spec.
        pad: electrical pad spec.
        pad_pitch: pitch between electrical pads.
        cross_section_metal: metal cross section for electrical routing.
        noutputs: number of outputs.
        use_cache: if True, read/write a GDS cache keyed by kwargs at
            ``$GFP_CACHE_DIR`` (default ``~/.cache/gfp``). Routing is slow
            (~30-60 s) so this avoids rebuilding on repeated runs. Bump
            ``_CACHE_VERSION`` or delete the cache dir to invalidate after
            routing-logic changes.
    """
    cache = (
        _cache_path(
            "switch_nxn_with_fiber_array",
            dict(
                coupler=coupler,
                spacing=spacing,
                bend_s=bend_s,
                bend_s_xsize=bend_s_xsize,
                cross_section=cross_section,
                pad=pad,
                pad_pitch=pad_pitch,
                cross_section_metal=cross_section_metal,
                noutputs=noutputs,
            ),
        )
        if use_cache
        else None
    )
    if cache is not None and cache.exists():
        return gf.import_gds(cache)

    c = gf.Component()

    switch = c << switch_nxn(
        coupler=coupler,
        spacing=spacing,
        bend_s=bend_s,
        bend_s_xsize=bend_s_xsize,
        cross_section=cross_section,
        noutputs=noutputs,
    )
    c.add_ports(switch.ports.filter(port_type="optical"))

    # Collect and add electrical ports for heater pads
    electrical_ports = list(switch.ports.filter(port_type="electrical", orientation=90))
    electrical_ports.sort(key=lambda p: p.center[0])

    if electrical_ports:
        npads = len(electrical_ports)
        pads = c << gf.components.array(
            component=pad,
            columns=npads,
            column_pitch=pad_pitch,
        )
        pads.x = switch.x
        pads.ymin = switch.ymax + 900

        pad_ports = list(pads.ports.filter(orientation=270))[:npads]
        pad_ports.sort(key=lambda p: p.center[0])

        # Add routing area polygon so the A* grid covers the full region
        routing_area_layer = (235, 4)
        margin = 50.0
        all_pts = [p.center for p in electrical_ports] + [p.center for p in pad_ports]
        xs = [pt[0] for pt in all_pts]
        ys = [pt[1] for pt in all_pts]
        c.add_polygon(
            [
                (min(xs) - margin, min(ys) - margin),
                (max(xs) + margin, min(ys) - margin),
                (max(xs) + margin, max(ys) + margin),
                (min(xs) - margin, max(ys) + margin),
            ],
            layer=routing_area_layer,
        )

        n = min(npads, len(electrical_ports))
        route_bundle_3d(
            c,
            ports1=electrical_ports[:n],
            ports2=pad_ports[:n],
            grid_unit=10.0,
            clearance=5.0,
        )

        # Remove routing area marker
        c.remove_layers(layers=[routing_area_layer])

        for i, port in enumerate(pad_ports):
            c.add_port(name=f"e{i}", port=port)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        c.write_gds(cache)
    return c
