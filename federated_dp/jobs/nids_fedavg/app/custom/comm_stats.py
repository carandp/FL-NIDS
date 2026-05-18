from __future__ import annotations

import pickle
from typing import Dict

from nvflare.apis.event_type import EventType
from nvflare.apis.filter import Filter
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.utils import fobs


class _CommStats:
    def __init__(self) -> None:
        self.bytes_out_total = 0
        self.bytes_in_total = 0
        self.per_site: Dict[str, Dict[str, int]] = {}

    def add(self, site: str, direction: str, size: int) -> None:
        if direction == "out":
            self.bytes_out_total += size
        else:
            self.bytes_in_total += size
        site_stats = self.per_site.setdefault(site, {"out": 0, "in": 0})
        site_stats[direction] += size

    def format_summary(self) -> str:
        header = "Data transfer summary (server perspective)"
        lines = [" "]
        lines.append("=" * len(header))
        lines.append(header)
        lines.append("=" * len(header))
        lines.append(f"Total sent to clients : {_format_bytes(self.bytes_out_total)}")
        lines.append(f"Total received        : {_format_bytes(self.bytes_in_total)}")
        lines.append(f"Grand total           : {_format_bytes(self.bytes_out_total + self.bytes_in_total)}")

        if self.per_site:
            lines.append("")
            lines.append("Per-client totals")
            lines.append("Client        To Client        From Client      Total")
            lines.append("------------  --------------  --------------  --------------")
            for site in sorted(self.per_site):
                out_b = self.per_site[site]["out"]
                in_b = self.per_site[site]["in"]
                total = out_b + in_b
                lines.append(
                    f"{site:<12}  {_format_bytes(out_b):>14}  {_format_bytes(in_b):>14}  {_format_bytes(total):>14}"
                )

        lines.append("=" * len(header))
        return "\n".join(lines)


_STATS = _CommStats()


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def _estimate_size(shareable: Shareable) -> int:
    try:
        return len(fobs.dumps(shareable))
    except Exception:
        try:
            return len(pickle.dumps(shareable, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 0


def _get_peer_name(fl_ctx: FLContext) -> str:
    peer_ctx = fl_ctx.get_peer_context()
    if peer_ctx:
        for key in (FLContextKey.SITE_NAME, FLContextKey.CLIENT_NAME):
            val = peer_ctx.get_prop(key)
            if val:
                return val
    for key in (FLContextKey.SITE_NAME, FLContextKey.CLIENT_NAME):
        val = fl_ctx.get_prop(key)
        if val:
            return val
    identity = fl_ctx.get_identity_name()
    return identity if identity else "unknown"


class CommStatsFilter(Filter):
    def __init__(self, direction: str):
        super().__init__()
        if direction not in {"in", "out"}:
            raise ValueError("direction must be 'in' or 'out'")
        self.direction = direction

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        size = _estimate_size(shareable)
        site = _get_peer_name(fl_ctx)
        _STATS.add(site, self.direction, size)
        return shareable


class CommStatsReporter(FLComponent):
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type in {EventType.END_RUN, EventType.JOB_COMPLETED}:
            self.log_info(fl_ctx, _STATS.format_summary())
