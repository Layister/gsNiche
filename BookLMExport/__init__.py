from __future__ import annotations

__all__ = ["export_results_tree"]


def export_results_tree(*args, **kwargs):
    from .export_results import export_results_tree as _export_results_tree

    return _export_results_tree(*args, **kwargs)
