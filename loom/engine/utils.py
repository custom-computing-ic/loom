"""Shared utilities for graph-engine matching."""


def deduplicate_matches(matches):
    """Remove matches covering the same host vertices, preserving order."""
    unique = {}
    for match in matches:
        unique.setdefault(frozenset(match.values()), match)
    return list(unique.values())
