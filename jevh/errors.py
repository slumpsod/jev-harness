"""Exceptions shared across the harness."""


class Cancelled(Exception):
    """The user stopped the turn. Raised wherever work notices the turn's cancel event."""
