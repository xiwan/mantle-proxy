#!/usr/bin/env python3
"""Compatibility entry point for the Mantle SigV4 proxy."""

from mantle_proxy.server import main


if __name__ == "__main__":
    raise SystemExit(main())
