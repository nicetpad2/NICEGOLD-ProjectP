# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - FastAPI Server Module (Compatibility)
════════════════════════════════════════════════════════════════════════════════

FastAPI server implementation for model serving and API endpoints.
This is a compatibility module that imports from server.py.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

# Import from the main server module for compatibility
from .server import FastAPIServer

__all__ = ["FastAPIServer"]
