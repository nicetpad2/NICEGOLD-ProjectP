# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - API Module
═══════════════════════════════════════════════════════════════════════════════

API endpoints and web services for ProjectP.

Author: NICEGOLD Team
Version: 3.0
Created: 2025-01-05
"""

from .dashboard import DashboardServer
from .endpoints import APIEndpoints
from .fastapi_server import FastAPIServer as FastAPIServerCompat
from .server import FastAPIServer

__all__ = ["FastAPIServer", "FastAPIServerCompat", "APIEndpoints", "DashboardServer"]
