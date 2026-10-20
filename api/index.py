"""
api/index.py
------------
Vercel Serverless Function entrypoint.
Maps Vercel function routing directly to the FastAPI application.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, handler, application

__all__ = ["app", "handler", "application"]
