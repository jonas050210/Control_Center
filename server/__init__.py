"""Local FastAPI backend that replaces the former Streamlit dashboard.

The package is intentionally import-light: ``server.app`` builds the ASGI
application, while the remaining modules hold pure logic (scene building,
mini-games, analytics) that can be unit-tested without a web server.
"""
