"""Gunicorn configuration for the OrnaVision API.

The app was previously served by Flask's development server with debug=True.
That server is single-process, its reloader restarted the whole app (reloading
~1.5 GB of models) whenever a background task wrote metadata.json or an upload
landed, and it has no request timeout. Those restarts are what the platform
surfaced as intermittent 503s.
"""

import multiprocessing
import os

bind = "0.0.0.0:{}".format(os.getenv("PORT", "5000"))

# One worker only: each worker loads its own copy of ConvNeXt x2 + CLIP, which
# is well over a gigabyte of RAM. Two workers on a small container means OOM,
# and an OOM-killed worker is a 503. Concurrency comes from threads instead -
# torch and the Qdrant HTTP client both release the GIL while they work.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
worker_class = "gthread"
threads = int(os.getenv("WEB_THREADS", str(max(4, min(8, multiprocessing.cpu_count() * 2)))))

# Image search runs a CNN forward pass and a Qdrant query; the first request
# after a cold start may also wait on model loading. The 30s default killed
# those requests mid-flight.
timeout = int(os.getenv("WEB_TIMEOUT", "300"))
graceful_timeout = 60
keepalive = 5

# Recycle periodically so a slow leak in a long-lived worker cannot accumulate.
max_requests = int(os.getenv("MAX_REQUESTS", "2000"))
max_requests_jitter = 200

# Models are loaded lazily by a background warmup thread inside each worker, so
# preloading buys nothing and would only make master restarts slower.
preload_app = False

accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
