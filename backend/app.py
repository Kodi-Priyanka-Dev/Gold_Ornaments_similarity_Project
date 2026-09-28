import os
import sys

# Ensure unbuffered stdout/stderr so logs appear immediately in terminal.
# errors='replace' keeps a print() of an emoji/unicode filename (common in
# scraped inventory image names) from crashing the request with
# UnicodeEncodeError on Windows' cp1252 console - it prints '?' instead.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True, errors='replace')
        sys.stderr.reconfigure(line_buffering=True, errors='replace')
    except Exception:
        pass

# Make `api` / `services` importable no matter how the app is launched
# (python backend/app.py, gunicorn backend.app:app, or gunicorn --chdir backend app:app).
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import time
from flask import Flask, jsonify, send_from_directory, request, g
from flask_cors import CORS

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

# httpx (used by huggingface_hub) logs every HEAD/GET request the CLIP model
# loader makes while probing for optional config files - dozens of lines of
# noise per startup, all expected 404s. Silence it so the terminal only shows
# this app's own logs (search results, warmup, errors).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
# The Azure SDK logs full request/response headers for every blob download at
# INFO, and sentence-transformers shows a "Batches" progress bar per encode
# whenever its logger is at INFO.
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from api.routes import api_blueprint

# Serve static files directly from the frontend folder
frontend_dir = os.path.abspath(os.path.join(BACKEND_DIR, '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')

# Secure CORS configuration based on allowed origins environment variable
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if allowed_origins:
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
else:
    print("WARNING: ALLOWED_ORIGINS environment variable is not set. CORS is currently fully open to all domains (Access-Control-Allow-Origin: *). This is a security risk in production.")
    CORS(app) # Fallback if not configured

app.register_blueprint(api_blueprint, url_prefix='/api')

# Uploads used to be unbounded (MAX_CONTENT_LENGTH = None, a 10 GB form buffer).
# A single oversized upload could exhaust container memory and take the whole
# process down, which is one of the ways the site ended up returning 503.
# Bulk inventory upload still needs a generous ceiling, so it is configurable.
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_MB', '512')) * 1024 * 1024
app.config['MAX_FORM_PARTS'] = int(os.getenv('MAX_FORM_PARTS', '5000'))
app.config['MAX_FORM_MEMORY_SIZE'] = int(os.getenv('MAX_FORM_MEMORY_MB', '64')) * 1024 * 1024

# Let browsers reuse the frontend assets instead of re-downloading them on every
# page load. style.css/script.js are already cache-busted with ?v= query strings.
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = int(os.getenv('STATIC_CACHE_SECONDS', '3600'))


# Raw image bytes are served on every result thumbnail (dozens per search) and
# werkzeug's own access log already prints one line per request for them, so
# they are excluded here to keep this log readable - everything else (search,
# uploads, compare, admin) is logged in full both ways.
_QUIET_PATH_PREFIXES = ('/api/images/', '/static/')
_QUIET_EXTENSIONS = ('.css', '.js', '.png', '.jpg', '.jpeg', '.ico', '.svg', '.woff', '.woff2')


def _is_quiet_path(path):
    return path.startswith(_QUIET_PATH_PREFIXES) or path.endswith(_QUIET_EXTENSIONS)


@app.before_request
def _log_request_in():
    """Print what the frontend just sent this backend: method, path, and a
    compact view of query/form/JSON body (file fields shown as name+size, not
    their bytes)."""
    g._t0 = time.time()
    if _is_quiet_path(request.path):
        return
    bits = [f"\n>>> FRONTEND -> BACKEND  {request.method} {request.path}"]
    if request.args:
        bits.append(f"    query:  {dict(request.args)}")
    if request.form:
        bits.append(f"    form:   {dict(request.form)}")
    if request.files:
        files = {k: f"{f.filename!r} ({f.mimetype}, {request.content_length or 0}B total)"
                 for k, f in request.files.items()}
        bits.append(f"    files:  {files}")
    if request.is_json:
        body = request.get_json(silent=True)
        if body is not None:
            bits.append(f"    json:   {body}")
    print("\n".join(bits))


@app.after_request
def _log_request_out(response):
    """Print what the backend is sending back: status, timing, and payload
    size, so the round trip is visible without opening browser dev tools."""
    if _is_quiet_path(request.path):
        return response
    elapsed_ms = int((time.time() - g.get('_t0', time.time())) * 1000)
    size = response.calculate_content_length()
    size_txt = f"{size}B" if size is not None else "?"
    print(f"<<< BACKEND -> FRONTEND  {request.method} {request.path}  "
          f"status={response.status_code}  time={elapsed_ms}ms  size={size_txt}")
    return response


@app.after_request
def no_cache_html(response):
    """HTML must not be cached, otherwise a deploy that bumps the ?v= markers in
    index.html would not reach browsers holding a cached copy of the page."""
    if response.mimetype == 'text/html':
        response.headers['Cache-Control'] = 'no-cache, must-revalidate'
    return response


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/admin')
def serve_admin():
    return send_from_directory(app.static_folder, 'admin.html')


@app.route('/compare')
def serve_compare():
    return send_from_directory(app.static_folder, 'compare.html')


@app.route('/health')
@app.route('/healthz')
def health():
    """Liveness probe. Deliberately does no model or database work so the
    platform can tell 'process is up' from 'models are still loading'."""
    return jsonify({"status": "ok"}), 200


@app.route('/ready')
def ready():
    """Readiness probe: reports whether the ML models have finished loading."""
    from services.similarity_search import model_status, models_ready
    ready_now = models_ready()
    return jsonify({
        "ready": ready_now,
        "models": model_status(),
    }), (200 if ready_now else 503)


@app.route('/favicon.ico')
def serve_favicon():
    return '', 204


@app.route('/.well-known/<path:path>')
def serve_well_known(path):
    return '', 204


def start_warmup():
    """Load the ML models in a background thread so the server starts answering
    requests immediately instead of blocking for minutes on first import."""
    try:
        from services.similarity_search import warmup
        warmup(background=True)
    except Exception as e:  # never let warmup failure stop the web server
        print(f"Warning: could not start model warmup: {e}")


# Runs under gunicorn too, not just `python app.py`.
start_warmup()


if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    # debug/reloader must stay off: the reloader loads every model twice and
    # restarts the process whenever a background task writes metadata.json or an
    # upload lands, which is what made the site drop out intermittently.
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    print("---------------------------------------")
    print(" Running with model: convnext_base")
    print(f" Listening on 0.0.0.0:{port} (debug={debug})")
    print(" For production use: gunicorn -c gunicorn_conf.py app:app")
    print("---------------------------------------")
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False, threaded=True)
