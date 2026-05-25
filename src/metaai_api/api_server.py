import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, cast

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
import time as time_module

from metaai_api import MetaAI
from metaai_api.database import db, DATA_DIR

logger = logging.getLogger(__name__)

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment variables from {env_path}")

# ── Admin auth config ──────────────────────────────────────────────
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", os.urandom(24).hex())
logger.info(f"Admin user: {ADMIN_USERNAME} (password {'SET' if ADMIN_PASSWORD else 'NOT SET - auth disabled'})")

# Refresh interval (seconds) for keeping lsd/fb_dtsg/cookies fresh
DEFAULT_REFRESH_SECONDS = 3600
REFRESH_SECONDS = int(os.getenv("META_AI_REFRESH_INTERVAL_SECONDS", DEFAULT_REFRESH_SECONDS))

# Request timeout (seconds) - prevents infinite hangs on long-running operations
# Increased to 180s to accommodate video generation (60s) + polling (120s) + overhead
DEFAULT_REQUEST_TIMEOUT = 180
REQUEST_TIMEOUT = int(os.getenv("META_AI_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT))

# CORS configuration
DEFAULT_ALLOWED_ORIGINS = ["*"]
CORS_ALLOWED_ORIGINS_ENV = os.getenv("META_AI_CORS_ALLOWED_ORIGINS", "")
CORS_ALLOWED_ORIGINS = [
    origin.strip() for origin in CORS_ALLOWED_ORIGINS_ENV.split(",")
] if CORS_ALLOWED_ORIGINS_ENV else DEFAULT_ALLOWED_ORIGINS


class TokenCache:
    """Thread-safe cache for Meta cookies and tokens."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cookies: Dict[str, str] = {}
        self._last_refresh: float = 0.0

    async def load_seed(self) -> None:
        seed = {
            "datr": os.getenv("META_AI_DATR", ""),
            "abra_sess": os.getenv("META_AI_ABRA_SESS", ""),
            "ecto_1_sess": os.getenv("META_AI_ECTO_1_SESS", ""),
            "dpr": os.getenv("META_AI_DPR", ""),
            "wd": os.getenv("META_AI_WD", ""),
            "_js_datr": os.getenv("META_AI_JS_DATR", ""),
            "abra_csrf": os.getenv("META_AI_ABRA_CSRF", ""),
            "rd_challenge": os.getenv("META_AI_RD_CHALLENGE", ""),
        }
        # Only datr is truly required. abra_sess is optional (some regions like Indonesia don't have it)
        missing = [k for k in ("datr",) if not seed.get(k)]
        if missing:
            raise RuntimeError(f"Missing required seed cookies: {', '.join(missing)}")
        
        # Log if abra_sess is missing (it's optional but recommended)
        if not seed.get("abra_sess"):
            logging.warning("abra_sess cookie not found - some features may have reduced functionality. This is common in certain regions like Indonesia.")
        async with self._lock:
            self._cookies = {k: v for k, v in seed.items() if v}
            self._last_refresh = 0.0

    async def refresh_if_needed(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_refresh) < REFRESH_SECONDS:
            return
        async with self._lock:
            if not force and (time.time() - self._last_refresh) < REFRESH_SECONDS:
                return
            try:
                # Create MetaAI with current cookies (cookie-based auth only)
                ai = MetaAI(cookies=dict(self._cookies))
                self._cookies = getattr(ai, "cookies", self._cookies)
                self._last_refresh = time.time()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cookie refresh failed: %s", exc)
                if force:
                    raise

    async def refresh_after_error(self) -> None:
        await self.refresh_if_needed(force=True)

    async def snapshot(self) -> Dict[str, str]:
        async with self._lock:
            return dict(self._cookies)


cache = TokenCache()
refresh_task: Optional[asyncio.Task] = None
app = FastAPI(title="Meta AI API Service", version="0.1.0")

# Serve static UI files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")

# ── Admin auth middleware (not decorated — registered at end of file for correct ordering) ──
PUBLIC_PATHS = {"/healthz", "/api/login", "/api/logout", "/api/auth/check", "/static/login.html", "/static/", "/favicon.ico"}


async def auth_middleware(request: Request, call_next):
    if ADMIN_PASSWORD:
        path = request.url.path.rstrip("/") or "/"
        if path.startswith("/static/") and path != "/static/login.html":
            return await call_next(request)
        if path in PUBLIC_PATHS:
            return await call_next(request)
        if path.startswith("/api/") and path in {"/api/login", "/api/logout", "/api/auth/check"}:
            return await call_next(request)
        if not request.session.get("authenticated"):
            if path.startswith("/api/"):
                return JSONResponse(status_code=401, content={"error": "Authentication required"})
            return RedirectResponse(url="/static/login.html")
    return await call_next(request)


# ── Admin auth endpoints ───────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
async def login(request: Request, body: LoginRequest):
    if not ADMIN_PASSWORD:
        request.session["authenticated"] = True
        request.session["username"] = "admin"
        return {"success": True}
    if body.username == ADMIN_USERNAME and body.password == ADMIN_PASSWORD:
        request.session["authenticated"] = True
        request.session["username"] = body.username
        return {"success": True}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/api/logout")
async def logout(request: Request):
    request.session.clear()
    return {"success": True}


@app.get("/api/auth/check")
async def auth_check(request: Request):
    return {
        "authenticated": request.session.get("authenticated", False),
        "username": request.session.get("username"),
        "auth_enabled": bool(ADMIN_PASSWORD),
    }

# Exception handler for unhandled exceptions - returns JSON
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Convert any unhandled exception to JSON response."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if logger.level == logging.DEBUG else "An unexpected error occurred"
        }
    )

# Middleware to log all requests
async def log_requests(request: Request, call_next):
    start_time = time_module.time()
    logger.warning(f"[REQUEST] {request.method} {request.url.path} - Content-Type: {request.headers.get('content-type', 'none')}")
    
    try:
        response = await call_next(request)
        process_time = time_module.time() - start_time
        logger.warning(f"[RESPONSE] {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}s")
        return response
    except Exception as exc:
        logger.error(f"[ERROR] {request.method} {request.url.path} - {exc}")
        raise


def _get_proxies() -> Optional[Dict[str, str]]:
    http_proxy = os.getenv("META_AI_PROXY_HTTP")
    https_proxy = os.getenv("META_AI_PROXY_HTTPS")
    if not http_proxy and not https_proxy:
        return None
    proxies: Dict[str, str] = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    return proxies


class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    new_conversation: bool = False
    media_ids: Optional[list] = None
    attachment_metadata: Optional[dict] = None  # {'file_size': int, 'mime_type': str}


class ImageRequest(BaseModel):
    prompt: str
    new_conversation: bool = False
    media_ids: Optional[list] = None
    attachment_metadata: Optional[dict] = None  # {'file_size': int, 'mime_type': str}
    orientation: Optional[str] = None  # 'VERTICAL', 'LANDSCAPE' (not HORIZONTAL), or 'SQUARE'
    num_images: int = Field(1, ge=1, le=4)  # Number of images to generate (1-4)


class VideoRequest(BaseModel):
    prompt: str
    media_ids: Optional[list] = None
    attachment_metadata: Optional[dict] = None  # {'file_size': int, 'mime_type': str}
    auto_poll: bool = True  # Auto-poll for video URLs (default: True)
    max_poll_attempts: int = Field(15, ge=1, le=60)  # Max polling attempts
    poll_wait_seconds: int = Field(3, ge=1, le=30)  # Seconds between polls
    # Deprecated parameters (kept for backwards compatibility)
    orientation: Optional[str] = None
    wait_before_poll: int = Field(10, ge=0, le=60)
    max_attempts: int = Field(30, ge=1, le=60)
    wait_seconds: int = Field(5, ge=1, le=30)
    verbose: bool = False


class VideoExtendRequest(BaseModel):
    media_id: str
    source_media_url: Optional[str] = None
    conversation_id: Optional[str] = None
    auto_poll: bool = True
    max_poll_attempts: int = Field(15, ge=1, le=60)
    poll_wait_seconds: int = Field(3, ge=1, le=30)


class ImageUploadResponse(BaseModel):
    success: bool
    media_id: Optional[str] = None
    upload_session_id: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    error: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: float
    updated_at: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobStatus] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> JobStatus:
        now = time.time()
        job_id = str(uuid.uuid4())
        job = JobStatus(job_id=job_id, status="pending", created_at=now, updated_at=now)
        async with self._lock:
            self._jobs[job_id] = job
        return job

    async def set_running(self, job_id: str) -> None:
        await self._update(job_id, status="running")

    async def set_result(self, job_id: str, result: Dict[str, Any]) -> None:
        await self._update(job_id, status="succeeded", result=result, error=None)

    async def set_error(self, job_id: str, error: str) -> None:
        await self._update(job_id, status="failed", error=error)

    async def get(self, job_id: str) -> JobStatus:
        async with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    async def _update(self, job_id: str, **fields: Any) -> None:
        async with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id].copy(update=fields)
            job.updated_at = time.time()
            self._jobs[job_id] = job


jobs = JobStore()

# Global MetaAI instance (initialized once at startup)
_meta_ai_instance: Optional[MetaAI] = None


async def get_cookies() -> Dict[str, str]:
    await cache.refresh_if_needed()
    return await cache.snapshot()


@app.get("/")
async def root(request: Request):
    if ADMIN_PASSWORD and not request.session.get("authenticated"):
        return RedirectResponse(url="/static/login.html")
    return RedirectResponse(url="/static/index.html")


@app.get("/api/cookies")
async def get_cookies_endpoint():
    c = await cache.snapshot()
    return {"cookies": c}


class CookieUpdateRequest(BaseModel):
    cookies: Dict[str, str]


@app.put("/api/cookies")
async def update_cookies(body: CookieUpdateRequest):
    global _meta_ai_instance
    new_cookies = body.cookies
    async with cache._lock:
        cache._cookies = dict(new_cookies)
        cache._last_refresh = 0.0
    try:
        _meta_ai_instance = MetaAI(cookies=new_cookies, proxy=_get_proxies())
    except Exception as exc:
        logger.warning(f"MetaAI re-init after cookie update failed: {exc}")
    return {"success": True, "message": "Cookies updated"}


@app.post("/api/cookies/refresh")
async def refresh_cookies_endpoint():
    await cache.refresh_if_needed(force=True)
    return {"success": True, "message": "Cookies refreshed"}


@app.get("/api/uploads")
async def list_uploads():
    return {"uploads": db.get_uploads()}


@app.post("/api/uploads")
async def create_upload(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are allowed")
    ext = Path(file.filename or "image.jpg").suffix or ".jpg"
    local_name = f"{uuid.uuid4()}{ext}"
    local_path = DATA_DIR / "uploads" / local_name
    content = await file.read()
    with open(local_path, "wb") as f:
        f.write(content)
    upload_id = db.add_upload(
        filename=local_name,
        original_name=file.filename or local_name,
        mime_type=file.content_type or "image/jpeg",
        file_size=len(content),
    )
    ai = _meta_ai_instance
    if ai:
        try:
            meta_result = await asyncio.wait_for(
                run_in_threadpool(ai.upload_image, str(local_path)),
                timeout=60,
            )
            if isinstance(meta_result, dict):
                media_id = meta_result.get("media_id")
                session_id = meta_result.get("upload_session_id")
                if media_id:
                    db.update_upload_media_id(upload_id, media_id, session_id)
        except Exception as exc:
            logger.warning(f"Meta AI upload for {upload_id} failed: {exc}")
    upload = db.get_upload(upload_id)
    return upload


@app.get("/api/uploads/{upload_id}/file")
async def get_upload_file(upload_id: int):
    upload = db.get_upload(upload_id)
    if not upload:
        raise HTTPException(404, "Upload not found")
    filepath = DATA_DIR / "uploads" / upload["filename"]
    if not filepath.exists():
        raise HTTPException(404, "File not found on disk")
    return FileResponse(
        str(filepath),
        media_type=upload["mime_type"],
        filename=upload["original_name"],
    )


@app.delete("/api/uploads/{upload_id}")
async def delete_upload_endpoint(upload_id: int):
    ok = db.delete_upload(upload_id)
    if not ok:
        raise HTTPException(404, "Upload not found")
    return {"success": True}


@app.get("/api/generations")
async def list_generations():
    return {"generations": db.get_generations()}


class GenerationCreateRequest(BaseModel):
    prompt: str
    generation_type: str = "video"
    input_media_ids: Optional[list] = None


class GenerationRecordRequest(BaseModel):
    prompt: str
    generation_type: str = "video"
    input_media_ids: Optional[list] = None
    result_json: Optional[dict] = None
    video_urls: Optional[list] = None
    status: str = "completed"


@app.post("/api/generations")
async def create_generation(body: GenerationCreateRequest):
    gen_id = db.add_generation(
        prompt=body.prompt,
        generation_type=body.generation_type,
        input_media_ids=body.input_media_ids,
        status="pending",
    )
    asyncio.create_task(_run_ui_generation_job(gen_id, body))
    return {"id": gen_id, "status": "pending"}


@app.post("/api/generations/record")
async def record_generation(body: GenerationRecordRequest):
    gen_id = db.add_generation(
        prompt=body.prompt,
        generation_type=body.generation_type,
        input_media_ids=body.input_media_ids,
        result_json=body.result_json,
        video_urls=body.video_urls,
        status=body.status,
    )
    return {"id": gen_id, "status": body.status}


@app.delete("/api/generations/{gen_id}")
async def delete_generation_endpoint(gen_id: int):
    ok = db.delete_generation(gen_id)
    if not ok:
        raise HTTPException(404, "Generation not found")
    return {"success": True}


@app.get("/api/config")
async def get_config():
    active = {}
    if _meta_ai_instance and hasattr(_meta_ai_instance, "generation_api"):
        gen = _meta_ai_instance.generation_api
        active = {
            "active_doc_ids": dict(gen._doc_ids),
            "active_doc_id_sources": dict(gen._doc_id_sources),
        }
    return {"config": db.get_all_config(), "active": active}


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, str]


@app.put("/api/config")
async def update_config(body: ConfigUpdateRequest):
    for key, value in body.config.items():
        db.set_config(key, value)
    _apply_config_to_meta_ai()
    return {"success": True, "config": db.get_all_config()}


@app.get("/api/download")
async def download_file(url: str, name: str = "download.mp4"):
    try:
        import requests as sync_requests
        resp = await run_in_threadpool(
            lambda: sync_requests.get(url, timeout=30, allow_redirects=True)
        )
        resp.raise_for_status()
        content = resp.content
        import io
        return StreamingResponse(
            io.BytesIO(content),
            media_type=resp.headers.get("content-type", "application/octet-stream"),
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )
    except Exception as exc:
        raise HTTPException(502, f"Download failed: {exc}")


@app.on_event("startup")
async def _startup() -> None:
    await cache.load_seed()
    # Skip initial refresh to avoid unnecessary token fetching
    # Tokens will be refreshed on-demand if needed
    # await cache.refresh_if_needed(force=True)
    
    # Initialize global MetaAI instance to prevent repeated token extraction
    global _meta_ai_instance, refresh_task
    logger.info("Initializing global MetaAI instance...")
    
    try:
        _meta_ai_instance = MetaAI(proxy=_get_proxies())
        
        # Log token status (handle case where extraction failed)
        if _meta_ai_instance.access_token:
            logger.info(f"MetaAI instance initialized with access token: {_meta_ai_instance.access_token[:50]}...")
        else:
            logger.warning("MetaAI instance initialized but access token extraction failed (may be rate-limited). Will retry in background.")
    except Exception as init_exc:  # noqa: BLE001
        logger.error(f"Failed to initialize MetaAI instance: {init_exc}")
        logger.warning("Server will start without MetaAI instance. API requests will fail until initialization succeeds.")
        _meta_ai_instance = None
    
    # Apply any saved config from DB to the running instance
    _apply_config_to_meta_ai()

    refresh_task = asyncio.create_task(_refresh_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    global refresh_task
    if refresh_task:
        refresh_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresh_task


@app.post("/chat")
async def chat(body: ChatRequest) -> Dict[str, Any]:
    if body.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported via HTTP JSON; set stream=false")
    # Use global MetaAI instance
    if _meta_ai_instance is None:
        raise HTTPException(status_code=503, detail="MetaAI instance not initialized yet. Server may be rate-limited. Please try again in a moment.")
    ai = _meta_ai_instance
    try:
        return cast(Dict[str, Any], await run_in_threadpool(
            ai.prompt,
            body.message,
            stream=False,
            new_conversation=body.new_conversation,
            media_ids=body.media_ids,
            attachment_metadata=body.attachment_metadata
        ))
    except Exception as exc:  # noqa: BLE001
        await cache.refresh_after_error()
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/image")
async def image(body: ImageRequest) -> Dict[str, Any]:
    """Generate images from text prompts."""
    if _meta_ai_instance is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "MetaAI instance not initialized",
                "detail": "Server is initializing or rate-limited. Please try again in a moment."
            }
        )
    ai = _meta_ai_instance
    try:
        # Determine number of images: use 4 for image-to-image, 1 for text-to-image
        num_images = 4 if body.media_ids else body.num_images
        
        # Use the new generation API with timeout protection
        result = await asyncio.wait_for(
            run_in_threadpool(
                ai.generate_image_new,
                prompt=body.prompt,
                orientation=body.orientation or "VERTICAL",
                num_images=num_images,
                media_ids=body.media_ids,
                attachment_metadata=body.attachment_metadata
            ),
            timeout=REQUEST_TIMEOUT
        )
        return cast(Dict[str, Any], result)
    except asyncio.TimeoutError:
        logger.warning(f"Image generation timeout after {REQUEST_TIMEOUT}s for prompt: {body.prompt[:50]}...")
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "Image generation timeout",
                "detail": f"Request exceeded {REQUEST_TIMEOUT} second timeout. The generation may still be processing."
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Image generation error: {exc}")
        await cache.refresh_after_error()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
                "detail": "Image generation failed"
            }
        )


@app.post("/video")
async def video(body: VideoRequest) -> Dict[str, Any]:
    """Generate videos from text prompts (auto-polls for URLs by default)."""
    if _meta_ai_instance is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "MetaAI instance not initialized",
                "detail": "Server is initializing or rate-limited. Please try again in a moment."
            }
        )
    ai = _meta_ai_instance
    try:
        # Use the new generation API with auto-polling support
        result = await asyncio.wait_for(
            run_in_threadpool(
                ai.generate_video_new,
                prompt=body.prompt,
                auto_poll=body.auto_poll,
                max_poll_attempts=body.max_poll_attempts,
                poll_wait_seconds=body.poll_wait_seconds,
                media_ids=body.media_ids,
                attachment_metadata=body.attachment_metadata
            ),
            timeout=REQUEST_TIMEOUT
        )
        if isinstance(result, dict):
            rsl = result.get("success"), result.get("status"), len(result.get("video_urls", [])), result.get("error", "")[:100]
            logger.warning(f"Video gen result → success={rsl[0]} status={rsl[1]} urls={rsl[2]} error={rsl[3]}")
        return cast(Dict[str, Any], result)
    except asyncio.TimeoutError:
        logger.warning(f"Video generation timeout after {REQUEST_TIMEOUT}s for prompt: {body.prompt[:50]}...")
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "Video generation timeout",
                "detail": f"Request exceeded {REQUEST_TIMEOUT} second timeout. Use /video/async for longer operations."
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Video generation error: {exc}")
        await cache.refresh_after_error()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
                "detail": "Video generation failed"
            }
        )


@app.post("/video/async")
async def video_async(body: VideoRequest) -> Dict[str, str]:
    job = await jobs.create()
    asyncio.create_task(_run_video_job(job.job_id, body))
    return {"job_id": job.job_id, "status": "pending"}


@app.get("/video/jobs/{job_id}")
async def video_job_status(job_id: str) -> Dict[str, Any]:
    try:
        job = await jobs.get(job_id)
        return job.dict()
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.post("/video/extend")
async def video_extend(body: VideoExtendRequest) -> Dict[str, Any]:
    """Extend an existing video using source media_id."""
    if _meta_ai_instance is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "MetaAI instance not initialized",
                "detail": "Server is initializing or rate-limited. Please try again in a moment.",
            },
        )

    ai = _meta_ai_instance
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(
                ai.extend_video,
                media_id=body.media_id,
                source_media_url=body.source_media_url,
                conversation_id=body.conversation_id,
                auto_poll=body.auto_poll,
                max_poll_attempts=body.max_poll_attempts,
                poll_wait_seconds=body.poll_wait_seconds,
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return cast(Dict[str, Any], result)
    except asyncio.TimeoutError:
        logger.warning(f"Video extend timeout after {REQUEST_TIMEOUT}s for media_id: {body.media_id}")
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "Video extend timeout",
                "detail": f"Request exceeded {REQUEST_TIMEOUT} second timeout.",
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Video extend error: {exc}")
        await cache.refresh_after_error()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
                "detail": "Video extend failed",
            },
        )


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """Upload an image to Meta AI for use in conversations or media generation."""
    import tempfile
    import os
    
    # Create temporary file to save the upload
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"metaai_upload_{uuid.uuid4()}_{file.filename}")
    
    try:
        # Save uploaded file to temporary location
        content = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        # Use global MetaAI instance
        if _meta_ai_instance is None:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "MetaAI instance not initialized",
                    "detail": "Server is initializing or rate-limited. Please try again in a moment."
                }
            )
        ai = _meta_ai_instance
        
        # Upload with timeout protection
        result = await asyncio.wait_for(
            run_in_threadpool(ai.upload_image, temp_path),
            timeout=60
        )
        
        return cast(Dict[str, Any], result)
    
    except asyncio.TimeoutError:
        logger.warning(f"Image upload timeout after 60s for file: {file.filename}")
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "Upload timeout",
                "detail": "Image upload exceeded 60 second timeout. Please try again."
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Image upload error: {exc}")
        await cache.refresh_after_error()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
                "detail": "Image upload failed"
            }
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:  # noqa: BLE001
                pass


@app.get("/healthz")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


async def _run_video_job(job_id: str, body: VideoRequest) -> None:
    logger.info(f"[JOB {job_id}] Starting video generation job")
    await jobs.set_running(job_id)
    # Use global MetaAI instance
    if _meta_ai_instance is None:
        await jobs.set_error(job_id, "MetaAI instance not initialized yet. Server may be rate-limited.")
        return
    ai = _meta_ai_instance
    try:
        logger.info(f"[JOB {job_id}] Calling generate_video_new with prompt: {body.prompt[:100]}...")
        result = await run_in_threadpool(
            ai.generate_video_new,
            prompt=body.prompt,
            media_ids=body.media_ids,
            attachment_metadata=body.attachment_metadata
        )
        
        logger.info(f"[JOB {job_id}] Video generation completed")
        logger.info(f"[JOB {job_id}] Result success: {result.get('success', False)}")
        logger.info(f"[JOB {job_id}] Video URLs count: {len(result.get('video_urls', []))}")
        logger.info(f"[JOB {job_id}] Result status: {result.get('status', 'UNKNOWN')}")
        
        # Check if video generation actually succeeded AND we have video URLs
        video_urls = result.get('video_urls', [])
        status = result.get('status')
        if status == "READY" and video_urls and len(video_urls) > 0:
            logger.info(f"[JOB {job_id}] Marking as SUCCEEDED with {len(video_urls)} video(s)")
            for idx, url in enumerate(video_urls, 1):
                logger.info(f"[JOB {job_id}] Video URL {idx}: {url[:150]}...")
            await jobs.set_result(job_id, result)
        else:
            # Video generation failed or no videos generated - mark job as failed
            if status == "PROCESSING":
                error_msg = result.get('error') or 'Video generation is still processing and no playable URLs are available yet.'
                logger.warning(f"[JOB {job_id}] Marking as FAILED (not ready): {error_msg}")
            elif result.get('has_graphql_errors'):
                error_msg = result.get('error') or 'GraphQL validation failed during video generation.'
                logger.warning(f"[JOB {job_id}] Marking as FAILED (graphql): {error_msg}")
            else:
                error_msg = result.get('error') or 'Video generation failed without playable video URLs.'
                logger.warning(f"[JOB {job_id}] Marking as FAILED: {error_msg}")
                logger.debug(f"[JOB {job_id}] Full result: {result}")
            await jobs.set_error(job_id, error_msg)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[JOB {job_id}] Exception occurred: {exc}", exc_info=True)
        await cache.refresh_after_error()
        await jobs.set_error(job_id, str(exc))


async def _refresh_loop() -> None:
    # If initial token extraction failed, retry after a short delay
    global _meta_ai_instance
    
    # If MetaAI instance creation completely failed, retry after delay
    if _meta_ai_instance is None:
        logger.info("MetaAI instance not initialized. Waiting 30 seconds before retry...")
        await asyncio.sleep(30)
        try:
            logger.info("Retrying MetaAI instance initialization...")
            _meta_ai_instance = MetaAI(proxy=_get_proxies())
            if _meta_ai_instance and _meta_ai_instance.access_token:
                logger.info(f"MetaAI instance successfully initialized: {_meta_ai_instance.access_token[:50]}...")
            else:
                logger.warning("MetaAI instance created but token extraction failed. Will retry in next cycle.")
        except Exception as init_exc:  # noqa: BLE001
            logger.error(f"Failed to initialize MetaAI instance on retry: {init_exc}")
            logger.info("Will retry in next refresh cycle.")
    
    # If token extraction failed but instance exists, retry immediately
    elif not _meta_ai_instance.access_token:
        logger.info("Initial token extraction failed. Waiting 30 seconds before retry...")
        await asyncio.sleep(30)
        try:
            logger.info("Retrying access token extraction...")
            _meta_ai_instance.access_token = _meta_ai_instance.extract_access_token_from_page()
            if _meta_ai_instance.access_token:
                logger.info(f"Access token successfully extracted: {_meta_ai_instance.access_token[:50]}...")
            else:
                logger.warning("Token extraction retry failed. Will retry in next refresh cycle.")
        except Exception as token_exc:  # noqa: BLE001
            logger.error(f"Failed to extract access token on retry: {token_exc}")
    
    while True:
        try:
            await cache.refresh_if_needed(force=True)
            
            # Refresh access token for the global MetaAI instance
            if _meta_ai_instance:
                logger.info("Refreshing access token for global MetaAI instance...")
                try:
                    new_token = _meta_ai_instance.extract_access_token_from_page()
                    if new_token:
                        _meta_ai_instance.access_token = new_token
                        logger.info(f"Access token refreshed: {_meta_ai_instance.access_token[:50]}...")
                    else:
                        logger.warning("Token refresh returned None. Keeping existing token.")
                except Exception as token_exc:  # noqa: BLE001
                    logger.error(f"Failed to refresh access token: {token_exc}")
            else:
                # Try to recreate MetaAI instance if it's still None
                logger.info("MetaAI instance is None. Attempting to recreate...")
                try:
                    _meta_ai_instance = MetaAI(proxy=_get_proxies())
                    if _meta_ai_instance and _meta_ai_instance.access_token:
                        logger.info(f"MetaAI instance recreated successfully: {_meta_ai_instance.access_token[:50]}...")
                except Exception as recreate_exc:  # noqa: BLE001
                    logger.error(f"Failed to recreate MetaAI instance: {recreate_exc}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Background refresh failed: %s", exc)
        await asyncio.sleep(REFRESH_SECONDS)


def _apply_config_to_meta_ai() -> None:
    """Apply DB config values to the running MetaAI instance."""
    ai = _meta_ai_instance
    if ai is None or not hasattr(ai, "generation_api"):
        return
    gen_api = ai.generation_api
    doc_id_keys = [
        ("doc_id_text_to_image", "TEXT_TO_IMAGE"),
        ("doc_id_text_to_video", "TEXT_TO_VIDEO"),
        ("doc_id_image_alt", "IMAGE_ALT"),
        ("doc_id_extend_video", "EXTEND_VIDEO"),
        ("doc_id_fetch_conversation", "FETCH_CONVERSATION"),
        ("doc_id_fetch_media", "FETCH_MEDIA"),
        ("doc_id_poll_media", "POLL_MEDIA"),
    ]
    changed = False
    for cfg_key, doc_key in doc_id_keys:
        val = db.get_config(cfg_key)
        if val and len(val) == 32 and val.isalnum():
            if gen_api._doc_ids.get(doc_key) != val:
                gen_api._doc_ids[doc_key] = val
                gen_api._doc_id_sources[doc_key] = "db_config"
                logger.info("Applied doc_id[%s] from DB config: %s", doc_key, val)
                changed = True
    if changed:
        gen_api._log_active_doc_ids()


async def _run_ui_generation_job(gen_id: int, body: GenerationCreateRequest) -> None:
    logger.info(f"[UI-GEN {gen_id}] Starting {body.generation_type} generation")
    db.update_generation(gen_id, status="processing")
    if _meta_ai_instance is None:
        db.update_generation(gen_id, status="failed")
        return
    ai = _meta_ai_instance
    try:
        media_ids = body.input_media_ids
        if media_ids:
            resolved = []
            for mid in media_ids:
                if isinstance(mid, int):
                    up = db.get_upload(mid)
                    if up and up.get("media_id"):
                        resolved.append(up["media_id"])
                else:
                    resolved.append(mid)
            media_ids = resolved if resolved else None

        if body.generation_type == "image":
            result = await run_in_threadpool(
                ai.generate_image_new,
                prompt=body.prompt,
                orientation="VERTICAL",
                num_images=1,
                media_ids=media_ids,
            )
        else:
            result = await run_in_threadpool(
                ai.generate_video_new,
                prompt=body.prompt,
                auto_poll=True,
                max_poll_attempts=30,
                poll_wait_seconds=3,
                media_ids=media_ids,
            )
        success = result.get("success", False) if isinstance(result, dict) else False
        video_urls = result.get("video_urls", []) if isinstance(result, dict) else []
        media_urls = result.get("media_urls", []) if isinstance(result, dict) else []
        urls = video_urls or media_urls or []
        status = "completed" if (success and urls) else "failed"
        db.update_generation(
            gen_id,
            status=status,
            result_json=result if isinstance(result, dict) else {},
            video_urls=urls,
        )
        logger.info(f"[UI-GEN {gen_id}] Completed with status {status}, {len(urls)} file(s)")
    except Exception as exc:
        logger.error(f"[UI-GEN {gen_id}] Failed: {exc}")
        db.update_generation(gen_id, status="failed")


# ── Middleware registration (order matters!) ───────────────────────
# add_middleware inserts at position 0. To get execution order from
# outermost→innermost as CORS→Session→auth→log→router, we add in
# REVERSE order (innermost first, outermost last) so that after the
# internal reversal the stack is built correctly:
app.add_middleware(BaseHTTPMiddleware, dispatch=log_requests)    # 1 - innermost
app.add_middleware(BaseHTTPMiddleware, dispatch=auth_middleware)  # 2
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, max_age=86400)  # 3
app.add_middleware(                                                # 4 - outermost
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
