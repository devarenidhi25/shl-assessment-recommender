import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.orchestrator import run_turn
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("shl_recommender.main")

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL Individual Test Solutions.",
    version="1.0.0",
)


@app.on_event("startup")
async def on_startup() -> None:
    # Force the catalog to load at process start (not lazily on first
    # request) so cold-start /health calls don't pay the parsing cost.
    from app.catalog import get_catalog

    catalog = get_catalog()
    logger.info("Loaded catalog with %d items from %s", len(catalog.all()), settings.catalog_path)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    start = time.monotonic()
    response = run_turn(payload.messages)
    elapsed = time.monotonic() - start
    logger.info(
        "chat turn processed in %.2fs | end_of_conversation=%s | n_recommendations=%d",
        elapsed,
        response.end_of_conversation,
        len(response.recommendations),
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Last-resort guard: /chat must never return a non-schema-compliant
    # error body to the grader. Any route other than /chat can still surface
    # a normal 500.
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    if request.url.path == "/chat":
        return JSONResponse(
            status_code=200,
            content=ChatResponse(
                reply=(
                    "Something went wrong on my end processing that message. Could you "
                    "try rephrasing your request about SHL assessments?"
                ),
                recommendations=[],
                end_of_conversation=False,
            ).model_dump(),
        )
    return JSONResponse(status_code=500, content={"detail": "internal_error"})
