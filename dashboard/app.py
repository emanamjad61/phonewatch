"""FastAPI-powered web dashboard for PhoneWatch."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import CONFIG_PATH, LiveDetectionController, analytics_payload, benchmark_payload, model_payload


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"


class LiveControlPayload(BaseModel):
    """Validated live-control request body."""

    mode: str = "meme"
    confidence: float = Field(default=0.5, ge=0.1, le=1.0)
    camera: int = Field(default=0, ge=0)


def create_app(config_path: str | Path = CONFIG_PATH) -> FastAPI:
    """Build the FastAPI app instance."""
    controller = LiveDetectionController(config_path)

    app = FastAPI(title="PhoneWatch", docs_url=None, redoc_url=None)
    app.state.controller = controller
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/bootstrap")
    def bootstrap() -> dict:
        return {
            "defaults": controller.defaults(),
            "live": controller.snapshot(),
        }

    @app.get("/api/live/state")
    def live_state() -> dict:
        return controller.snapshot()

    @app.post("/api/live/start")
    def live_start(payload: LiveControlPayload) -> dict:
        return controller.start(mode=payload.mode, confidence=payload.confidence, camera=payload.camera)

    @app.post("/api/live/stop")
    def live_stop() -> dict:
        return controller.stop()

    @app.post("/api/live/reset")
    def live_reset() -> dict:
        return controller.reset()

    @app.get("/api/live/stream")
    def live_stream() -> StreamingResponse:
        return StreamingResponse(
            controller.stream_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/api/analytics")
    def analytics(q: str = "") -> dict:
        return analytics_payload(query=q)

    @app.get("/api/model")
    def model() -> dict:
        return model_payload()

    @app.post("/api/benchmark")
    def benchmark(frames: int = 30) -> dict:
        return benchmark_payload(n_frames=max(5, min(120, int(frames))))

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.on_event("shutdown")
    def _shutdown() -> None:
        controller.close()

    return app


app = create_app()


def run_dashboard_server(config_path: str | Path = CONFIG_PATH, host: str = "127.0.0.1", port: int = 8501) -> None:
    """Run the web dashboard server."""
    uvicorn.run(create_app(config_path), host=host, port=port, log_level="info")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PhoneWatch web dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="Port to listen on (default: 8501)")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to config.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dashboard_server(config_path=args.config, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
