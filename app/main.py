from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
    )
    application.include_router(router)
    return application


app = create_app()
