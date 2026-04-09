from fastapi import APIRouter, Depends, WebSocket

from app.dependencies import AppDependencies, get_deps
from app.ws_kws import handle_kws_ws

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/kws")
async def ws_kws(
    websocket: WebSocket,
    deps: AppDependencies = Depends(get_deps),
):
    await handle_kws_ws(websocket, deps.service)
