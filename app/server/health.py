from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter
from loguru import logger

from ..models import HealthCheckResponse
from ..services import GeminiClientPool, LMDBConversationStore

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    pool = GeminiClientPool()
    db = LMDBConversationStore()

    try:
        await pool.init()
    except Exception as e:
        logger.error(f"Failed to initialize Gemini clients: {e}")
        return HealthCheckResponse(ok=False, error=str(e))

    client_status = pool.status()

    if not all(client_status.values()):
        logger.warning("One or more Gemini clients not running")

    stat = db.stats()
    if not stat:
        logger.error("Failed to retrieve LMDB conversation store stats")
        return HealthCheckResponse(
            ok=False, error="LMDB conversation store unavailable", clients=client_status
        )

    return HealthCheckResponse(ok=all(client_status.values()), storage=stat, clients=client_status)


@router.get("/pool/status")
async def get_pool_status() -> Dict[str, Any]:
    """Get detailed pool status including all clients and cooldowns."""
    pool = GeminiClientPool()

    # Get pool stats
    stats = pool.get_pool_stats()

    # Get detailed client list
    clients: List[Dict[str, Any]] = []
    for client in pool.clients:
        clients.append({
            "id": client.id,
            "running": client.running(),
            "proxy": client.proxy[:50] + "..." if client.proxy and len(client.proxy) > 50 else client.proxy,
        })

    # Get cooldown status
    cooldowns = pool.get_cooldown_status()

    # Get banned accounts
    banned_accounts = list(pool._banned_accounts)

    return {
        "stats": stats,
        "clients": clients,
        "cooldowns": cooldowns,
        "banned_accounts": banned_accounts,
        "timestamp": datetime.now().isoformat(),
    }
