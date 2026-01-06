from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel

from ..models import HealthCheckResponse
from ..services import GeminiClientPool, LMDBConversationStore

router = APIRouter()


class DeleteByDomainRequest(BaseModel):
    """Request body for deleting clients by domain."""
    domain: str


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

    # Get detailed client list with statistics
    clients: List[Dict[str, Any]] = []
    for client in pool.clients:
        client_info = {
            "id": client.id,
            "running": client.running(),
            "proxy": client.proxy[:50] + "..." if client.proxy and len(client.proxy) > 50 else client.proxy,
        }
        # Add client statistics
        client_stats = client.get_stats()
        client_info.update(client_stats)
        clients.append(client_info)

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


@router.get("/pool/domains")
async def get_pool_domains() -> Dict[str, Any]:
    """Get all unique email domains in the pool with their counts."""
    pool = GeminiClientPool()
    domains = pool.get_domains()
    return {
        "success": True,
        "domains": domains,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/pool/remove-stopped")
async def remove_stopped_clients() -> Dict[str, Any]:
    """Remove all stopped (non-running) clients from the pool."""
    pool = GeminiClientPool()
    result = pool.remove_stopped_clients()
    return {
        "success": True,
        **result,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/pool/remove-by-domain")
async def remove_clients_by_domain(request: DeleteByDomainRequest) -> Dict[str, Any]:
    """Remove all clients matching a specific email domain from the pool."""
    pool = GeminiClientPool()
    result = pool.remove_clients_by_domain(request.domain)
    return {
        "success": True,
        **result,
        "timestamp": datetime.now().isoformat(),
    }


@router.delete("/pool/client/{client_id:path}")
async def remove_single_client(client_id: str) -> Dict[str, Any]:
    """Remove a single client from the pool by ID (email)."""
    pool = GeminiClientPool()
    success = pool.remove_client(client_id)
    return {
        "success": success,
        "client_id": client_id,
        "message": f"Client {client_id} removed" if success else f"Client {client_id} not found",
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/pool/reinit-all")
async def reinit_all_clients() -> Dict[str, Any]:
    """
    Reinitialize stopped clients in the pool (one-click login).
    Already running clients will be skipped.
    """
    pool = GeminiClientPool()
    result = await pool.reinit_all_clients()
    return {
        "success": True,
        **result,
        "success_count": len(result["success"]),
        "failed_count": len(result["failed"]),
        "skipped_count": len(result.get("skipped", [])),
        "timestamp": datetime.now().isoformat(),
    }
