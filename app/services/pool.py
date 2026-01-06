import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from gemini_webapi.exceptions import AuthError
from loguru import logger

from ..utils import g_config
from ..utils.config import GeminiClientSettings
from ..utils.singleton import Singleton
from .client import GeminiClientWrapper

# Maximum number of clients to initialize concurrently
MAX_CONCURRENT_INIT = 30

# Cooldown duration for usage limit exceeded (24 hours)
USAGE_LIMIT_COOLDOWN_HOURS = 24

# Default interval for auto-refresh from database (5 minutes)
DEFAULT_AUTO_REFRESH_INTERVAL = 300


class GeminiClientPool(metaclass=Singleton):
    """Pool of GeminiClient instances identified by unique ids."""

    def __init__(self) -> None:
        self._clients: List[GeminiClientWrapper] = []
        self._id_map: Dict[str, GeminiClientWrapper] = {}
        self._round_robin: deque[GeminiClientWrapper] = deque()
        self._restart_locks: Dict[str, asyncio.Lock] = {}
        self._db_loaded: bool = False
        # Model cooldowns: {(client_id, model_name): expiry_time}
        self._model_cooldowns: Dict[Tuple[str, str], datetime] = {}
        # Auto-refresh task
        self._auto_refresh_task: Optional[asyncio.Task] = None
        self._auto_refresh_interval: int = DEFAULT_AUTO_REFRESH_INTERVAL
        # Banned accounts set (to avoid reloading them)
        self._banned_accounts: Set[str] = set()

        # Load from config.yaml initially (may be empty if using database)
        self._load_from_config()

    def _load_from_config(self) -> None:
        """Load clients from config.yaml."""
        for c in g_config.gemini.clients:
            self._add_client(c)

    def _add_client(self, settings: GeminiClientSettings) -> None:
        """Add a single client from settings."""
        if settings.id in self._id_map:
            logger.debug(f"Client {settings.id} already exists, skipping")
            return

        client = GeminiClientWrapper(
            client_id=settings.id,
            secure_1psid=settings.secure_1psid,
            secure_1psidts=settings.secure_1psidts,
            proxy=settings.proxy,
        )
        self._clients.append(client)
        self._id_map[settings.id] = client
        self._round_robin.append(client)
        self._restart_locks[settings.id] = asyncio.Lock()

    async def load_from_database(self) -> int:
        """
        Load clients from MySQL database.

        Returns:
            Number of clients loaded from database.
        """
        if not g_config.database.enabled:
            logger.debug("Database loading disabled in config")
            return 0

        try:
            from .mysql_store import initialize_mysql_store

            db_config = g_config.database
            store = await initialize_mysql_store(
                host=db_config.host,
                port=db_config.port,
                user=db_config.user,
                password=db_config.password,
                database=db_config.database,
            )

            if store is None:
                logger.warning("Failed to initialize MySQL store")
                return 0

            accounts = await store.get_gemini_accounts()

            if not accounts:
                logger.warning("No Gemini accounts found in database")
                return 0

            loaded_count = 0
            for account in accounts:
                if account.id not in self._id_map:
                    self._add_client(account)
                    loaded_count += 1

            self._db_loaded = True
            logger.info(f"Loaded {loaded_count} Gemini clients from database")
            return loaded_count

        except Exception as e:
            logger.exception(f"Failed to load clients from database: {e}")
            return 0

    async def init(self) -> None:
        """Initialize all clients in the pool concurrently with batch limit."""
        # Try to load from database first if enabled
        if g_config.database.enabled and not self._db_loaded:
            db_count = await self.load_from_database()
            if db_count > 0:
                logger.info(f"Using {db_count} clients from database")

        # Check if we have any clients
        if len(self._clients) == 0:
            raise ValueError(
                "No Gemini clients configured. "
                "Please configure clients in config.yaml or enable database loading."
            )

        # Use semaphore to limit concurrent initializations
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_INIT)

        async def init_client(client: GeminiClientWrapper) -> bool:
            """Initialize a single client and return success status."""
            async with semaphore:
                if client.running():
                    return True
                try:
                    await client.init(
                        timeout=g_config.gemini.timeout,
                        auto_refresh=g_config.gemini.auto_refresh,
                        verbose=g_config.gemini.verbose,
                        refresh_interval=g_config.gemini.refresh_interval,
                    )
                    return client.running()
                except AuthError as e:
                    logger.error(f"AuthError for client {client.id}: {e} - disabling account cookies")
                    # Disable account cookies in database
                    await self.handle_auth_error(client.id)
                    return False
                except Exception:
                    logger.exception(f"Failed to initialize client {client.id}")
                    return False

        # Initialize all clients concurrently (max {MAX_CONCURRENT_INIT} at a time)
        logger.info(f"Initializing {len(self._clients)} clients concurrently (max {MAX_CONCURRENT_INIT} per batch)...")
        results = await asyncio.gather(
            *[init_client(client) for client in self._clients],
            return_exceptions=True
        )

        success_count = sum(1 for r in results if r is True)

        if success_count == 0:
            raise RuntimeError("Failed to initialize any Gemini clients")

        logger.info(f"Initialized {success_count}/{len(self._clients)} Gemini clients")

    async def acquire(self, client_id: Optional[str] = None) -> GeminiClientWrapper:
        """Return a healthy client by id or using round-robin."""
        if not self._round_robin:
            raise RuntimeError("No Gemini clients configured")

        if client_id:
            client = self._id_map.get(client_id)
            if not client:
                raise ValueError(f"Client id {client_id} not found")
            if await self._ensure_client_ready(client):
                return client
            raise RuntimeError(
                f"Gemini client {client_id} is not running and could not be restarted"
            )

        for _ in range(len(self._round_robin)):
            client = self._round_robin[0]
            self._round_robin.rotate(-1)
            if await self._ensure_client_ready(client):
                return client

        raise RuntimeError("No Gemini clients are currently available")

    async def _ensure_client_ready(self, client: GeminiClientWrapper) -> bool:
        """Make sure the client is running, attempting a restart if needed."""
        if client.running():
            return True

        lock = self._restart_locks.get(client.id)
        if lock is None:
            return False  # Should not happen

        async with lock:
            if client.running():
                return True

            try:
                await client.init(
                    timeout=g_config.gemini.timeout,
                    auto_refresh=g_config.gemini.auto_refresh,
                    verbose=g_config.gemini.verbose,
                    refresh_interval=g_config.gemini.refresh_interval,
                )
                logger.info(f"Restarted Gemini client {client.id} after it stopped.")
                return True
            except Exception:
                logger.exception(f"Failed to restart Gemini client {client.id}")
                return False

    async def refresh_from_database(self) -> int:
        """
        Reload clients from database, adding new ones.

        Returns:
            Number of new clients added.
        """
        if not g_config.database.enabled:
            return 0

        try:
            from .mysql_store import get_mysql_store

            store = get_mysql_store()
            if store is None:
                return 0

            accounts = await store.get_gemini_accounts()
            new_clients: List[GeminiClientWrapper] = []

            for account in accounts:
                if account.id not in self._id_map:
                    self._add_client(account)
                    new_clients.append(self._id_map[account.id])

            if not new_clients:
                return 0

            # Use semaphore to limit concurrent initializations
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_INIT)

            async def init_new_client(client: GeminiClientWrapper) -> bool:
                """Initialize a single new client."""
                async with semaphore:
                    try:
                        await client.init(
                            timeout=g_config.gemini.timeout,
                            auto_refresh=g_config.gemini.auto_refresh,
                            verbose=g_config.gemini.verbose,
                            refresh_interval=g_config.gemini.refresh_interval,
                        )
                        return True
                    except AuthError as e:
                        logger.error(f"AuthError for new client {client.id}: {e} - disabling account cookies")
                        await self.handle_auth_error(client.id)
                        return False
                    except Exception:
                        logger.exception(f"Failed to initialize new client {client.id}")
                        return False

            # Initialize all new clients concurrently (max {MAX_CONCURRENT_INIT} at a time)
            logger.info(f"Initializing {len(new_clients)} new clients concurrently (max {MAX_CONCURRENT_INIT} per batch)...")
            results = await asyncio.gather(
                *[init_new_client(client) for client in new_clients],
                return_exceptions=True
            )

            added_count = sum(1 for r in results if r is True)

            if added_count > 0:
                logger.info(f"Added {added_count} new clients from database refresh")

            return added_count

        except Exception as e:
            logger.exception(f"Failed to refresh clients from database: {e}")
            return 0

    @property
    def clients(self) -> List[GeminiClientWrapper]:
        """Return managed clients."""
        return self._clients

    def status(self) -> Dict[str, bool]:
        """Return running status for each client."""
        return {client.id: client.running() for client in self._clients}

    def remove_client(self, client_id: str) -> bool:
        """
        Remove a client from the pool.

        Args:
            client_id: Client identifier (email)

        Returns:
            True if client was removed, False if not found.
        """
        if client_id not in self._id_map:
            logger.warning(f"Client {client_id} not found in pool")
            return False

        client = self._id_map.pop(client_id)

        # Remove from clients list
        if client in self._clients:
            self._clients.remove(client)

        # Remove from round robin
        if client in self._round_robin:
            self._round_robin.remove(client)

        # Remove restart lock
        if client_id in self._restart_locks:
            del self._restart_locks[client_id]

        # Close the client
        try:
            asyncio.create_task(client.close())
        except Exception as e:
            logger.warning(f"Error closing client {client_id}: {e}")

        logger.info(f"Removed client {client_id} from pool")
        return True

    def add_model_cooldown(self, client_id: str, model_name: str, hours: int = USAGE_LIMIT_COOLDOWN_HOURS) -> None:
        """
        Add a cooldown for a specific client + model combination.

        Args:
            client_id: Client identifier (email)
            model_name: Model name that hit the limit
            hours: Cooldown duration in hours (default 24)
        """
        expiry_time = datetime.now() + timedelta(hours=hours)
        self._model_cooldowns[(client_id, model_name)] = expiry_time
        logger.warning(f"Added {hours}h cooldown for {client_id} on model {model_name}, expires at {expiry_time}")

    def is_model_in_cooldown(self, client_id: str, model_name: str) -> bool:
        """
        Check if a client + model combination is in cooldown.

        Args:
            client_id: Client identifier (email)
            model_name: Model name to check

        Returns:
            True if in cooldown, False otherwise.
        """
        key = (client_id, model_name)
        if key not in self._model_cooldowns:
            return False

        expiry_time = self._model_cooldowns[key]
        if datetime.now() >= expiry_time:
            # Cooldown expired, remove it
            del self._model_cooldowns[key]
            logger.info(f"Cooldown expired for {client_id} on model {model_name}")
            return False

        return True

    def get_cooldown_status(self) -> Dict[str, str]:
        """Return current cooldown status."""
        now = datetime.now()
        result = {}
        expired_keys = []

        for (client_id, model_name), expiry_time in self._model_cooldowns.items():
            if now >= expiry_time:
                expired_keys.append((client_id, model_name))
            else:
                remaining = expiry_time - now
                result[f"{client_id}:{model_name}"] = f"Expires in {remaining}"

        # Clean up expired cooldowns
        for key in expired_keys:
            del self._model_cooldowns[key]

        return result

    async def handle_usage_limit_exceeded(self, client_id: str, model_name: str) -> None:
        """
        Handle USAGE_LIMIT_EXCEEDED error: add cooldown and remove client from pool.

        Args:
            client_id: Client identifier (email)
            model_name: Model name that hit the limit
        """
        logger.warning(f"Handling usage limit exceeded for {client_id} on model {model_name}")

        # Add 24h cooldown for this client + model
        self.add_model_cooldown(client_id, model_name, USAGE_LIMIT_COOLDOWN_HOURS)

        # Remove client from pool
        self.remove_client(client_id)

    async def handle_account_banned(self, client_id: str) -> None:
        """
        Handle ACCOUNT_BANNED error: disable account in database and remove from pool.

        Args:
            client_id: Client identifier (email)
        """
        logger.error(f"Handling account banned for {client_id}")

        # Add to banned set to prevent reloading
        self._banned_accounts.add(client_id)

        # Remove from pool first
        self.remove_client(client_id)

        # Disable in database
        try:
            from .mysql_store import get_mysql_store

            store = get_mysql_store()
            if store:
                await store.disable_account_web(client_id)
                logger.warning(f"Disabled Gemini Web access for banned account: {client_id}")
            else:
                logger.warning("MySQL store not available, cannot disable account in database")
        except Exception as e:
            logger.error(f"Failed to disable account {client_id} in database: {e}")

    async def handle_auth_error(self, client_id: str) -> None:
        """
        Handle AuthError (expired cookies): disable account cookies in database and remove from pool.

        Args:
            client_id: Client identifier (email)
        """
        logger.error(f"Handling auth error (expired cookies) for {client_id}")

        # Add to banned set to prevent reloading
        self._banned_accounts.add(client_id)

        # Remove from pool first
        self.remove_client(client_id)

        # Disable cookies in database
        try:
            from .mysql_store import get_mysql_store

            store = get_mysql_store()
            if store:
                await store.disable_account_web(client_id)
                logger.warning(f"Disabled Gemini cookies for account with expired auth: {client_id}")
            else:
                logger.warning("MySQL store not available, cannot disable account cookies in database")
        except Exception as e:
            logger.error(f"Failed to disable cookies for {client_id} in database: {e}")

    async def handle_ip_blocked(self, client_id: str) -> None:
        """
        Handle IP_TEMPORARILY_BLOCKED error: change proxy and reload client.

        Args:
            client_id: Client identifier (email)
        """
        logger.warning(f"Handling IP blocked for {client_id}")

        # Remove from pool first
        self.remove_client(client_id)

        try:
            from .mysql_store import get_mysql_store

            store = get_mysql_store()
            if not store:
                logger.warning("MySQL store not available, cannot update proxy")
                return

            # Get a new random proxy port
            new_port = await store.get_random_proxy_port()
            if not new_port:
                logger.warning(f"Cannot get new proxy port for {client_id}")
                return

            # Update proxy in database
            success = await store.update_account_proxy(client_id, new_port)
            if not success:
                logger.error(f"Failed to update proxy for {client_id}")
                return

            # Get updated account info and reload
            account_info = await store.get_account_info(client_id)
            if not account_info:
                logger.error(f"Cannot get account info for {client_id}")
                return

            # Build new proxy URL
            proxy_url = store._build_proxy_url(new_port, account_info.get("proxy_country"))

            # Create new client settings
            new_settings = GeminiClientSettings(
                id=client_id,
                secure_1psid=account_info["gemini_1psid"],
                secure_1psidts=account_info["gemini_1psidts"],
                proxy=proxy_url,
            )

            # Add back to pool
            self._add_client(new_settings)

            # Initialize the new client
            new_client = self._id_map.get(client_id)
            if new_client:
                try:
                    await new_client.init(
                        timeout=g_config.gemini.timeout,
                        auto_refresh=g_config.gemini.auto_refresh,
                        verbose=g_config.gemini.verbose,
                        refresh_interval=g_config.gemini.refresh_interval,
                    )
                    logger.info(f"Reloaded client {client_id} with new proxy port {new_port}")
                except Exception as e:
                    logger.error(f"Failed to initialize reloaded client {client_id}: {e}")
                    self.remove_client(client_id)

        except Exception as e:
            logger.exception(f"Failed to handle IP blocked for {client_id}: {e}")

    async def start_auto_refresh(self, interval: int = DEFAULT_AUTO_REFRESH_INTERVAL) -> None:
        """
        Start the auto-refresh task that periodically loads new accounts from database.

        Args:
            interval: Refresh interval in seconds (default 5 minutes)
        """
        if self._auto_refresh_task and not self._auto_refresh_task.done():
            logger.warning("Auto-refresh task already running")
            return

        self._auto_refresh_interval = interval
        self._auto_refresh_task = asyncio.create_task(self._auto_refresh_loop())
        logger.info(f"Started auto-refresh task with interval {interval}s")

    async def stop_auto_refresh(self) -> None:
        """Stop the auto-refresh task."""
        if self._auto_refresh_task:
            self._auto_refresh_task.cancel()
            try:
                await self._auto_refresh_task
            except asyncio.CancelledError:
                pass
            self._auto_refresh_task = None
            logger.info("Stopped auto-refresh task")

    async def _auto_refresh_loop(self) -> None:
        """Internal loop that periodically refreshes accounts from database."""
        while True:
            try:
                await asyncio.sleep(self._auto_refresh_interval)
                
                # Clean up expired cooldowns
                self._cleanup_expired_cooldowns()
                
                # Refresh from database
                added = await self._refresh_accounts_from_db()
                if added > 0:
                    logger.info(f"Auto-refresh: added {added} new accounts to pool")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in auto-refresh loop: {e}")
                # Continue running even if there's an error
                await asyncio.sleep(60)  # Wait a bit before retrying

    def _cleanup_expired_cooldowns(self) -> None:
        """Clean up expired cooldowns from the cooldown map."""
        now = datetime.now()
        expired_keys = [
            key for key, expiry in self._model_cooldowns.items()
            if now >= expiry
        ]
        for key in expired_keys:
            del self._model_cooldowns[key]
            logger.debug(f"Cleaned up expired cooldown for {key}")

    async def _refresh_accounts_from_db(self) -> int:
        """
        Refresh accounts from database, adding new ones that are not in pool.

        Returns:
            Number of new accounts added.
        """
        if not g_config.database.enabled:
            return 0

        try:
            from .mysql_store import get_mysql_store

            store = get_mysql_store()
            if store is None:
                return 0

            accounts = await store.get_gemini_accounts()
            new_clients: List[GeminiClientWrapper] = []

            for account in accounts:
                # Skip if already in pool
                if account.id in self._id_map:
                    continue
                    
                # Skip banned accounts
                if account.id in self._banned_accounts:
                    logger.debug(f"Skipping banned account: {account.id}")
                    continue
                    
                # Skip accounts in model cooldown (check all models)
                # We don't skip based on model cooldown here because
                # the account might be usable for other models
                
                self._add_client(account)
                new_clients.append(self._id_map[account.id])

            if not new_clients:
                return 0

            # Use semaphore to limit concurrent initializations
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_INIT)

            async def init_new_client(client: GeminiClientWrapper) -> bool:
                """Initialize a single new client."""
                async with semaphore:
                    try:
                        await client.init(
                            timeout=g_config.gemini.timeout,
                            auto_refresh=g_config.gemini.auto_refresh,
                            verbose=g_config.gemini.verbose,
                            refresh_interval=g_config.gemini.refresh_interval,
                        )
                        return True
                    except AuthError as e:
                        logger.error(f"AuthError for client {client.id}: {e} - disabling account cookies")
                        await self.handle_auth_error(client.id)
                        return False
                    except Exception:
                        logger.exception(f"Failed to initialize new client {client.id}")
                        # Remove failed client from pool
                        self.remove_client(client.id)
                        return False

            # Initialize all new clients concurrently
            if new_clients:
                logger.info(f"Auto-refresh: initializing {len(new_clients)} new clients...")
                results = await asyncio.gather(
                    *[init_new_client(client) for client in new_clients],
                    return_exceptions=True
                )

                added_count = sum(1 for r in results if r is True)
                return added_count

            return 0

        except Exception as e:
            logger.exception(f"Failed to refresh accounts from database: {e}")
            return 0

    def get_pool_stats(self) -> dict:
        """
        Get statistics about the pool.

        Returns:
            Dictionary with pool statistics.
        """
        running_count = sum(1 for c in self._clients if c.running())
        return {
            "total_clients": len(self._clients),
            "running_clients": running_count,
            "stopped_clients": len(self._clients) - running_count,
            "banned_accounts": len(self._banned_accounts),
            "active_cooldowns": len(self._model_cooldowns),
            "auto_refresh_running": self._auto_refresh_task is not None and not self._auto_refresh_task.done(),
            "auto_refresh_interval": self._auto_refresh_interval,
        }

    def remove_stopped_clients(self) -> dict:
        """
        Remove all stopped (non-running) clients from the pool.

        Returns:
            Dictionary with removal results.
        """
        stopped_clients = [c for c in self._clients if not c.running()]
        removed = []
        failed = []

        for client in stopped_clients:
            try:
                if self.remove_client(client.id):
                    removed.append(client.id)
                else:
                    failed.append(client.id)
            except Exception as e:
                logger.error(f"Error removing stopped client {client.id}: {e}")
                failed.append(client.id)

        logger.info(f"Removed {len(removed)} stopped clients from pool")
        return {
            "removed": removed,
            "failed": failed,
            "removed_count": len(removed),
            "failed_count": len(failed),
        }

    def remove_clients_by_domain(self, domain: str) -> dict:
        """
        Remove all clients matching a specific email domain from the pool.

        Args:
            domain: Email domain to match (e.g., 'gmail.com', 'example.com')

        Returns:
            Dictionary with removal results.
        """
        if not domain:
            return {"removed": [], "failed": [], "removed_count": 0, "failed_count": 0, "error": "Domain is required"}

        # Normalize domain
        domain = domain.lower().strip()
        if domain.startswith("@"):
            domain = domain[1:]

        matching_clients = [c for c in self._clients if c.id.lower().endswith(f"@{domain}")]
        removed = []
        failed = []

        for client in matching_clients:
            try:
                if self.remove_client(client.id):
                    removed.append(client.id)
                else:
                    failed.append(client.id)
            except Exception as e:
                logger.error(f"Error removing client {client.id} by domain: {e}")
                failed.append(client.id)

        logger.info(f"Removed {len(removed)} clients with domain @{domain} from pool")
        return {
            "removed": removed,
            "failed": failed,
            "removed_count": len(removed),
            "failed_count": len(failed),
            "domain": domain,
        }

    def get_domains(self) -> dict:
        """
        Get all unique email domains in the pool with their counts.

        Returns:
            Dictionary mapping domains to their client counts.
        """
        domain_counts: Dict[str, int] = {}
        for client in self._clients:
            email = client.id.lower()
            if "@" in email:
                domain = email.split("@")[1]
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return domain_counts

    def record_request(self, client_id: str, success: bool = True, error: str | None = None) -> None:
        """
        Record a request for a specific client.

        Args:
            client_id: Client identifier (email)
            success: Whether the request was successful
            error: Error message if the request failed
        """
        client = self._id_map.get(client_id)
        if client:
            client.record_request(success, error)

    def get_client_stats(self, client_id: str) -> Optional[dict]:
        """
        Get statistics for a specific client.

        Args:
            client_id: Client identifier (email)

        Returns:
            Client statistics dictionary or None if not found.
        """
        client = self._id_map.get(client_id)
        if client:
            return client.get_stats()
        return None

    def get_all_client_stats(self) -> Dict[str, dict]:
        """
        Get statistics for all clients.

        Returns:
            Dictionary mapping client IDs to their statistics.
        """
        return {client.id: client.get_stats() for client in self._clients}

    async def reinit_all_clients(self) -> dict:
        """
        Reinitialize stopped clients by fetching fresh cookies from database (one-click login).
        Already running clients will be skipped.
        Stopped clients will be removed and recreated with fresh cookies from database.

        Returns:
            Dictionary with reinitialization results.
        """
        results = {
            "success": [],
            "failed": [],
            "skipped": [],
            "total": len(self._clients),
        }

        if not self._clients:
            return results

        # Separate running and stopped clients
        stopped_client_ids = [c.id for c in self._clients if not c.running()]
        running_clients = [c for c in self._clients if c.running()]

        # Add running clients to skipped list
        for client in running_clients:
            results["skipped"].append(client.id)

        if not stopped_client_ids:
            logger.info(f"All {len(running_clients)} clients are already running, nothing to reinitialize")
            return results

        # Fetch fresh cookies from database
        db_accounts: Dict[str, "GeminiClientSettings"] = {}
        if g_config.database.enabled:
            try:
                from .mysql_store import get_mysql_store
                store = get_mysql_store()
                if store:
                    accounts = await store.get_gemini_accounts()
                    db_accounts = {acc.id: acc for acc in accounts}
                    logger.info(f"Fetched {len(db_accounts)} accounts with fresh cookies from database")
            except Exception as e:
                logger.error(f"Failed to fetch accounts from database: {e}")
                # Return early if database fetch failed
                for client_id in stopped_client_ids:
                    results["failed"].append({"id": client_id, "error": "Failed to fetch cookies from database"})
                return results

        # Remove stopped clients first
        for client_id in stopped_client_ids:
            self.remove_client(client_id)

        # Use semaphore to limit concurrent initializations
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_INIT)

        async def create_and_init_client(client_id: str) -> tuple[str, bool, str]:
            """Create and initialize a client with fresh cookies from database."""
            async with semaphore:
                try:
                    if client_id not in db_accounts:
                        return (client_id, False, "No cookies found in database")

                    db_account = db_accounts[client_id]

                    # Create new client with fresh cookies
                    new_client = GeminiClientWrapper(
                        client_id=db_account.id,
                        secure_1psid=db_account.secure_1psid,
                        secure_1psidts=db_account.secure_1psidts,
                        proxy=db_account.proxy,
                    )

                    # Add to pool
                    self._clients.append(new_client)
                    self._id_map[db_account.id] = new_client
                    self._round_robin.append(new_client)
                    self._restart_locks[db_account.id] = asyncio.Lock()

                    # Initialize
                    await new_client.init(
                        timeout=g_config.gemini.timeout,
                        auto_refresh=g_config.gemini.auto_refresh,
                        verbose=g_config.gemini.verbose,
                        refresh_interval=g_config.gemini.refresh_interval,
                    )
                    logger.info(f"Successfully reinitialized client {client_id} with fresh cookies")
                    return (client_id, True, "")
                except AuthError as e:
                    logger.error(f"AuthError for client {client_id}: {e}")
                    # Remove failed client from pool
                    self.remove_client(client_id)
                    return (client_id, False, f"AuthError: {str(e)}")
                except Exception as e:
                    logger.exception(f"Failed to reinitialize client {client_id}")
                    # Remove failed client from pool
                    self.remove_client(client_id)
                    return (client_id, False, str(e))

        # Reinitialize only stopped clients concurrently
        logger.info(f"Reinitializing {len(stopped_client_ids)} stopped clients with fresh cookies from database (skipping {len(running_clients)} already running)...")
        init_results = await asyncio.gather(
            *[create_and_init_client(client_id) for client_id in stopped_client_ids],
            return_exceptions=True
        )

        for result in init_results:
            if isinstance(result, Exception):
                logger.error(f"Unexpected error during reinit: {result}")
                results["failed"].append({"id": "unknown", "error": str(result)})
            else:
                client_id, success, error = result
                if success:
                    results["success"].append(client_id)
                else:
                    results["failed"].append({"id": client_id, "error": error})

        logger.info(f"Reinitialization complete: {len(results['success'])} success, {len(results['failed'])} failed, {len(results['skipped'])} skipped (already running)")
        return results
