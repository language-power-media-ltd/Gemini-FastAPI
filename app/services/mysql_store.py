"""MySQL database connection for reading Gemini account credentials from Node.js project."""

import asyncio
import json
import random
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from ..utils.config import GeminiClientSettings

# Try to import aiomysql, but don't fail if not installed
try:
    import aiomysql

    AIOMYSQL_AVAILABLE = True
except ImportError:
    AIOMYSQL_AVAILABLE = False
    logger.warning("aiomysql not installed. Database-based account loading disabled.")


class MySQLAccountStore:
    """MySQL-based storage for Gemini account credentials."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        pool_size: int = 5,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.pool_size = pool_size
        self._pool: Optional["aiomysql.Pool"] = None

    async def _ensure_pool(self) -> "aiomysql.Pool":
        """Ensure connection pool is initialized."""
        if not AIOMYSQL_AVAILABLE:
            raise RuntimeError("aiomysql is not installed")

        if self._pool is None or self._pool.closed:
            self._pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                minsize=1,
                maxsize=self.pool_size,
                charset="utf8mb4",
                autocommit=True,
            )
            logger.info(f"MySQL connection pool created: {self.host}:{self.port}/{self.database}")
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool and not self._pool.closed:
            self._pool.close()
            await self._pool.wait_closed()
            logger.info("MySQL connection pool closed")

    async def get_gemini_accounts(self) -> list[GeminiClientSettings]:
        """
        Fetch all enabled Gemini accounts with valid cookies from the database.

        Returns:
            List of GeminiClientSettings objects ready to be used by GeminiClientPool.
        """
        if not AIOMYSQL_AVAILABLE:
            logger.warning("aiomysql not available, returning empty account list")
            return []

        pool = await self._ensure_pool()

        # Query accounts that have both gemini_1psid and gemini_1psidts set and are enabled
        sql = """
            SELECT 
                id,
                email,
                gemini_1psid,
                gemini_1psidts,
                proxy_port,
                proxy_country
            FROM accounts
            WHERE 
                `enable` = 1
                AND gemini_1psid IS NOT NULL 
                AND gemini_1psid != ''
                AND gemini_1psidts IS NOT NULL 
                AND gemini_1psidts != ''
        """

        accounts: list[GeminiClientSettings] = []

        try:
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(sql)
                    rows = await cursor.fetchall()

                    for row in rows:
                        # Build proxy URL if proxy info is available
                        proxy_url = None
                        proxy_port = row.get("proxy_port")
                        proxy_country = row.get("proxy_country")
                        
                        logger.debug(f"Account {row.get('email')}: proxy_port={proxy_port}, proxy_country={proxy_country}")
                        
                        if proxy_port and proxy_country:
                            # Use the google_login_proxy config from root config.json
                            proxy_url = self._build_proxy_url(proxy_port, proxy_country)
                            logger.debug(f"Built proxy URL for {row.get('email')}: {proxy_url[:50] if proxy_url else None}...")

                        # Use email or id as the client identifier
                        client_id = row.get("email") or f"account-{row['id']}"

                        try:
                            account = GeminiClientSettings(
                                id=client_id,
                                secure_1psid=row["gemini_1psid"],
                                secure_1psidts=row["gemini_1psidts"],
                                proxy=proxy_url,
                            )
                            accounts.append(account)
                            logger.info(f"Added account {client_id} with proxy: {'Yes' if proxy_url else 'No'}")
                        except Exception as e:
                            logger.warning(f"Failed to create GeminiClientSettings for {client_id}: {e}")
                            continue

            logger.info(f"Loaded {len(accounts)} Gemini accounts from database")
            return accounts

        except Exception as e:
            logger.error(f"Failed to fetch Gemini accounts from database: {e}")
            return []

    def _build_proxy_url(self, port: int, country: Optional[str]) -> Optional[str]:
        """Build proxy URL from port and country code."""
        # Try to read proxy config from root config.json
        try:
            root_config_path = Path(__file__).parent.parent.parent.parent / "config.json"
            if root_config_path.exists():
                with open(root_config_path) as f:
                    config = json.load(f)

                google_login_proxy = config.get("google_login_proxy", {})
                if google_login_proxy.get("enabled"):
                    base_url = google_login_proxy.get("url", "")
                    username = google_login_proxy.get("username", "")
                    password = google_login_proxy.get("password", "")

                    if base_url and username and password:
                        # Parse base URL to extract host
                        from urllib.parse import urlparse, quote

                        parsed = urlparse(base_url)
                        host = parsed.hostname

                        if host:
                            # URL encode the password in case it contains special characters
                            encoded_password = quote(password, safe="")
                            
                            # Build username with country suffix if provided
                            # Format: user-{username}-country-{country} (matching Node.js project)
                            if country:
                                full_username = f"user-{username}-country-{country}"
                            else:
                                full_username = f"user-{username}"
                            
                            # Format: http://user:pass@host:port
                            proxy_url = f"http://{full_username}:{encoded_password}@{host}:{port}"
                            logger.debug(f"Built proxy URL for port {port}: http://{full_username}:***@{host}:{port}")
                            return proxy_url

        except Exception as e:
            logger.warning(f"Failed to build proxy URL: {e}")

        return None

    async def update_account_cookies(
        self,
        email: str,
        secure_1psid: str,
        secure_1psidts: str,
    ) -> bool:
        """
        Update Gemini cookies for an account.

        Args:
            email: Account email (used as identifier)
            secure_1psid: New __Secure-1PSID cookie value
            secure_1psidts: New __Secure-1PSIDTS cookie value

        Returns:
            True if update was successful, False otherwise.
        """
        if not AIOMYSQL_AVAILABLE:
            return False

        pool = await self._ensure_pool()

        sql = """
            UPDATE accounts
            SET 
                gemini_1psid = %s,
                gemini_1psidts = %s,
                updated_at = UNIX_TIMESTAMP() * 1000
            WHERE email = %s
        """

        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(sql, (secure_1psid, secure_1psidts, email))
                    affected = cursor.rowcount

                    if affected > 0:
                        logger.debug(f"Updated Gemini cookies for account: {email}")
                        return True
                    else:
                        logger.warning(f"No account found with email: {email}")
                        return False

        except Exception as e:
            logger.error(f"Failed to update Gemini cookies for {email}: {e}")
            return False

    async def disable_account_web(self, email: str) -> bool:
        """
        Disable Gemini Web access for an account (set gemini_1psid and gemini_1psidts to empty).

        Args:
            email: Account email (used as identifier)

        Returns:
            True if update was successful, False otherwise.
        """
        if not AIOMYSQL_AVAILABLE:
            return False

        pool = await self._ensure_pool()

        sql = """
            UPDATE accounts
            SET 
                gemini_1psid = '',
                gemini_1psidts = '',
                updated_at = UNIX_TIMESTAMP() * 1000
            WHERE email = %s
        """

        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(sql, (email,))
                    affected = cursor.rowcount

                    if affected > 0:
                        logger.warning(f"Disabled Gemini Web access for banned account: {email}")
                        return True
                    else:
                        logger.warning(f"No account found with email: {email}")
                        return False

        except Exception as e:
            logger.error(f"Failed to disable Gemini Web for {email}: {e}")
            return False

    async def update_account_proxy(self, email: str, new_proxy_port: int) -> bool:
        """
        Update proxy port for an account.

        Args:
            email: Account email (used as identifier)
            new_proxy_port: New proxy port number

        Returns:
            True if update was successful, False otherwise.
        """
        if not AIOMYSQL_AVAILABLE:
            return False

        pool = await self._ensure_pool()

        sql = """
            UPDATE accounts
            SET 
                proxy_port = %s,
                updated_at = UNIX_TIMESTAMP() * 1000
            WHERE email = %s
        """

        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(sql, (new_proxy_port, email))
                    affected = cursor.rowcount

                    if affected > 0:
                        logger.info(f"Updated proxy port for account {email}: {new_proxy_port}")
                        return True
                    else:
                        logger.warning(f"No account found with email: {email}")
                        return False

        except Exception as e:
            logger.error(f"Failed to update proxy for {email}: {e}")
            return False

    async def get_random_proxy_port(self) -> Optional[int]:
        """
        Get a random proxy port from the configured range.

        Returns:
            Random port number, or None if config is not available.
        """
        try:
            root_config_path = Path(__file__).parent.parent.parent.parent / "config.json"
            if root_config_path.exists():
                with open(root_config_path) as f:
                    config = json.load(f)

                google_login_proxy = config.get("google_login_proxy", {})
                if google_login_proxy.get("enabled"):
                    port_min = google_login_proxy.get("port_range_min", 10001)
                    port_max = google_login_proxy.get("port_range_max", 63000)
                    return random.randint(port_min, port_max)

        except Exception as e:
            logger.warning(f"Failed to get random proxy port: {e}")

        return None

    async def get_account_info(self, email: str) -> Optional[dict]:
        """
        Get account info from database.

        Args:
            email: Account email

        Returns:
            Account info dict or None if not found.
        """
        if not AIOMYSQL_AVAILABLE:
            return None

        pool = await self._ensure_pool()

        sql = """
            SELECT 
                id,
                email,
                gemini_1psid,
                gemini_1psidts,
                proxy_port,
                proxy_country
            FROM accounts
            WHERE email = %s
        """

        try:
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(sql, (email,))
                    row = await cursor.fetchone()
                    return row

        except Exception as e:
            logger.error(f"Failed to get account info for {email}: {e}")
            return None


# Global instance (lazy initialization)
_mysql_store: Optional[MySQLAccountStore] = None


def get_mysql_store() -> Optional[MySQLAccountStore]:
    """Get or create the global MySQL store instance."""
    global _mysql_store
    return _mysql_store


async def initialize_mysql_store(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> Optional[MySQLAccountStore]:
    """Initialize the global MySQL store instance."""
    global _mysql_store

    if not AIOMYSQL_AVAILABLE:
        logger.warning("aiomysql not available, skipping MySQL store initialization")
        return None

    _mysql_store = MySQLAccountStore(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )

    # Test connection
    try:
        await _mysql_store._ensure_pool()
        return _mysql_store
    except Exception as e:
        logger.error(f"Failed to initialize MySQL store: {e}")
        _mysql_store = None
        return None
