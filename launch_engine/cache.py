import json
import time
from typing import Optional
import aiosqlite
from datetime import datetime

from launch_engine.core.validation import ValidationResult


def _validation_result_to_dict(result: ValidationResult) -> dict:
    """Convert ValidationResult to a JSON-serializable dict.

    Converts datetime objects to ISO format strings.
    """
    data = result.model_dump()
    # Convert datetime objects to ISO format strings
    if isinstance(data.get("checked_at"), datetime):
        data["checked_at"] = data["checked_at"].isoformat()
    if isinstance(data.get("evidence", {}).get("checked_at"), datetime):
        data["evidence"]["checked_at"] = data["evidence"]["checked_at"].isoformat()
    return data


def _dict_to_validation_result(data: dict) -> ValidationResult:
    """Convert a dict to ValidationResult.

    Converts ISO format strings back to datetime objects.
    """
    # Convert ISO format strings back to datetime objects
    if isinstance(data.get("checked_at"), str):
        data["checked_at"] = datetime.fromisoformat(data["checked_at"])
    if isinstance(data.get("evidence", {}).get("checked_at"), str):
        data["evidence"]["checked_at"] = datetime.fromisoformat(
            data["evidence"]["checked_at"]
        )
    return ValidationResult(**data)


class SQLiteCache:
    """SQLite-based cache with TTL, version, and context_hash support."""

    def __init__(self, db_path: str):
        """Initialize the cache with a database path.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create the cache table if it doesn't exist."""
        self.db = await aiosqlite.connect(self.db_path)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                version TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """)
        await self.db.commit()

    async def get(self, key: str) -> Optional[ValidationResult]:
        """Retrieve a cached result if it exists and hasn't expired.

        Args:
            key: The cache key to look up.

        Returns:
            The cached ValidationResult if found and not expired, None otherwise.
        """
        if self.db is None:
            await self.initialize()

        cursor = await self.db.execute(
            "SELECT result_json, expires_at FROM cache WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        await cursor.close()

        if row is None:
            return None

        result_json, expires_at = row
        now = int(time.time())

        # Check if expired
        if now >= expires_at:
            # Delete expired entry
            await self.db.execute("DELETE FROM cache WHERE key = ?", (key,))
            await self.db.commit()
            return None

        # Deserialize and return the result
        result_dict = json.loads(result_json)
        return _dict_to_validation_result(result_dict)

    async def set(
        self, key: str, result: ValidationResult, ttl: int, context_hash: str
    ) -> None:
        """Store a result in the cache with TTL and context hash.

        Args:
            key: The cache key.
            result: The ValidationResult to cache.
            ttl: Time to live in seconds.
            context_hash: Hash of the context for validation.
        """
        if self.db is None:
            await self.initialize()

        now = int(time.time())
        expires_at = now + ttl
        result_dict = _validation_result_to_dict(result)
        result_json = json.dumps(result_dict)

        # Use INSERT OR REPLACE to handle updates
        await self.db.execute(
            """
            INSERT OR REPLACE INTO cache
            (key, result_json, expires_at, version, context_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                result_json,
                expires_at,
                result.adapter_version,
                context_hash,
                now,
            ),
        )
        await self.db.commit()

    async def clear(self) -> None:
        """Clear all cache entries."""
        if self.db is None:
            await self.initialize()

        await self.db.execute("DELETE FROM cache")
        await self.db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self.db is not None:
            await self.db.close()
            self.db = None
