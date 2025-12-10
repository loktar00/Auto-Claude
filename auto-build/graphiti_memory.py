"""
Graphiti Memory Integration
===========================

Provides persistent knowledge graph memory using Graphiti with FalkorDB backend.
This is an OPTIONAL enhancement layer - when disabled, falls back to file-based memory.

Key Features:
- Session insight storage as episodes
- Codebase knowledge persistence
- Cross-session context retrieval
- Graceful fallback when unavailable

Architecture:
- Uses lazy initialization - doesn't connect until first use
- All operations are async with proper error handling
- Falls back to file-based memory on any failure
- Stores spec-specific memories using spec name as group_id

Usage:
    from graphiti_memory import GraphitiMemory, is_graphiti_enabled

    if is_graphiti_enabled():
        memory = GraphitiMemory(spec_dir, project_dir)
        await memory.save_session_insights(session_num, insights)
        context = await memory.get_relevant_context("authentication patterns")
        await memory.close()
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from graphiti_config import (
    GraphitiConfig,
    GraphitiState,
    is_graphiti_enabled,
    EPISODE_TYPE_SESSION_INSIGHT,
    EPISODE_TYPE_CODEBASE_DISCOVERY,
    EPISODE_TYPE_PATTERN,
    EPISODE_TYPE_GOTCHA,
)

# Configure logging
logger = logging.getLogger(__name__)

# Maximum results to return for context queries (avoid overwhelming agent context)
MAX_CONTEXT_RESULTS = 10

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1


class GraphitiMemory:
    """
    Manages Graphiti-based persistent memory for auto-build sessions.

    This class provides a high-level interface for:
    - Storing session insights as episodes
    - Recording codebase discoveries (file purposes, patterns, gotchas)
    - Retrieving relevant context for new sessions
    - Searching across all stored knowledge

    All operations are async and include error handling with fallback behavior.
    The integration is OPTIONAL - if Graphiti is disabled or unavailable,
    operations gracefully no-op or return empty results.
    """

    def __init__(self, spec_dir: Path, project_dir: Path):
        """
        Initialize Graphiti memory manager.

        Args:
            spec_dir: Spec directory (used as namespace/group_id)
            project_dir: Project root directory
        """
        self.spec_dir = spec_dir
        self.project_dir = project_dir
        self.config = GraphitiConfig.from_env()
        self.state: Optional[GraphitiState] = None
        self._graphiti = None  # Lazy initialization
        self._driver = None
        self._initialized = False
        self._available = False

        # Load existing state if available
        self.state = GraphitiState.load(spec_dir)

        # Check availability
        self._available = self.config.is_valid()

    @property
    def is_enabled(self) -> bool:
        """Check if Graphiti integration is enabled and configured."""
        return self._available

    @property
    def is_initialized(self) -> bool:
        """Check if Graphiti has been initialized for this spec."""
        return self._initialized and self.state is not None and self.state.initialized

    @property
    def group_id(self) -> str:
        """Get the group ID for this spec (uses spec folder name)."""
        return self.spec_dir.name

    async def initialize(self) -> bool:
        """
        Initialize the Graphiti client and build indices.

        This is called lazily on first operation. Handles connection
        errors gracefully by disabling Graphiti for the session.

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True

        if not self._available:
            logger.info("Graphiti not available - skipping initialization")
            return False

        try:
            # Import Graphiti here to avoid import errors when disabled
            from graphiti_core import Graphiti
            from graphiti_core.driver.falkordb_driver import FalkorDriver

            # Initialize FalkorDB driver
            self._driver = FalkorDriver(
                host=self.config.falkordb_host,
                port=self.config.falkordb_port,
                password=self.config.falkordb_password or None,
                database=self.config.database,
            )

            # Initialize Graphiti with the driver
            self._graphiti = Graphiti(graph_driver=self._driver)

            # Build indices (first time only)
            if not self.state or not self.state.indices_built:
                logger.info("Building Graphiti indices and constraints...")
                await self._graphiti.build_indices_and_constraints()

                # Update state
                if not self.state:
                    self.state = GraphitiState()

                self.state.initialized = True
                self.state.indices_built = True
                self.state.database = self.config.database
                self.state.created_at = datetime.now(timezone.utc).isoformat()
                self.state.save(self.spec_dir)

            self._initialized = True
            logger.info(f"Graphiti initialized for spec: {self.group_id}")
            return True

        except ImportError as e:
            logger.warning(
                f"Graphiti packages not installed: {e}. "
                "Install with: pip install graphiti-core[falkordb]"
            )
            self._available = False
            return False

        except Exception as e:
            logger.warning(f"Failed to initialize Graphiti: {e}")
            self._record_error(f"Initialization failed: {e}")
            self._available = False
            return False

    async def close(self) -> None:
        """
        Close the Graphiti client and clean up connections.

        Should be called when done with memory operations.
        """
        if self._graphiti:
            try:
                await self._graphiti.close()
                logger.info("Graphiti connection closed")
            except Exception as e:
                logger.warning(f"Error closing Graphiti: {e}")
            finally:
                self._graphiti = None
                self._driver = None
                self._initialized = False

    async def save_session_insights(
        self,
        session_num: int,
        insights: dict,
    ) -> bool:
        """
        Save session insights as a Graphiti episode.

        Args:
            session_num: Session number (1-indexed)
            insights: Dictionary containing session learnings with keys:
                - chunks_completed: list[str]
                - discoveries: dict
                - what_worked: list[str]
                - what_failed: list[str]
                - recommendations_for_next_session: list[str]

        Returns:
            True if saved successfully
        """
        if not await self._ensure_initialized():
            return False

        try:
            from graphiti_core.nodes import EpisodeType

            # Build episode content
            episode_content = {
                "type": EPISODE_TYPE_SESSION_INSIGHT,
                "session_number": session_num,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **insights,
            }

            # Add as episode
            await self._graphiti.add_episode(
                name=f"session_{session_num:03d}",
                episode_body=json.dumps(episode_content),
                source=EpisodeType.text,
                source_description=f"Auto-build session insight for {self.group_id}",
                reference_time=datetime.now(timezone.utc),
                group_id=self.group_id,
            )

            # Update state
            if self.state:
                self.state.last_session = session_num
                self.state.episode_count += 1
                self.state.save(self.spec_dir)

            logger.info(f"Saved session {session_num} insights to Graphiti")
            return True

        except Exception as e:
            logger.warning(f"Failed to save session insights: {e}")
            self._record_error(f"Save session insights failed: {e}")
            return False

    async def save_codebase_discoveries(
        self,
        discoveries: dict[str, str],
    ) -> bool:
        """
        Save codebase discoveries (file purposes) to the knowledge graph.

        Args:
            discoveries: Dictionary mapping file paths to their purposes
                Example: {
                    "src/api/auth.py": "Handles JWT authentication",
                    "src/models/user.py": "User database model"
                }

        Returns:
            True if saved successfully
        """
        if not await self._ensure_initialized():
            return False

        if not discoveries:
            return True

        try:
            from graphiti_core.nodes import EpisodeType

            # Build episode content
            episode_content = {
                "type": EPISODE_TYPE_CODEBASE_DISCOVERY,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files": discoveries,
            }

            # Add as episode
            await self._graphiti.add_episode(
                name=f"codebase_discovery_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                episode_body=json.dumps(episode_content),
                source=EpisodeType.text,
                source_description=f"Codebase file discoveries for {self.group_id}",
                reference_time=datetime.now(timezone.utc),
                group_id=self.group_id,
            )

            # Update state
            if self.state:
                self.state.episode_count += 1
                self.state.save(self.spec_dir)

            logger.info(f"Saved {len(discoveries)} codebase discoveries to Graphiti")
            return True

        except Exception as e:
            logger.warning(f"Failed to save codebase discoveries: {e}")
            self._record_error(f"Save discoveries failed: {e}")
            return False

    async def save_pattern(self, pattern: str) -> bool:
        """
        Save a code pattern to the knowledge graph.

        Args:
            pattern: Description of the code pattern

        Returns:
            True if saved successfully
        """
        if not await self._ensure_initialized():
            return False

        try:
            from graphiti_core.nodes import EpisodeType

            episode_content = {
                "type": EPISODE_TYPE_PATTERN,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pattern": pattern,
            }

            await self._graphiti.add_episode(
                name=f"pattern_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                episode_body=json.dumps(episode_content),
                source=EpisodeType.text,
                source_description=f"Code pattern for {self.group_id}",
                reference_time=datetime.now(timezone.utc),
                group_id=self.group_id,
            )

            if self.state:
                self.state.episode_count += 1
                self.state.save(self.spec_dir)

            logger.info(f"Saved pattern to Graphiti: {pattern[:50]}...")
            return True

        except Exception as e:
            logger.warning(f"Failed to save pattern: {e}")
            self._record_error(f"Save pattern failed: {e}")
            return False

    async def save_gotcha(self, gotcha: str) -> bool:
        """
        Save a gotcha (pitfall) to the knowledge graph.

        Args:
            gotcha: Description of the pitfall to avoid

        Returns:
            True if saved successfully
        """
        if not await self._ensure_initialized():
            return False

        try:
            from graphiti_core.nodes import EpisodeType

            episode_content = {
                "type": EPISODE_TYPE_GOTCHA,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gotcha": gotcha,
            }

            await self._graphiti.add_episode(
                name=f"gotcha_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                episode_body=json.dumps(episode_content),
                source=EpisodeType.text,
                source_description=f"Gotcha/pitfall for {self.group_id}",
                reference_time=datetime.now(timezone.utc),
                group_id=self.group_id,
            )

            if self.state:
                self.state.episode_count += 1
                self.state.save(self.spec_dir)

            logger.info(f"Saved gotcha to Graphiti: {gotcha[:50]}...")
            return True

        except Exception as e:
            logger.warning(f"Failed to save gotcha: {e}")
            self._record_error(f"Save gotcha failed: {e}")
            return False

    async def get_relevant_context(
        self,
        query: str,
        num_results: int = MAX_CONTEXT_RESULTS,
    ) -> list[dict]:
        """
        Search for relevant context based on a query.

        Args:
            query: Search query (e.g., "authentication patterns", "database models")
            num_results: Maximum number of results to return

        Returns:
            List of relevant context items with keys:
                - content: str
                - score: float
                - type: str (episode type)
        """
        if not await self._ensure_initialized():
            return []

        try:
            results = await self._graphiti.search(
                query=query,
                group_ids=[self.group_id],
                num_results=min(num_results, MAX_CONTEXT_RESULTS),
            )

            context_items = []
            for result in results:
                # Extract content from result
                content = getattr(result, 'content', None) or getattr(result, 'fact', None) or str(result)

                context_items.append({
                    "content": content,
                    "score": getattr(result, 'score', 0.0),
                    "type": getattr(result, 'type', 'unknown'),
                })

            logger.info(f"Found {len(context_items)} relevant context items for: {query[:50]}...")
            return context_items

        except Exception as e:
            logger.warning(f"Failed to search context: {e}")
            self._record_error(f"Search context failed: {e}")
            return []

    async def get_session_history(
        self,
        limit: int = 5,
    ) -> list[dict]:
        """
        Get recent session insights from the knowledge graph.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session insight summaries
        """
        if not await self._ensure_initialized():
            return []

        try:
            # Search for session insights
            results = await self._graphiti.search(
                query="session insight completed chunks recommendations",
                group_ids=[self.group_id],
                num_results=limit * 2,  # Get more to filter
            )

            sessions = []
            for result in results:
                content = getattr(result, 'content', None) or getattr(result, 'fact', None)
                if content and EPISODE_TYPE_SESSION_INSIGHT in str(content):
                    try:
                        # Try to parse as JSON
                        data = json.loads(content) if isinstance(content, str) else content
                        if data.get('type') == EPISODE_TYPE_SESSION_INSIGHT:
                            sessions.append(data)
                    except (json.JSONDecodeError, TypeError):
                        continue

            # Sort by session number and return latest
            sessions.sort(key=lambda x: x.get('session_number', 0), reverse=True)
            return sessions[:limit]

        except Exception as e:
            logger.warning(f"Failed to get session history: {e}")
            return []

    def get_status_summary(self) -> dict:
        """
        Get a summary of Graphiti memory status.

        Returns:
            Dict with status information
        """
        return {
            "enabled": self.is_enabled,
            "initialized": self.is_initialized,
            "database": self.config.database if self.is_enabled else None,
            "host": f"{self.config.falkordb_host}:{self.config.falkordb_port}" if self.is_enabled else None,
            "group_id": self.group_id,
            "episode_count": self.state.episode_count if self.state else 0,
            "last_session": self.state.last_session if self.state else None,
            "errors": len(self.state.error_log) if self.state else 0,
        }

    async def _ensure_initialized(self) -> bool:
        """
        Ensure Graphiti is initialized, attempting initialization if needed.

        Returns:
            True if initialized and ready
        """
        if self._initialized:
            return True

        if not self._available:
            return False

        return await self.initialize()

    def _record_error(self, error_msg: str) -> None:
        """Record an error in the state."""
        if not self.state:
            self.state = GraphitiState()

        self.state.record_error(error_msg)
        self.state.save(self.spec_dir)


# Convenience function for getting a memory manager
def get_graphiti_memory(spec_dir: Path, project_dir: Path) -> GraphitiMemory:
    """
    Get a GraphitiMemory instance for the given spec.

    This is the main entry point for other modules.

    Args:
        spec_dir: Spec directory
        project_dir: Project root directory

    Returns:
        GraphitiMemory instance
    """
    return GraphitiMemory(spec_dir, project_dir)


async def test_graphiti_connection() -> tuple[bool, str]:
    """
    Test if FalkorDB is available and Graphiti can connect.

    Returns:
        Tuple of (success: bool, message: str)
    """
    config = GraphitiConfig.from_env()

    if not config.enabled:
        return False, "Graphiti not enabled (GRAPHITI_ENABLED not set to true)"

    if not config.openai_api_key:
        return False, "OpenAI API key not set (required for Graphiti embeddings)"

    try:
        from graphiti_core import Graphiti
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        # Try to connect
        driver = FalkorDriver(
            host=config.falkordb_host,
            port=config.falkordb_port,
            password=config.falkordb_password or None,
            database=config.database,
        )

        graphiti = Graphiti(graph_driver=driver)

        # Try a simple operation
        await graphiti.build_indices_and_constraints()
        await graphiti.close()

        return True, f"Connected to FalkorDB at {config.falkordb_host}:{config.falkordb_port}"

    except ImportError as e:
        return False, f"Graphiti packages not installed: {e}"

    except Exception as e:
        return False, f"Connection failed: {e}"
