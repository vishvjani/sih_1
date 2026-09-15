from supabase import create_client, Client
from src.app.config import config
import logging

logger = logging.getLogger(__name__)

_supabase_client: Client | None = None

def get_supabase() -> Client | None:
    """Returns a Supabase client, or None if credentials are not configured."""
    global _supabase_client
    if _supabase_client is None:
        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            logger.warning("Supabase credentials not configured. Database persistence disabled.")
            return None
        try:
            _supabase_client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info("Supabase client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            return None
    return _supabase_client
