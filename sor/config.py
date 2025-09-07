"""bot/config.py - Aiogram 3.x uyumlu optimal config yönetimi

Binance ve Aiogram için yapılandırma sınıfı. Default değerler ile gelir,
.env dosyasındaki değerlerle override edilir.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Environment variables'ı yükle
load_dotenv()

# Logging yapılandırması
logger = logging.getLogger(__name__)
logger.setLevel(logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")))

# Global cache instance
_CONFIG_INSTANCE: Optional["BotConfig"] = None


@dataclass
class BotConfig:
    """Aiogram 3.x uyumlu bot yapılandırma sınıfı."""

    # ========================
    # 🤖 TELEGRAM BOT SETTINGS
    # ========================
    TELEGRAM_TOKEN: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    DEFAULT_LOCALE: str = field(default_factory=lambda: os.getenv("DEFAULT_LOCALE", "en"))
    ADMIN_IDS: List[int] = field(default_factory=lambda: [
        int(x.strip()) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()
    ])
    
    # Aiogram specific settings
    AIOGRAM_REDIS_HOST: str = field(default_factory=lambda: os.getenv("AIOGRAM_REDIS_HOST", "localhost"))
    AIOGRAM_REDIS_PORT: int = field(default_factory=lambda: int(os.getenv("AIOGRAM_REDIS_PORT", "6379")))
    AIOGRAM_REDIS_DB: int = field(default_factory=lambda: int(os.getenv("AIOGRAM_REDIS_DB", "0")))
    
    # FSM storage settings
    USE_REDIS_FSM: bool = field(default_factory=lambda: os.getenv("USE_REDIS_FSM", "true").lower() == "true")
    FSM_STORAGE_TTL: int = field(default_factory=lambda: int(os.getenv("FSM_STORAGE_TTL", "3600")))
    
    # Webhook settings (eğer kullanılıyorsa)
    WEBHOOK_HOST: str = field(default_factory=lambda: os.getenv("WEBHOOK_HOST", ""))
    WEBHOOK_PATH: str = field(default_factory=lambda: os.getenv("WEBHOOK_PATH", ""))
    WEBHOOK_SECRET: str = field(default_factory=lambda: os.getenv("WEBHOOK_SECRET", ""))
    WEBAPP_HOST: str = field(default_factory=lambda: os.getenv("WEBAPP_HOST", "0.0.0.0"))
    WEBAPP_PORT: int = field(default_factory=lambda: int(os.getenv("WEBAPP_PORT", "3001")))

    # ========================
    # 🔐 BINANCE API SETTINGS
    # ========================
    BINANCE_API_KEY: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    BINANCE_API_SECRET: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    BINANCE_BASE_URL: str = field(default_factory=lambda: os.getenv("BINANCE_BASE_URL", "https://api.binance.com"))
    BINANCE_FAPI_URL: str = field(default_factory=lambda: os.getenv("BINANCE_FAPI_URL", "https://fapi.binance.com"))
    BINANCE_WS_URL: str = field(default_factory=lambda: os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws"))

    # ========================
    # ⚙️ TECHNICAL SETTINGS
    # ========================
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Rate limiting
    REQUEST_TIMEOUT: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "30")))
    MAX_REQUESTS_PER_MINUTE: int = field(default_factory=lambda: int(os.getenv("MAX_REQUESTS_PER_MINUTE", "1200")))
    
    # Database settings
    DATABASE_URL: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    USE_DATABASE: bool = field(default_factory=lambda: os.getenv("USE_DATABASE", "false").lower() == "true")
    
    # Cache settings
    CACHE_TTL: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "300")))
    MAX_CACHE_SIZE: int = field(default_factory=lambda: int(os.getenv("MAX_CACHE_SIZE", "1000")))

    # ========================
    # 📊 TRADING SETTINGS
    # ========================
    SCAN_SYMBOLS: List[str] = field(default_factory=lambda: [
        symbol.strip() for symbol in os.getenv(
            "SCAN_SYMBOLS", 
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,TRXUSDT,CAKEUSDT,SUIUSDT,PEPEUSDT,ARPAUSDT,TURBOUSDT"
        ).split(",") if symbol.strip()
    ])
    
    ENABLE_TRADING: bool = field(default_factory=lambda: os.getenv("ENABLE_TRADING", "false").lower() == "true")
    TRADING_STRATEGY: str = field(default_factory=lambda: os.getenv("TRADING_STRATEGY", "conservative"))
    MAX_LEVERAGE: int = field(default_factory=lambda: int(os.getenv("MAX_LEVERAGE", "3")))
    
    # Alert settings
    ALERT_PRICE_CHANGE_PERCENT: float = field(default_factory=lambda: float(os.getenv("ALERT_PRICE_CHANGE_PERCENT", "5.0")))
    ENABLE_PRICE_ALERTS: bool = field(default_factory=lambda: os.getenv("ENABLE_PRICE_ALERTS", "true").lower() == "true")
    ALERT_COOLDOWN: int = field(default_factory=lambda: int(os.getenv("ALERT_COOLDOWN", "300")))

    # ========================
    # 🛠️ METHODS
    # ========================

    @classmethod
    def load(cls) -> "BotConfig":
        """Environment'dan config yükler."""
        return cls()

    def validate(self) -> bool:
        """Config değerlerini doğrular."""
        errors = []
        
        # Telegram bot validation
        if not self.TELEGRAM_TOKEN:
            errors.append("❌ TELEGRAM_TOKEN gereklidir")
        
        # Binance validation (eğer trading enabled ise)
        if self.ENABLE_TRADING:
            if not self.BINANCE_API_KEY:
                errors.append("❌ BINANCE_API_KEY gereklidir (trading enabled)")
            if not self.BINANCE_API_SECRET:
                errors.append("❌ BINANCE_API_SECRET gereklidir (trading enabled)")
        
        if errors:
            raise ValueError("\n".join(errors))
        
        return True

    def is_admin(self, user_id: int) -> bool:
        """Kullanıcının admin olup olmadığını kontrol eder."""
        return user_id in self.ADMIN_IDS

    def to_dict(self) -> Dict[str, Any]:
        """Config'i dict olarak döndürür (debug/log amaçlı)."""
        sensitive_fields = {"TELEGRAM_TOKEN", "BINANCE_API_KEY", "BINANCE_API_SECRET", "WEBHOOK_SECRET"}
        return {
            k: "***HIDDEN***" if k in sensitive_fields and getattr(self, k) else getattr(self, k)
            for k in self.__dataclass_fields__
        }


async def get_config() -> BotConfig:
    """Global config instance'ını döndürür.
    
    Aiogram 3.x ile uyumlu async singleton pattern.
    """
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = BotConfig.load()
        _CONFIG_INSTANCE.validate()
        logger.info("✅ Bot config yüklendi ve doğrulandı")
    return _CONFIG_INSTANCE


# Sync fonksiyonlar global instance'dan çekiyor
def get_telegram_token() -> str:
    """Aiogram için Telegram bot token'ını döndürür."""
    if _CONFIG_INSTANCE is None:
        raise RuntimeError("Config henüz yüklenmemiş. Lütfen önce get_config() çağırın.")
    return _CONFIG_INSTANCE.TELEGRAM_TOKEN


def get_admins() -> List[int]:
    """Admin kullanıcı ID'lerini döndürür."""
    if _CONFIG_INSTANCE is None:
        raise RuntimeError("Config henüz yüklenmemiş. Lütfen önce get_config() çağırın.")
    return _CONFIG_INSTANCE.ADMIN_IDS


# Redis configuration for Aiogram
def get_redis_config() -> Dict[str, Any]:
    """Aiogram için Redis konfigürasyonu döndürür."""
    if _CONFIG_INSTANCE is None:
        raise RuntimeError("Config henüz yüklenmemiş. Lütfen önce get_config() çağırın.")
    
    config = _CONFIG_INSTANCE
    return {
        "host": config.AIOGRAM_REDIS_HOST,
        "port": config.AIOGRAM_REDIS_PORT,
        "db": config.AIOGRAM_REDIS_DB,
    }