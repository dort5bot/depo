"""
main.py - Telegram Bot Ana Giriş Noktası (FİNAL)
----------------------------------------
Aiogram 3.x + Router pattern + Webhook + Render uyumlu.
- Yeni config yapısına tam uyumlu
- Gelişmiş error handling
- Health check endpoints
- Graceful shutdown
free tier platformlarla tam uyumludur.
"""

import os
import asyncio
import logging
import signal
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import web
from dotenv import load_dotenv
from aiogram import BaseMiddleware
from aiogram import Bot, Dispatcher, Router
from aiogram.types import Update, Message
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import BaseFilter
from aiogram.types import ErrorEvent

from utils.handler_loader import load_handlers, clear_handler_cache
from utils.binance.binance_a import BinanceAPI
from utils.binance.binance_request import BinanceHTTPClient
from utils.binance.binance_circuit_breaker import CircuitBreaker
from config import BotConfig, get_config, get_telegram_token, get_admins

# ---------------------------------------------------------------------
# Config & Logging Setup
# ---------------------------------------------------------------------
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Global instances
bot: Optional[Bot] = None
dispatcher: Optional[Dispatcher] = None
binance_api: Optional[BinanceAPI] = None
app_config: Optional[BotConfig] = None
runner: Optional[web.AppRunner] = None

# Graceful shutdown flag
shutdown_event = asyncio.Event()

# ---------------------------------------------------------------------
# Signal Handling for Graceful Shutdown
# ---------------------------------------------------------------------
def handle_shutdown(signum, frame) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

# Register signal handlers
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# ---------------------------------------------------------------------
# Error Handler
# ---------------------------------------------------------------------
async def error_handler(event: ErrorEvent) -> None:
    """Global error handler for aiogram."""
    logger.error(f"❌ Error handling update: {event.exception}")
    
    try:
        if event.update.message:
            await event.update.message.answer("❌ Bir hata oluştu, lütfen daha sonra tekrar deneyin.")
    except Exception as e:
        logger.error(f"❌ Failed to send error message: {e}")

# ---------------------------------------------------------------------
# Rate Limiting Filter
# ---------------------------------------------------------------------
class RateLimitFilter(BaseFilter):
    """Rate limiting filter for messages."""
    
    def __init__(self, rate: float = 1.0):
        self.rate = rate
        self.last_called = 0.0

    async def __call__(self, message: Message) -> bool:
        current_time = asyncio.get_event_loop().time()
        if current_time - self.last_called >= self.rate:
            self.last_called = current_time
            return True
        return False

# ---------------------------------------------------------------------
# Lifespan Management (Async Context Manager)
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan():
    """Manage application lifecycle with async context manager."""
    global bot, dispatcher, binance_api, app_config
    
    try:
        # Load configuration
        app_config = await get_config()
        
        # Initialize bot with default properties
        bot = Bot(
            token=get_telegram_token(),
            default=DefaultBotProperties(
                parse_mode=ParseMode.HTML,
                timeout=app_config.REQUEST_TIMEOUT
            )
        )
        
        # Initialize dispatcher with main router and error handler
        main_router = Router()
        dispatcher = Dispatcher()
        dispatcher.include_router(main_router)
        dispatcher.errors.register(error_handler)
        
        # Initialize Binance API (only if trading is enabled)
        if app_config.ENABLE_TRADING:
            http_client = BinanceHTTPClient(
                api_key=app_config.BINANCE_API_KEY,
                secret_key=app_config.BINANCE_API_SECRET,
                base_url=app_config.BINANCE_BASE_URL,
                timeout=app_config.REQUEST_TIMEOUT
            )
            
            circuit_breaker = CircuitBreaker(
                failure_threshold=3,  # Default value
                recovery_timeout=60,  # Default value
                half_open_attempts=2
            )
            
            binance_api = BinanceAPI(http_client, circuit_breaker)
            logger.info("✅ Binance API initialized (trading enabled)")
        else:
            binance_api = None
            logger.info("ℹ️ Binance API not initialized (trading disabled)")
        
        logger.info("✅ Application components initialized")
        yield
        
    except Exception as e:
        logger.error(f"❌ Application initialization failed: {e}")
        raise
    finally:
        # Cleanup resources
        cleanup_tasks = []
        
        if binance_api:
            cleanup_tasks.append(binance_api.close())
        
        if bot and hasattr(bot, 'session'):
            cleanup_tasks.append(bot.session.close())
        
        if cleanup_tasks:
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Cleanup task failed: {result}")
        
        logger.info("🛑 Application resources cleaned up")

# ---------------------------------------------------------------------
# Health Check Endpoints
# ---------------------------------------------------------------------
async def health_check(request: web.Request) -> web.Response:
    """Health check endpoint for Render and monitoring."""
    global app_config
    
    try:
        services_status = await check_services()
        return web.json_response({
            "status": "healthy",
            "service": "telegram-bot",
            "platform": "render" if "RENDER" in os.environ else "local",
            "environment": "production" if not app_config.DEBUG else "development",
            "services": services_status,
            "timestamp": asyncio.get_event_loop().time()
        })
    except Exception as e:
        return web.json_response({
            "status": "unhealthy",
            "error": str(e)
        }, status=500)

async def readiness_check(request: web.Request) -> web.Response:
    """Readiness check for Kubernetes and load balancers."""
    global bot, binance_api, app_config
    
    if bot and app_config:
        # Binance API is only required if trading is enabled
        if app_config.ENABLE_TRADING and not binance_api:
            return web.json_response({"status": "not_ready"}, status=503)
        return web.json_response({"status": "ready"})
    else:
        return web.json_response({"status": "not_ready"}, status=503)

async def version_info(request: web.Request) -> web.Response:
    """Version and system information endpoint."""
    return web.json_response(await get_system_info())

# ---------------------------------------------------------------------
# Webhook Setup Functions
# ---------------------------------------------------------------------
async def on_startup(bot: Bot) -> None:
    """Execute on application startup."""
    global app_config
    
    try:
        # Clear handler cache and load handlers
        await clear_handler_cache()
        load_results = await load_handlers(dispatcher)
        
        if load_results["failed"] > 0:
            logger.warning(f"⚠️ {load_results['failed']} handlers failed to load")
        
        # Set webhook if webhook is configured
        if app_config.WEBHOOK_HOST and app_config.WEBHOOK_PATH:
            webhook_url = f"{app_config.WEBHOOK_HOST}{app_config.WEBHOOK_PATH}/{get_telegram_token()}"
            await bot.delete_webhook(drop_pending_updates=True)
            await bot.set_webhook(webhook_url, secret_token=app_config.WEBHOOK_SECRET)
            logger.info(f"✅ Webhook set successfully: {webhook_url}")
        else:
            logger.info("ℹ️ Webhook not configured, using polling mode")
        
        logger.info(f"🌐 Health check: http://{app_config.WEBAPP_HOST}:{app_config.WEBAPP_PORT}/health")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

async def on_shutdown(bot: Bot) -> None:
    """Execute on application shutdown."""
    logger.info("🛑 Shutting down application...")
    
    try:
        # Delete webhook if it was set
        if app_config.WEBHOOK_HOST and app_config.WEBHOOK_PATH:
            await bot.delete_webhook()
            logger.info("✅ Webhook deleted")
    except Exception as e:
        logger.warning(f"⚠️ Failed to delete webhook: {e}")


# Middleware desteği 
class ExampleMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        # Pre-processing
        result = await handler(event, data)
        # Post-processing
        return result

# Dispatcher'a ekleme
dispatcher.update.outer_middleware(ExampleMiddleware())


# ---------------------------------------------------------------------
# Main Application Factory
# ---------------------------------------------------------------------
async def create_app() -> web.Application:
    """Create and configure aiohttp web application."""
    global bot, dispatcher, app_config
    
    # Initialize components
    async with lifespan():
        # Create aiohttp app
        app = web.Application()
        
        # Register routes
        app.router.add_get("/", health_check)
        app.router.add_get("/health", health_check)
        app.router.add_get("/ready", readiness_check)
        app.router.add_get("/version", version_info)
        
        # Configure webhook handler if webhook is enabled
        if app_config.WEBHOOK_HOST and app_config.WEBHOOK_PATH:
            webhook_handler = SimpleRequestHandler(
                dispatcher=dispatcher,
                bot=bot,
                secret_token=app_config.WEBHOOK_SECRET
            )
            
            # Register webhook endpoint
            webhook_path = f"{app_config.WEBHOOK_PATH}/{get_telegram_token()}"
            webhook_handler.register(app, path=webhook_path)
            logger.info(f"📨 Webhook endpoint configured: {webhook_path}")
        
        # Setup startup/shutdown hooks
        app.on_startup.append(lambda app: on_startup(bot))
        app.on_shutdown.append(lambda app: on_shutdown(bot))
        
        # Setup aiogram application
        setup_application(app, dispatcher, bot=bot)
        
        logger.info(f"🚀 Application configured on port {app_config.WEBAPP_PORT}")
        
        return app

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
async def check_services() -> Dict[str, Any]:
    """Check connectivity to all external services."""
    global bot, binance_api, app_config
    
    services_status = {}
    
    # Check Telegram API
    try:
        if bot:
            me = await bot.get_me()
            services_status["telegram"] = {
                "status": "connected",
                "bot_username": me.username,
                "bot_id": me.id,
                "first_name": me.first_name
            }
        else:
            services_status["telegram"] = {"status": "disconnected", "error": "Bot not initialized"}
    except Exception as e:
        services_status["telegram"] = {
            "status": "disconnected",
            "error": str(e)
        }
    
    # Check Binance API (only if trading is enabled)
    if app_config.ENABLE_TRADING:
        try:
            if binance_api:
                ping_result = await binance_api.ping()
                services_status["binance"] = {
                    "status": "connected" if ping_result else "disconnected",
                    "ping": ping_result,
                    "trading_enabled": True
                }
            else:
                services_status["binance"] = {"status": "disconnected", "error": "Binance API not initialized", "trading_enabled": True}
        except Exception as e:
            services_status["binance"] = {
                "status": "disconnected",
                "error": str(e),
                "trading_enabled": True
            }
    else:
        services_status["binance"] = {
            "status": "disabled",
            "trading_enabled": False
        }
    
    return services_status

async def get_system_info() -> Dict[str, Any]:
    """Get system information and status."""
    global app_config
    
    return {
        "version": "1.0.0",
        "platform": "render" if "RENDER" in os.environ else "local",
        "environment": "production" if not app_config.DEBUG else "development",
        "python_version": os.sys.version,
        "aiohttp_version": aiohttp.__version__,
        "debug_mode": app_config.DEBUG,
        "trading_enabled": app_config.ENABLE_TRADING,
        "services": await check_services()
    }

# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------
async def main() -> None:
    """Main application entry point."""
    global app_config, runner
    
    try:
        # Load configuration
        app_config = await get_config()
        
        # Platform detection
        platform = "Render" if "RENDER" in os.environ else "Local"
        logger.info(f"🏗️ Platform detected: {platform}")
        logger.info(f"🌐 Environment: {'production' if not app_config.DEBUG else 'development'}")
        logger.info(f"🚪 Port: {app_config.WEBAPP_PORT}")
        logger.info(f"🏠 Host: {app_config.WEBAPP_HOST}")
        logger.info(f"🤖 Trading enabled: {app_config.ENABLE_TRADING}")
        
        # Create and run application
        app = await create_app()
        runner = web.AppRunner(app)
        
        await runner.setup()
        site = web.TCPSite(runner, host=app_config.WEBAPP_HOST, port=app_config.WEBAPP_PORT)
        await site.start()
        
        logger.info(f"✅ Server started successfully on port {app_config.WEBAPP_PORT}")
        logger.info("🤖 Bot is now running...")
        logger.info(f"📊 Health check: http://{app_config.WEBAPP_HOST}:{app_config.WEBAPP_PORT}/health")
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        logger.info("👋 Shutdown signal received, exiting...")
        
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"🚨 Critical error: {e}")
        raise
    finally:
        # Ensure proper cleanup
        try:
            if runner:
                await runner.cleanup()
            logger.info("✅ Application cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

if __name__ == "__main__":
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Application terminated by user")
    except Exception as e:
        logger.critical(f"💥 Fatal error: {e}")
        exit(1)
