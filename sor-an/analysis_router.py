# analysis/analysis_router.py

"""
Hybrid API Management - TAM ASYNC
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv

from analysis.analysis_schema_manager import load_module_run_function, load_analysis_schema
from utils.apikey_manager import APIKeyManager, BaseManager
from analysis.analysis_helpers import AnalysisHelpers

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# ==========================================================
# ✅ ASYNC ANALYSIS HELPERS WRAPPER - EN BAŞTA TANIMLA
# ==========================================================

class AsyncAnalysisHelpers:
    """Async wrapper for AnalysisHelpers - TANIM EN BAŞTA"""
    
    @staticmethod
    async def get_timestamp_async() -> float:
        """Async timestamp alma"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, AnalysisHelpers.get_timestamp)
    
    @staticmethod
    async def validate_output_async(output: Dict[str, Any]) -> bool:
        """Async output validation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, AnalysisHelpers.validate_output, output)
    
    @staticmethod
    async def create_fallback_output_async(module_name: str, reason: str = "Error") -> Dict[str, Any]:
        """Async fallback output"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            AnalysisHelpers.create_fallback_output, 
            module_name, 
            reason
        )

# Kullanım için alias
AnalysisHelpersAsync = AsyncAnalysisHelpers

# ==========================================================
# 📦 RESPONSE MODELLERİ
# ==========================================================

class AnalysisResponse(BaseModel):
    """Standart analiz response modeli"""
    symbol: str
    score: float
    signal: str
    confidence: Optional[float] = None
    components: Optional[Dict[str, float]] = None
    explain: Optional[str] = None
    module: Optional[str] = None
    api_source: str
    user_id: Optional[int] = None
    timestamp: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "score": 0.75,
                "signal": "bullish",
                "confidence": 0.8,
                "api_source": "apikey_manager",
                "user_id": 123,
                "timestamp": 1633046400.0
            }
        }

class ErrorResponse(BaseModel):
    """Hata response modeli"""
    error: str
    detail: Optional[str] = None
    module: Optional[str] = None

# ==========================================================
# 🔐 HYBRID API KEY YÖNETİMİ
# ==========================================================

# Sistem API'si (fallback için - .env'den)
SYSTEM_BINANCE_API = {
    "api_key": os.getenv("BINANCE_API_KEY"),
    "api_secret": os.getenv("BINANCE_API_SECRET"),
    "source": "system"
}

async def get_user_credentials(
    x_user_id: Optional[int] = Header(None, description="Kullanıcı ID"),
    x_api_key: Optional[str] = Header(None, description="Binance API Key (opsiyonel)"),
    x_api_secret: Optional[str] = Header(None, description="Binance API Secret (opsiyonel)")
) -> Dict[str, str]:
    """
    🔄 HYBRID CREDENTIAL MANAGEMENT:
    1. Önce header'dan gelen API key/secret
    2. Sonra APIKeyManager'dan kullanıcının kayıtlı API'si  
    3. En son sistem API'si (fallback)
    """
    
    # ✅ 1. HEADER'DAN GELEN API (en yüksek öncelik)
    if x_api_key and x_api_secret:
        logger.info(f"🔑 Using header API credentials for user: {x_user_id}")
        return {
            "api_key": x_api_key,
            "api_secret": x_api_secret, 
            "source": "header",
            "user_id": x_user_id
        }
    
    # ✅ 2. APIKeyManager'DAN KULLANICI API'Sİ
    if x_user_id:
        try:
            # ✅ ASYNC DÜZELTME: await ekle
            api_manager = await APIKeyManager.get_instance()
            user_creds = await api_manager.get_apikey(x_user_id)
            
            if user_creds and user_creds[0] and user_creds[1]:
                api_key, secret_key = user_creds
                logger.info(f"🔑 Using APIKeyManager credentials for user: {x_user_id}")
                return {
                    "api_key": api_key,
                    "api_secret": secret_key,
                    "source": "apikey_manager", 
                    "user_id": x_user_id
                }
            else:
                logger.warning(f"⚠️ No valid API credentials found in APIKeyManager for user: {x_user_id}")
        except Exception as e:
            logger.error(f"❌ APIKeyManager error for user {x_user_id}: {e}")
            # Fallback to system API
    
    # ✅ 3. SİSTEM API'Sİ (FALLBACK - RATE LİMİT RİSKLİ!)
    if SYSTEM_BINANCE_API["api_key"] and SYSTEM_BINANCE_API["api_secret"]:
        logger.warning(f"⚠️ Using SYSTEM API as fallback for user: {x_user_id}")
        return {
            **SYSTEM_BINANCE_API,
            "user_id": x_user_id,
            "warning": "Using shared system API - rate limits may apply"
        }
    
    # ❌ Hiçbir API bulunamadı
    logger.error(f"❌ No API credentials found for user: {x_user_id}")
    raise HTTPException(
        status_code=400, 
        detail={
            "error": "No Binance API credentials found",
            "solutions": [
                "Provide x-user-id + registered API keys in APIKeyManager",
                "Provide x-api-key + x-api-secret headers", 
                "Set system API keys in .env file"
            ]
        }
    )
    

# ==========================================================
# ✅ ASYNC UTILITY FONKSİYONLARI
# ==========================================================

async def load_analysis_schema_async(yaml_path: str = "analysis_metric_schema.yaml"):
    """Async schema yükleme"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: load_analysis_schema(yaml_path))

async def load_module_run_function_async(module_file: str):
    """Async modül run function yükleme"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: load_module_run_function(module_file))

# ==========================================================
# 📦 ASYNC MODÜL YÜKLEYİCİ
# ==========================================================

async def create_endpoint_async(run_func, module_name: str):
    """✅ TAM ASYNC endpoint creator"""
    
    async def endpoint(
        symbol: str = Query(..., description="Trading symbol (e.g., BTCUSDT)"),
        priority: Optional[str] = Query(None, description="Analysis priority: basic, pro, expert"),
        credentials: Dict = Depends(get_user_credentials)
    ):
        try:
            # ✅ ASYNC INPUT VALIDATION
            if not symbol or not symbol.strip():
                raise HTTPException(
                    status_code=400, 
                    detail="Symbol parameter is required and cannot be empty"
                )
            
            symbol = symbol.strip().upper()
            logger.info(f"🔍 Starting ASYNC analysis: {module_name} for {symbol}, user: {credentials.get('user_id')}")
            
            # ✅ ARTIK DOĞRU: AnalysisHelpersAsync TANIMLI
            timestamp = await AnalysisHelpersAsync.get_timestamp_async()
            
            # ✅ ASYNC CREDENTIALS ENTEGRASYONU
            try:
                if asyncio.iscoroutinefunction(run_func):
                    result = await run_func(
                        symbol=symbol, 
                        priority=priority,
                        credentials=credentials
                    )
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        lambda: run_func(symbol=symbol, priority=priority, credentials=credentials)
                    )
                logger.debug(f"✅ Used credentials-aware execution for {module_name}")
                
            except TypeError as e:
                if "unexpected keyword argument 'credentials'" in str(e):
                    logger.debug(f"🔄 Using legacy execution (no credentials) for {module_name}")
                    if asyncio.iscoroutinefunction(run_func):
                        result = await run_func(symbol=symbol, priority=priority)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, 
                            lambda: run_func(symbol=symbol, priority=priority)
                        )
                else:
                    raise
            
            # ✅ ASYNC RESULT VALIDATION & ENHANCEMENT
            if not isinstance(result, dict):
                logger.error(f"❌ Invalid result type from {module_name}: {type(result)}")
                raise HTTPException(
                    status_code=500, 
                    detail="Invalid response format from analysis module"
                )
            
            # Temel alanları kontrol et ve zenginleştir
            if 'symbol' not in result:
                result['symbol'] = symbol
            if 'module' not in result:
                result['module'] = module_name
            if 'timestamp' not in result:
                result['timestamp'] = timestamp
            
            # API source bilgisini response'a ekle
            result["api_source"] = credentials.get("source", "unknown")
            result["user_id"] = credentials.get("user_id")
            
            # ✅ ARTIK DOĞRU: AnalysisHelpersAsync TANIMLI
            is_valid = await AnalysisHelpersAsync.validate_output_async(result)
            if not is_valid:
                logger.warning(f"⚠️ Output validation failed for {module_name}, using fallback")
                result = await AnalysisHelpersAsync.create_fallback_output_async(
                    module_name, 
                    "Output validation failed"
                )
                result["api_source"] = credentials.get("source", "unknown")
                result["user_id"] = credentials.get("user_id")
                result["timestamp"] = timestamp
            
            logger.info(f"✅ ASYNC Analysis completed: {module_name} for {symbol}, score: {result.get('score', 'N/A')}")
            return result
            
        except HTTPException:
            raise
        except asyncio.CancelledError:
            logger.warning(f"⏹️ ASYNC Analysis cancelled: {module_name} for {symbol}")
            raise HTTPException(status_code=499, detail="Analysis cancelled by client")
        except Exception as e:
            logger.error(f"❌ ASYNC Analysis error in {module_name} for {symbol}: {str(e)}", exc_info=True)
            
            # ✅ ARTIK DOĞRU: AnalysisHelpersAsync TANIMLI
            fallback_result = await AnalysisHelpersAsync.create_fallback_output_async(
                module_name, 
                f"Analysis error: {str(e)}"
            )
            fallback_result["api_source"] = credentials.get("source", "unknown")
            fallback_result["user_id"] = credentials.get("user_id")
            fallback_result["timestamp"] = await AnalysisHelpersAsync.get_timestamp_async()
            
            return fallback_result
    
    return endpoint

# ==========================================================
# 🔁 ASYNC ROUTE OLUŞTURMA
# ==========================================================

async def initialize_routes_async():
    """Async route initialization"""
    try:
        schema = await load_analysis_schema_async()
        logger.info(f"📦 ASYNC Loading {len(schema.modules)} analysis modules...")
        
        route_count = 0
        
        for module in schema.modules:
            try:
                route_path = f"/{module.command}"
                module_file = module.file
                
                # ✅ ASYNC MODÜL YÜKLEME
                run_function = await load_module_run_function_async(module_file)
                
                # ✅ ASYNC ENDPOINT OLUŞTURMA
                endpoint = await create_endpoint_async(run_function, module.name)
                
                # Route'u ekle
                router.add_api_route(
                    path=route_path,
                    endpoint=endpoint,
                    methods=["GET"],
                    summary=module.name,
                    description=f"{module.objective or ''} (API: {module.api_type})",
                    tags=["analysis"],
                    response_model=AnalysisResponse,
                    responses={
                        400: {"model": ErrorResponse, "description": "Bad Request"},
                        500: {"model": ErrorResponse, "description": "Internal Server Error"}
                    }
                )
                
                route_count += 1
                logger.info(f"✅ ASYNC Route created: {route_path} -> {module.name}")
                
            except Exception as e:
                logger.error(f"❌ ASYNC Failed to create route for {module.name}: {str(e)}")
                continue
        
        logger.info(f"🎯 ASYNC Total routes created: {route_count}")
        return route_count
        
    except Exception as e:
        logger.error(f"❌ ASYNC Critical error during route initialization: {e}")
        return 0

# ==========================================================
# 🚀 STARTUP EVENT
# ==========================================================

@router.on_event("startup")
async def startup_event():
    """✅ TAM ASYNC startup event"""
    try:
        logger.info("🚀 ASYNC Initializing APIKeyManager...")
        
        # ✅ ASYNC DATABASE INITIALIZATION
        if not BaseManager._db_initialized:
            success = await BaseManager.initialize_database()
            if not success:
                logger.warning("⚠️ ASYNC Database initialization failed - continuing with limited functionality")
            else:
                logger.info("✅ ASYNC Database initialized successfully")
        
        # ✅ ASYNC APIKeyManager INITIALIZATION
        api_manager = await APIKeyManager.get_instance()
        logger.info("✅ ASYNC APIKeyManager initialized successfully")
        
        # ✅ ASYNC ROUTE INITIALIZATION
        route_count = await initialize_routes_async()
        logger.info(f"🚀 ASYNC Startup completed: {route_count} routes initialized")
        
    except Exception as e:
        logger.error(f"❌ ASYNC Startup error: {e}")

# ==========================================================
# 🏠 ROUTE ENDPOINT'LERİ - TEK TANIM!
# ==========================================================

@router.get("/health")
async def health_check():
    """✅ ASYNC sistem sağlık kontrolü"""
    timestamp = await AnalysisHelpersAsync.get_timestamp_async()
    return {
        "status": "healthy",
        "timestamp": timestamp,
        "async": True,
        "initialized_routes": len([r for r in router.routes if hasattr(r, 'methods')]) - 3  # /, /health, /modules
    }

@router.get("/modules")
async def list_modules():
    """✅ ASYNC modül listesi"""
    schema = await load_analysis_schema_async()
    return {
        "modules": [
            {
                "name": module.name,
                "command": module.command,
                "objective": module.objective,
                "api_type": module.api_type,
                "async": True
            }
            for module in schema.modules
        ],
        "count": len(schema.modules),
        "timestamp": await AnalysisHelpersAsync.get_timestamp_async()
    }

@router.get("/")
async def root():
    """✅ TEK ASYNC ana sayfa - DUPLICATE YOK!"""
    timestamp = await AnalysisHelpersAsync.get_timestamp_async()
    
    # Async route listesi
    endpoints = []
    for route in router.routes:
        if hasattr(route, "path") and route.path not in ["/", "/health", "/modules"]:
            endpoints.append({
                "path": route.path,
                "methods": route.methods,
                "name": getattr(route, "summary", "Unknown"),
                "async": True
            })
    
    return {
        "message": "ASYNC Analysis API Service",
        "version": "1.0.0",
        "timestamp": timestamp,
        "async": True,
        "endpoints": endpoints,
        "health_check": "/health",
        "modules_list": "/modules"
    }