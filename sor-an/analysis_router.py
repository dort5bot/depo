# analysis/analysis_router.py

"""
analysis_router.py
Version: 2.0.0 - Tam Async Optimized
Hybrid API Management - TAM ASYNC
BaseModule, Core ve Helpers ile tam uyumlu
multi-user destekli ve tüm analiz modülleriyle tam uyumlu
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Query, HTTPException, Depends, Header, BackgroundTasks
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

# ✅ TAM ASYNC IMPORTS - YENİ YAPILANDIRMA
from analysis.analysis_core import (
    AnalysisAggregator, 
    get_aggregator,
    AnalysisResult,
    AnalysisStatus
)
from analysis.analysis_helpers import (
    AnalysisOutput, 
    AnalysisUtilities,
    utility_functions
)
from utils.apikey_manager import APIKeyManager, BaseManager

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# ==========================================================
# 📦 RESPONSE MODELLERİ - GÜNCELLENMİŞ
# ==========================================================

class AnalysisResponse(BaseModel):
    """Standart analiz response modeli - AnalysisOutput ile uyumlu"""
    symbol: str
    score: float
    signal: str
    confidence: Optional[float] = None
    components: Dict[str, float] = {}
    explain: str
    module: str
    api_source: str
    user_id: Optional[int] = None
    timestamp: float
    fallback: Optional[bool] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "score": 0.75,
                "signal": "bullish",
                "confidence": 0.8,
                "components": {"rsi": 0.7, "macd": 0.8},
                "explain": "Strong bullish trend detected",
                "module": "trend_analysis",
                "api_source": "apikey_manager",
                "user_id": 123,
                "timestamp": 1633046400.0
            }
        }

class MultiAnalysisResponse(BaseModel):
    """Çoklu analiz response modeli"""
    symbol: str
    results: Dict[str, AnalysisResponse]
    composite_scores: Dict[str, float]
    summary: Dict[str, Any]
    user_id: Optional[int] = None
    timestamp: float

class ErrorResponse(BaseModel):
    """Hata response modeli"""
    error: str
    detail: Optional[str] = None
    module: Optional[str] = None
    timestamp: float

# ==========================================================
# 🔐 HYBRID API KEY YÖNETİMİ - GÜNCELLENMİŞ
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
) -> Dict[str, Any]:
    """
    🔄 HYBRID CREDENTIAL MANAGEMENT - TAM ASYNC:
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
    
    # ✅ 2. APIKeyManager'DAN KULLANICI API'Sİ - TAM ASYNC
    if x_user_id:
        try:
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
# ✅ ASYNC UTILITY FONKSİYONLARI - YENİ
# ==========================================================

async def validate_output_async(output: Dict[str, Any]) -> bool:
    """Async output validation - AnalysisUtilities ile uyumlu"""
    return utility_functions.validate_output(output)

async def create_fallback_output_async(module_name: str, reason: str = "Error") -> Dict[str, Any]:
    """Async fallback output - AnalysisUtilities ile uyumlu"""
    return utility_functions.create_fallback_output(module_name, reason)

async def normalize_output_async(raw_output: Dict[str, Any], module_name: str, symbol: str) -> Dict[str, Any]:
    """Async output normalization - AnalysisOutput schema ile uyumlu"""
    try:
        # AnalysisOutput schema validation
        validated_output = AnalysisOutput(
            **raw_output,
            module=module_name,
            timestamp=asyncio.get_event_loop().time()
        )
        return validated_output.dict()
    except ValidationError as e:
        logger.warning(f"Output validation failed for {module_name}: {e}")
        fallback = await create_fallback_output_async(module_name, f"Validation error: {str(e)}")
        fallback["symbol"] = symbol
        return fallback

# ==========================================================
# 🔄 ASYNC MODÜL YÖNETİCİSİ - YENİ
# ==========================================================

class AsyncModuleManager:
    """TAM ASYNC modül yöneticisi - BaseModule ve Core ile uyumlu"""
    
    def __init__(self):
        self.aggregator: Optional[AnalysisAggregator] = None
        self._initialized = False
    
    async def initialize(self):
        """Async initializer"""
        if self._initialized:
            return
            
        self.aggregator = await get_aggregator()
        self._initialized = True
        logger.info("✅ AsyncModuleManager initialized successfully")
    
    async def get_module_analysis(
        self, 
        module_name: str, 
        symbol: str, 
        credentials: Dict[str, Any],
        priority: Optional[str] = None
    ) -> AnalysisResponse:
        """TAM ASYNC modül analizi - BaseModule ile uyumlu"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # ✅ AnalysisCore ile tam entegre async analiz
            raw_result = await self.aggregator.get_module_analysis(module_name, symbol)
            
            # ✅ Output normalization
            normalized_result = await normalize_output_async(raw_result, module_name, symbol)
            
            # ✅ Credentials entegrasyonu
            normalized_result["api_source"] = credentials.get("source", "unknown")
            normalized_result["user_id"] = credentials.get("user_id")
            
            # ✅ Response validation
            return AnalysisResponse(**normalized_result)
            
        except Exception as e:
            logger.error(f"Module analysis failed for {module_name}: {e}")
            fallback_data = await create_fallback_output_async(module_name, str(e))
            fallback_data["api_source"] = credentials.get("source", "unknown")
            fallback_data["user_id"] = credentials.get("user_id")
            fallback_data["symbol"] = symbol
            return AnalysisResponse(**fallback_data)
    
    async def get_comprehensive_analysis(
        self, 
        symbol: str, 
        credentials: Dict[str, Any]
    ) -> MultiAnalysisResponse:
        """TAM ASYNC kapsamlı analiz - AnalysisCore ile uyumlu"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # ✅ AnalysisCore comprehensive analysis
            comprehensive_result = await self.aggregator.get_comprehensive_analysis(symbol)
            
            # ✅ Format results
            formatted_results = {}
            for module_result in comprehensive_result.get('module_analyses', {}).get('results', []):
                if isinstance(module_result, AnalysisResult) and module_result.status == AnalysisStatus.COMPLETED:
                    module_data = module_result.data
                    if isinstance(module_data, dict):
                        module_data["api_source"] = credentials.get("source", "unknown")
                        module_data["user_id"] = credentials.get("user_id")
                        formatted_results[module_result.module_name] = AnalysisResponse(**module_data)
            
            return MultiAnalysisResponse(
                symbol=symbol,
                results=formatted_results,
                composite_scores=comprehensive_result.get('composite_scores', {}),
                summary=comprehensive_result.get('summary', {}),
                user_id=credentials.get("user_id"),
                timestamp=asyncio.get_event_loop().time()
            )
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed for {symbol}: {e}")
            return MultiAnalysisResponse(
                symbol=symbol,
                results={},
                composite_scores={},
                summary={"error": str(e)},
                user_id=credentials.get("user_id"),
                timestamp=asyncio.get_event_loop().time()
            )

# Global instance
module_manager = AsyncModuleManager()

# ==========================================================
# 🎯 ASYNC ENDPOINT'LER - YENİ YAPILANDIRMA
# ==========================================================

@router.get("/health")
async def health_check():
    """✅ TAM ASYNC sistem sağlık kontrolü - AnalysisCore ile uyumlu"""
    try:
        aggregator = await get_aggregator()
        health_status, health_checks = await aggregator.comprehensive_health_check()
        
        return {
            "status": "healthy" if health_status else "degraded",
            "timestamp": asyncio.get_event_loop().time(),
            "async": True,
            "health_checks": health_checks,
            "aggregator_initialized": aggregator._is_running,
            "module_count": len(aggregator.schema.modules) if aggregator.schema else 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": asyncio.get_event_loop().time(),
            "error": str(e),
            "async": True
        }

@router.get("/modules")
async def list_modules():
    """✅ TAM ASYNC modül listesi - AnalysisCore ile uyumlu"""
    try:
        aggregator = await get_aggregator()
        
        if not aggregator.schema:
            return {
                "modules": [],
                "count": 0,
                "timestamp": asyncio.get_event_loop().time(),
                "error": "Schema not loaded"
            }
        
        return {
            "modules": [
                {
                    "name": module.name,
                    "command": module.command,
                    "objective": module.objective,
                    "api_type": module.api_type,
                    "async": True,
                    "file": module.file
                }
                for module in aggregator.schema.modules
            ],
            "count": len(aggregator.schema.modules),
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"Module list failed: {e}")
        return {
            "modules": [],
            "count": 0,
            "timestamp": asyncio.get_event_loop().time(),
            "error": str(e)
        }

@router.get("/analysis/{module_name}")
async def run_module_analysis(
    module_name: str,
    symbol: str = Query(..., description="Trading symbol (e.g., BTCUSDT)"),
    priority: Optional[str] = Query(None, description="Analysis priority: basic, pro, expert"),
    credentials: Dict[str, Any] = Depends(get_user_credentials)
):
    """✅ TAM ASYNC tek modül analizi - BaseModule ile tam uyumlu"""
    logger.info(f"🔍 Starting ASYNC analysis: {module_name} for {symbol}, user: {credentials.get('user_id')}")
    
    try:
        result = await module_manager.get_module_analysis(
            module_name=module_name,
            symbol=symbol,
            credentials=credentials,
            priority=priority
        )
        
        logger.info(f"✅ ASYNC Analysis completed: {module_name} for {symbol}, score: {result.score}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ASYNC Analysis error in {module_name} for {symbol}: {str(e)}")
        fallback_data = await create_fallback_output_async(module_name, f"Analysis error: {str(e)}")
        fallback_data["api_source"] = credentials.get("source", "unknown")
        fallback_data["user_id"] = credentials.get("user_id")
        fallback_data["symbol"] = symbol
        return AnalysisResponse(**fallback_data)

@router.get("/analysis/comprehensive")
async def run_comprehensive_analysis(
    symbol: str = Query(..., description="Trading symbol (e.g., BTCUSDT)"),
    credentials: Dict[str, Any] = Depends(get_user_credentials)
):
    """✅ TAM ASYNC kapsamlı analiz - Tüm modüller + composite scores"""
    logger.info(f"🔍 Starting COMPREHENSIVE ASYNC analysis for {symbol}, user: {credentials.get('user_id')}")
    
    try:
        result = await module_manager.get_comprehensive_analysis(symbol, credentials)
        
        logger.info(f"✅ COMPREHENSIVE ASYNC Analysis completed for {symbol}, modules: {len(result.results)}")
        return result
        
    except Exception as e:
        logger.error(f"❌ COMPREHENSIVE ASYNC Analysis error for {symbol}: {str(e)}")
        return MultiAnalysisResponse(
            symbol=symbol,
            results={},
            composite_scores={},
            summary={"error": str(e)},
            user_id=credentials.get("user_id"),
            timestamp=asyncio.get_event_loop().time()
        )

@router.get("/trend/{symbol}")
async def get_trend_strength(
    symbol: str,
    credentials: Dict[str, Any] = Depends(get_user_credentials)
):
    """✅ TAM ASYNC trend strength analizi - AnalysisCore composite ile uyumlu"""
    try:
        aggregator = await get_aggregator()
        trend_result = await aggregator.get_trend_strength(symbol)
        
        # Format response
        response_data = {
            "symbol": symbol,
            "score": trend_result.get('score', 0.5),
            "signal": "bullish" if trend_result.get('score', 0.5) > 0.6 else "bearish" if trend_result.get('score', 0.5) < 0.4 else "neutral",
            "confidence": trend_result.get('confidence', 0.0),
            "components": trend_result.get('components', {}),
            "explain": trend_result.get('explanation', 'Trend analysis'),
            "module": "trend_strength",
            "api_source": credentials.get("source", "unknown"),
            "user_id": credentials.get("user_id"),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        return AnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Trend analysis failed for {symbol}: {e}")
        fallback_data = await create_fallback_output_async("trend_strength", str(e))
        fallback_data["api_source"] = credentials.get("source", "unknown")
        fallback_data["user_id"] = credentials.get("user_id")
        fallback_data["symbol"] = symbol
        return AnalysisResponse(**fallback_data)

# ==========================================================
# 🔄 MULTI-USER ENDPOINT'LER - YENİ
# ==========================================================

@router.get("/user/analytics")
async def get_user_analytics(
    credentials: Dict[str, Any] = Depends(get_user_credentials)
):
    """✅ TAM ASYNC kullanıcı analitikleri - Multi-user support"""
    user_id = credentials.get("user_id")
    
    try:
        aggregator = await get_aggregator()
        performance_metrics = aggregator.get_performance_metrics()
        
        return {
            "user_id": user_id,
            "api_source": credentials.get("source", "unknown"),
            "performance": {
                "total_executions": performance_metrics.get("total_executions", 0),
                "average_execution_time": performance_metrics.get("average_execution_time", 0),
                "success_rate": performance_metrics.get("success_rate", 0),
                "cache_hits": getattr(aggregator, '_cache_hits', 0),
                "cache_misses": getattr(aggregator, '_cache_misses', 0)
            },
            "available_modules": len(aggregator.schema.modules) if aggregator.schema else 0,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"User analytics failed for {user_id}: {e}")
        return {
            "user_id": user_id,
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }

@router.get("/batch-analysis")
async def run_batch_analysis(
    symbols: str = Query(..., description="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)"),
    modules: str = Query(..., description="Comma-separated module names"),
    credentials: Dict[str, Any] = Depends(get_user_credentials),
    background_tasks: BackgroundTasks = None
):
    """✅ TAM ASYNC batch analiz - Multi-user, multi-symbol support"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        module_list = [m.strip() for m in modules.split(',')]
        user_id = credentials.get("user_id")
        
        logger.info(f"🔍 Starting BATCH ASYNC analysis for {len(symbol_list)} symbols, {len(module_list)} modules, user: {user_id}")
        
        results = {}
        
        # ✅ ASYNC paralel işleme
        for symbol in symbol_list:
            symbol_results = {}
            tasks = []
            
            for module_name in module_list:
                task = module_manager.get_module_analysis(
                    module_name=module_name,
                    symbol=symbol,
                    credentials=credentials
                )
                tasks.append((module_name, task))
            
            # Tüm modülleri paralel çalıştır
            for module_name, task in tasks:
                try:
                    result = await task
                    symbol_results[module_name] = result
                except Exception as e:
                    logger.error(f"Batch analysis failed for {module_name} on {symbol}: {e}")
                    fallback_data = await create_fallback_output_async(module_name, str(e))
                    fallback_data["api_source"] = credentials.get("source", "unknown")
                    fallback_data["user_id"] = user_id
                    fallback_data["symbol"] = symbol
                    symbol_results[module_name] = AnalysisResponse(**fallback_data)
            
            results[symbol] = symbol_results
        
        logger.info(f"✅ BATCH ASYNC Analysis completed for {len(symbol_list)} symbols")
        return {
            "user_id": user_id,
            "api_source": credentials.get("source", "unknown"),
            "results": results,
            "summary": {
                "total_symbols": len(symbol_list),
                "total_modules": len(module_list),
                "successful_analyses": sum(len(symbol_results) for symbol_results in results.values())
            },
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return {
            "error": str(e),
            "user_id": credentials.get("user_id"),
            "timestamp": asyncio.get_event_loop().time()
        }

# ==========================================================
# 🚀 STARTUP EVENT - GÜNCELLENMİŞ
# ==========================================================

@router.on_event("startup")
async def startup_event():
    """✅ TAM ASYNC startup event - Tüm bileşenlerle uyumlu"""
    try:
        logger.info("🚀 ASYNC Initializing Analysis System...")
        
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
        
        # ✅ ASYNC ANALYSIS AGGREGATOR INITIALIZATION
        aggregator = await get_aggregator()
        logger.info("✅ ASYNC AnalysisAggregator initialized successfully")
        
        # ✅ ASYNC MODULE MANAGER INITIALIZATION
        await module_manager.initialize()
        logger.info("✅ ASYNC ModuleManager initialized successfully")
        
        # ✅ SCHEMA VALIDATION
        if aggregator.schema:
            logger.info(f"✅ ASYNC Schema loaded: {len(aggregator.schema.modules)} modules")
        else:
            logger.warning("⚠️ ASYNC Schema not loaded - some features may be limited")
        
        logger.info("🚀 ASYNC Analysis System startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ ASYNC Startup error: {e}")

# ==========================================================
# 🏠 ROOT ENDPOINT - GÜNCELLENMİŞ
# ==========================================================

@router.get("/")
async def root():
    """✅ TEK ASYNC ana sayfa - Tüm endpoint'lerle uyumlu"""
    try:
        aggregator = await get_aggregator()
        module_count = len(aggregator.schema.modules) if aggregator.schema else 0
        
        endpoints = [
            {"path": "/health", "methods": ["GET"], "name": "System Health Check", "async": True},
            {"path": "/modules", "methods": ["GET"], "name": "List Available Modules", "async": True},
            {"path": "/analysis/{module_name}", "methods": ["GET"], "name": "Single Module Analysis", "async": True},
            {"path": "/analysis/comprehensive", "methods": ["GET"], "name": "Comprehensive Analysis", "async": True},
            {"path": "/trend/{symbol}", "methods": ["GET"], "name": "Trend Strength Analysis", "async": True},
            {"path": "/user/analytics", "methods": ["GET"], "name": "User Analytics", "async": True},
            {"path": "/batch-analysis", "methods": ["GET"], "name": "Batch Analysis", "async": True}
        ]
        
        return {
            "message": "ASYNC Analysis API Service - Multi User Support",
            "version": "2.0.0",
            "timestamp": asyncio.get_event_loop().time(),
            "async": True,
            "modules_available": module_count,
            "aggregator_status": "running" if aggregator._is_running else "stopped",
            "multi_user_support": True,
            "endpoints": endpoints
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return {
            "message": "ASYNC Analysis API Service",
            "version": "2.0.0", 
            "timestamp": asyncio.get_event_loop().time(),
            "async": True,
            "error": "Initialization in progress"
        }