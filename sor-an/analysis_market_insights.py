# analysis/analysis_market_insights.py
# =================================================
# dosya iki ana sınıf içerir
# - MarketMetricsBuilder
# - MarketScanner
#
# Usage:
# from analysis.analysis_market_insights import MarketMetricsBuilder, MarketScanner
# builder = MarketMetricsBuilder(aggregator)
# scanner = MarketScanner(aggregator)
# kripto para piyasası hakkında genel istatistiksel analizler ve içgörüler üreten yardımcı bir modüldür.
"""
analiz sonuçlarını işleyip piyasanın genel durumu, altcoin performansı ve risk iştahı gibi göstergeleri hesaplamak ve raporlamak

| Gösterge                               | Açıklama                                       | Ne Ölçer                                                                      |
| -------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------- |
| **MMI (Market Momentum Index)**        | BTC ve ETH trend skorlarının ortalaması        | Genel piyasa yönü (boğa/ayı)                                                  |
| **ADI (Altcoin Dominance Index)**      | Altcoin ortalama skorları – MMI farkı          | Altcoin'lerin BTC/ETH'ye göre güçlü/zayıf olma durumu                         |
| **SSP (Stablecoin Sentiment Premium)** | Stablecoin'lerin likidite skoruna dayalı ölçüm | Piyasadaki risk iştahı (para stablecoinlerde mi yoksa riskli varlıklarda mı?) |

| Bileşen                    | Amacı                                                    |
| -------------------------- | -------------------------------------------------------- |
| **MarketMetricsBuilder**   | Piyasa genelini ölçen metrikler üretir (MMI, ADI, SSP)   |
| **MarketScanner**          | Belirli skor/metriklere göre sembolleri tarar ve sıralar |
| **create_market_report()** | Hızlı piyasa raporu üretimi sağlar                       |
| **scan_top_performers()**  | En iyi performanslı coinleri listeler                    |


"""


import logging
from statistics import mean
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Analysis helpers import
try:
    from analysis.analysis_helpers import AnalysisHelpers, AnalysisOutput
except ImportError:
    # Fallback için basit bir AnalysisOutput sınıfı
    from pydantic import BaseModel
    class AnalysisOutput(BaseModel):
        score: float = 0.5
        signal: str = "neutral"
        confidence: float = 0.0
        components: Dict[str, float] = {}
        explain: str = ""
        timestamp: float = 0.0
        module: str = ""
    
    class AnalysisHelpers:
        @staticmethod
        def normalize_score(score: float) -> float:
            return max(0.0, min(1.0, score))

logger = logging.getLogger(__name__)

class MarketMetricsBuilder:
    """✅ TAM ASYNC market metrics builder"""
    
    def __init__(self, aggregator):
        self.agg = aggregator
        self.helpers = AnalysisHelpers()
    
    async def _extract_score_async(self, data: Any, fallback: float = 0.5) -> float:
        """✅ ASYNC skor çıkarma"""
        try:
            loop = asyncio.get_event_loop()
            
            def sync_extract():
                if isinstance(data, AnalysisOutput):
                    return self.helpers.normalize_score(data.score)
                elif isinstance(data, dict):
                    composite_scores = data.get("composite_scores", {})
                    if composite_scores:
                        trend_data = composite_scores.get("trend_strength", {})
                        if isinstance(trend_data, dict) and "score" in trend_data:
                            return self.helpers.normalize_score(trend_data["score"])
                    
                    for key in ["trend_strength", "score", "market_health", "liquidity_score"]:
                        if key in data and isinstance(data[key], (int, float)):
                            return self.helpers.normalize_score(data[key])
                    
                    analysis_data = data.get("analysis", {})
                    if isinstance(analysis_data, dict) and "score" in analysis_data:
                        return self.helpers.normalize_score(analysis_data["score"])
                
                return self.helpers.normalize_score(fallback)
            
            return await loop.run_in_executor(None, sync_extract)
            
        except Exception as e:
            logger.warning(f"Async score extraction failed: {e}")
            return await self._normalize_score_async(fallback)
    
    async def _extract_liquidity_score_async(self, data: Any) -> float:
        """✅ ASYNC likidite skoru çıkarma"""
        try:
            loop = asyncio.get_event_loop()
            
            def sync_extract():
                if isinstance(data, AnalysisOutput):
                    return self.helpers.normalize_score(data.score)
                elif isinstance(data, dict):
                    composite_scores = data.get("composite_scores", {})
                    liquidity_data = composite_scores.get("liquidity_pressure", {})
                    if isinstance(liquidity_data, dict) and "score" in liquidity_data:
                        return self.helpers.normalize_score(liquidity_data["score"])
                    
                    for key in ["liquidity_score", "liquidity_pressure", "market_health", "score"]:
                        if key in data and isinstance(data[key], (int, float)):
                            return self.helpers.normalize_score(data[key])
                
                return 0.5
            
            return await loop.run_in_executor(None, sync_extract)
            
        except Exception as e:
            logger.warning(f"Async liquidity extraction failed: {e}")
            return 0.5
    
    async def _normalize_score_async(self, score: float) -> float:
        """✅ ASYNC skor normalizasyonu"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.helpers.normalize_score(score))
    
    async def get_mmi(self) -> float:
        """✅ ASYNC Market Momentum Index"""
        try:
            # ✅ PARALEL ASYNC CALLS
            btc_task = asyncio.create_task(self.agg.get_comprehensive_analysis("BTCUSDT"))
            eth_task = asyncio.create_task(self.agg.get_comprehensive_analysis("ETHUSDT"))
            
            btc_data, eth_data = await asyncio.gather(btc_task, eth_task)
            
            btc_score_task = asyncio.create_task(self._extract_score_async(btc_data))
            eth_score_task = asyncio.create_task(self._extract_score_async(eth_data))
            
            btc_score, eth_score = await asyncio.gather(btc_score_task, eth_score_task)
            
            mmi = (btc_score + eth_score) / 2.0
            return round(await self._normalize_score_async(mmi), 4)
            
        except Exception as e:
            logger.error(f"Async MMI calculation failed: {e}")
            return 0.5

    async def get_adi(self, altcoins: List[str] = None) -> float:
        """✅ ASYNC Altcoin Dominance Index"""
        try:
            if altcoins is None:
                altcoins = ["SOLUSDT", "BNBUSDT", "MATICUSDT", "AVAXUSDT", 
                           "ADAUSDT", "OPUSDT", "DOGEUSDT"]
            
            mmi_task = asyncio.create_task(self.get_mmi())
            
            # ✅ ASYNC PARALEL ALTCOIN ANALİZLERİ
            altcoin_tasks = []
            for symbol in altcoins:
                task = asyncio.create_task(
                    self._process_altcoin_async(symbol)
                )
                altcoin_tasks.append(task)
            
            mmi, altcoin_results = await asyncio.gather(
                mmi_task,
                asyncio.gather(*altcoin_tasks, return_exceptions=True)
            )
            
            alt_scores = [score for score in altcoin_results if isinstance(score, (int, float))]
            
            if not alt_scores:
                return 0.0
            
            alt_avg = mean(alt_scores)
            adi = alt_avg - mmi
            return round(adi, 4)
            
        except Exception as e:
            logger.error(f"Async ADI calculation failed: {e}")
            return 0.0
    
    async def _process_altcoin_async(self, symbol: str) -> float:
        """✅ ASYNC altcoin işleme"""
        try:
            data = await self.agg.get_comprehensive_analysis(symbol)
            return await self._extract_score_async(data)
        except Exception as e:
            logger.debug(f"Async altcoin processing failed for {symbol}: {e}")
            return 0.5

    async def get_ssp(self, stables: List[str] = None) -> float:
        """✅ ASYNC Stablecoin Sentiment Premium"""
        try:
            if stables is None:
                stables = ["USDT", "FDUSD", "USDC"]
            
            # ✅ ASYNC PARALEL STABLECOIN ANALİZLERİ
            stable_tasks = []
            for stable in stables:
                task = asyncio.create_task(
                    self._process_stablecoin_async(stable)
                )
                stable_tasks.append(task)
            
            stable_results = await asyncio.gather(*stable_tasks, return_exceptions=True)
            stable_scores = [score for score in stable_results if isinstance(score, (int, float))]
            
            if not stable_scores:
                return 0.0
            
            avg_score = mean(stable_scores)
            ssp = (0.5 - avg_score) * 200
            return round(ssp, 2)
            
        except Exception as e:
            logger.error(f"Async SSP calculation failed: {e}")
            return 0.0
    
    async def _process_stablecoin_async(self, stable: str) -> float:
        """✅ ASYNC stablecoin işleme"""
        try:
            data = await self.agg.get_comprehensive_analysis(stable)
            return await self._extract_liquidity_score_async(data)
        except Exception as e:
            logger.debug(f"Async stablecoin processing failed for {stable}: {e}")
            return 0.5

    async def build_report(self) -> Dict[str, Any]:
        """✅ TAM ASYNC piyasa raporu"""
        try:
            # ✅ PARALEL METRIC HESAPLAMALARI
            mmi_task = asyncio.create_task(self.get_mmi())
            adi_task = asyncio.create_task(self.get_adi())
            ssp_task = asyncio.create_task(self.get_ssp())
            
            mmi, adi, ssp = await asyncio.gather(mmi_task, adi_task, ssp_task)
            
            # ✅ ASYNC TREND ANALİZİ
            trend_status = await self._analyze_trend_async(mmi)
            altcoin_status = await self._analyze_altcoins_async(adi)
            risk_appetite = await self._analyze_risk_async(ssp)
            market_health = await self._calculate_market_health_async(mmi, abs(adi), abs(ssp/100))
            
            summary = (f"Piyasa {trend_status} eğiliminde (MMI: {mmi}). "
                      f"{altcoin_status} (ADI: {adi}). "
                      f"{risk_appetite} (SSP: {ssp}).")
            
            return {
                "mmi": mmi,
                "adi": adi, 
                "ssp": ssp,
                "summary": summary,
                "timestamp": await self._get_iso_timestamp_async(),
                "trend_status": trend_status,
                "altcoin_status": altcoin_status,
                "risk_appetite": risk_appetite,
                "market_health": market_health
            }
            
        except Exception as e:
            logger.error(f"Async market report build failed: {e}")
            return await self._get_fallback_report_async()
    
    async def _analyze_trend_async(self, mmi: float) -> str:
        """✅ ASYNC trend analizi"""
        if mmi > 0.7:
            return "güçlü boğa"
        elif mmi > 0.6:
            return "boğa"
        elif mmi < 0.3:
            return "güçlü ayı"
        elif mmi < 0.4:
            return "ayı"
        else:
            return "kararsız"
    
    async def _analyze_altcoins_async(self, adi: float) -> str:
        """✅ ASYNC altcoin analizi"""
        if adi > 0.1:
            return "Altcoinler çok güçlü"
        elif adi > 0.05:
            return "Altcoinler önde"
        elif adi < -0.1:
            return "Altcoinler çok zayıf"
        elif adi < -0.05:
            return "Altcoinler geride"
        else:
            return "Altcoinler dengeli"
    
    async def _analyze_risk_async(self, ssp: float) -> str:
        """✅ ASYNC risk analizi"""
        if ssp < -20:
            return "Çok güçlü risk iştahı"
        elif ssp < -10:
            return "Güçlü risk iştahı"
        elif ssp > 20:
            return "Güçlü riskten kaçış"
        elif ssp > 10:
            return "Riskten kaçış"
        else:
            return "Dengeli risk iştahı"
    
    async def _calculate_market_health_async(self, mmi: float, altcoin_volatility: float, risk_volatility: float) -> str:
        """✅ ASYNC piyasa sağlığı hesaplama"""
        loop = asyncio.get_event_loop()
        
        def sync_calculate():
            stability_score = 1.0 - (altcoin_volatility + risk_volatility) / 2.0
            overall_health = (mmi + stability_score) / 2.0
            
            if overall_health > 0.7:
                return "excellent"
            elif overall_health > 0.6:
                return "good"
            elif overall_health > 0.5:
                return "moderate"
            elif overall_health > 0.4:
                return "weak"
            else:
                return "poor"
        
        return await loop.run_in_executor(None, sync_calculate)
    
    async def _get_iso_timestamp_async(self) -> str:
        """✅ ASYNC timestamp alma"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: datetime.now().isoformat())
    
    async def _get_fallback_report_async(self) -> Dict[str, Any]:
        """✅ ASYNC fallback rapor"""
        return {
            "mmi": 0.5,
            "adi": 0.0,
            "ssp": 0.0,
            "summary": "Piyasa verileri geçici olarak kullanılamıyor",
            "timestamp": await self._get_iso_timestamp_async(),
            "trend_status": "belirsiz",
            "altcoin_status": "belirsiz", 
            "risk_appetite": "belirsiz",
            "market_health": "unknown",
            "fallback": True
        }
        

class MarketScanner:
    """✅ TAM ASYNC market scanner"""
    
    def __init__(self, aggregator):
        self.agg = aggregator
        self.helpers = AnalysisHelpers()
    
    async def _extract_analysis_score_async(self, data: Any) -> float:
        """✅ ASYNC analiz skoru çıkarma"""
        try:
            loop = asyncio.get_event_loop()
            
            def sync_extract():
                if isinstance(data, AnalysisOutput):
                    return self.helpers.normalize_score(data.score)
                elif isinstance(data, dict):
                    return self.helpers.normalize_score(data.get("score", 0.5))
                else:
                    return 0.5
            
            return await loop.run_in_executor(None, sync_extract)
        except Exception:
            return 0.5

    async def scan_module(self, module_name: str, symbols: List[str]) -> List[Tuple[str, float]]:
        """✅ ASYNC modül tarama"""
        # ✅ PARALEL SEMBOL ANALİZLERİ
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                self._process_symbol_module_async(symbol, module_name)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Geçerli sonuçları filtrele
        valid_results = []
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                valid_results.append(result)
        
        # Skora göre sırala
        valid_results.sort(key=lambda x: x[1], reverse=True)
        return valid_results
    
    async def _process_symbol_module_async(self, symbol: str, module_name: str) -> Tuple[str, float]:
        """✅ ASYNC sembol modül işleme"""
        try:
            analysis_data = await self.agg.get_module_analysis(module_name, symbol)
            score = await self._extract_analysis_score_async(analysis_data)
            return (symbol, score)
        except Exception as e:
            logger.debug(f"Async module scan failed for {symbol}.{module_name}: {e}")
            return (symbol, 0.5)

    async def scan_composite(self, score_name: str, symbols: List[str]) -> List[Tuple[str, float]]:
        """✅ ASYNC bileşik skor tarama"""
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                self._process_symbol_composite_async(symbol, score_name)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                valid_results.append(result)
        
        valid_results.sort(key=lambda x: x[1], reverse=True)
        return valid_results
    
    async def _process_symbol_composite_async(self, symbol: str, score_name: str) -> Tuple[str, float]:
        """✅ ASYNC sembol bileşik skor işleme"""
        try:
            if hasattr(self.agg, 'composite_engine'):
                composite_data = await self.agg.composite_engine.calculate_single_score(score_name, symbol)
            else:
                comprehensive_data = await self.agg.get_comprehensive_analysis(symbol)
                composite_scores = comprehensive_data.get("composite_scores", {}) if isinstance(comprehensive_data, dict) else {}
                composite_data = composite_scores.get(score_name, {})
            
            score = await self._extract_analysis_score_async(composite_data)
            return (symbol, score)
        except Exception as e:
            logger.debug(f"Async composite scan failed for {symbol}.{score_name}: {e}")
            return (symbol, 0.5)


    async def top(self, score_name: str, symbols: List[str], n: int = 5) -> List[Tuple[str, float]]:
        """En yüksek skora sahip sembolleri bul"""
        try:
            scanned_data = await self.scan_composite(score_name, symbols)
            return scanned_data[:n]
        except Exception as e:
            logger.error(f"Top scan failed for {score_name}: {e}")
            return []

    async def bottom(self, score_name: str, symbols: List[str], n: int = 5) -> List[Tuple[str, float]]:
        """En düşük skora sahip sembolleri bul"""
        try:
            scanned_data = await self.scan_composite(score_name, symbols)
            return scanned_data[-n:][::-1]  # Düşükten yükseğe sırala
        except Exception as e:
            logger.error(f"Bottom scan failed for {score_name}: {e}")
            return []


    async def get_symbol_rank(self, symbol: str, score_name: str, comparison_symbols: List[str]) -> Dict[str, Any]:
        """✅ TAM ASYNC sembol sıralama hesaplama"""
        try:
            all_data = await self.scan_composite(score_name, comparison_symbols)
            
            # Sembolün sırasını bul
            symbol_rank = None
            symbol_score = 0.0
            
            for rank, (sym, score) in enumerate(all_data, 1):
                if sym == symbol:
                    symbol_rank = rank
                    symbol_score = score
                    break
            
            if symbol_rank is None:
                # Sembol listede yoksa, skorunu async çıkar
                symbol_data = await self.agg.get_comprehensive_analysis(symbol)
                composite_scores = symbol_data.get("composite_scores", {}) if isinstance(symbol_data, dict) else {}
                target_score_data = composite_scores.get(score_name, {})
                symbol_score = await self._extract_analysis_score_async(target_score_data)
                
                # Yeni sıralama hesapla (CPU-bound ama hafif)
                better_count = sum(1 for _, score in all_data if score > symbol_score)
                symbol_rank = better_count + 1
            
            total_symbols = len(comparison_symbols)
            percentile = (total_symbols - symbol_rank) / total_symbols * 100 if total_symbols > 0 else 0
            
            return {
                "symbol": symbol,
                "score_name": score_name,
                "rank": symbol_rank,
                "total_symbols": total_symbols,
                "percentile": round(percentile, 1),
                "score": symbol_score,
                "performance": "top" if percentile >= 80 else "bottom" if percentile <= 20 else "middle"
            }
            
        except Exception as e:
            logger.error(f"Symbol rank calculation failed for {symbol}.{score_name}: {e}")
            return {
                "symbol": symbol,
                "score_name": score_name,
                "rank": None,
                "total_symbols": 0,
                "percentile": 0,
                "score": 0,
                "performance": "unknown",
                "error": str(e)
            }

    async def scan_multiple_scores(self, score_names: List[str], symbols: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """✅ TAM ASYNC birden fazla skor tarama"""
        try:
            # Hepsini aynı anda çalıştır
            tasks = [
                asyncio.create_task(self.scan_composite(score_name, symbols))
                for score_name in score_names
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Sonuçları eşleştir
            return {
                score_name: (res if isinstance(res, list) else [])
                for score_name, res in zip(score_names, results)
            }
        except Exception as e:
            logger.error(f"Multiple scores scan failed: {e}")
            return {name: [] for name in score_names}


# Convenience functions
async def create_market_report(aggregator) -> Dict[str, Any]:
    """Hızlı piyasa raporu oluşturma fonksiyonu"""
    builder = MarketMetricsBuilder(aggregator)
    return await builder.build_report()

async def scan_top_performers(aggregator, score_name: str, symbols: List[str], n: int = 10) -> List[Tuple[str, float]]:
    """En iyi performans gösteren sembolleri bul"""
    scanner = MarketScanner(aggregator)
    return await scanner.top(score_name, symbols, n)