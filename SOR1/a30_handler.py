# handlers/a30_handler.py
"""
OPTIMIZED COMMAND HANDLER - SADECE GERÇEK VERİ
"""

import logging
import math
from typing import Dict, List, Any
from aiogram import Router, types
from analysis.analysis_core import AnalysisCore

logger = logging.getLogger(__name__)
router = Router(name="command_router")

# ✅ TÜM KOMUTLAR TEK YERDE
COMMANDS = {
    "/t": {
        "scores": [
            "trend_momentum_composite",
            "volatility_composite", 
            "regime_composite",
            "risk_composite",
            "core_macro"
        ],
        "columns": ["Trend", "Vol", "Rejim", "Risk", "Toplam"]
    },
    "/tt": {
        "scores": ["trend_momentum_composite"],
        "columns": ["Trend"]},
    "/tv": {
        "scores": ["volatility_composite"],
        "columns": ["Vol"]},
    "/tre": {
        "scores": ["regime_composite"],
        "columns": ["Rejim"]},
    "/tr": {
        "scores": ["risk_composite"],
        "columns": ["Risk"]},
    "/tcc": {
        "scores": ["core_macro"],
        "columns": ["Toplam"]},   
    "/ts": {
        "scores": [
            "sentiment_composite",
            "flow_dynamics_composite", 
            "market_sentiment_macro"
        ],
        "columns": ["Sentiment", "Flow", "Toplam"]
    }
}

class SimpleCommandHandler:
    """Optimized command handler - SADECE GERÇEK VERİ"""
    
    def __init__(self):
        self.analysis_core = AnalysisCore()
        self.commands = COMMANDS
        
        # ✅ DEFAULT TAKİP LİSTESİ - YENİ
        self.default_watchlist = [
            "BTCUSDT", "BNBUSDT", "SOLUSDT", "CAKEUSDT", "PEPEUSDT", "ARPAUSDT"
        ]
        
        logger.info("✅ Command Handler initialized - SADECE GERÇEK VERİ")
    
    async def handle(self, text: str) -> Dict[str, Any]:
        """Tüm komutları işle - SADECE GERÇEK VERİ"""
        parts = text.strip().split()
        if not parts or parts[0] not in self.commands:
            return None
            
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        logger.info(f"🔄 Processing: {cmd}, args: {args}")
        
        try:
            # Sembolleri belirle
            symbols = await self._get_symbols(args)
            if not symbols:
                return {"error": "Geçersiz sembol veya argüman"}
            
            # ✅ HER SEMBOL İÇİN AYRI HESAPLAMA - SADECE GERÇEK VERİ
            symbol_scores = {}
            all_calculated_scores = set()
            failed_symbols = []
            
            for symbol in symbols:
                result = await self.analysis_core.analyze_symbol(
                    symbol=symbol,
                    required_composites=self.commands[cmd]["scores"]
                )
                
                if result and "error" not in result:
                    composites = result.get("composites", {})
                    scores = self._extract_scores(composites, cmd, symbol)
                    
                    # ✅ SADECE GERÇEK VERİ KONTROLÜ: Tüm değerler NaN ise başarısız say
                    if self._has_real_data(scores):
                        symbol_scores[symbol] = scores
                        all_calculated_scores.update(composites.keys())
                        logger.info(f"✅ {symbol} - GERÇEK VERİ ile hesaplandı")
                    else:
                        failed_symbols.append(symbol)
                        logger.warning(f"❌ {symbol} - HİÇ GERÇEK VERİ YOK, atlandı")
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"❌ {symbol} - Hesaplama başarısız: {result.get('error', 'Bilinmeyen hata')}")
            
            if failed_symbols:
                logger.warning(f"📊 Başarısız semboller: {failed_symbols}")
            
            if not symbol_scores:
                return {"error": "Hiçbir sembol için GERÇEK VERİ bulunamadı"}
            
            return {
                "command": cmd,
                "symbols": list(symbol_scores.keys()),
                "symbol_scores": symbol_scores,
                "columns": self.commands[cmd]["columns"],
                "calculated_scores": len(all_calculated_scores),
                "failed_symbols": failed_symbols
            }
            
        except Exception as e:
            logger.error(f"❌ Command failed: {e}")
            return {"error": f"İşlem hatası: {str(e)}"}

    def _extract_scores(self, composites: Dict, cmd: str, symbol: str) -> Dict[str, float]:
        """Composites'den skorları çıkar - SADECE GERÇEK VERİ"""
        command_config = self.commands[cmd]
        scores = {}
        
        for i, metric in enumerate(command_config["scores"]):
            column_name = command_config["columns"][i] if i < len(command_config["columns"]) else metric
            raw_value = composites.get(metric)
            
            # 🔥 SADECE GERÇEK VERİ - SENTETİK/SIFIR YOK
            if raw_value is None:
                scores[column_name] = float('nan')
                logger.debug(f"📊 {symbol} - {metric}: VERİ YOK (None)")
            elif isinstance(raw_value, float) and math.isnan(raw_value):
                scores[column_name] = float('nan')
                logger.debug(f"📊 {symbol} - {metric}: VERİ YOK (NaN)")
            elif isinstance(raw_value, (int, float)):
                # GERÇEK VERİ - yuvarla ve kaydet
                scores[column_name] = round(raw_value, 3)
                logger.debug(f"📊 {symbol} - {metric}: {raw_value:.3f} (GERÇEK)")
            else:
                # Geçersiz veri tipi
                scores[column_name] = float('nan')
                logger.warning(f"📊 {symbol} - {metric}: GEÇERSİZ VERİ TİPİ {type(raw_value)}")
                
        return scores

    def _has_real_data(self, scores: Dict[str, float]) -> bool:
        """Skorlarda gerçek veri var mı kontrol et"""
        for value in scores.values():
            if isinstance(value, (int, float)) and not math.isnan(value):
                return True
        return False
    
    async def _get_symbols(self, args: List[str]) -> List[str]:
        """Sembol listesini oluştur"""
        if not args:
            return self.default_watchlist
        
        first_arg = args[0].upper()
        
        if first_arg.isdigit():
            count = min(int(first_arg), 20)
            return await self._get_trending_symbols(count)
        else:
            return [self._normalize_symbol(first_arg)]
    
    def _normalize_symbol(self, symbol_input: str) -> str:
        """Sembol normalizasyonu"""
        clean = symbol_input.upper().strip()
        
        if clean.endswith('USDT') or clean.endswith('FDUSD') or clean.endswith('BUSD'):
            return clean
        return f"{clean}USDT"
    
    async def _get_trending_symbols(self, count: int) -> List[str]:
        """Trend sembolleri getir - SADECE GERÇEK VERİ"""
        try:
            aggregator = await self.analysis_core._get_aggregator()
            ticker_data = await aggregator.get_data('futures_ticker_24hr')
            
            if not ticker_data or not isinstance(ticker_data, list):
                logger.warning("❌ Trend verisi alınamadı, fallback kullanılıyor")
                return self.default_watchlist[:count]
            
            # USDT pair'lerini filtrele ve sırala
            usdt_pairs = [
                symbol for symbol in ticker_data 
                if isinstance(symbol, dict) and symbol.get('symbol', '').endswith('USDT')
            ]
            
            if not usdt_pairs:
                logger.warning("❌ USDT pair bulunamadı, fallback kullanılıyor")
                return self.default_watchlist[:count]
            
            # Hacime göre sırala
            sorted_symbols = sorted(
                usdt_pairs, 
                key=lambda x: float(x.get('quoteVolume', 0)), 
                reverse=True
            )
            
            trending_symbols = [symbol['symbol'] for symbol in sorted_symbols[:count]]
            logger.info(f"📈 Gerçek trend sembolleri: {len(trending_symbols)}")
            return trending_symbols
            
        except Exception as e:
            logger.error(f"❌ Trend sembol alımı başarısız: {e}")
            return self.default_watchlist[:count]

# ✅ TEK HANDLER INSTANCE
handler = SimpleCommandHandler()

# ✅ OPTIMIZED FORMAT FONKSİYONU - NaN DOSTU
def format_table_response(result: Dict[str, Any]) -> str:
    """Otomatik tablo oluştur - SADECE GERÇEK VERİ"""
    symbol_scores = result["symbol_scores"]
    columns = result["columns"]
    
    # Header
    header = "Sembol  " + "  ".join([f"{col:10}" for col in columns])
      
    lines = [
        f"📊 <b>{result['command'].upper()}</b> - SADECE GERÇEK VERİ",
        "─" * (10 + len(columns) * 12),
        f"<b>{header}</b>",
        "─" * (10 + len(columns) * 12)
    ]
    
    # Sıralı semboller (NaN'lar en sona)
    sorted_symbols = sorted(
        symbol_scores.items(),
        key=lambda x: float('-inf') if any(math.isnan(v) for v in x[1].values()) else x[1].get("Toplam", 0),
        reverse=True
    )
    
    # Satırlar
    for symbol, scores in sorted_symbols:
        display_symbol = symbol.replace('USDT', '')
        score_cells = []
        
        for col in columns:
            value = scores.get(col, float('nan'))
            if isinstance(value, float) and math.isnan(value):
                score_cells.append(f"❌ ---")  # VERİ YOK
            else:
                score_cells.append(f"{get_icon(col, value)} {value:+.2f}")
        
        line = f"{display_symbol:6}  " + "  ".join(score_cells)
        lines.append(line)
    
    # Özet
    real_symbols = [s for s in symbol_scores.keys() if any(not math.isnan(v) for v in symbol_scores[s].values())]
    best_symbol = "N/A"
    best_score = 0
    
    if real_symbols:
        # En iyi sembolü bul (NaN olmayanlar arasından)
        valid_symbols = {s: scores for s, scores in symbol_scores.items() 
                        if not math.isnan(scores.get("Toplam", float('nan')))}
        if valid_symbols:
            best_symbol_data = max(valid_symbols.items(), key=lambda x: x[1].get("Toplam", 0))
            best_symbol = best_symbol_data[0].replace('USDT', '')
            best_score = best_symbol_data[1].get("Toplam", 0)
    
    failed_count = len(result.get('failed_symbols', []))
    
    lines.extend([
        "─" * (10 + len(columns) * 12),
        f"<b>Özet:</b> {len(real_symbols)}/{len(symbol_scores)} sembol | " +
        f"<b>Başarısız:</b> {failed_count} | " +
        f"<b>En İyi:</b> {best_symbol} ({best_score:+.2f})",
        f"<i>{result['calculated_scores']} gerçek metrik hesaplandı</i>"
    ])
    
    if failed_count > 0:
        lines.append(f"<i>Başarısız semboller: {', '.join([s.replace('USDT', '') for s in result.get('failed_symbols', [])])}</i>")
    
    return "\n".join(lines)

def get_icon(column: str, score: float) -> str:
    """İkon belirle"""
    if math.isnan(score):
        return "❌"  # VERİ YOK
    
    if column == "Risk":
        return "🔴" if score > 0.3 else "🟠" if score > 0.1 else "🟡" if score > -0.1 else "🟢"
    elif column in ["Trend", "Rejim", "Sentiment", "Toplam"]:
        return "🟢" if score > 0.3 else "🟡" if score > 0.1 else "⚪" if score > -0.1 else "🟠" if score > -0.3 else "🔴"
    elif column in ["Vol", "Flow"]:
        return "⚡" if abs(score) > 0.4 else "🔸" if abs(score) > 0.2 else "💤"
    else:
        return "🟢" if score > 0.3 else "🟡" if score > 0.1 else "⚪" if score > -0.1 else "🟠" if score > -0.3 else "🔴"

# ✅ MESSAGE HANDLER
@router.message()
async def handle_all_messages(message: types.Message):
    """Tüm mesajları işle - SADECE GERÇEK VERİ"""
    text = message.text or ""
    
    if not text.startswith('/'):
        return
    
    result = await handler.handle(text)
    
    if result is None:
        await message.answer("❌ Desteklenmeyen komut: /t, /tt, /tv, /tre, /tr, /tcc, /ts")
        return
        
    if "error" in result:
        await message.answer(f"⚠️ {result['error']}")
        return
    
    response = format_table_response(result)
    await message.answer(response, parse_mode="HTML")

# Test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        handler = SimpleCommandHandler()
        
        for test_text in ["/t", "/t bnb", "/t 5", "/ts", "/ts eth"]:
            print(f"\n🔹 Testing: {test_text}")
            result = await handler.handle(test_text)
            if result:
                real_symbols = len([s for s, scores in result['symbol_scores'].items() 
                                  if any(not math.isnan(v) for v in scores.values())])
                print(f"✅ {result['command']} - {real_symbols}/{len(result['symbols'])} gerçek sembol")
                if result.get('failed_symbols'):
                    print(f"❌ Başarısız: {result['failed_symbols']}")
    
    asyncio.run(test())
    
    