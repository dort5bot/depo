# handlers/a31_handler.py
"""
OPTIMIZED COMMAND HANDLER - CORE UYUMLU
core > run_full_pipeline

1. handler.handle("/t BTC") çağrılır
2. parts = ["/t", "BTC"]
3. cmd = "/t", args = ["BTC"]
4. symbols = ["BTCUSDT"] (normalize edilmiş)
5. required_scores = ["trend", "vol", "core"]
6. Her symbol için:
   await run_full_pipeline_async(
       symbol="BTCUSDT",
       requested_scores=["trend", "vol", "core"]
   )
7. CORE'dan Beklenen Yanıt:
    result = await run_full_pipeline_async(...)
   {
    "composites": {
        "trend": 0.45,
        "vol": 0.32,
        "core": 0.78
    },
    "macros": {
        "trend": 0.42,
        "vol": 0.30,
        "core": 0.75
    },
    "timestamp": "...",
    ... diğer veriler
}
"""

import logging
import math
from typing import Dict, List, Any
from aiogram import Router, types
from analysis.a_core import run_full_pipeline_async

logger = logging.getLogger(__name__)
router = Router(name="command_router")

# ✅ TÜM KOMUTLAR - SADECE SCORES LİSTESİ
COMMANDS = { #COMMANDS dict'i sadece scores listesi içeriyor
    "/t": ["trend", "vol", "core"],
    "/tt": ["trend"],
    "/tv": ["vol"], 
    "/tcc": ["core"]
}

class SimpleCommandHandler:
    """Optimized command handler - CORE UYUMLU"""
    
    def __init__(self):
        self.commands = COMMANDS
        
        # ✅ DEFAULT TAKİP LİSTESİ
        self.default_watchlist = [
            "BTCUSDT", "BNBUSDT", "SOLUSDT", "CAKEUSDT", "PEPEUSDT", "ARPAUSDT"
        ]
        
        logger.info("✅ Command Handler initialized - CORE UYUMLU")
    
    async def handle(self, text: str) -> Dict[str, Any]:
        """Tüm komutları işle"""
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
            
            # ✅ HER SEMBOL İÇİN CORE PIPELINE ÇAĞIR
            symbol_scores = {}
            failed_symbols = []
            
            for symbol in symbols:
                result = await self._analyze_symbol(
                    symbol=symbol,
                    required_scores=self.commands[cmd]
                )
                
                if result and "error" not in result:
                    # ✅ CORE ÇIKTISINDAN SCORELARI ÇIKAR
                    scores = self._extract_scores(result, cmd, symbol)
                    
                    # GERÇEK VERİ KONTROLÜ
                    if self._has_real_data(scores):
                        symbol_scores[symbol] = scores
                        logger.info(f"✅ {symbol} - Core ile hesaplandı")
                    else:
                        failed_symbols.append(symbol)
                        logger.warning(f"❌ {symbol} - HİÇ GERÇEK VERİ YOK, atlandı")
                else:
                    failed_symbols.append(symbol)
                    error_msg = result.get("error", "Bilinmeyen hata") if result else "No result"
                    logger.warning(f"❌ {symbol} - Core hesaplama başarısız: {error_msg}")
            
            if failed_symbols:
                logger.warning(f"📊 Başarısız semboller: {failed_symbols}")
            
            if not symbol_scores:
                return {"error": "Hiçbir sembol için GERÇEK VERİ bulunamadı"}
            
            return {
                "command": cmd,
                "symbols": list(symbol_scores.keys()),
                "symbol_scores": symbol_scores,
                "scores": self.commands[cmd],  # ✅ SCORES listesi yeterli
                "failed_symbols": failed_symbols,
            }
            
        except Exception as e:
            logger.error(f"❌ Command failed: {e}")
            return {"error": f"İşlem hatası: {str(e)}"}

    # CORE'u Çağırma Noktası:
    async def _analyze_symbol(self, symbol: str, required_scores: List[str]) -> Dict[str, Any]:
        """Core pipeline'ını direkt çağır"""
        try:
            result = await run_full_pipeline_async(
                symbol=symbol,
                requested_scores=required_scores
            )
            return result
        except Exception as e:
            logger.error(f"❌ Core analysis failed for {symbol}: {e}")
            return {"error": str(e)}

    # Skor Çıkarma İşlemi:
    def _extract_scores(self, result: Dict, cmd: str, symbol: str) -> Dict[str, float]:
        """Core'dan gelen skorları çıkar - BASİT VERSİYON"""
        required_scores = self.commands[cmd]
        scores = {}
        
        # ✅ SCORE ISIMLERINI BÜYÜK HARF YAP (görsel için)
        score_names = [s.upper() for s in required_scores]
        
        composites = result.get("composites", {})
        macros = result.get("macros", {})
        
        for i, metric in enumerate(required_scores):
            display_name = score_names[i]
            
            # Önce composites'te ara, sonra macros'ta
            if metric in composites:
                raw_value = composites[metric]
            elif metric in macros:
                raw_value = macros[metric]
            else:
                raw_value = None
            
            # 🔥 GERÇEK VERİ KONTROLÜ
            if raw_value is None:
                scores[display_name] = float('nan')
                logger.debug(f"📊 {symbol} - {metric}: VERİ YOK (None)")
            elif isinstance(raw_value, float) and math.isnan(raw_value):
                scores[display_name] = float('nan')
                logger.debug(f"📊 {symbol} - {metric}: VERİ YOK (NaN)")
            elif isinstance(raw_value, (int, float)):
                # GERÇEK VERİ - yuvarla ve kaydet
                scores[display_name] = round(raw_value, 3)
                logger.debug(f"📊 {symbol} - {metric}: {raw_value:.3f} (GERÇEK)")
            else:
                # Geçersiz veri tipi
                scores[display_name] = float('nan')
                logger.warning(f"📊 {symbol} - {metric}: GEÇERSİZ VERİ TİPİ {type(raw_value)}")
                
        return scores

    def _has_real_data(self, scores: Dict[str, float]) -> bool:
        """Skorlarda gerçek veri var mı kontrol et"""
        return any(
            isinstance(value, (int, float)) and not math.isnan(value) 
            for value in scores.values()
        )
    
    async def _get_symbols(self, args: List[str]) -> List[str]:
        """Sembol listesini oluştur"""
        if not args:
            return self.default_watchlist
        
        first_arg = args[0].upper()
        
        if first_arg.isdigit():
            count = min(int(first_arg), 20)
            return self.default_watchlist[:count]
        else:
            return [self._normalize_symbol(first_arg)]
    
    def _normalize_symbol(self, symbol_input: str) -> str:
        """Sembol normalizasyonu"""
        clean = symbol_input.upper().strip()
        return clean if clean.endswith('USDT') else f"{clean}USDT"

# ✅ TEK HANDLER INSTANCE
handler = SimpleCommandHandler()

# ✅ OPTIMIZED FORMAT FONKSİYONU
def format_table_response(result: Dict[str, Any]) -> str:
    """Core sonuçlarını formatla - BASİT VERSİYON"""
    symbol_scores = result["symbol_scores"]
    scores = result["scores"]
    
    # ✅ BAŞLIKLARI BÜYÜK HARF YAP
    headers = [s.upper() for s in scores]
    
    # Header
    header = "Sembol  " + "  ".join([f"{h:10}" for h in headers])
    
    lines = [
        f"📊 <b>{result['command'].upper()}</b> - CORE ANALİZ",
        "─" * (10 + len(headers) * 12),
        f"<b>{header}</b>",
        "─" * (10 + len(headers) * 12)
    ]
    
    # Satırlar - Basit sıralama
    for symbol, scores_dict in symbol_scores.items():
        display_symbol = symbol.replace('USDT', '')
        score_cells = []
        
        for header in headers:
            value = scores_dict.get(header, float('nan'))
            if isinstance(value, float) and math.isnan(value):
                score_cells.append("❌ ---")
            else:
                score_cells.append(f"{get_icon(header, value)} {value:+.2f}")
        
        line = f"{display_symbol:6}  " + "  ".join(score_cells)
        lines.append(line)
    
    # Özet
    failed_count = len(result.get('failed_symbols', []))
    success_count = len(symbol_scores)
    
    lines.extend([
        "─" * (10 + len(headers) * 12),
        f"<b>Özet:</b> {success_count}/{success_count+failed_count} sembol | " +
        f"<b>Başarısız:</b> {failed_count}"
    ])
    
    if failed_count > 0:
        lines.append(f"<i>Başarısız: {', '.join([s.replace('USDT', '') for s in result.get('failed_symbols', [])])}</i>")
    
    return "\n".join(lines)

def get_icon(column: str, score: float) -> str:
    """İkon belirle - BASİT VERSİYON"""
    if math.isnan(score):
        return "❌"
    
    column_lower = column.lower()
    
    if column_lower == "trend":
        return "🟢" if score > 0.3 else "🟡" if score > 0.1 else "⚪" if score > -0.1 else "🟠" if score > -0.3 else "🔴"
    elif column_lower == "vol":
        return "⚡" if abs(score) > 0.4 else "🔸" if abs(score) > 0.2 else "💤"
    elif column_lower == "core":
        return "🟢" if score > 0.3 else "🟡" if score > 0.1 else "⚪" if score > -0.1 else "🟠" if score > -0.3 else "🔴"
    else:
        return "🔹"

# ✅ MESSAGE HANDLER
@router.message()
async def handle_all_messages(message: types.Message):
    """Tüm mesajları işle"""
    text = message.text or ""
    
    if not text.startswith('/'):
        return
    
    result = await handler.handle(text)
    
    if result is None:
        await message.answer("❌ Desteklenmeyen komut: /t, /tt, /tv, /tcc")
        return
        
    if "error" in result:
        await message.answer(f"⚠️ {result['error']}")
        return
    
    response = format_table_response(result)
    await message.answer(response, parse_mode="HTML")