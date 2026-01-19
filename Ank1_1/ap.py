import pandas as pd
import numpy as np
import asyncio
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# --- Geliştirilmiş BarContextBuilder ---
class BarContextBuilder:
    """DB'den gelen veriyi analiz motoru için stabilize eder."""
    def __init__(self, df_raw: pd.DataFrame, index_symbols: List[str]):
        self.df_raw = df_raw.copy()
        self.index_symbols = index_symbols

    def select_reference_ts(self) -> pd.DataFrame:
        if self.df_raw.empty: return pd.DataFrame()
        ts_coverage = self.df_raw.groupby("ts")["symbol"].nunique().sort_index()
        if ts_coverage.empty: return pd.DataFrame()
        ref_ts = ts_coverage.index[-1]
        return self.df_raw[self.df_raw["ts"] <= ref_ts]

    def build_pivots(self):
        df = (
            self.df_raw
            .groupby(["ts", "symbol"])
            .agg(
                price=("price", "last"),
                volume=("volume", "sum"),
                open_interest=("open_interest", "last"),
                funding_rate=("funding_rate", "last"),
            )
            .reset_index()
            .sort_values("ts")
        )
        # Pivot tabloları
        prices = df.pivot(index="ts", columns="symbol", values="price")
        volumes = df.pivot(index="ts", columns="symbol", values="volume").fillna(0)
        oi = df.pivot(index="ts", columns="symbol", values="open_interest")
        funding = df.pivot(index="ts", columns="symbol", values="funding_rate")
        return prices, volumes, oi, funding

# --- Hibrit ve Hacim Ağırlıklı MarketContextEngine ---
class MarketContextEngine:
    """Alt Power (AP) – Hibrit (Core + Satellite) ve Hacim Ağırlıklı Sürüm"""

    @staticmethod
    def scale_0_100(value: float, min_val: float, max_val: float) -> float:
        if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
            return float("nan")
        norm = (value - min_val) / (max_val - min_val)
        return float(np.clip(norm * 100, 0, 100))

    def calculate_alt_power(self, data_bundle: Dict, static_basket: List[str]) -> Dict[str, float]:
        df_micro = data_bundle.get("micro")
        df_macro = data_bundle.get("macro")

        if df_micro is None or df_micro.empty:
            return self._nan_result("NO_MICRO_DATA")

        # 1. HİBRİT SEPET OLUŞTURMA (Satellite Selection)
        # Son timestamp'teki en yüksek hacimli ilk 5 altcoini bul (Sabit basket dışındakiler)
        latest_ts = df_micro['ts'].max()
        df_latest = df_micro[df_micro['ts'] == latest_ts]
        
        # Trend olanları seç (Hacim ve Fiyat performansı kombinasyonu)
        dynamic_candidates = df_latest[~df_latest['symbol'].isin(static_basket + ["BTCUSDT"])]
        top_movers = dynamic_candidates.sort_values("volume", ascending=False).head(5)['symbol'].tolist()
        
        full_basket = list(set(static_basket + top_movers))
        logger.info(f"Hybrid Basket: Core({len(static_basket)}) + Satellite({len(top_movers)})")

        # 2. STABİLİZASYON
        builder = BarContextBuilder(df_micro, full_basket)
        stable_df = builder.select_reference_ts()
        if stable_df.empty: return self._nan_result("NO_STABLE_BAR")

        prices, volumes, oi, funding = builder.build_pivots()
        valid_alts = [s for s in full_basket if s in prices.columns and s != "BTCUSDT"]

        # 3. DİNAMİK VOLATİLİTE VE HACİM AĞIRLIKLARI
        # Sadece son barın hacmini alarak ağırlık oluştur
        last_volumes = volumes[valid_alts].iloc[-1]
        total_vol = last_volumes.sum()
        vol_weights = last_volumes / total_vol if total_vol > 0 else 1/len(valid_alts)

        market_vol = prices[valid_alts].pct_change(fill_method=None).std().median()
        dyn_range = market_vol if not (pd.isna(market_vol) or market_vol == 0) else 0.005

        # 4. ALT vs BTC (Hacim Ağırlıklı Return)
        btc_ret = prices["BTCUSDT"].pct_change(fill_method=None).iloc[-1]
        alt_rets_raw = prices[valid_alts].pct_change(fill_method=None).iloc[-1]
        
        # Hacim ağırlıklı altcoin getirisi (Gerçek piyasa gücü)
        alt_ret_weighted = (alt_rets_raw * vol_weights).sum()

        dom_bias = 1.0
        if df_macro is not None and len(df_macro) >= 2:
            if df_macro["btc_dom"].iloc[0] > df_macro["btc_dom"].iloc[1]:
                dom_bias = 0.85 # Dominance artarken baskıla

        v_btc = self.scale_0_100(alt_ret_weighted - btc_ret, -dyn_range, dyn_range) * dom_bias

        # 5. SHORT-TERM MOMENTUM (Hacim Ağırlıklı)
        mom_short_raw = prices[valid_alts].pct_change(periods=1, fill_method=None).iloc[-1]
        mom_mid_raw = prices[valid_alts].pct_change(periods=12, fill_method=None).iloc[-1]
        
        weighted_mom = ((mom_short_raw * 0.7 + mom_mid_raw * 0.3) * vol_weights).sum()
        v_short = self.scale_0_100(weighted_mom, -dyn_range * 1.5, dyn_range * 1.5)

        # 6. STRUCTURAL POWER (OI + Funding)
        oi_change = oi[valid_alts].pct_change(periods=6, fill_method=None).iloc[-1].median()
        avg_fund = funding[valid_alts].iloc[-1].median()
        
        # Funding eşiklerini daralttık (Daha hassas: %0.03 ve %-0.01 aralığı)
        oi_score = self.scale_0_100(oi_change, -0.03, 0.03)
        fund_score = self.scale_0_100(avg_fund, 0.0003, -0.0001) 
        
        v_long = (oi_score * 0.6 + fund_score * 0.4) if not (pd.isna(oi_score) or pd.isna(fund_score)) else float("nan")

        return {
            "alt_vs_btc_short": v_btc,
            "alt_short_term": v_short,
            "coin_long_term": v_long,
            "hybrid_info": {"satellite_count": len(top_movers), "satellites": top_movers},
            "macro_regime": "BTC-Focused" if dom_bias < 1.0 else "Risk-On"
        }

    @staticmethod
    def _nan_result(reason: str):
        return {
            "alt_vs_btc_short": float("nan"), "alt_short_term": float("nan"),
            "coin_long_term": float("nan"), "macro_regime": reason
          }
