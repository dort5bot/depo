# tests/test_analysis.py
"""
MAPS Analysis System Comprehensive Test Suite
Test edilen bileşenler:
- AnalysisAggregator
- MetricEngine 
- MetricResolver
- SchemaManager
- ModuleLoader
- HealthChecker
- Tüm metric kategorileri

# Tüm testler
python tests/test_analysis.py

# Sadece belirli testler
pytest tests/test_analysis.py::TestAnalysisAggregator -v

# Hata ayıklama modunda
pytest tests/test_analysis.py -v --pdb

# Gerekli kütüphaneler
pip install pytest pytest-asyncio pandas numpy

# Test çalıştırma
python tests/test_analysis.py
# veya
pytest tests/test_analysis.py -v

# Özel test grupları
pytest tests/test_analysis.py::TestAnalysisAggregator -v
pytest tests/test_analysis.py::TestMetricEngine -v

"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Test için importlar
# Proje kökünü Python path'ine ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.analysis_a import AnalysisAggregator, create_analysis_aggregator
from analysis.metric_engine import MetricEngine
from analysis.metric_resolver import MetricResolver
from analysis.schema_manager import SchemaManager
from analysis.module_loader import ModuleLoader
from analysis.health_checker import UnifiedHealthChecker, ComponentType, HealthStatus



#Ortak fixture yap
@pytest.fixture
async def aggregator():
    agg = create_analysis_aggregator()
    yield agg
    if hasattr(agg, 'cleanup'):
        agg.cleanup()  # cleanup() async değil, sync



@pytest.mark.asyncio
async def test_aggregator_initialization(aggregator):
    # aggregator async fixture'tan geliyor
    instance = await aggregator  # eğer aggregator bir async generator ise
    assert hasattr(instance, 'schema')




# # 🔥 test METODLAR:
class TestDataGenerator:
    """Test verisi oluşturma yardımcı sınıfı"""
    
    @staticmethod
    def generate_ohlcv_data(rows=100, start_price=100, volatility=0.02):
        """OHLCV test verisi oluştur"""
        dates = pd.date_range(start='2024-01-01', periods=rows, freq='1h')
        
        prices = [start_price]
        for i in range(1, rows):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.1))  # Price can't go below 0.1
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': [abs(np.random.normal(1000, 200)) for _ in range(rows)]
        })
        
        return df
    
    @staticmethod
    def generate_orderbook_data(rows=50):
        """Order book test verisi oluştur"""
        return {
            'bid_price': [100 - i*0.1 for i in range(rows)],
            'bid_size': [abs(np.random.normal(10, 2)) for _ in range(rows)],
            'ask_price': [100 + i*0.1 for i in range(rows)],
            'ask_size': [abs(np.random.normal(8, 1.5)) for _ in range(rows)]
        }
    
    
    
    @staticmethod
    def generate_derivatives_data(rows=100):
        """Türev piyasası test verisi"""
        return {
            'funding_rate': np.random.normal(0.0001, 0.0005, rows),
            'open_interest': np.random.uniform(1000000, 5000000, rows),
            'long_short_ratio': np.random.uniform(0.5, 1.5, rows),
            'liquidations_long': np.random.exponential(100000, rows),
            'liquidations_short': np.random.exponential(80000, rows),
            'volume_buy': np.random.uniform(50000, 200000, rows),
            'volume_sell': np.random.uniform(50000, 200000, rows)
        }
    
    @staticmethod  
    def generate_onchain_data(rows=50):
        """On-chain test verisi"""
        return {
            'etf_flow': np.random.normal(0, 1000, rows),
            'exchange_netflow': np.random.normal(-500, 500, rows),
            'stablecoin_flow': np.random.normal(0, 2000, rows),
            'realized_profit': np.random.uniform(0, 5000, rows),
            'realized_loss': np.random.uniform(0, 3000, rows),
            'market_cap': np.random.uniform(1e9, 2e9, rows),
            'realized_cap': np.random.uniform(8e8, 1.5e9, rows)
        }
    
    @staticmethod
    def generate_microstructure_data(rows=200):
        """Microstructure test verisi"""
        return {
            'bid_price': np.cumsum(np.random.normal(0, 0.1, rows)) + 100,
            'bid_size': np.random.exponential(10, rows),
            'ask_price': np.cumsum(np.random.normal(0, 0.1, rows)) + 100.1,
            'ask_size': np.random.exponential(8, rows),
            'trade_volume': np.random.exponential(50, rows),
            'trade_price': np.cumsum(np.random.normal(0, 0.05, rows)) + 100.05
        }
    
    @staticmethod
    def generate_composite_test_data(rows=100):
        """Composite metric test verisi"""
        ohlcv = TestDataGenerator.generate_ohlcv_data(rows)
        derivatives = TestDataGenerator.generate_derivatives_data(rows)
        
        return {
            'ohlcv': ohlcv,
            'derivatives': derivatives,
            'timestamp': pd.date_range(start='2024-01-01', periods=rows, freq='1h')
        }
        


class TestAnalysisAggregator:
    """AnalysisAggregator test sınıfı"""
    
    @pytest.fixture
    async def aggregator(self):
        """Test için AnalysisAggregator instance'ı"""
        agg = create_analysis_aggregator()
        yield agg
        # Cleanup
        if hasattr(agg, 'cleanup'):
            await agg.cleanup()
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Örnek OHLCV verisi"""
        return TestDataGenerator.generate_ohlcv_data(rows=50)
    
    @pytest.fixture
    def sample_users(self):
        """Test kullanıcıları"""
        return ["test_user_1", "test_user_2", "test_user_3"]
    
    @pytest.fixture
    def sample_symbols(self):
        """Test sembolleri"""
        return ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

    # 🔹 TEMEL FONKSİYONEL TESTLER
    
    @pytest.mark.asyncio
    async def test_aggregator_initialization(self, aggregator):
        """Aggregator başlatma testi"""
        assert aggregator is not None
        assert hasattr(aggregator, 'schema')
        assert hasattr(aggregator, 'engine')
        assert hasattr(aggregator, 'resolver')
        assert hasattr(aggregator, 'health_checker')
        assert aggregator.settings['enabled'] == True
    
    @pytest.mark.asyncio
    async def test_symbol_validation(self, aggregator):
        """Sembol validasyon testi"""
        # Geçerli semboller (config'den alınan)
        valid_symbols = aggregator.get_available_symbols()
        if valid_symbols:
            assert aggregator.validate_symbol(valid_symbols[0]) == True
        
        # Geçersiz sembol
        assert aggregator.validate_symbol("INVALID_SYMBOL_XYZ") == False
    
    @pytest.mark.asyncio
    async def test_module_listing(self, aggregator):
        """Modül listeleme testi"""
        modules = aggregator.schema.list_modules()
        assert isinstance(modules, list)
        assert len(modules) > 0
        
        # Tüm modüllerin geçerli olup olmadığını kontrol et
        for module_name in modules:
            module_info = aggregator.schema.get_module(module_name)
            assert module_info is not None
            assert 'name' in module_info
            assert 'metrics' in module_info
    
    # 🔹 METRİK HESAPLAMA TESTLERİ
    
    @pytest.mark.asyncio
    async def test_single_module_execution(self, aggregator):
        """Tek modül çalıştırma testi"""
        modules = aggregator.schema.list_modules()
        if not modules:
            pytest.skip("No modules available for testing")
        
        # İlk modülü test et
        module_name = modules[0]
        result = await aggregator.run_module(
            user_id="test_user",
            module_name=module_name,
            symbol="BTCUSDT",
            interval="1h"
        )
        
        assert module_name in result
        assert isinstance(result[module_name], dict)
        
        # Modülün metriklerini kontrol et
        module_info = aggregator.schema.get_module(module_name)
        expected_metrics = []
        for metric_group in module_info.get('metrics', {}).values():
            expected_metrics.extend(metric_group)
        
        # Bazı metriklerin hesaplanıp hesaplanmadığını kontrol et
        calculated_metrics = list(result[module_name].keys())
        assert len(calculated_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_all_modules_execution(self, aggregator):
        """Tüm modülleri çalıştırma testi"""
        result = await aggregator.run_all_modules(
            user_id="test_user",
            symbol="BTCUSDT", 
            interval="1h"
        )
        
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Tüm modüllerin sonuçlarının birleştirilip birleştirilmediğini kontrol et
        modules = aggregator.schema.list_modules()
        for module_name in modules:
            module_info = aggregator.schema.get_module(module_name)
            for metric_group in module_info.get('metrics', {}).values():
                for metric_name in metric_group:
                    expected_key = f"{module_name}.{metric_name}"
                    # Tüm metrikler olmayabilir (hata vs.), ama bazıları olmalı
                    if expected_key in result:
                        assert result[expected_key] is not None
    
    @pytest.mark.asyncio 
    async def test_batch_module_execution(self, aggregator):
        """Batch modül çalıştırma testi"""
        modules = aggregator.schema.list_modules()[:3]  # İlk 3 modül
        
        result = await aggregator.run_module_batch(
            user_id="test_user",
            modules=modules,
            symbol="BTCUSDT",
            interval="1h"
        )
        
        assert isinstance(result, dict)
        
        # Batch sonuçlarının doğru şekilde birleştirildiğini kontrol et
        for module_name in modules:
            module_info = aggregator.schema.get_module(module_name)
            for metric_group in module_info.get('metrics', {}).values():
                for metric_name in metric_group:
                    expected_key = f"{module_name}.{metric_name}"
                    if expected_key in result:
                        assert result[expected_key] is not None
    
    @pytest.mark.asyncio
    async def test_multi_user_execution(self, aggregator, sample_users):
        """Çoklu kullanıcı testi"""
        result = await aggregator.run_multi_user(
            user_ids=sample_users,
            symbol="BTCUSDT",
            interval="1h"
        )
        
        assert isinstance(result, dict)
        assert len(result) == len(sample_users)
        
        for user_id in sample_users:
            assert user_id in result
            assert isinstance(result[user_id], dict)
    
    # 🔹 COMPOSITE ANALİZ TESTLERİ
    
    @pytest.mark.asyncio
    async def test_composite_analysis(self, aggregator):
        """Composite analiz testi"""
        result = await aggregator.run_composite_analysis(
            user_id="test_user",
            symbol="BTCUSDT",
            interval="1h"
        )
        
        assert isinstance(result, dict)
        assert 'composite_scores' in result
        assert 'base_metrics' in result
        
        composite_scores = result['composite_scores']
        expected_composites = ['trend_strength', 'volatility_regime', 'risk_level', 'market_regime']
        
        for score_name in expected_composites:
            if score_name in composite_scores:
                score_value = composite_scores[score_name]
                # Skorların beklenen aralıkta olup olmadığını kontrol et
                if score_name != 'market_regime':  #market_regime string döner
                    assert isinstance(score_value, (int, float, np.number))
    
    # 🔹 PERFORMANS ve SAĞLIK TESTLERİ
    
    @pytest.mark.asyncio
    async def test_system_health(self, aggregator):
        """Sistem sağlık kontrolü testi"""
        health_data = await aggregator.get_system_health()
        
        assert isinstance(health_data, dict)
        assert 'overall_status' in health_data
        assert 'components' in health_data
        assert 'performance_summary' in health_data
        
        # Health status geçerli olmalı
        assert health_data['overall_status'] in ['healthy', 'degraded', 'critical', 'offline']
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, aggregator):
        """Performans izleme testi"""
        # Birkaç işlem yap
        await aggregator.run_all_modules("test_user", "BTCUSDT", "1h")
        
        health_data = await aggregator.get_system_health()
        performance_data = health_data.get('performance_summary', {})
        
        # Performans verilerinin var olduğunu kontrol et
        assert isinstance(performance_data, dict)
    
    # 🔹 HATA DURUMU TESTLERİ
    
    @pytest.mark.asyncio
    async def test_invalid_module(self, aggregator):
        """Geçersiz modül testi"""
        with pytest.raises(Exception):
            await aggregator.run_module(
                user_id="test_user",
                module_name="INVALID_MODULE_NAME",
                symbol="BTCUSDT",
                interval="1h"
            )
    
    @pytest.mark.asyncio
    async def test_invalid_symbol(self, aggregator):
        """Geçersiz sembol testi"""
        # Bu test, sembol validasyonunun çalışıp çalışmadığını kontrol eder
        invalid_symbol = "INVALID_SYMBOL_123"
        
        if not aggregator.validate_symbol(invalid_symbol):
            # Geçersiz sembolle çalışmaya çalışırsa hata vermeli
            with pytest.raises(Exception):
                await aggregator.run_module(
                    user_id="test_user",
                    module_name=aggregator.schema.list_modules()[0],
                    symbol=invalid_symbol,
                    interval="1h"
                )
    
    @pytest.mark.asyncio
    async def test_metric_timeout_handling(self, aggregator):
        """Metric timeout handling testi"""
        # Bu test, timeout durumlarının doğru şekilde handle edilip edilmediğini kontrol eder
        # Özel olarak yavaş çalışan bir metric test edilebilir
        pass  # Implement later with specific timeout tests


class TestMetricEngine:
    """MetricEngine test sınıfı"""
    
    @pytest.fixture
    def metric_engine(self):
        return MetricEngine()
    
    @pytest.fixture
    def sample_data(self):
        return TestDataGenerator.generate_ohlcv_data(rows=20)
    
    def test_metric_computation(self, metric_engine, sample_data):
        """Temel metric hesaplama testi"""
        def simple_metric(data):
            return data['close'].mean()
        
        result = metric_engine.compute(
            module_name="test_module",
            metric_name="test_metric",
            func=simple_metric,
            data=sample_data
        )
        
        assert isinstance(result, float)
        assert result == sample_data['close'].mean()
    
    def test_last_valid_fallback(self, metric_engine):
        """Last valid fallback testi"""
        # Hata veren bir metric fonksiyonu
        def failing_metric(data):
            raise ValueError("Test error")
        
        # İlk çağrı - hata vermeli
        result1 = metric_engine.compute(
            module_name="test_module",
            metric_name="failing_metric",
            func=failing_metric,
            data={},
            use_last_valid=False,
            default=42.0
        )
        assert result1 == 42.0
        
        # Last valid ile çağrı
        result2 = metric_engine.compute(
            module_name="test_module", 
            metric_name="failing_metric",
            func=failing_metric,
            data={},
            use_last_valid=True,
            default=100.0
        )
        assert result2 == 42.0  # Last valid değeri kullanmalı


class TestMetricResolver:
    """MetricResolver test sınıfı"""
    
    @pytest.fixture
    def resolver(self):
        return MetricResolver()
    
    def test_metric_resolution(self, resolver):
        """Metric çözümleme testi"""
        # Klasik metrikleri test et
        classical_metrics = ['EMA', 'RSI', 'MACD', 'ATR']
        
        for metric_name in classical_metrics:
            try:
                func = resolver.resolve(metric_name)
                assert callable(func)
            except ValueError:
                # Metric bulunamayabilir, bu normal
                pass
    
    def test_metric_normalization(self, resolver):
        """Metric isim normalizasyon testi"""
        # Farklı formatlardaki metric isimlerini test et
        test_cases = [
            'rsi', 'RSI', 'R_S_I',  # RSI için farklı formatlar
            'macd', 'MACD',
            'ema', 'EMA'
        ]
        
        for metric_name in test_cases:
            try:
                func = resolver.resolve(metric_name)
                if func:
                    assert callable(func)
            except ValueError:
                # Bulunamayan metricler normal
                pass


class TestSchemaManager:
    """SchemaManager test sınıfı"""
    
    @pytest.fixture
    def schema_manager(self):
        return SchemaManager()
    
    def test_schema_loading(self, schema_manager):
        """Schema yükleme testi"""
        modules = schema_manager.list_modules()
        assert isinstance(modules, list)
        assert len(modules) > 0
    
    def test_module_filtering(self, schema_manager):
        """Modül filtreleme testi"""
        # Data model'e göre filtrele
        pandas_modules = schema_manager.filter_by_data_model('pandas')
        numpy_modules = schema_manager.filter_by_data_model('numpy')
        
        assert isinstance(pandas_modules, list)
        assert isinstance(numpy_modules, list)
    
    def test_module_grouping(self, schema_manager):
        """Modül gruplama testi"""
        grouped = schema_manager.group_by_data_source()
        assert isinstance(grouped, dict)
        
        for source, modules in grouped.items():
            assert isinstance(source, str)
            assert isinstance(modules, list)


class TestHealthChecker:
    """HealthChecker test sınıfı"""
    
    @pytest.fixture
    def health_checker(self, aggregator):
        return UnifiedHealthChecker(aggregator)
    
    @pytest.mark.asyncio
    async def test_health_check(self, health_checker):
        """Sağlık kontrolü testi"""
        health_data = await health_checker.comprehensive_health_check()
        
        assert isinstance(health_data, dict)
        assert 'overall_status' in health_data
        assert health_data['overall_status'] in [s.value for s in HealthStatus]
    
    def test_performance_tracking(self, health_checker):
        """Performans izleme testi"""
        # Bazı performans verileri ekle
        health_checker.track_performance(ComponentType.METRIC_ENGINE, 0.15, True)
        health_checker.track_performance(ComponentType.DATA_PROVIDER, 0.08, True)
        health_checker.track_metric_performance("RSI", 0.05)
        
        # Performans özetini al
        performance_summary = health_checker.get_performance_summary()
        assert isinstance(performance_summary, dict)


class TestAdvancedScenarios:
    """İleri seviye test senaryoları"""
    
    @pytest.fixture
    async def aggregator(self):
        agg = create_analysis_aggregator()
        yield agg
        if hasattr(agg, 'cleanup'):
            agg.cleanup()
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, aggregator):
        """Cache mekanizması testi"""
        # İlk çağrı
        start_time = datetime.now()
        result1 = await aggregator.run_module(
            user_id="cache_test_user",
            module_name=aggregator.schema.list_modules()[0],
            symbol="BTCUSDT",
            interval="1h"
        )
        
        # Aynı parametrelerle ikinci çağrı (cache hit olmalı)
        result2 = await aggregator.run_module(
            user_id="cache_test_user", 
            module_name=aggregator.schema.list_modules()[0],
            symbol="BTCUSDT",
            interval="1h"
        )
        
        # Sonuçlar aynı olmalı
        assert result1 == result2
        
        # Performans istatistiklerinde cache hit artmış olmalı
        health_data = await aggregator.get_system_health()
        cache_stats = health_data.get('cache_effectiveness', {})
        assert cache_stats.get('cache_hits', 0) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_metrics(self, aggregator):
        """Streaming metrik hesaplama testi"""
        modules = aggregator.schema.list_modules()[:3]
        results = []
        
        async for result in aggregator.stream_metrics_calculation(
            user_id="stream_test_user",
            modules=modules,
            symbol="ETHUSDT",
            interval="1h"
        ):
            results.append(result)
            assert isinstance(result, dict)
            assert len(result) == 1  # Her seferinde bir modül sonucu
        
        assert len(results) == len(modules)
    
    @pytest.mark.asyncio
    async def test_memory_management(self, aggregator):
        """Bellek yönetimi testi"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Çoklu büyük işlem
        tasks = []
        for i in range(10):
            task = aggregator.run_all_modules(
                user_id=f"memory_test_{i}",
                symbol="BTCUSDT", 
                interval="1h"
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Bellek temizleme
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Bellek sızıntısı olmamalı (50MB altında artış)
        assert memory_increase < 50, f"Bellek sızıntısı: {memory_increase:.2f}MB"
        


class TestErrorScenarios:
    """Hata senaryoları testleri"""
    
    @pytest.fixture
    async def aggregator(self):
        agg = create_analysis_aggregator()
        yield agg
        if hasattr(agg, 'cleanup'):
            agg.cleanup()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_scenario(self, aggregator):
        """Circuit breaker mekanizması testi"""
        # Bu test için özel olarak hata üreten bir senaryo gerekebilir
        # Şimdilik placeholder - gerçek implementasyon için mock'lar gerekli
        pass
    
    @pytest.mark.asyncio 
    async def test_metric_timeout_handling(self, aggregator):
        """Metric timeout handling testi"""
        # Yavaş çalışan metric simülasyonu
        async def slow_metric(data):
            await asyncio.sleep(35)  # Timeout süresinden fazla
            return 42.0
        
        # MetricResolver'ı geçici olarak değiştir
        original_resolve = aggregator.resolver.resolve
        aggregator.resolver.resolve = lambda x: slow_metric
        
        try:
            result = await aggregator.run_module(
                user_id="timeout_test_user",
                module_name=aggregator.schema.list_modules()[0],
                symbol="BTCUSDT",
                interval="1h"
            )
            # Timeout olmalı ve default değer dönmeli
            assert result is not None
        except asyncio.TimeoutError:
            # Timeout exception'ı da kabul edilebilir
            pass
        finally:
            # Orijinal resolver'ı geri yükle
            aggregator.resolver.resolve = original_resolve
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self, aggregator):
        """Geçersiz veri handling testi"""
        # NaN veya None veri ile test
        invalid_data = {
            'open': [np.nan, np.nan, np.nan],
            'high': [None, None, None],
            'low': [0, 0, 0],
            'close': [100, 100, 100],
            'volume': [0, 0, 0]
        }
        
        # Bu test için özel data provider mock'u gerekli
        # Şimdilik placeholder
        pass
        


# 🔹 ENTEGRASYON TESTLERİ

class TestIntegrationScenarios:
    """Entegrasyon senaryo testleri"""
    
    @pytest.fixture
    async def test_system(self):
        """Tam test sistemi"""
        aggregator = create_analysis_aggregator()
        yield aggregator
        if hasattr(aggregator, 'cleanup'):
            aggregator.cleanup()
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, test_system):
        """Tam iş akışı testi"""
        # 1. Sistem sağlığını kontrol et
        health = await test_system.get_system_health()
        assert health['overall_status'] != 'critical'
        
        # 2. Tüm modülleri çalıştır
        all_results = await test_system.run_all_modules(
            user_id="integration_test_user",
            symbol="BTCUSDT",
            interval="1h"
        )
        assert isinstance(all_results, dict)
        assert len(all_results) > 0
        
        # 3. Composite analiz yap
        composite_results = await test_system.run_composite_analysis(
            user_id="integration_test_user", 
            symbol="BTCUSDT",
            interval="1h"
        )
        assert 'composite_scores' in composite_results
        
        # 4. Performansı kontrol et
        final_health = await test_system.get_system_health()
        assert 'performance_summary' in final_health
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self, test_system):
        """Yüksek yük senaryosu testi"""
        users = [f"load_test_user_{i}" for i in range(5)]
        symbols = test_system.get_default_symbols(3)
        
        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Paralel olarak çoklu kullanıcı ve sembol testi
        tasks = []
        for user in users:
            for symbol in symbols:
                task = test_system.run_all_modules(user, symbol, "1h")
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Tüm görevlerin tamamlandığını kontrol et (exception'lar normal)
        assert len(results) == len(users) * len(symbols)
        
        # Sistemin hala sağlıklı olduğunu kontrol et
        health = await test_system.get_system_health()
        assert health['overall_status'] != 'offline'
        
        
    @pytest.mark.asyncio
    async def test_mixed_workload_scenario(self, test_system):
        """Karışık iş yükü senaryosu"""
        # Farklı modül tipleriyle test
        modules_by_intensity = {
            'low': [],
            'medium': [], 
            'high': []
        }
        
        for module_name in test_system.schema.list_modules():
            module_info = test_system.schema.get_module(module_name)
            intensity = module_info.get('compute_intensity', 'medium')
            modules_by_intensity[intensity].append(module_name)
        
        # Farklı intensity'lerde paralel test
        tasks = []
        for intensity, module_list in modules_by_intensity.items():
            if module_list:
                task = test_system.run_module_batch(
                    user_id=f"intensity_{intensity}_user",
                    modules=module_list[:2],  # İlk 2 modül
                    symbol="BTCUSDT",
                    interval="1h"
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Tüm görevler tamamlanmış olmalı
        assert len(results) == len([m for m in modules_by_intensity.values() if m])
    
    @pytest.mark.asyncio
    async def test_data_persistence_scenario(self, test_system):
        """Veri kalıcılığı senaryosu"""
        # Aynı kullanıcı için ardışık istekler
        user_id = "persistence_test_user"
        symbol = "ETHUSDT"
        
        # İlk istek
        result1 = await test_system.run_all_modules(user_id, symbol, "1h")
        
        # Kısa süre sonra ikinci istek (cache etkisini test etmek için)
        await asyncio.sleep(1)
        result2 = await test_system.run_all_modules(user_id, symbol, "1h")
        
        # Sonuçlar tutarlı olmalı (aynı veya benzer)
        assert len(result1) == len(result2)
        
        # Anahtar metrikler aynı olmalı
        common_keys = set(result1.keys()) & set(result2.keys())
        assert len(common_keys) > 0
        
        
        


# 🔹 TEST ÇALIŞTIRICI

def run_all_tests():
    """Tüm testleri çalıştır"""
    import subprocess
    import sys
    
    print("🧪 MAPS Analysis System Test Suite")
    print("=" * 50)
    
    # Test komutunu oluştur
    cmd = [
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--tb=short",
        "-x"  # İlk hata durumunda dur
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Tüm testler başarıyla geçti!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Bazı testler başarısız oldu!")
        return False


if __name__ == "__main__":
    # Testleri çalıştır
    success = run_all_tests()
    
    # Hızlı manuel test
    if success:
        print("\n🚀 Hızlı Manuel Test:")
        print("-" * 30)
        
        async def quick_test():
            aggregator = create_analysis_aggregator()
            try:
                # Hızlı sistem kontrolü
                health = await aggregator.get_system_health()
                print(f"🏥 Sistem Sağlığı: {health['overall_status']}")
                
                # Modül listesi
                modules = aggregator.schema.list_modules()
                print(f"📦 Modül Sayısı: {len(modules)}")
                
                # Hızlı metric testi
                if modules:
                    result = await aggregator.run_module(
                        user_id="quick_test_user",
                        module_name=modules[0],
                        symbol="BTCUSDT", 
                        interval="1h"
                    )
                    print(f"✅ İlk modül testi başarılı: {list(result.keys())[0]}")
                
            except Exception as e:
                print(f"❌ Hızlı test hatası: {e}")
            finally:
                if hasattr(aggregator, 'cleanup'):
                    aggregator.cleanup()
        
        asyncio.run(quick_test())
    
    sys.exit(0 if success else 1)