1 trend_moment.py  + c_trend.py  eklenen dosya (tam kod ve config):  →  Polars  
veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
analysis_helpers.py ile tam uyumlu, gereken tam kodları ver"
"

2 volat_regime.py  + c_volat.py  eklenen dosya (tam kod ve config):  →  Polars  
veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
analysis_helpers.py ile tam uyumlu, gereken tam kodları ver"
"

3 deriv_sentim.py  + c_deriv.py  eklenen dosya (tam kod ve config):  →  Pandas  
veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
analysis_helpers.py ile tam uyumlu, gereken tam kodları ver"
"

4 order_micros.py  + c_order.py  eklenen dosya (tam kod ve config):  →  Numpy  
veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
analysis_helpers.py ile tam uyumlu, gereken tam kodları ver"
"

5 corr_lead.py  + c_corr.py  eklenen dosya (tam kod ve config):  →  Polars  
veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
analysis_helpers.py ile tam uyumlu, gereken tam kodları ver"
"

6 onchain.py  + c_onchain.py  eklenen dosya (tam kod ve config):  →  Pandas  
veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
analysis_helpers.py ile tam uyumlu, gereken tam kodları ver"
"

7 risk_expos.py  + c_risk.py  eklenen dosya (tam kod ve config):  →  Polars  
veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
analysis_helpers.py ile tam uyumlu, gereken tam kodları ver"
"

8 microalpha.py  + c_micro.py  eklenen dosya (tam kod ve config):  →  Numpy  
veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
analysis_helpers.py ile tam uyumlu, gereken tam kodları ver"
"

9 port_alloc.py  + c_portalloc.py  eklenen dosya (tam kod ve config):  →  Polars  
veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
analysis_helpers.py ile tam uyumlu, gereken tam kodları ver"
"

10 regime_anomal.py  + ❌ Hayır (içeride tanımla)  eklenen dosya (tam kod ve config):  →  Polars  
veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
analysis_helpers.py ile tam uyumlu, gereken tam kodları ver"



*
    ├──analysis/    # kaynak veriyi analiz eden modul
        ├── analysis_base_module.py   #
        ├── analysis_core.py          # analysis modülünde Ana aggregator
        ├── analysis_router.py        # FastAPI router, analysis_metric_schema ile dinamik router
        ├── analysis_schema_manager.py   # Dynamic module loading: Schema yöneticisi
        ├── analysis_market_insights.py  # piyasa trendlerini, altcoin gücünü ve yatırımcı risk iştahını ölçe
        ├── analysis_metric_schema.yaml  # modüllerin schemaları: dinamik router, özet bilgiler, endpointler, job tipi
        ├── analysis_helpers.py # Merkezi Analysis Helper Sınıfı
        │
        ├── __init__.py
        ├── corr_lead.py
        ├── deriv_sentim.py
        ├── microalpha.py
        ├── onchain.py
        ├── order_micros.py
        ├── port_alloc.py
        ├── regime_anomal.py
        ├── risk_expos.py
        ├── trend_moment.py
        └── volat_regime.py
        │
        ├──config/  #modüller için Hybrid config Yapi
            ├──  c_corr.py
            ├──  c_deriv.py
            ├──  c_micro.py
            ├──  c_onchain.py
            ├──  c_order.py
            ├──  c_portalloc.py
            ├──  c_risk.py
            ├──  c_trend.py
            ├──  c_volat.py
            ├──  cm_base.py
            └── cm_loader.py
        │   
        ├── composite/  # bileşik skorlar modülü
            ├── composite_engine.py          # composite - Ana bileşik skor motoru
            ├── composite_strategies.py      # Skor stratejileri
            ├── composite_config.yaml        # Tüm bileşik skor tanımları
            ├── composite_optimizer.py       # Ağırlık optimizasyonu
    │

          A. Trend & Momentum (TA)   →  Polars  veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
B. Piyasa Rejimi (Volatilite & Yapı)   →  Polars  ,k

1
eklenen dosyalar (tam kod ve config): deriv_sentim.py  + c_deriv.py →  Pandas  veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
eklenen dosyalar (tam kod ve config): order_micros.py  + c_order.py →  Numpy  veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
eklenen dosyalar (tam kod ve config): corr_lead.py  + c_corr.py →  Polars  veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
2
eklenen dosyalar (tam kod ve config): onchain.py  + c_onchain.py →  Pandas  veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
eklenen dosyalar (tam kod ve config): risk_expos.py  + c_risk.py →  Polars  veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
eklenen dosyalar (tam kod ve config): microalpha.py  + c_micro.py →  Numpy  veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı

3
eklenen dosyalar (tam kod ve config): port_alloc.py  + c_portalloc.py →  Polars  veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı
4
eklenen dosyalar (tam kod ): regime_anomal.py   →  Polars  veri modelinde, tam async, multi user yapı da mı, değilse gerekli kodları ver, isimlendirmeleri koru örnek: compute_metrics > compute_metrics olmalı

> buna bağlı modüller şu veri modellri ile çalışıyor. 
    bu dosya buna uyumlu  mu
Polars
Pandas
Numpy
asyncio.to_thread

          
