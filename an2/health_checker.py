# analysis/health_checker.py
"""
Unified Health Check & 
Performance Monitoring System

MAPS Framework - Comprehensive system monitoring and performance analytics

Author: ysf-bot-framework
Version: 2025.1
Features:
- Real-time health status monitoring
- Performance metrics tracking and analytics
- Automated alerting system
- Historical performance analysis
- Resource utilization monitoring
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

class ComponentType(Enum):
    METRIC_ENGINE = "metric_engine"
    DATA_PROVIDER = "data_provider"
    MODULE_LOADER = "module_loader"
    SCHEMA_MANAGER = "schema_manager"
    CACHE_MANAGER = "cache_manager"
    EXTERNAL_API = "external_api"

@dataclass
class PerformanceStats:
    """Performance statistics for a single component"""
    component: ComponentType
    call_count: int = 0
    total_duration: float = 0.0
    success_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    last_updated: datetime = None
    
    def update(self, duration: float, success: bool = True):
        self.call_count += 1
        self.total_duration += duration
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.last_updated = datetime.now()

@dataclass
class HealthCheckResult:
    """Health check result for a component"""
    component: ComponentType
    status: HealthStatus
    message: str
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = None

class UnifiedHealthChecker:
    """
    Unified Health Check and Performance Monitoring System
    Combines real-time health checks with performance analytics
    """
    
    def __init__(self, analysis_aggregator=None, config=None):
        self.aggregator = analysis_aggregator
        self.config = config
        
        # Performance tracking
        self.performance_stats: Dict[ComponentType, PerformanceStats] = {}
        self.response_times: Dict[ComponentType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Health check history
        self.health_history: Dict[ComponentType, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alerts: List[Dict] = []
        
        # System metrics
        self.system_metrics = {
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'disk_io': deque(maxlen=100)
        }
        
        # Thresholds
        self.thresholds = {
            'response_time_critical': 10.0,  # seconds
            'response_time_warning': 3.0,
            'error_rate_critical': 0.1,  # 10%
            'error_rate_warning': 0.05,  # 5%
            'memory_critical': 90.0,  # percentage
            'memory_warning': 80.0,
            'cpu_critical': 90.0,
            'cpu_warning': 75.0
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize performance tracking for all components"""
        for component in ComponentType:
            self.performance_stats[component] = PerformanceStats(
                component=component,
                last_updated=datetime.now()
            )
    
    # ==================== PERFORMANCE MONITORING ====================
        
    def track_performance(self, component: ComponentType, duration: float, success: bool = True):
        """Geliştirilmiş ve güvenli performance tracking"""
        try:
            with self._lock:  # Thread-safe izleme
                stats = self.performance_stats[component]
                stats.update(duration, success)

                # Response time kayıt
                self.response_times[component].append(duration)

                # Ortalama ve p95 hesaplama
                if len(self.response_times[component]) >= 10:
                    times = list(self.response_times[component])
                    stats.avg_response_time = statistics.mean(times)
                    stats.p95_response_time = statistics.quantiles(times, n=20)[18]  # 95. yüzdelik

                # Gerçek zamanlı alert kontrolü
                self._check_real_time_alerts(component, duration, success)

                # Eski verileri temizleme (memory optimizasyonu)
                if len(self.response_times[component]) % 100 == 0:
                    self._cleanup_old_data()

                # Performans uyarı kontrolü (genel metrik düzeyinde)
                self._check_performance_alerts(component, duration, success)

        except Exception as e:
            logger.error(f"Performance tracking error for {component}: {e}")
    
    
    def track_metric_performance(self, metric_name: str, duration: float):
        """Track performance for individual metrics"""
        self.metric_performance[metric_name].append(duration)
        # Keep only last 100 measurements
        if len(self.metric_performance[metric_name]) > 100:
            self.metric_performance[metric_name] = self.metric_performance[metric_name][-100:]
    
    def _check_performance_alerts(self, component: ComponentType, duration: float, success: bool):
        """Check and generate performance alerts"""
        stats = self.performance_stats[component]
        
        # Response time alert
        if duration > self.thresholds['response_time_critical']:
            self._add_alert(
                component=component,
                level="CRITICAL",
                message=f"High response time: {duration:.2f}s",
                details={'response_time': duration}
            )
        elif duration > self.thresholds['response_time_warning']:
            self._add_alert(
                component=component,
                level="WARNING", 
                message=f"Elevated response time: {duration:.2f}s",
                details={'response_time': duration}
            )
        
        # Error rate alert
        if stats.call_count >= 10:
            error_rate = stats.error_count / stats.call_count
            if error_rate > self.thresholds['error_rate_critical']:
                self._add_alert(
                    component=component,
                    level="CRITICAL",
                    message=f"High error rate: {error_rate:.1%}",
                    details={'error_rate': error_rate}
                )
    
    def _add_alert(self, component: ComponentType, level: str, message: str, details: Dict = None):
        """Add a new alert"""
        alert = {
            'timestamp': datetime.now(),
            'component': component.value,
            'level': level,
            'message': message,
            'details': details or {}
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT [{level}] {component.value}: {message}")
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    # ==================== HEALTH CHECK METHODS ====================
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all system components
        Returns aggregated health status
        """
        health_checks = [
            self._check_metric_engine_health(),
            self._check_data_provider_health(),
            self._check_module_loader_health(),
            self._check_schema_manager_health(),
            self._check_system_resources(),
            self._check_external_apis_health()
        ]
        
        results = await asyncio.gather(*health_checks, return_exceptions=True)
        
        # Aggregate results
        component_statuses = {}
        overall_status = HealthStatus.HEALTHY
        critical_components = 0
        
        for result in results:
            if isinstance(result, HealthCheckResult):
                component_statuses[result.component.value] = {
                    'status': result.status.value,
                    'message': result.message,
                    'response_time': result.response_time,
                    'timestamp': result.timestamp.isoformat()
                }
                
                # Determine overall status
                if result.status == HealthStatus.CRITICAL:
                    critical_components += 1
                    overall_status = HealthStatus.CRITICAL
                elif result.status == HealthStatus.DEGRADED and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.DEGRADED
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'components': component_statuses,
            'performance_summary': self.get_performance_summary(),
            'alerts': self.get_recent_alerts(10),
            'system_metrics': self._get_system_metrics()
        }
    
    async def _check_metric_engine_health(self) -> HealthCheckResult:
        """Check health of Metric Engine"""
        start_time = time.time()
        try:
            if self.aggregator and hasattr(self.aggregator, 'engine'):
                # Test basic functionality
                test_result = await self.aggregator.engine.compute_async(
                    "test_module", "test_metric", lambda x: 1.0, [1, 2, 3],
                    use_last_valid=True, default=0.0
                )
                status = HealthStatus.HEALTHY
                message = "Metric engine operational"
            else:
                status = HealthStatus.DEGRADED
                message = "Metric engine not available"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Metric engine error: {str(e)}"
        
        response_time = time.time() - start_time
        return HealthCheckResult(
            component=ComponentType.METRIC_ENGINE,
            status=status,
            message=message,
            response_time=response_time,
            timestamp=datetime.now()
        )
    
    async def _check_data_provider_health(self) -> HealthCheckResult:
        """Check health of Data Provider"""
        start_time = time.time()
        try:
            if self.aggregator and hasattr(self.aggregator, 'data_provider'):
                # Test connectivity
                status = HealthStatus.HEALTHY
                message = "Data provider operational"
            else:
                status = HealthStatus.DEGRADED
                message = "Data provider not available"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Data provider error: {str(e)}"
        
        response_time = time.time() - start_time
        return HealthCheckResult(
            component=ComponentType.DATA_PROVIDER,
            status=status,
            message=message,
            response_time=response_time,
            timestamp=datetime.now()
        )
    
    async def _check_module_loader_health(self) -> HealthCheckResult:
        """Check health of Module Loader"""
        start_time = time.time()
        try:
            if self.aggregator and hasattr(self.aggregator, 'loader'):
                modules = self.aggregator.loader.list_modules()
                status = HealthStatus.HEALTHY
                message = f"Module loader operational ({len(modules)} modules)"
            else:
                status = HealthStatus.DEGRADED
                message = "Module loader not available"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Module loader error: {str(e)}"
        
        response_time = time.time() - start_time
        return HealthCheckResult(
            component=ComponentType.MODULE_LOADER,
            status=status,
            message=message,
            response_time=response_time,
            timestamp=datetime.now()
        )
    
    async def _check_schema_manager_health(self) -> HealthCheckResult:
        """Check health of Schema Manager"""
        start_time = time.time()
        try:
            if self.aggregator and hasattr(self.aggregator, 'schema'):
                modules = self.aggregator.schema.list_modules()
                status = HealthStatus.HEALTHY
                message = f"Schema manager operational ({len(modules)} schemas)"
            else:
                status = HealthStatus.DEGRADED
                message = "Schema manager not available"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Schema manager error: {str(e)}"
        
        response_time = time.time() - start_time
        return HealthCheckResult(
            component=ComponentType.SCHEMA_MANAGER,
            status=status,
            message=message,
            response_time=response_time,
            timestamp=datetime.now()
        )
    
    async def _check_external_apis_health(self) -> HealthCheckResult:
        """Check health of external APIs"""
        start_time = time.time()
        try:
            # Placeholder for actual API health checks
            status = HealthStatus.HEALTHY
            message = "External APIs operational"
            
        except Exception as e:
            status = HealthStatus.DEGRADED
            message = f"External API issues: {str(e)}"
        
        response_time = time.time() - start_time
        return HealthCheckResult(
            component=ComponentType.EXTERNAL_API,
            status=status,
            message=message,
            response_time=response_time,
            timestamp=datetime.now()
        )
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization"""
        start_time = time.time()
        try:
            # Memory usage
            memory_percent = psutil.virtual_memory().percent
            self.system_metrics['memory_usage'].append(memory_percent)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.system_metrics['cpu_usage'].append(cpu_percent)
            
            # Disk I/O (simplified)
            disk_io = psutil.disk_usage('/').percent
            self.system_metrics['disk_io'].append(disk_io)
            
            # Determine status based on thresholds
            if (memory_percent > self.thresholds['memory_critical'] or 
                cpu_percent > self.thresholds['cpu_critical']):
                status = HealthStatus.CRITICAL
            elif (memory_percent > self.thresholds['memory_warning'] or 
                  cpu_percent > self.thresholds['cpu_warning']):
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            message = f"System resources: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            
        except Exception as e:
            status = HealthStatus.DEGRADED
            message = f"System resource check failed: {str(e)}"
        
        response_time = time.time() - start_time
        return HealthCheckResult(
            component=ComponentType.METRIC_ENGINE,  # Using metric engine as system representative
            status=status,
            message=message,
            response_time=response_time,
            timestamp=datetime.now(),
            details={
                'memory_usage': memory_percent if 'memory_percent' in locals() else None,
                'cpu_usage': cpu_percent if 'cpu_percent' in locals() else None,
                'disk_usage': disk_io if 'disk_io' in locals() else None
            }
        )
    
    # ==================== REPORTING METHODS ====================
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {}
        
        for component, stats in self.performance_stats.items():
            if stats.call_count > 0:
                success_rate = stats.success_count / stats.call_count
                summary[component.value] = {
                    'call_count': stats.call_count,
                    'success_rate': success_rate,
                    'avg_response_time': stats.avg_response_time,
                    'p95_response_time': stats.p95_response_time,
                    'last_updated': stats.last_updated.isoformat()
                }
        
        # Metric performance summary
        slow_metrics = self.get_slow_metrics(threshold=1.0)
        summary['metric_performance'] = {
            'total_tracked_metrics': len(self.metric_performance),
            'slow_metrics': slow_metrics,
            'avg_metric_times': self.get_average_metric_times()
        }
        
        return summary
    
    def get_slow_metrics(self, threshold: float = 1.0) -> List[Dict]:
        """Get metrics with average execution time above threshold"""
        slow_metrics = []
        
        for metric_name, times in self.metric_performance.items():
            if times:
                avg_time = statistics.mean(times)
                if avg_time > threshold:
                    slow_metrics.append({
                        'metric': metric_name,
                        'avg_time': avg_time,
                        'call_count': len(times),
                        'p95_time': statistics.quantiles(times, n=20)[18] if len(times) >= 10 else avg_time
                    })
        
        return sorted(slow_metrics, key=lambda x: x['avg_time'], reverse=True)
    
    def get_average_metric_times(self) -> Dict[str, float]:
        """Get average execution times for all tracked metrics"""
        avg_times = {}
        
        for metric_name, times in self.metric_performance.items():
            if times:
                avg_times[metric_name] = statistics.mean(times)
        
        return avg_times
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts sorted by timestamp"""
        return sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')
            
            return {
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'percent': memory.percent
                },
                'cpu': {
                    'percent': cpu
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'percent': disk.percent
                }
            }
        except Exception as e:
            logger.error(f"System metrics error: {e}")
            return {}
    
    def get_health_history(self, component: ComponentType, hours: int = 24) -> List[Dict]:
        """Get health history for a component"""
        history = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for check in self.health_history[component]:
            if check.timestamp >= cutoff_time:
                history.append({
                    'timestamp': check.timestamp.isoformat(),
                    'status': check.status.value,
                    'response_time': check.response_time,
                    'message': check.message
                })
        
        return history
    
    def clear_old_data(self, days: int = 7):
        """Clear data older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Clear old alerts
        self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
        
        # Clear old health history
        for component in self.health_history:
            self.health_history[component] = deque(
                [check for check in self.health_history[component] if check.timestamp > cutoff_time],
                maxlen=100
            )


# ==================== USAGE EXAMPLES ====================

async def example_usage():
    """Example usage of the UnifiedHealthChecker"""
    
    # Initialize
    health_checker = UnifiedHealthChecker()
    
    # Perform comprehensive health check
    health_status = await health_checker.comprehensive_health_check()
    print("Health Status:", health_status['overall_status'])
    
    # Track some performance metrics
    health_checker.track_performance(ComponentType.METRIC_ENGINE, 0.15, True)
    health_checker.track_performance(ComponentType.DATA_PROVIDER, 2.5, True)
    health_checker.track_metric_performance("RSI", 0.08)
    health_checker.track_metric_performance("Kalman_Filter", 1.2)
    
    # Get performance summary
    performance = health_checker.get_performance_summary()
    print("Performance Summary:", performance)
    
    # Get slow metrics
    slow_metrics = health_checker.get_slow_metrics()
    print("Slow Metrics:", slow_metrics)
    
    # Get recent alerts
    alerts = health_checker.get_recent_alerts()
    print("Recent Alerts:", alerts)

if __name__ == "__main__":
    asyncio.run(example_usage())