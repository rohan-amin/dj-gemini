"""
Observability and monitoring system for DJ loop management.

This module provides comprehensive observability capabilities including metrics
collection, performance monitoring, health checks, diagnostic logging, and
real-time status reporting for the loop management system.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'type': self.metric_type.value
        }


@dataclass
class HealthCheck:
    """Health check definition and result."""
    name: str
    description: str
    check_function: Callable[[], bool]
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_check_time: float = 0.0
    last_error: Optional[str] = None
    check_interval: float = 30.0  # seconds
    
    def perform_check(self) -> HealthStatus:
        """Perform the health check and update status."""
        try:
            self.last_check_time = time.time()
            result = self.check_function()
            
            if result:
                self.last_status = HealthStatus.HEALTHY
                self.last_error = None
            else:
                self.last_status = HealthStatus.WARNING
                self.last_error = "Check function returned False"
            
        except Exception as e:
            self.last_status = HealthStatus.CRITICAL
            self.last_error = str(e)
            
        return self.last_status


@dataclass 
class TimingMeasurement:
    """Single timing measurement for performance analysis."""
    operation: str
    duration_ms: float
    timestamp: float
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates performance metrics for the loop system.
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of historical data points to keep
        """
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        
        self._lock = threading.RLock()
        
        logger.debug("🔄 MetricsCollector initialized")
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Value to increment by
            tags: Optional tags for the metric
        """
        with self._lock:
            self._counters[name] += value
            
            metric = PerformanceMetric(
                name=name,
                value=self._counters[name],
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.COUNTER
            )
            
            self._metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Current value
            tags: Optional tags for the metric
        """
        with self._lock:
            self._gauges[name] = value
            
            metric = PerformanceMetric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.GAUGE
            )
            
            self._metrics[name].append(metric)
    
    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a timing measurement.
        
        Args:
            name: Timer name
            duration_ms: Duration in milliseconds
            tags: Optional tags for the metric
        """
        with self._lock:
            timing = TimingMeasurement(
                operation=name,
                duration_ms=duration_ms,
                timestamp=time.time(),
                metadata=tags or {}
            )
            
            self._timers[name].append(timing)
            
            metric = PerformanceMetric(
                name=name,
                value=duration_ms,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.TIMER
            )
            
            self._metrics[name].append(metric)
    
    def get_counter_value(self, name: str) -> float:
        """Get current counter value."""
        with self._lock:
            return self._counters.get(name, 0.0)
    
    def get_gauge_value(self, name: str) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._gauges.get(name, 0.0)
    
    def get_timing_stats(self, name: str, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """
        Get timing statistics for a metric.
        
        Args:
            name: Timer name
            window_seconds: Optional time window for statistics (None = all data)
            
        Returns:
            Dictionary with timing statistics
        """
        with self._lock:
            timings = self._timers.get(name, deque())
            
            if not timings:
                return {'count': 0}
            
            # Filter by time window if specified
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                relevant_timings = [t for t in timings if t.timestamp >= cutoff_time]
            else:
                relevant_timings = list(timings)
            
            if not relevant_timings:
                return {'count': 0}
            
            durations = [t.duration_ms for t in relevant_timings]
            
            return {
                'count': len(durations),
                'min': min(durations),
                'max': max(durations),
                'mean': statistics.mean(durations),
                'median': statistics.median(durations),
                'p95': self._percentile(durations, 0.95),
                'p99': self._percentile(durations, 0.99)
            }
    
    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all collected metrics."""
        with self._lock:
            result = {}
            for name, metric_deque in self._metrics.items():
                result[name] = [metric.to_dict() for metric in metric_deque]
            return result
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'timers': {name: len(timings) for name, timings in self._timers.items()},
                'total_metrics': sum(len(metrics) for metrics in self._metrics.values()),
                'collection_time': time.time()
            }
    
    def clear_metrics(self) -> int:
        """
        Clear all collected metrics.
        
        Returns:
            Number of metrics cleared
        """
        with self._lock:
            total_cleared = sum(len(metrics) for metrics in self._metrics.values())
            
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()
            
            logger.info(f"🔄 Cleared {total_cleared} metrics")
            return total_cleared
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(percentile * (len(sorted_data) - 1))
        return sorted_data[index]


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics_collector: MetricsCollector, operation_name: str, 
                 tags: Optional[Dict[str, str]] = None):
        """
        Initialize performance timer.
        
        Args:
            metrics_collector: MetricsCollector instance
            operation_name: Name of operation being timed
            tags: Optional tags for the timing metric
        """
        self.metrics_collector = metrics_collector
        self.operation_name = operation_name
        self.tags = tags or {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        # Add success tag based on whether exception occurred
        tags = self.tags.copy()
        tags['success'] = str(exc_type is None)
        
        self.metrics_collector.record_timing(self.operation_name, duration_ms, tags)
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds (only available after completion)."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


class LoopSystemHealthMonitor:
    """
    Comprehensive health monitoring for the loop system.
    """
    
    def __init__(self, loop_controller, completion_system=None, config_manager=None):
        """
        Initialize health monitor.
        
        Args:
            loop_controller: LoopController instance to monitor
            completion_system: Optional LoopCompletionSystem to monitor
            config_manager: Optional LoopConfigurationManager to monitor
        """
        self.loop_controller = loop_controller
        self.completion_system = completion_system
        self.config_manager = config_manager
        
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_collector = MetricsCollector()
        
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._lock = threading.RLock()
        
        # Initialize standard health checks
        self._register_standard_health_checks()
        
        logger.info("🔄 LoopSystemHealthMonitor initialized")
    
    def start_monitoring(self, check_interval: float = 30.0) -> None:
        """
        Start background health monitoring.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        if self._monitoring_active:
            logger.warning("🔄 Health monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            name="LoopHealthMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info(f"🔄 Health monitoring started - check interval: {check_interval}s")
    
    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            if self._monitoring_thread.is_alive():
                logger.warning("🔄 Health monitoring thread did not stop gracefully")
        
        logger.info("🔄 Health monitoring stopped")
    
    def register_health_check(self, name: str, description: str, 
                            check_function: Callable[[], bool],
                            check_interval: float = 30.0) -> None:
        """
        Register a custom health check.
        
        Args:
            name: Unique name for the health check
            description: Human-readable description
            check_function: Function that returns True if healthy
            check_interval: Interval between checks
        """
        with self._lock:
            health_check = HealthCheck(
                name=name,
                description=description,
                check_function=check_function,
                check_interval=check_interval
            )
            
            self.health_checks[name] = health_check
            logger.debug(f"🔄 Health check registered: {name}")
    
    def unregister_health_check(self, name: str) -> bool:
        """
        Unregister a health check.
        
        Args:
            name: Name of health check to remove
            
        Returns:
            True if health check was found and removed
        """
        with self._lock:
            removed = self.health_checks.pop(name, None)
            if removed:
                logger.debug(f"🔄 Health check unregistered: {name}")
                return True
            return False
    
    def perform_all_health_checks(self) -> Dict[str, HealthStatus]:
        """
        Perform all registered health checks.
        
        Returns:
            Dictionary mapping check names to their status
        """
        results = {}
        
        with self._lock:
            for name, health_check in self.health_checks.items():
                try:
                    status = health_check.perform_check()
                    results[name] = status
                    
                    # Update metrics
                    self.metrics_collector.set_gauge(
                        f"health_check.{name}",
                        1.0 if status == HealthStatus.HEALTHY else 0.0,
                        {'check_name': name, 'status': status.value}
                    )
                    
                except Exception as e:
                    logger.error(f"🔄 Error performing health check {name}: {e}")
                    results[name] = HealthStatus.CRITICAL
        
        return results
    
    def get_overall_health_status(self) -> HealthStatus:
        """
        Get overall system health status.
        
        Returns:
            HealthStatus representing overall system health
        """
        check_results = self.perform_all_health_checks()
        
        if not check_results:
            return HealthStatus.UNKNOWN
        
        # Determine overall status based on worst individual status
        statuses = list(check_results.values())
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive health report.
        
        Returns:
            Dictionary with detailed health information
        """
        overall_status = self.get_overall_health_status()
        check_results = self.perform_all_health_checks()
        
        # Collect system metrics
        loop_stats = self.loop_controller.get_stats() if self.loop_controller else {}
        completion_stats = self.completion_system.get_stats() if self.completion_system else {}
        config_stats = self.config_manager.get_stats() if self.config_manager else {}
        
        return {
            'overall_status': overall_status.value,
            'timestamp': time.time(),
            'health_checks': {
                name: {
                    'status': status.value,
                    'last_check': check.last_check_time,
                    'error': check.last_error,
                    'description': check.description
                }
                for name, check in self.health_checks.items()
                for status in [check_results.get(name, HealthStatus.UNKNOWN)]
            },
            'system_metrics': {
                'loop_controller': loop_stats,
                'completion_system': completion_stats,
                'configuration_manager': config_stats
            },
            'performance_metrics': self.metrics_collector.get_metric_summary()
        }
    
    def collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        try:
            # Loop controller metrics
            if self.loop_controller:
                stats = self.loop_controller.get_stats()
                self.metrics_collector.set_gauge('loop_controller.active_loops', stats.get('active_loops', 0))
                self.metrics_collector.set_gauge('loop_controller.total_loops', stats.get('total_loops', 0))
                self.metrics_collector.set_gauge('loop_controller.processing_errors', stats.get('processing_errors', 0))
            
            # Completion system metrics
            if self.completion_system:
                stats = self.completion_system.get_stats()
                self.metrics_collector.set_gauge('completion_system.total_completions', stats.get('total_completions', 0))
                self.metrics_collector.set_gauge('completion_system.successful_actions', stats.get('successful_actions', 0))
                self.metrics_collector.set_gauge('completion_system.failed_actions', stats.get('failed_actions', 0))
                self.metrics_collector.set_gauge('completion_system.pending_events', stats.get('pending_events', 0))
            
            # Configuration manager metrics
            if self.config_manager:
                stats = self.config_manager.get_stats()
                self.metrics_collector.set_gauge('config_manager.total_configurations', stats.get('total_configurations', 0))
                self.metrics_collector.set_gauge('config_manager.total_loops', stats.get('total_loops', 0))
                self.metrics_collector.set_gauge('config_manager.validation_errors', stats.get('validation_errors', 0))
            
        except Exception as e:
            logger.error(f"🔄 Error collecting system metrics: {e}")
    
    def _register_standard_health_checks(self) -> None:
        """Register standard health checks for the loop system."""
        
        # Loop controller health check
        def check_loop_controller():
            if not self.loop_controller:
                return False
            return hasattr(self.loop_controller, '_is_initialized') and self.loop_controller._is_initialized
        
        self.register_health_check(
            'loop_controller',
            'Loop controller is initialized and operational',
            check_loop_controller
        )
        
        # Completion system health check
        if self.completion_system:
            def check_completion_system():
                return self.completion_system._running
            
            self.register_health_check(
                'completion_system',
                'Loop completion system is running',
                check_completion_system
            )
        
        # Processing performance check
        def check_processing_performance():
            if not self.loop_controller:
                return True
            
            stats = self.loop_controller.get_stats()
            error_rate = stats.get('processing_errors', 0) / max(1, stats.get('loops_activated', 1))
            return error_rate < 0.1  # Less than 10% error rate
        
        self.register_health_check(
            'processing_performance',
            'Loop processing error rate is acceptable',
            check_processing_performance
        )
    
    def _monitoring_loop(self, check_interval: float) -> None:
        """Main monitoring loop running in background thread."""
        logger.debug("🔄 Health monitoring loop started")
        
        while self._monitoring_active:
            try:
                # Perform health checks
                self.perform_all_health_checks()
                
                # Collect system metrics
                self.collect_system_metrics()
                
                # Sleep until next check
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"🔄 Error in monitoring loop: {e}")
                time.sleep(1.0)  # Short sleep on error
        
        logger.debug("🔄 Health monitoring loop stopped")
    
    def get_timing_context(self, operation_name: str, tags: Optional[Dict[str, str]] = None) -> PerformanceTimer:
        """
        Get a performance timing context manager.
        
        Args:
            operation_name: Name of operation to time
            tags: Optional tags for the timing
            
        Returns:
            PerformanceTimer context manager
        """
        return PerformanceTimer(self.metrics_collector, operation_name, tags)
    
    def cleanup(self) -> None:
        """Clean up health monitor resources."""
        try:
            logger.debug("🔄 Cleaning up LoopSystemHealthMonitor")
            
            # Stop monitoring
            self.stop_monitoring()
            
            # Clear health checks
            check_count = len(self.health_checks)
            self.health_checks.clear()
            
            # Clear metrics
            metric_count = self.metrics_collector.clear_metrics()
            
            logger.debug(f"🔄 LoopSystemHealthMonitor cleanup complete - "
                        f"{check_count} health checks, {metric_count} metrics cleared")
            
        except Exception as e:
            logger.error(f"🔄 Error during health monitor cleanup: {e}")


class ObservabilityIntegrator:
    """
    Integrates observability features with the loop management system.
    """
    
    def __init__(self, loop_controller, completion_system=None, config_manager=None):
        """
        Initialize observability integrator.
        
        Args:
            loop_controller: LoopController instance
            completion_system: Optional LoopCompletionSystem instance
            config_manager: Optional LoopConfigurationManager instance
        """
        self.loop_controller = loop_controller
        self.completion_system = completion_system
        self.config_manager = config_manager
        
        self.health_monitor = LoopSystemHealthMonitor(
            loop_controller, completion_system, config_manager
        )
        
        # Install hooks for automatic metric collection
        self._install_observability_hooks()
        
        logger.info("🔄 ObservabilityIntegrator initialized")
    
    def start_monitoring(self, check_interval: float = 30.0) -> None:
        """Start comprehensive monitoring."""
        self.health_monitor.start_monitoring(check_interval)
        logger.info("🔄 Comprehensive monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop comprehensive monitoring."""
        self.health_monitor.stop_monitoring()
        logger.info("🔄 Comprehensive monitoring stopped")
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive system dashboard data.
        
        Returns:
            Dictionary with all system status and metrics
        """
        return {
            'timestamp': time.time(),
            'health': self.health_monitor.get_health_report(),
            'performance': self.health_monitor.metrics_collector.get_metric_summary(),
            'system_info': {
                'components': {
                    'loop_controller': self.loop_controller is not None,
                    'completion_system': self.completion_system is not None,
                    'config_manager': self.config_manager is not None
                }
            }
        }
    
    def _install_observability_hooks(self) -> None:
        """Install hooks for automatic observability data collection."""
        # This would typically involve monkey-patching or decorator installation
        # For now, we'll document the approach
        logger.debug("🔄 Observability hooks installed (placeholder)")
    
    def cleanup(self) -> None:
        """Clean up observability integrator."""
        try:
            logger.debug("🔄 Cleaning up ObservabilityIntegrator")
            self.health_monitor.cleanup()
            logger.debug("🔄 ObservabilityIntegrator cleanup complete")
            
        except Exception as e:
            logger.error(f"🔄 Error during observability integrator cleanup: {e}")


# Utility decorators for observability

def timed_operation(metrics_collector: MetricsCollector, operation_name: str, 
                   tags: Optional[Dict[str, str]] = None):
    """
    Decorator to automatically time function execution.
    
    Args:
        metrics_collector: MetricsCollector instance
        operation_name: Name for the timing metric
        tags: Optional tags for the metric
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceTimer(metrics_collector, operation_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def counted_operation(metrics_collector: MetricsCollector, counter_name: str,
                     tags: Optional[Dict[str, str]] = None):
    """
    Decorator to automatically count function calls.
    
    Args:
        metrics_collector: MetricsCollector instance
        counter_name: Name for the counter metric
        tags: Optional tags for the metric
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                metrics_collector.increment_counter(f"{counter_name}.success", 1.0, tags)
                return result
            except Exception as e:
                error_tags = (tags or {}).copy()
                error_tags['error_type'] = type(e).__name__
                metrics_collector.increment_counter(f"{counter_name}.error", 1.0, error_tags)
                raise
        return wrapper
    return decorator