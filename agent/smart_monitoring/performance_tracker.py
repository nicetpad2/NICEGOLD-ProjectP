"""
Performance Tracker for Smart Monitoring
ติดตาม performance metrics และ trends
"""

from typing import Dict, Any, List
import time
from datetime import datetime, timedelta

class PerformanceTracker:
    """ติดตาม performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.performance_baseline = {}
        
    def track_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track performance metrics"""
        timestamp = datetime.now()
        
        performance_data = {
            'timestamp': timestamp.isoformat(),
            'metrics': metrics,
            'trends': self._calculate_trends(metrics)
        }
        
        self.metrics_history.append(performance_data)
        
        # Keep only last 100 entries
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return performance_data
    
    def _calculate_trends(self, current_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Calculate performance trends"""
        if len(self.metrics_history) < 2:
            return {}
        
        trends = {}
        previous_metrics = self.metrics_history[-1]['metrics']
        
        for key, value in current_metrics.items():
            if key in previous_metrics and isinstance(value, (int, float)):
                prev_value = previous_metrics[key]
                if prev_value > 0:
                    change = (value - prev_value) / prev_value * 100
                    if change > 5:
                        trends[key] = 'increasing'
                    elif change < -5:
                        trends[key] = 'decreasing'
                    else:
                        trends[key] = 'stable'
        
        return trends
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        return {
            'latest_metrics': latest['metrics'],
            'trends': latest['trends'],
            'history_length': len(self.metrics_history),
            'last_update': latest['timestamp']
        }
