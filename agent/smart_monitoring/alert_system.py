from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
"""
Alert System for Smart Monitoring
ระบบแจ้งเตือนสำหรับการ monitoring
"""


logger = logging.getLogger(__name__)

class AlertSystem:
    """ระบบแจ้งเตือน"""

    def __init__(self):
        self.active_alerts = []
        self.alert_history = []
        self.alert_rules = {}
        self.notification_channels = []

    def create_alert(self, alert_type: str, severity: str, message: str, 
                    metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """สร้าง alert ใหม่"""
        alert = {
            'id': f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]}", 
            'type': alert_type, 
            'severity': severity, 
            'message': message, 
            'metadata': metadata or {}, 
            'created_at': datetime.now().isoformat(), 
            'status': 'active', 
            'acknowledged': False, 
            'resolved': False
        }

        self.active_alerts.append(alert)
        self.alert_history.append(alert.copy())

        # Log alert
        log_level = self._get_log_level(severity)
        logger.log(log_level, f"Alert created: [{severity}] {message}")

        return alert

    def _get_log_level(self, severity: str) -> int:
        """แปลง severity เป็น log level"""
        severity_mapping = {
            'critical': logging.CRITICAL, 
            'high': logging.ERROR, 
            'medium': logging.WARNING, 
            'low': logging.INFO
        }
        return severity_mapping.get(severity, logging.INFO)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """ยืนยันว่าได้รับ alert แล้ว"""
        for alert in self.active_alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False

    def resolve_alert(self, alert_id: str, resolution_note: str = None) -> bool:
        """แก้ไข alert"""
        for i, alert in enumerate(self.active_alerts):
            if alert['id'] == alert_id:
                alert['resolved'] = True
                alert['resolved_at'] = datetime.now().isoformat()
                alert['resolution_note'] = resolution_note
                alert['status'] = 'resolved'

                # Move to history and remove from active
                self.active_alerts.pop(i)

                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False

    def get_active_alerts(self, severity: str = None) -> List[Dict[str, Any]]:
        """ดึง active alerts"""
        if severity:
            return [alert for alert in self.active_alerts if alert.get('severity') == severity]
        return self.active_alerts.copy()

    def get_alert_summary(self) -> Dict[str, Any]:
        """สรุป alerts"""
        severity_counts = {}
        for alert in self.active_alerts:
            severity = alert.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'total_active': len(self.active_alerts), 
            'total_history': len(self.alert_history), 
            'severity_breakdown': severity_counts, 
            'unacknowledged': len([a for a in self.active_alerts if not a.get('acknowledged', False)]), 
            'last_alert': self.alert_history[ - 1]['created_at'] if self.alert_history else None
        }

    def clear_resolved_alerts(self):
        """เคลียร์ alerts ที่แก้ไขแล้ว"""
        # Keep only active alerts
        self.active_alerts = [alert for alert in self.active_alerts if not alert.get('resolved', False)]
        logger.info("Resolved alerts cleared from active list")

    def auto_resolve_old_alerts(self, hours: int = 24):
        """แก้ไข alerts เก่าอัตโนมัติ"""
        cutoff_time = datetime.now() - timedelta(hours = hours)
        cutoff_str = cutoff_time.isoformat()

        resolved_count = 0
        for alert in self.active_alerts[:]:  # Use slice copy for safe iteration
            if alert['created_at'] < cutoff_str:
                self.resolve_alert(alert['id'], f"Auto - resolved after {hours} hours")
                resolved_count += 1

        if resolved_count > 0:
            logger.info(f"Auto - resolved {resolved_count} old alerts")

        return resolved_count

    def create_threshold_alert(self, metric_name: str, current_value: float, 
                             threshold: float, comparison: str = 'greater') -> Dict[str, Any]:
        """สร้าง threshold alert"""
        if comparison == 'greater' and current_value > threshold:
            severity = self._calculate_threshold_severity(current_value, threshold, comparison)
            message = f"{metric_name} exceeded threshold: {current_value} > {threshold}"
        elif comparison == 'less' and current_value < threshold:
            severity = self._calculate_threshold_severity(current_value, threshold, comparison)
            message = f"{metric_name} below threshold: {current_value} < {threshold}"
        else:
            return {}  # No alert needed

        return self.create_alert(
            alert_type = 'threshold', 
            severity = severity, 
            message = message, 
            metadata = {
                'metric': metric_name, 
                'current_value': current_value, 
                'threshold': threshold, 
                'comparison': comparison
            }
        )

    def _calculate_threshold_severity(self, value: float, threshold: float, comparison: str) -> str:
        """คำนวณ severity สำหรับ threshold alert"""
        if comparison == 'greater':
            ratio = value / threshold
        else:  # less
            ratio = threshold / value

        if ratio >= 2.0:
            return 'critical'
        elif ratio >= 1.5:
            return 'high'
        elif ratio >= 1.2:
            return 'medium'
        else:
            return 'low'

    def add_alert_rule(self, rule_name: str, condition: Dict[str, Any]):
        """เพิ่ม alert rule"""
        self.alert_rules[rule_name] = {
            'condition': condition, 
            'created_at': datetime.now().isoformat(), 
            'enabled': True
        }
        logger.info(f"Alert rule added: {rule_name}")

    def evaluate_alert_rules(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ประเมิน alert rules"""
        triggered_alerts = []

        for rule_name, rule in self.alert_rules.items():
            if not rule.get('enabled', True):
                continue

            condition = rule['condition']
            metric_name = condition.get('metric')
            threshold = condition.get('threshold')
            comparison = condition.get('comparison', 'greater')

            if metric_name in metrics:
                current_value = metrics[metric_name]

                # Check if condition is met
                if ((comparison == 'greater' and current_value > threshold) or
                    (comparison == 'less' and current_value < threshold) or
                    (comparison == 'equal' and current_value == threshold)):

                    alert = self.create_threshold_alert(metric_name, current_value, threshold, comparison)
                    if alert:
                        alert['rule_name'] = rule_name
                        triggered_alerts.append(alert)

        return triggered_alerts

    def get_alert_statistics(self, days: int = 7) -> Dict[str, Any]:
        """สถิติ alerts"""
        cutoff_time = datetime.now() - timedelta(days = days)
        cutoff_str = cutoff_time.isoformat()

        recent_alerts = [alert for alert in self.alert_history if alert['created_at'] >= cutoff_str]

        type_counts = {}
        severity_counts = {}

        for alert in recent_alerts:
            alert_type = alert.get('type', 'unknown')
            severity = alert.get('severity', 'unknown')

            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'period_days': days, 
            'total_alerts': len(recent_alerts), 
            'type_breakdown': type_counts, 
            'severity_breakdown': severity_counts, 
            'average_per_day': len(recent_alerts) / days if days > 0 else 0, 
            'current_active': len(self.active_alerts)
        }