"""
Performance Profiler
===================

Advanced performance analysis and profiling for ProjectP optimization.
"""

import os
import time
import psutil
import functools
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
from datetime import datetime
import cProfile
import pstats
from io import StringIO

class PerformanceProfiler:
    """
    Advanced performance profiling and analysis for project optimization.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.performance_data = {}
        self.memory_snapshots = []
        self.profiling_results = {}
        
    def profile_function_performance(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a specific function's performance.
        """
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # CPU profiling
        profiler = cProfile.Profile()
        
        # Time measurement
        start_time = time.time()
        
        try:
            # Run profiling
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            end_time = time.time()
            
            # Memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate profile stats
            stats_stream = StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats()
            
            profile_data = {
                'execution_time': end_time - start_time,
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_delta_mb': memory_after - memory_before,
                'function_name': func.__name__,
                'success': True,
                'result_size': self._estimate_result_size(result),
                'profile_stats': stats_stream.getvalue(),
                'timestamp': datetime.now().isoformat()
            }
            
            return profile_data
            
        except Exception as e:
            return {
                'execution_time': time.time() - start_time,
                'memory_before_mb': memory_before,
                'memory_after_mb': process.memory_info().rss / 1024 / 1024,
                'function_name': func.__name__,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def profile_project_pipeline(self) -> Dict[str, Any]:
        """
        Profile the entire project pipeline performance.
        """
        print("🔍 Starting comprehensive pipeline performance profiling...")
        
        pipeline_performance = {
            'overall_performance': self._analyze_overall_performance(),
            'memory_usage_patterns': self._analyze_memory_patterns(),
            'cpu_usage_patterns': self._analyze_cpu_patterns(),
            'io_performance': self._analyze_io_performance(),
            'bottleneck_analysis': self._identify_performance_bottlenecks(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
        
        return pipeline_performance
    
    def _analyze_overall_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance metrics."""
        process = psutil.Process()
        
        return {
            'current_memory_usage_mb': process.memory_info().rss / 1024 / 1024,
            'current_cpu_percent': process.cpu_percent(),
            'thread_count': process.num_threads(),
            'open_files_count': len(process.open_files()) if hasattr(process, 'open_files') else 0,
            'system_memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'system_memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_cpu_count': psutil.cpu_count(),
            'system_cpu_percent': psutil.cpu_percent(interval=1)
        }
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        try:
            # Take memory snapshot
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_patterns = {
                'current_rss_mb': memory_info.rss / 1024 / 1024,
                'current_vms_mb': memory_info.vms / 1024 / 1024,
                'memory_growth_rate': self._calculate_memory_growth_rate(),
                'potential_memory_leaks': self._detect_potential_memory_leaks(),
                'memory_efficiency_score': self._calculate_memory_efficiency()
            }
            
            # Store snapshot for trend analysis
            self.memory_snapshots.append({
                'timestamp': datetime.now().isoformat(),
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024
            })
            
            # Keep only last 100 snapshots
            if len(self.memory_snapshots) > 100:
                self.memory_snapshots = self.memory_snapshots[-100:]
            
            return memory_patterns
            
        except Exception as e:
            return {
                'error': f"Memory analysis failed: {e}",
                'memory_efficiency_score': 0.5
            }
    
    def _analyze_cpu_patterns(self) -> Dict[str, Any]:
        """Analyze CPU usage patterns."""
        try:
            process = psutil.Process()
            
            # Get CPU times
            cpu_times = process.cpu_times()
            
            return {
                'user_time': cpu_times.user,
                'system_time': cpu_times.system,
                'cpu_percent': process.cpu_percent(),
                'cpu_efficiency_score': self._calculate_cpu_efficiency(),
                'context_switches': getattr(process, 'num_ctx_switches', lambda: None)(),
                'cpu_optimization_potential': self._assess_cpu_optimization_potential()
            }
            
        except Exception as e:
            return {
                'error': f"CPU analysis failed: {e}",
                'cpu_efficiency_score': 0.5
            }
    
    def _analyze_io_performance(self) -> Dict[str, Any]:
        """Analyze I/O performance patterns."""
        try:
            process = psutil.Process()
            
            # Get I/O counters if available
            io_counters = getattr(process, 'io_counters', lambda: None)()
            
            if io_counters:
                return {
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count,
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                    'io_efficiency_score': self._calculate_io_efficiency(io_counters),
                    'io_optimization_recommendations': self._generate_io_recommendations(io_counters)
                }
            else:
                return {
                    'io_counters_available': False,
                    'io_efficiency_score': 0.7  # Default moderate score
                }
                
        except Exception as e:
            return {
                'error': f"I/O analysis failed: {e}",
                'io_efficiency_score': 0.5
            }
    
    def _identify_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify potential performance bottlenecks."""
        bottlenecks = []
        
        # Memory bottlenecks
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 80:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'high',
                'description': f"High memory usage: {memory_usage:.1f}%",
                'recommendation': 'Optimize memory-intensive operations'
            })
        
        # CPU bottlenecks
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 90:
            bottlenecks.append({
                'type': 'cpu',
                'severity': 'high',
                'description': f"High CPU usage: {cpu_usage:.1f}%",
                'recommendation': 'Optimize CPU-intensive operations or use parallel processing'
            })
        
        # I/O bottlenecks (check for large files)
        large_files = self._find_large_files()
        if large_files:
            bottlenecks.append({
                'type': 'io',
                'severity': 'medium',
                'description': f"Found {len(large_files)} large files that may impact I/O",
                'recommendation': 'Consider file size optimization or streaming for large files',
                'files': large_files[:5]  # Show top 5
            })
        
        return bottlenecks
    
    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        # Check for common optimization patterns
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files[:20]:  # Limit to first 20 files for performance
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common inefficient patterns
                if 'pd.concat' in content and 'for ' in content:
                    opportunities.append({
                        'type': 'pandas_optimization',
                        'file': str(py_file),
                        'description': 'Potential inefficient pandas concatenation in loop',
                        'recommendation': 'Use list comprehension and single concat call',
                        'impact': 'high'
                    })
                
                if '.iterrows()' in content:
                    opportunities.append({
                        'type': 'pandas_optimization',
                        'file': str(py_file),
                        'description': 'iterrows() usage detected',
                        'recommendation': 'Replace with vectorized operations',
                        'impact': 'high'
                    })
                
                if 'np.where' not in content and 'if ' in content:
                    opportunities.append({
                        'type': 'vectorization',
                        'file': str(py_file),
                        'description': 'Potential vectorization opportunity',
                        'recommendation': 'Consider using numpy vectorized operations',
                        'impact': 'medium'
                    })
                
            except Exception as e:
                continue
        
        return opportunities
    
    def _calculate_memory_growth_rate(self) -> float:
        """Calculate memory growth rate from snapshots."""
        if len(self.memory_snapshots) < 2:
            return 0.0
        
        try:
            recent = self.memory_snapshots[-5:]  # Last 5 snapshots
            if len(recent) < 2:
                return 0.0
            
            growth = recent[-1]['rss_mb'] - recent[0]['rss_mb']
            return growth / len(recent)
        
        except Exception:
            return 0.0
    
    def _detect_potential_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        leaks = []
        
        if len(self.memory_snapshots) >= 10:
            # Check for consistent growth
            recent_snapshots = self.memory_snapshots[-10:]
            growth_trend = []
            
            for i in range(1, len(recent_snapshots)):
                growth = recent_snapshots[i]['rss_mb'] - recent_snapshots[i-1]['rss_mb']
                growth_trend.append(growth)
            
            # If memory consistently grows
            positive_growth = sum(1 for g in growth_trend if g > 0)
            if positive_growth / len(growth_trend) > 0.7:  # 70% positive growth
                leaks.append({
                    'type': 'consistent_growth',
                    'severity': 'medium',
                    'description': 'Memory consistently growing over time',
                    'growth_rate_mb_per_snapshot': sum(growth_trend) / len(growth_trend)
                })
        
        return leaks
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score (0-1)."""
        try:
            system_memory = psutil.virtual_memory()
            process = psutil.Process()
            
            # Base efficiency on memory usage relative to available
            usage_ratio = process.memory_info().rss / system_memory.available
            
            # Lower ratio = higher efficiency
            efficiency = max(0.0, 1.0 - min(1.0, usage_ratio * 2))
            
            return efficiency
        
        except Exception:
            return 0.5
    
    def _calculate_cpu_efficiency(self) -> float:
        """Calculate CPU efficiency score (0-1)."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Moderate CPU usage (30-70%) is considered efficient
            if 30 <= cpu_percent <= 70:
                return 0.9
            elif cpu_percent < 30:
                return 0.6  # Under-utilized
            else:
                return max(0.1, 1.0 - (cpu_percent - 70) / 30)
        
        except Exception:
            return 0.5
    
    def _calculate_io_efficiency(self, io_counters) -> float:
        """Calculate I/O efficiency score."""
        try:
            # Simple heuristic based on read/write ratio
            total_ops = io_counters.read_count + io_counters.write_count
            if total_ops == 0:
                return 0.8
            
            # Balanced read/write is generally good
            read_ratio = io_counters.read_count / total_ops
            
            # Optimal ratio is around 0.6-0.8 (more reads than writes for ML)
            if 0.6 <= read_ratio <= 0.8:
                return 0.9
            else:
                return 0.7
        
        except Exception:
            return 0.5
    
    def _generate_io_recommendations(self, io_counters) -> List[str]:
        """Generate I/O optimization recommendations."""
        recommendations = []
        
        try:
            if io_counters.write_count > io_counters.read_count * 2:
                recommendations.append("Consider reducing write operations or batching writes")
            
            if io_counters.read_bytes > 1024 * 1024 * 1024:  # > 1GB
                recommendations.append("Large amount of data read - consider streaming or chunking")
            
            if io_counters.write_bytes > 1024 * 1024 * 1024:  # > 1GB
                recommendations.append("Large amount of data written - consider compression")
        
        except Exception:
            pass
        
        if not recommendations:
            recommendations.append("I/O patterns appear normal")
        
        return recommendations
    
    def _assess_cpu_optimization_potential(self) -> Dict[str, Any]:
        """Assess CPU optimization potential."""
        return {
            'parallel_processing_opportunity': True,
            'vectorization_opportunity': True,
            'algorithm_optimization_opportunity': True,
            'estimated_improvement': '10-30%'
        }
    
    def _find_large_files(self) -> List[Dict[str, Any]]:
        """Find large files that might impact performance."""
        large_files = []
        
        try:
            for file_path in self.project_root.rglob("*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    if size > 100 * 1024 * 1024:  # > 100MB
                        large_files.append({
                            'path': str(file_path),
                            'size_mb': size / 1024 / 1024,
                            'type': file_path.suffix
                        })
            
            # Sort by size
            large_files.sort(key=lambda x: x['size_mb'], reverse=True)
            
        except Exception as e:
            print(f"Error finding large files: {e}")
        
        return large_files
    
    def _estimate_result_size(self, result) -> Dict[str, Any]:
        """Estimate the size of a function result."""
        try:
            import sys
            
            size_info = {
                'type': type(result).__name__,
                'size_bytes': sys.getsizeof(result)
            }
            
            # Additional info for common types
            if hasattr(result, 'shape'):  # numpy array, pandas DataFrame
                size_info['shape'] = str(result.shape)
            elif hasattr(result, '__len__'):
                size_info['length'] = len(result)
            
            return size_info
            
        except Exception:
            return {'type': 'unknown', 'size_bytes': 0}
    
    def generate_performance_report(self, output_path: str = None) -> str:
        """Generate comprehensive performance report."""
        if not output_path:
            output_path = self.project_root / "performance_report.json"
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'system_info': self._analyze_overall_performance(),
                'memory_analysis': self._analyze_memory_patterns(),
                'cpu_analysis': self._analyze_cpu_patterns(),
                'io_analysis': self._analyze_io_performance(),
                'bottlenecks': self._identify_performance_bottlenecks(),
                'optimization_opportunities': self._identify_optimization_opportunities(),
                'performance_score': self._calculate_overall_performance_score()
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error generating performance report: {e}")
            return ""
    
    def _calculate_overall_performance_score(self) -> float:
        """Calculate overall performance score (0-1)."""
        try:
            memory_score = self._calculate_memory_efficiency()
            cpu_score = self._calculate_cpu_efficiency()
            
            # Weight the scores
            overall_score = (memory_score * 0.4 + cpu_score * 0.4 + 0.6 * 0.2)  # Base score
            
            # Penalize for bottlenecks
            bottlenecks = self._identify_performance_bottlenecks()
            bottleneck_penalty = min(0.3, len(bottlenecks) * 0.1)
            
            return max(0.0, min(1.0, overall_score - bottleneck_penalty))
            
        except Exception:
            return 0.5
