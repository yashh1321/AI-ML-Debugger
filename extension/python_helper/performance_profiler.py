import json
import os
import sys
import time
import threading
import psutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any
import sqlite3

class PerformanceProfiler:
    """Resource and performance profiling for ML training"""

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.db_path = os.path.join(workspace_path, '.vscode', 'ai_debugger_performance.db')
        self.monitoring = False
        self.monitor_thread = None
        self.ensure_db()

    def ensure_db(self):
        """Create performance database if it doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_percent REAL,
                memory_percent REAL,
                memory_used_gb REAL,
                gpu_count INTEGER,
                gpu_utilization TEXT,
                gpu_memory_used TEXT,
                gpu_memory_total TEXT,
                disk_io_read_mb REAL,
                disk_io_write_mb REAL,
                network_sent_mb REAL,
                network_recv_mb REAL,
                process_info TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timeline_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                description TEXT,
                duration_ms REAL,
                file_path TEXT,
                line_number INTEGER,
                function_name TEXT,
                metadata TEXT
            )
        ''')

        # Add performance indices for faster O(log N) querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_logs_time ON performance_logs (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_events_time ON timeline_events (timestamp)')

        conn.commit()
        conn.close()

    def get_gpu_info(self) -> Dict:
        """Get GPU information using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            gpus = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpus.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'utilization': float(parts[2]),
                                'memory_used': float(parts[3]),
                                'memory_total': float(parts[4]),
                                'temperature': float(parts[5])
                            })

            return {
                'available': len(gpus) > 0,
                'count': len(gpus),
                'gpus': gpus
            }
        except Exception:
            return {'available': False, 'count': 0, 'gpus': []}

    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
        disk_io_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0

        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0
        network_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0

        # GPU info
        gpu_info = self.get_gpu_info()

        # Process info for Python processes
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'cmdline': ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else ''
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'disk_io_read_mb': disk_io_read_mb,
            'disk_io_write_mb': disk_io_write_mb,
            'network_sent_mb': network_sent_mb,
            'network_recv_mb': network_recv_mb,
            'gpu_info': gpu_info,
            'python_processes': python_processes
        }

    def log_performance(self, metrics: Dict):
        """Log performance metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        gpu_info = metrics['gpu_info']
        cursor.execute('''
            INSERT INTO performance_logs
            (cpu_percent, memory_percent, memory_used_gb, gpu_count,
             gpu_utilization, gpu_memory_used, gpu_memory_total,
             disk_io_read_mb, disk_io_write_mb, network_sent_mb, network_recv_mb,
             process_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['cpu_percent'],
            metrics['memory_percent'],
            metrics['memory_used_gb'],
            gpu_info['count'],
            json.dumps([gpu['utilization'] for gpu in gpu_info['gpus']]),
            json.dumps([gpu['memory_used'] for gpu in gpu_info['gpus']]),
            json.dumps([gpu['memory_total'] for gpu in gpu_info['gpus']]),
            metrics['disk_io_read_mb'],
            metrics['disk_io_write_mb'],
            metrics['network_sent_mb'],
            metrics['network_recv_mb'],
            json.dumps(metrics['python_processes'])
        ))

        conn.commit()
        conn.close()

    def start_monitoring(self, interval: float = 2.0):
        """Start continuous performance monitoring"""
        if self.monitoring:
            return

        self.monitoring = True

        def monitor_loop():
            while self.monitoring:
                try:
                    metrics = self.get_system_metrics()
                    self.log_performance(metrics)
                    time.sleep(interval)
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def get_performance_history(self, hours: int = 1) -> List[Dict]:
        """Get performance history from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM performance_logs
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours))

        columns = [desc[0] for desc in cursor.description]
        history = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return history

    def log_timeline_event(self, event_type: str, description: str,
                          duration_ms: float = None, file_path: str = None,
                          line_number: int = None, function_name: str = None,
                          metadata: Dict = None):
        """Log a timeline event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO timeline_events
            (event_type, description, duration_ms, file_path, line_number, function_name, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_type,
            description,
            duration_ms,
            file_path,
            line_number,
            function_name,
            json.dumps(metadata) if metadata else None
        ))

        conn.commit()
        conn.close()

    def get_timeline_events(self, hours: int = 1) -> List[Dict]:
        """Get timeline events from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM timeline_events
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours))

        columns = [desc[0] for desc in cursor.description]
        events = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return events

    def analyze_performance_bottlenecks(self) -> Dict:
        """Analyze performance data for bottlenecks"""
        history = self.get_performance_history(1)
        events = self.get_timeline_events(1)

        if not history:
            return {'bottlenecks': [], 'recommendations': []}

        bottlenecks = []
        recommendations = []

        # Analyze CPU usage
        avg_cpu = sum(h['cpu_percent'] for h in history) / len(history)
        if avg_cpu > 80:
            bottlenecks.append({
                'type': 'cpu',
                'severity': 'high',
                'description': f'High CPU usage: {avg_cpu:.1f}%'
            })
            recommendations.append('Consider reducing batch size or using multiple workers')

        # Analyze Memory usage
        avg_memory = sum(h['memory_percent'] for h in history) / len(history)
        if avg_memory > 85:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'high',
                'description': f'High memory usage: {avg_memory:.1f}%'
            })
            recommendations.append('Consider reducing model size or batch size')

        # Analyze GPU utilization
        gpu_utils = []
        for h in history:
            if h['gpu_utilization']:
                try:
                    utils = json.loads(h['gpu_utilization'])
                    gpu_utils.extend(utils)
                except:
                    pass

        if gpu_utils:
            avg_gpu = sum(gpu_utils) / len(gpu_utils)
            if avg_gpu < 50:
                bottlenecks.append({
                    'type': 'gpu_underutilization',
                    'severity': 'medium',
                    'description': f'Low GPU utilization: {avg_gpu:.1f}%'
                })
                recommendations.append('Consider increasing batch size or optimizing data loading')

        return {
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'summary': {
                'avg_cpu': avg_cpu if history else 0,
                'avg_memory': avg_memory if history else 0,
                'avg_gpu': sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                'total_events': len(events)
            }
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python performance_profiler.py <command> [args...]")
        return

    command = sys.argv[1]
    workspace_path = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()

    profiler = PerformanceProfiler(workspace_path)

    if command == "current_metrics":
        metrics = profiler.get_system_metrics()
        print(json.dumps({'success': True, 'metrics': metrics}))

    elif command == "start_monitoring":
        interval = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        profiler.start_monitoring(interval)
        print(json.dumps({'success': True, 'message': 'Monitoring started'}))

    elif command == "stop_monitoring":
        profiler.stop_monitoring()
        print(json.dumps({'success': True, 'message': 'Monitoring stopped'}))

    elif command == "history":
        hours = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        history = profiler.get_performance_history(hours)
        print(json.dumps({'success': True, 'history': history}))

    elif command == "timeline":
        hours = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        events = profiler.get_timeline_events(hours)
        print(json.dumps({'success': True, 'events': events}))

    elif command == "analyze":
        analysis = profiler.analyze_performance_bottlenecks()
        print(json.dumps({'success': True, 'analysis': analysis}))

    elif command == "log_event":
        event_type = sys.argv[3]
        description = sys.argv[4]
        duration = float(sys.argv[5]) if len(sys.argv) > 5 else None
        profiler.log_timeline_event(event_type, description, duration)
        print(json.dumps({'success': True, 'message': 'Event logged'}))

    else:
        print(json.dumps({'success': False, 'error': f'Unknown command: {command}'}))

if __name__ == "__main__":
    main()