#!/usr/bin/env python3
"""
Inline Model Telemetry & Timeline for AI/ML Debugger
Chrome DevTools-style performance timeline for ML training
"""

import json
import sys
import traceback
import os
import time
import threading
import psutil
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import queue
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional GPU monitoring imports
try:
    import pynvml
    GPU_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class PerformanceTimeline:
    """Chrome DevTools-style performance timeline for ML training"""

    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path or os.getcwd()
        self.timeline_db_path = os.path.join(self.workspace_path, ".vscode", "performance-timeline.db")
        self.ensure_database()

        # Timeline state
        self.recording = False
        self.session_id = None
        self.start_time = None

        # Monitoring threads
        self.monitor_thread = None
        self.data_queue = queue.Queue()
        self.stop_monitoring = threading.Event()

        # Performance data collectors
        self.layer_timings = defaultdict(list)
        self.memory_timeline = deque(maxlen=1000)
        self.gpu_timeline = deque(maxlen=1000)
        self.cpu_timeline = deque(maxlen=1000)
        self.io_timeline = deque(maxlen=1000)

        # Event markers
        self.events = []
        self.markers = []

        # GPU handles
        self.gpu_handles = []
        if GPU_AVAILABLE:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
            except Exception as e:
                logger.warning(f"Could not initialize GPU monitoring: {e}")

    def ensure_database(self):
        """Create timeline database"""
        os.makedirs(os.path.dirname(self.timeline_db_path), exist_ok=True)

        with sqlite3.connect(self.timeline_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS timeline_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_ms INTEGER,
                    total_events INTEGER,
                    metadata TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp_ms INTEGER,
                    event_type TEXT,
                    event_name TEXT,
                    duration_ms INTEGER,
                    category TEXT,
                    thread_name TEXT,
                    process_id INTEGER,
                    details TEXT,
                    FOREIGN KEY (session_id) REFERENCES timeline_sessions (session_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp_ms INTEGER,
                    cpu_percent REAL,
                    memory_used_mb REAL,
                    memory_percent REAL,
                    gpu_utilization REAL,
                    gpu_memory_used_mb REAL,
                    gpu_memory_percent REAL,
                    io_read_mb REAL,
                    io_write_mb REAL,
                    network_sent_mb REAL,
                    network_recv_mb REAL,
                    FOREIGN KEY (session_id) REFERENCES timeline_sessions (session_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS layer_timings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp_ms INTEGER,
                    layer_name TEXT,
                    operation_type TEXT,
                    forward_time_ms REAL,
                    backward_time_ms REAL,
                    memory_allocated_mb REAL,
                    memory_freed_mb REAL,
                    input_shapes TEXT,
                    output_shapes TEXT,
                    parameters_count INTEGER,
                    FOREIGN KEY (session_id) REFERENCES timeline_sessions (session_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS timeline_markers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp_ms INTEGER,
                    marker_type TEXT,
                    label TEXT,
                    description TEXT,
                    line_number INTEGER,
                    file_path TEXT,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES timeline_sessions (session_id)
                )
            ''')

            # ⚡ Bolt: Performance Optimization
            # Added indexes on frequently queried columns (session_id and timestamp_ms)
            # This transforms O(N) full table scans into O(log N) lookups when retrieving
            # timeline data and performing range queries, significantly improving retrieval times
            # for sessions with high event volumes.

            # Indexes for performance_events
            conn.execute('CREATE INDEX IF NOT EXISTS idx_perf_events_session ON performance_events(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_perf_events_time ON performance_events(timestamp_ms)')

            # Indexes for system_metrics
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sys_metrics_session ON system_metrics(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sys_metrics_time ON system_metrics(timestamp_ms)')

            # Indexes for layer_timings
            conn.execute('CREATE INDEX IF NOT EXISTS idx_layer_timings_session ON layer_timings(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_layer_timings_time ON layer_timings(timestamp_ms)')

            # Indexes for timeline_markers
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timeline_markers_session ON timeline_markers(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timeline_markers_time ON timeline_markers(timestamp_ms)')

    def start_recording(self, session_id: str = None) -> Dict[str, Any]:
        """Start performance timeline recording"""
        try:
            if self.recording:
                return {"status": "error", "message": "Recording already in progress"}

            self.session_id = session_id or f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.start_time = time.time()
            self.recording = True
            self.stop_monitoring.clear()

            # Clear previous data
            self.layer_timings.clear()
            self.memory_timeline.clear()
            self.gpu_timeline.clear()
            self.cpu_timeline.clear()
            self.io_timeline.clear()
            self.events.clear()
            self.markers.clear()

            # Create session record
            with sqlite3.connect(self.timeline_db_path) as conn:
                conn.execute('''
                    INSERT INTO timeline_sessions
                    (session_id, start_time, metadata)
                    VALUES (?, ?, ?)
                ''', (self.session_id, datetime.now(), json.dumps({"pid": os.getpid()})))

            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()

            # Add session start marker
            self.add_marker("session_start", "Recording Started", "Performance timeline recording started")

            return {
                "status": "success",
                "session_id": self.session_id,
                "message": "Timeline recording started"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_recording(self) -> Dict[str, Any]:
        """Stop performance timeline recording"""
        try:
            if not self.recording:
                return {"status": "error", "message": "No recording in progress"}

            self.recording = False
            self.stop_monitoring.set()

            # Wait for monitoring thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)

            end_time = time.time()
            duration_ms = int((end_time - self.start_time) * 1000)

            # Add session end marker
            self.add_marker("session_end", "Recording Stopped", "Performance timeline recording stopped")

            # Update session record
            with sqlite3.connect(self.timeline_db_path) as conn:
                conn.execute('''
                    UPDATE timeline_sessions
                    SET end_time = ?, duration_ms = ?, total_events = ?
                    WHERE session_id = ?
                ''', (datetime.now(), duration_ms, len(self.events), self.session_id))

            # Process and save collected data
            self._save_timeline_data()

            return {
                "status": "success",
                "session_id": self.session_id,
                "duration_ms": duration_ms,
                "total_events": len(self.events),
                "message": "Timeline recording stopped"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def add_event(self, event_type: str, event_name: str, duration_ms: float = 0,
                  category: str = "general", details: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a performance event to the timeline"""
        try:
            if not self.recording:
                return {"status": "info", "message": "Not recording"}

            timestamp_ms = int((time.time() - self.start_time) * 1000)

            event = {
                "timestamp_ms": timestamp_ms,
                "event_type": event_type,
                "event_name": event_name,
                "duration_ms": duration_ms,
                "category": category,
                "thread_name": threading.current_thread().name,
                "process_id": os.getpid(),
                "details": details or {}
            }

            self.events.append(event)

            # Store in database
            with sqlite3.connect(self.timeline_db_path) as conn:
                conn.execute('''
                    INSERT INTO performance_events
                    (session_id, timestamp_ms, event_type, event_name, duration_ms,
                     category, thread_name, process_id, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.session_id,
                    timestamp_ms,
                    event_type,
                    event_name,
                    duration_ms,
                    category,
                    threading.current_thread().name,
                    os.getpid(),
                    json.dumps(details or {})
                ))

            return {"status": "success", "event_id": len(self.events) - 1}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def add_marker(self, marker_type: str, label: str, description: str = "",
                   line_number: int = None, file_path: str = None) -> Dict[str, Any]:
        """Add a timeline marker"""
        try:
            if not self.recording:
                return {"status": "info", "message": "Not recording"}

            timestamp_ms = int((time.time() - self.start_time) * 1000)

            marker = {
                "timestamp_ms": timestamp_ms,
                "marker_type": marker_type,
                "label": label,
                "description": description,
                "line_number": line_number,
                "file_path": file_path
            }

            self.markers.append(marker)

            # Store in database
            with sqlite3.connect(self.timeline_db_path) as conn:
                conn.execute('''
                    INSERT INTO timeline_markers
                    (session_id, timestamp_ms, marker_type, label, description,
                     line_number, file_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.session_id,
                    timestamp_ms,
                    marker_type,
                    label,
                    description,
                    line_number,
                    file_path,
                    json.dumps(marker)
                ))

            return {"status": "success", "marker_id": len(self.markers) - 1}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def log_layer_timing(self, layer_name: str, operation_type: str,
                        forward_time_ms: float = None, backward_time_ms: float = None,
                        memory_allocated_mb: float = None, memory_freed_mb: float = None,
                        input_shapes: List = None, output_shapes: List = None,
                        parameters_count: int = None) -> Dict[str, Any]:
        """Log layer-wise timing information"""
        try:
            if not self.recording:
                return {"status": "info", "message": "Not recording"}

            timestamp_ms = int((time.time() - self.start_time) * 1000)

            # Store timing data
            timing_data = {
                "timestamp_ms": timestamp_ms,
                "layer_name": layer_name,
                "operation_type": operation_type,
                "forward_time_ms": forward_time_ms,
                "backward_time_ms": backward_time_ms,
                "memory_allocated_mb": memory_allocated_mb,
                "memory_freed_mb": memory_freed_mb,
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
                "parameters_count": parameters_count
            }

            self.layer_timings[layer_name].append(timing_data)

            # Store in database
            with sqlite3.connect(self.timeline_db_path) as conn:
                conn.execute('''
                    INSERT INTO layer_timings
                    (session_id, timestamp_ms, layer_name, operation_type,
                     forward_time_ms, backward_time_ms, memory_allocated_mb, memory_freed_mb,
                     input_shapes, output_shapes, parameters_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.session_id,
                    timestamp_ms,
                    layer_name,
                    operation_type,
                    forward_time_ms,
                    backward_time_ms,
                    memory_allocated_mb,
                    memory_freed_mb,
                    json.dumps(input_shapes) if input_shapes else None,
                    json.dumps(output_shapes) if output_shapes else None,
                    parameters_count
                ))

            return {"status": "success"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _monitoring_loop(self):
        """Background monitoring loop for system metrics"""
        try:
            last_io_stats = psutil.disk_io_counters()
            last_net_stats = psutil.net_io_counters()

            while not self.stop_monitoring.is_set():
                timestamp_ms = int((time.time() - self.start_time) * 1000)

                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_used_mb = memory.used / (1024 * 1024)
                memory_percent = memory.percent

                # I/O Stats
                io_stats = psutil.disk_io_counters()
                net_stats = psutil.net_io_counters()

                io_read_mb = (io_stats.read_bytes - last_io_stats.read_bytes) / (1024 * 1024)
                io_write_mb = (io_stats.write_bytes - last_io_stats.write_bytes) / (1024 * 1024)

                net_sent_mb = (net_stats.bytes_sent - last_net_stats.bytes_sent) / (1024 * 1024)
                net_recv_mb = (net_stats.bytes_recv - last_net_stats.bytes_recv) / (1024 * 1024)

                last_io_stats = io_stats
                last_net_stats = net_stats

                # GPU Stats
                gpu_utilization = 0
                gpu_memory_used_mb = 0
                gpu_memory_percent = 0

                if GPU_AVAILABLE and self.gpu_handles:
                    try:
                        for handle in self.gpu_handles:
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                            gpu_utilization = max(gpu_utilization, util.gpu)
                            gpu_memory_used_mb += memory_info.used / (1024 * 1024)
                            gpu_memory_percent = max(gpu_memory_percent,
                                                   (memory_info.used / memory_info.total) * 100)
                    except Exception as e:
                        logger.warning(f"GPU monitoring error: {e}")

                # Store metrics
                metrics = {
                    "timestamp_ms": timestamp_ms,
                    "cpu_percent": cpu_percent,
                    "memory_used_mb": memory_used_mb,
                    "memory_percent": memory_percent,
                    "gpu_utilization": gpu_utilization,
                    "gpu_memory_used_mb": gpu_memory_used_mb,
                    "gpu_memory_percent": gpu_memory_percent,
                    "io_read_mb": io_read_mb,
                    "io_write_mb": io_write_mb,
                    "network_sent_mb": net_sent_mb,
                    "network_recv_mb": net_recv_mb
                }

                # Add to timelines
                self.cpu_timeline.append({"timestamp_ms": timestamp_ms, "value": cpu_percent})
                self.memory_timeline.append({"timestamp_ms": timestamp_ms, "value": memory_used_mb})
                self.gpu_timeline.append({"timestamp_ms": timestamp_ms, "value": gpu_utilization})
                self.io_timeline.append({"timestamp_ms": timestamp_ms, "read": io_read_mb, "write": io_write_mb})

                # Store in database
                try:
                    with sqlite3.connect(self.timeline_db_path) as conn:
                        conn.execute('''
                            INSERT INTO system_metrics
                            (session_id, timestamp_ms, cpu_percent, memory_used_mb, memory_percent,
                             gpu_utilization, gpu_memory_used_mb, gpu_memory_percent,
                             io_read_mb, io_write_mb, network_sent_mb, network_recv_mb)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            self.session_id,
                            timestamp_ms,
                            cpu_percent,
                            memory_used_mb,
                            memory_percent,
                            gpu_utilization,
                            gpu_memory_used_mb,
                            gpu_memory_percent,
                            io_read_mb,
                            io_write_mb,
                            net_sent_mb,
                            net_recv_mb
                        ))
                except Exception as e:
                    logger.warning(f"Database error in monitoring loop: {e}")

                # Sleep for monitoring interval
                time.sleep(0.5)  # 500ms interval

        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")

    def _save_timeline_data(self):
        """Save collected timeline data to database"""
        try:
            # Data is already being saved in real-time, this is for any final processing
            pass
        except Exception as e:
            logger.error(f"Error saving timeline data: {e}")

    def get_timeline_data(self, session_id: str = None,
                         time_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """Get timeline data for visualization"""
        try:
            target_session = session_id or self.session_id
            if not target_session:
                return {"status": "error", "message": "No session specified"}

            with sqlite3.connect(self.timeline_db_path) as conn:
                # Get session info
                session_info = conn.execute('''
                    SELECT * FROM timeline_sessions WHERE session_id = ?
                ''', (target_session,)).fetchone()

                if not session_info:
                    return {"status": "error", "message": "Session not found"}

                # Build time range filter
                time_filter = ""
                time_params = [target_session]
                if time_range:
                    time_filter = "AND timestamp_ms BETWEEN ? AND ?"
                    time_params.extend(time_range)

                # Get events
                events = conn.execute(f'''
                    SELECT * FROM performance_events
                    WHERE session_id = ? {time_filter}
                    ORDER BY timestamp_ms
                ''', time_params).fetchall()

                # Get system metrics
                metrics = conn.execute(f'''
                    SELECT * FROM system_metrics
                    WHERE session_id = ? {time_filter}
                    ORDER BY timestamp_ms
                ''', time_params).fetchall()

                # Get layer timings
                layer_timings = conn.execute(f'''
                    SELECT * FROM layer_timings
                    WHERE session_id = ? {time_filter}
                    ORDER BY timestamp_ms
                ''', time_params).fetchall()

                # Get markers
                markers = conn.execute(f'''
                    SELECT * FROM timeline_markers
                    WHERE session_id = ? {time_filter}
                    ORDER BY timestamp_ms
                ''', time_params).fetchall()

            # Format data for frontend
            timeline_data = {
                "session_info": {
                    "session_id": session_info[1],
                    "start_time": session_info[2],
                    "end_time": session_info[3],
                    "duration_ms": session_info[4],
                    "total_events": session_info[5]
                },
                "events": [
                    {
                        "timestamp_ms": event[2],
                        "event_type": event[3],
                        "event_name": event[4],
                        "duration_ms": event[5],
                        "category": event[6],
                        "thread_name": event[7],
                        "process_id": event[8],
                        "details": json.loads(event[9]) if event[9] else {}
                    }
                    for event in events
                ],
                "system_metrics": [
                    {
                        "timestamp_ms": metric[2],
                        "cpu_percent": metric[3],
                        "memory_used_mb": metric[4],
                        "memory_percent": metric[5],
                        "gpu_utilization": metric[6],
                        "gpu_memory_used_mb": metric[7],
                        "gpu_memory_percent": metric[8],
                        "io_read_mb": metric[9],
                        "io_write_mb": metric[10],
                        "network_sent_mb": metric[11],
                        "network_recv_mb": metric[12]
                    }
                    for metric in metrics
                ],
                "layer_timings": [
                    {
                        "timestamp_ms": timing[2],
                        "layer_name": timing[3],
                        "operation_type": timing[4],
                        "forward_time_ms": timing[5],
                        "backward_time_ms": timing[6],
                        "memory_allocated_mb": timing[7],
                        "memory_freed_mb": timing[8],
                        "input_shapes": json.loads(timing[9]) if timing[9] else None,
                        "output_shapes": json.loads(timing[10]) if timing[10] else None,
                        "parameters_count": timing[11]
                    }
                    for timing in layer_timings
                ],
                "markers": [
                    {
                        "timestamp_ms": marker[2],
                        "marker_type": marker[3],
                        "label": marker[4],
                        "description": marker[5],
                        "line_number": marker[6],
                        "file_path": marker[7]
                    }
                    for marker in markers
                ]
            }

            return {
                "status": "success",
                "timeline": timeline_data
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def analyze_performance_bottlenecks(self, session_id: str = None) -> Dict[str, Any]:
        """Analyze performance data to identify bottlenecks"""
        try:
            target_session = session_id or self.session_id
            if not target_session:
                return {"status": "error", "message": "No session specified"}

            timeline_data = self.get_timeline_data(target_session)
            if timeline_data["status"] != "success":
                return timeline_data

            timeline = timeline_data["timeline"]

            # Analyze bottlenecks
            bottlenecks = []
            recommendations = []

            # CPU bottlenecks
            cpu_data = [m["cpu_percent"] for m in timeline["system_metrics"]]
            if cpu_data:
                avg_cpu = sum(cpu_data) / len(cpu_data)
                max_cpu = max(cpu_data)

                if avg_cpu > 90:
                    bottlenecks.append({
                        "type": "cpu_high_usage",
                        "severity": "high",
                        "description": f"High average CPU usage: {avg_cpu:.1f}%",
                        "recommendation": "Consider reducing data preprocessing complexity or using GPU acceleration"
                    })

                if max_cpu > 95:
                    bottlenecks.append({
                        "type": "cpu_spike",
                        "severity": "medium",
                        "description": f"CPU usage spikes to {max_cpu:.1f}%",
                        "recommendation": "Check for blocking operations or inefficient data loading"
                    })

            # Memory bottlenecks
            memory_data = [m["memory_percent"] for m in timeline["system_metrics"]]
            if memory_data:
                avg_memory = sum(memory_data) / len(memory_data)
                max_memory = max(memory_data)

                if avg_memory > 85:
                    bottlenecks.append({
                        "type": "memory_high_usage",
                        "severity": "high",
                        "description": f"High average memory usage: {avg_memory:.1f}%",
                        "recommendation": "Reduce batch size or enable gradient checkpointing"
                    })

                if max_memory > 95:
                    bottlenecks.append({
                        "type": "memory_spike",
                        "severity": "critical",
                        "description": f"Memory usage spikes to {max_memory:.1f}%",
                        "recommendation": "Risk of OOM errors. Reduce model size or batch size immediately"
                    })

            # GPU bottlenecks
            gpu_data = [m["gpu_utilization"] for m in timeline["system_metrics"] if m["gpu_utilization"] > 0]
            if gpu_data:
                avg_gpu = sum(gpu_data) / len(gpu_data)

                if avg_gpu < 70:
                    bottlenecks.append({
                        "type": "gpu_underutilization",
                        "severity": "medium",
                        "description": f"Low GPU utilization: {avg_gpu:.1f}%",
                        "recommendation": "Increase batch size or check for CPU-GPU transfer bottlenecks"
                    })

            # I/O bottlenecks
            io_read_data = [m["io_read_mb"] for m in timeline["system_metrics"]]
            if io_read_data:
                max_io_read = max(io_read_data)

                if max_io_read > 100:  # More than 100MB/s reads
                    bottlenecks.append({
                        "type": "io_bottleneck",
                        "severity": "medium",
                        "description": f"High I/O read activity: {max_io_read:.1f} MB/s",
                        "recommendation": "Use data caching, faster storage, or increase num_workers in DataLoader"
                    })

            # Layer timing analysis
            layer_timings = timeline["layer_timings"]
            if layer_timings:
                # Find slowest layers
                layer_avg_times = defaultdict(list)
                for timing in layer_timings:
                    if timing["forward_time_ms"]:
                        layer_avg_times[timing["layer_name"]].append(timing["forward_time_ms"])

                slow_layers = []
                for layer_name, times in layer_avg_times.items():
                    avg_time = sum(times) / len(times)
                    if avg_time > 10:  # More than 10ms average
                        slow_layers.append((layer_name, avg_time))

                slow_layers.sort(key=lambda x: x[1], reverse=True)

                for layer_name, avg_time in slow_layers[:3]:  # Top 3 slow layers
                    bottlenecks.append({
                        "type": "slow_layer",
                        "severity": "medium",
                        "description": f"Layer '{layer_name}' averages {avg_time:.1f}ms",
                        "recommendation": f"Optimize layer '{layer_name}' implementation or reduce complexity"
                    })

            # General recommendations
            recommendations.extend([
                "Use mixed precision training (fp16) to reduce memory and increase speed",
                "Enable data loader prefetching and pin_memory=True",
                "Use torch.compile() for PyTorch 2.0+ performance improvements",
                "Consider gradient accumulation for effective larger batch sizes",
                "Profile individual operations with torch.profiler for detailed analysis"
            ])

            analysis = {
                "session_id": target_session,
                "bottlenecks": bottlenecks,
                "recommendations": recommendations[:5],  # Top 5
                "performance_summary": {
                    "avg_cpu_usage": sum(cpu_data) / len(cpu_data) if cpu_data else 0,
                    "avg_memory_usage": sum(memory_data) / len(memory_data) if memory_data else 0,
                    "avg_gpu_usage": sum(gpu_data) / len(gpu_data) if gpu_data else 0,
                    "total_events": len(timeline["events"]),
                    "total_markers": len(timeline["markers"])
                }
            }

            return {
                "status": "success",
                "analysis": analysis
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def export_timeline(self, session_id: str = None, format: str = "json") -> Dict[str, Any]:
        """Export timeline data in various formats"""
        try:
            target_session = session_id or self.session_id
            if not target_session:
                return {"status": "error", "message": "No session specified"}

            timeline_data = self.get_timeline_data(target_session)
            if timeline_data["status"] != "success":
                return timeline_data

            if format == "json":
                export_path = os.path.join(self.workspace_path, f"timeline_{target_session}.json")
                with open(export_path, 'w') as f:
                    json.dump(timeline_data["timeline"], f, indent=2, default=str)

                return {
                    "status": "success",
                    "export_path": export_path,
                    "format": "json"
                }

            elif format == "chrome_tracing":
                # Export in Chrome Tracing format for Chrome DevTools
                export_path = os.path.join(self.workspace_path, f"timeline_{target_session}_trace.json")

                chrome_events = []
                timeline = timeline_data["timeline"]

                # Convert events to Chrome Tracing format
                for event in timeline["events"]:
                    chrome_events.append({
                        "name": event["event_name"],
                        "cat": event["category"],
                        "ph": "X",  # Complete event
                        "ts": event["timestamp_ms"] * 1000,  # Chrome expects microseconds
                        "dur": event["duration_ms"] * 1000,
                        "pid": event["process_id"],
                        "tid": hash(event["thread_name"]) % 1000,
                        "args": event["details"]
                    })

                # Add markers as instant events
                for marker in timeline["markers"]:
                    chrome_events.append({
                        "name": marker["label"],
                        "cat": "marker",
                        "ph": "i",  # Instant event
                        "ts": marker["timestamp_ms"] * 1000,
                        "pid": 1,
                        "tid": 1,
                        "args": {"description": marker["description"]}
                    })

                with open(export_path, 'w') as f:
                    json.dump({"traceEvents": chrome_events}, f)

                return {
                    "status": "success",
                    "export_path": export_path,
                    "format": "chrome_tracing",
                    "instructions": "Load this file in Chrome DevTools Performance tab"
                }

            else:
                return {"status": "error", "message": f"Unsupported format: {format}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_sessions(self) -> Dict[str, Any]:
        """List all timeline sessions"""
        try:
            with sqlite3.connect(self.timeline_db_path) as conn:
                sessions = conn.execute('''
                    SELECT session_id, start_time, end_time, duration_ms, total_events
                    FROM timeline_sessions
                    ORDER BY start_time DESC
                    LIMIT 20
                ''').fetchall()

                session_list = [
                    {
                        "session_id": session[0],
                        "start_time": session[1],
                        "end_time": session[2],
                        "duration_ms": session[3],
                        "total_events": session[4]
                    }
                    for session in sessions
                ]

                return {
                    "status": "success",
                    "sessions": session_list
                }

        except Exception as e:
            return {"status": "error", "message": str(e)}

def main():
    """Main entry point for Performance Timeline"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Command required"}))
        return

    command = sys.argv[1]
    workspace_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        timeline = PerformanceTimeline(workspace_path)

        if command == "start_recording":
            session_id = sys.argv[3] if len(sys.argv) > 3 else None
            result = timeline.start_recording(session_id)

        elif command == "stop_recording":
            result = timeline.stop_recording()

        elif command == "add_event":
            event_type = sys.argv[3]
            event_name = sys.argv[4]
            duration_ms = float(sys.argv[5]) if len(sys.argv) > 5 else 0
            category = sys.argv[6] if len(sys.argv) > 6 else "general"
            details = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
            result = timeline.add_event(event_type, event_name, duration_ms, category, details)

        elif command == "add_marker":
            marker_type = sys.argv[3]
            label = sys.argv[4]
            description = sys.argv[5] if len(sys.argv) > 5 else ""
            line_number = int(sys.argv[6]) if len(sys.argv) > 6 else None
            file_path = sys.argv[7] if len(sys.argv) > 7 else None
            result = timeline.add_marker(marker_type, label, description, line_number, file_path)

        elif command == "log_layer_timing":
            layer_name = sys.argv[3]
            operation_type = sys.argv[4]
            forward_time_ms = float(sys.argv[5]) if len(sys.argv) > 5 else None
            backward_time_ms = float(sys.argv[6]) if len(sys.argv) > 6 else None
            result = timeline.log_layer_timing(layer_name, operation_type, forward_time_ms, backward_time_ms)

        elif command == "get_timeline":
            session_id = sys.argv[3] if len(sys.argv) > 3 else None
            result = timeline.get_timeline_data(session_id)

        elif command == "analyze_bottlenecks":
            session_id = sys.argv[3] if len(sys.argv) > 3 else None
            result = timeline.analyze_performance_bottlenecks(session_id)

        elif command == "export_timeline":
            session_id = sys.argv[3] if len(sys.argv) > 3 else None
            format_type = sys.argv[4] if len(sys.argv) > 4 else "json"
            result = timeline.export_timeline(session_id, format_type)

        elif command == "list_sessions":
            result = timeline.list_sessions()

        else:
            result = {"error": f"Unknown command: {command}"}

        print(json.dumps(result, default=str))

    except Exception as e:
        error_result = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    main()