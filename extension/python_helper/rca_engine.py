#!/usr/bin/env python3
"""
Auto-Root Cause Analysis (RCA) Engine for AI/ML Debugger
Analyzes training failures and provides intelligent diagnostics
"""

import numpy as np
import json
import sys
import traceback
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingFailurePattern:
    """Represents a pattern of training failure"""

    def __init__(self, name: str, description: str, symptoms: List[str],
                 recommendations: List[str], confidence_threshold: float = 0.7):
        self.name = name
        self.description = description
        self.symptoms = symptoms
        self.recommendations = recommendations
        self.confidence_threshold = confidence_threshold
        self.matches = 0
        self.total_checks = 0

class RCAEngine:
    """Root Cause Analysis Engine for ML training issues"""

    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path or os.getcwd()
        self.rca_db_path = os.path.join(self.workspace_path, ".vscode", "ai-debugger-rca.db")
        self.patterns_db_path = os.path.join(self.workspace_path, ".vscode", "failure-patterns.json")

        self.ensure_database()
        self.load_failure_patterns()

        # Training session state
        self.current_session = None
        self.metrics_history = []
        self.gradient_history = []
        self.loss_history = []
        self.anomaly_scores = []

    def ensure_database(self):
        """Create RCA database if it doesn't exist"""
        os.makedirs(os.path.dirname(self.rca_db_path), exist_ok=True)

        with sqlite3.connect(self.rca_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS rca_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT,
                    failure_type TEXT,
                    root_cause TEXT,
                    confidence REAL,
                    recommendations TEXT,
                    metadata TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    epoch INTEGER,
                    step INTEGER,
                    timestamp TIMESTAMP,
                    loss REAL,
                    accuracy REAL,
                    gradient_norm REAL,
                    learning_rate REAL,
                    memory_usage REAL,
                    gpu_utilization REAL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES rca_sessions (session_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TIMESTAMP,
                    anomaly_type TEXT,
                    severity TEXT,
                    description TEXT,
                    affected_layers TEXT,
                    confidence REAL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES rca_sessions (session_id)
                )
            ''')

    def load_failure_patterns(self):
        """Load known failure patterns"""
        self.failure_patterns = [
            TrainingFailurePattern(
                name="exploding_gradients",
                description="Gradients are growing exponentially during training",
                symptoms=[
                    "gradient_norm > 100",
                    "loss_increase_rate > 10",
                    "nan_in_gradients",
                    "model_weights_exploding"
                ],
                recommendations=[
                    "Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
                    "Reduce learning rate by factor of 10",
                    "Check for unstable activation functions (avoid large inputs to exp, sigmoid)",
                    "Consider using gradient accumulation instead of large batch sizes"
                ]
            ),

            TrainingFailurePattern(
                name="vanishing_gradients",
                description="Gradients are becoming too small to update weights effectively",
                symptoms=[
                    "gradient_norm < 1e-7",
                    "loss_plateau > 10_epochs",
                    "early_layer_gradients_near_zero",
                    "no_improvement_accuracy"
                ],
                recommendations=[
                    "Use residual connections (ResNet-style) or dense connections",
                    "Switch to ReLU or LeakyReLU activations instead of sigmoid/tanh",
                    "Consider using batch normalization or layer normalization",
                    "Initialize weights with Xavier/He initialization",
                    "Try different optimizers: Adam, RMSprop instead of SGD"
                ]
            ),

            TrainingFailurePattern(
                name="learning_rate_too_high",
                description="Learning rate is causing training instability",
                symptoms=[
                    "loss_oscillating",
                    "accuracy_fluctuating",
                    "gradient_norm_unstable",
                    "early_divergence"
                ],
                recommendations=[
                    "Reduce learning rate by factor of 10: lr = lr / 10",
                    "Use learning rate scheduling: StepLR, CosineAnnealingLR",
                    "Try adaptive optimizers: Adam, AdamW",
                    "Use warm-up phase for learning rate"
                ]
            ),

            TrainingFailurePattern(
                name="learning_rate_too_low",
                description="Learning rate is too small, causing slow convergence",
                symptoms=[
                    "loss_plateau_early",
                    "minimal_weight_updates",
                    "slow_convergence",
                    "gradient_norm_consistent_but_small"
                ],
                recommendations=[
                    "Increase learning rate by factor of 2-5",
                    "Use learning rate finder to identify optimal range",
                    "Consider cyclic learning rates",
                    "Check if model architecture is appropriate for the problem"
                ]
            ),

            TrainingFailurePattern(
                name="data_leakage",
                description="Training and validation data overlap or target leakage",
                symptoms=[
                    "perfect_training_accuracy",
                    "poor_validation_performance",
                    "val_loss_much_higher_than_train",
                    "model_memorizing"
                ],
                recommendations=[
                    "Check data splitting: ensure no overlap between train/val/test",
                    "Remove features that leak target information",
                    "Use temporal splits for time-series data",
                    "Apply proper cross-validation techniques"
                ]
            ),

            TrainingFailurePattern(
                name="overfitting",
                description="Model is memorizing training data instead of generalizing",
                symptoms=[
                    "train_val_loss_diverging",
                    "validation_accuracy_decreasing",
                    "perfect_training_metrics",
                    "high_model_complexity"
                ],
                recommendations=[
                    "Add regularization: L1/L2 weight decay",
                    "Use dropout layers: nn.Dropout(0.2-0.5)",
                    "Reduce model complexity: fewer layers/parameters",
                    "Increase training data or use data augmentation",
                    "Early stopping based on validation loss"
                ]
            ),

            TrainingFailurePattern(
                name="underfitting",
                description="Model is too simple to capture underlying patterns",
                symptoms=[
                    "both_train_val_loss_high",
                    "slow_learning",
                    "low_model_capacity",
                    "simple_patterns_not_learned"
                ],
                recommendations=[
                    "Increase model complexity: more layers/neurons",
                    "Use more sophisticated architectures",
                    "Reduce regularization strength",
                    "Train for more epochs",
                    "Check feature engineering and data preprocessing"
                ]
            ),

            TrainingFailurePattern(
                name="dead_relu_problem",
                description="ReLU neurons are dying and not activating",
                symptoms=[
                    "many_zero_activations",
                    "sparse_gradients",
                    "decreasing_active_neurons",
                    "poor_feature_learning"
                ],
                recommendations=[
                    "Use Leaky ReLU instead: nn.LeakyReLU(0.01)",
                    "Try ELU or Swish activations",
                    "Reduce learning rate",
                    "Use proper weight initialization",
                    "Add batch normalization"
                ]
            ),

            TrainingFailurePattern(
                name="batch_size_issues",
                description="Batch size is causing training problems",
                symptoms=[
                    "noisy_gradient_updates",
                    "slow_convergence",
                    "memory_issues",
                    "unstable_training"
                ],
                recommendations=[
                    "For small batches: increase batch size or use batch accumulation",
                    "For large batches: use learning rate scaling or warm-up",
                    "Try different batch sizes: 16, 32, 64, 128",
                    "Use gradient accumulation for effective larger batches"
                ]
            ),

            TrainingFailurePattern(
                name="data_preprocessing_issues",
                description="Data preprocessing is causing training problems",
                symptoms=[
                    "unusual_loss_patterns",
                    "poor_initial_performance",
                    "training_not_improving",
                    "weird_gradient_distributions"
                ],
                recommendations=[
                    "Check data normalization: StandardScaler, MinMaxScaler",
                    "Verify data loading: check shapes, dtypes, ranges",
                    "Inspect data augmentation: ensure it's not too aggressive",
                    "Check for NaN or infinite values in input data"
                ]
            )
        ]

        # Load custom patterns if they exist
        if os.path.exists(self.patterns_db_path):
            try:
                with open(self.patterns_db_path, 'r') as f:
                    custom_patterns_data = json.load(f)

                    for pattern_data in custom_patterns_data:
                        pattern = TrainingFailurePattern(
                            name=pattern_data.get('name', 'Unknown'),
                            description=pattern_data.get('description', ''),
                            symptoms=pattern_data.get('symptoms', []),
                            recommendations=pattern_data.get('recommendations', []),
                            confidence_threshold=pattern_data.get('confidence_threshold', 0.7)
                        )
                        self.failure_patterns.append(pattern)
            except Exception as e:
                logger.warning(f"Could not load custom patterns: {e}")

    def start_session(self, session_id: str) -> Dict[str, Any]:
        """Start a new RCA session"""
        try:
            self.current_session = session_id
            self.metrics_history = []
            self.gradient_history = []
            self.loss_history = []
            self.anomaly_scores = []

            with sqlite3.connect(self.rca_db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO rca_sessions
                    (session_id, start_time, status)
                    VALUES (?, ?, ?)
                ''', (session_id, datetime.now(), 'active'))

            return {
                "status": "success",
                "session_id": session_id,
                "message": "RCA session started"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def log_training_step(self, session_id: str, epoch: int, step: int,
                         metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Log training step metrics for analysis"""
        try:
            # Store metrics
            timestamp = datetime.now()

            with sqlite3.connect(self.rca_db_path) as conn:
                conn.execute('''
                    INSERT INTO training_metrics
                    (session_id, epoch, step, timestamp, loss, accuracy,
                     gradient_norm, learning_rate, memory_usage, gpu_utilization, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id, epoch, step, timestamp,
                    metrics.get('loss', 0),
                    metrics.get('accuracy', 0),
                    metrics.get('gradient_norm', 0),
                    metrics.get('learning_rate', 0),
                    metrics.get('memory_usage', 0),
                    metrics.get('gpu_utilization', 0),
                    json.dumps(metrics)
                ))

            # Update history
            self.metrics_history.append(metrics)
            if 'loss' in metrics:
                self.loss_history.append(metrics['loss'])

            # Perform real-time anomaly detection
            anomalies = self.detect_anomalies(metrics, epoch, step)

            return {
                "status": "success",
                "logged": True,
                "anomalies": anomalies
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def detect_anomalies(self, current_metrics: Dict[str, Any],
                        epoch: int, step: int) -> List[Dict[str, Any]]:
        """Real-time anomaly detection"""
        anomalies = []

        try:
            # Check for NaN or infinite values
            for key, value in current_metrics.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        anomalies.append({
                            "type": "invalid_value",
                            "severity": "critical",
                            "description": f"{key} contains NaN or infinite value",
                            "value": value,
                            "recommendation": f"Check computation of {key}, likely numerical instability"
                        })

            # Check gradient explosion
            if 'gradient_norm' in current_metrics:
                grad_norm = current_metrics['gradient_norm']
                if grad_norm > 100:
                    anomalies.append({
                        "type": "exploding_gradients",
                        "severity": "critical",
                        "description": f"Gradient norm ({grad_norm:.2f}) indicates exploding gradients",
                        "recommendation": "Apply gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)"
                    })
                elif grad_norm < 1e-7:
                    anomalies.append({
                        "type": "vanishing_gradients",
                        "severity": "warning",
                        "description": f"Gradient norm ({grad_norm:.2e}) indicates vanishing gradients",
                        "recommendation": "Consider residual connections, different activation functions, or weight initialization"
                    })

            # Check loss trends
            if len(self.loss_history) > 5:
                recent_losses = self.loss_history[-5:]
                if all(l1 < l2 for l1, l2 in zip(recent_losses[:-1], recent_losses[1:])):
                    anomalies.append({
                        "type": "increasing_loss",
                        "severity": "warning",
                        "description": "Loss has been increasing for 5 consecutive steps",
                        "recommendation": "Consider reducing learning rate or checking data quality"
                    })

            # Check for training stagnation
            if len(self.loss_history) > 20:
                recent_losses = self.loss_history[-20:]
                loss_variance = np.var(recent_losses)
                if loss_variance < 1e-8:
                    anomalies.append({
                        "type": "loss_plateau",
                        "severity": "warning",
                        "description": "Loss has plateaued for 20 steps",
                        "recommendation": "Consider increasing learning rate, changing optimizer, or checking model capacity"
                    })

            # Store anomalies in database
            if anomalies and self.current_session:
                with sqlite3.connect(self.rca_db_path) as conn:
                    for anomaly in anomalies:
                        conn.execute('''
                            INSERT INTO anomaly_detections
                            (session_id, timestamp, anomaly_type, severity, description, confidence, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            self.current_session,
                            datetime.now(),
                            anomaly['type'],
                            anomaly['severity'],
                            anomaly['description'],
                            0.8,  # Default confidence
                            json.dumps(anomaly)
                        ))

            return anomalies

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return []

    def analyze_failure(self, session_id: str, failure_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive failure analysis"""
        try:
            # Get session metrics
            with sqlite3.connect(self.rca_db_path) as conn:
                metrics_data = conn.execute('''
                    SELECT * FROM training_metrics
                    WHERE session_id = ?
                    ORDER BY timestamp
                ''', (session_id,)).fetchall()

                anomalies_data = conn.execute('''
                    SELECT * FROM anomaly_detections
                    WHERE session_id = ?
                    ORDER BY timestamp
                ''', (session_id,)).fetchall()

            # Pattern matching
            matched_patterns = []
            for pattern in self.failure_patterns:
                confidence = self.calculate_pattern_confidence(pattern, failure_info, metrics_data, anomalies_data)
                if confidence > pattern.confidence_threshold:
                    matched_patterns.append({
                        "pattern": pattern.name,
                        "description": pattern.description,
                        "confidence": confidence,
                        "recommendations": pattern.recommendations
                    })

            # Sort by confidence
            matched_patterns.sort(key=lambda x: x['confidence'], reverse=True)

            # Generate comprehensive analysis
            analysis = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "failure_analysis": {
                    "primary_cause": matched_patterns[0] if matched_patterns else None,
                    "secondary_causes": matched_patterns[1:3],
                    "confidence_score": matched_patterns[0]['confidence'] if matched_patterns else 0.0
                },
                "metrics_summary": self.summarize_metrics(metrics_data),
                "anomalies_summary": self.summarize_anomalies(anomalies_data),
                "recommendations": self.generate_recommendations(matched_patterns),
                "preventive_measures": self.suggest_preventive_measures(matched_patterns)
            }

            # Update session with results
            with sqlite3.connect(self.rca_db_path) as conn:
                conn.execute('''
                    UPDATE rca_sessions
                    SET end_time = ?, status = ?, failure_type = ?,
                        root_cause = ?, confidence = ?, recommendations = ?
                    WHERE session_id = ?
                ''', (
                    datetime.now(),
                    'analyzed',
                    matched_patterns[0]['pattern'] if matched_patterns else 'unknown',
                    matched_patterns[0]['description'] if matched_patterns else 'No clear pattern found',
                    matched_patterns[0]['confidence'] if matched_patterns else 0.0,
                    json.dumps(analysis['recommendations']),
                    session_id
                ))

            return {
                "status": "success",
                "analysis": analysis
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def calculate_pattern_confidence(self, pattern: TrainingFailurePattern,
                                   failure_info: Dict[str, Any],
                                   metrics_data: List, anomalies_data: List) -> float:
        """Calculate confidence score for a failure pattern"""
        try:
            matches = 0
            total_symptoms = len(pattern.symptoms)

            for symptom in pattern.symptoms:
                if self.check_symptom(symptom, failure_info, metrics_data, anomalies_data):
                    matches += 1

            confidence = matches / total_symptoms if total_symptoms > 0 else 0.0
            return confidence

        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {e}")
            return 0.0

    def check_symptom(self, symptom: str, failure_info: Dict[str, Any],
                     metrics_data: List, anomalies_data: List) -> bool:
        """Check if a specific symptom is present"""
        try:
            # Simple heuristic checking
            if symptom == "gradient_norm > 100":
                return any(float(row[7]) > 100 for row in metrics_data if row[7] is not None)
            elif symptom == "gradient_norm < 1e-7":
                return any(float(row[7]) < 1e-7 for row in metrics_data if row[7] is not None)
            elif symptom == "loss_increase_rate > 10":
                if len(metrics_data) > 1:
                    losses = [float(row[5]) for row in metrics_data if row[5] is not None]
                    if len(losses) > 1:
                        increases = [l2/l1 for l1, l2 in zip(losses[:-1], losses[1:]) if l1 > 0]
                        return any(inc > 10 for inc in increases)
            elif symptom == "nan_in_gradients":
                return "nan" in str(failure_info).lower() or any("nan" in str(row).lower() for row in anomalies_data)
            elif symptom == "loss_plateau > 10_epochs":
                # Check if loss variance is very low over many steps
                if len(metrics_data) > 50:
                    recent_losses = [float(row[5]) for row in metrics_data[-50:] if row[5] is not None]
                    if len(recent_losses) > 10:
                        return np.var(recent_losses) < 1e-6
            # Add more symptom checks as needed

            return False

        except Exception as e:
            logger.error(f"Error checking symptom {symptom}: {e}")
            return False

    def summarize_metrics(self, metrics_data: List) -> Dict[str, Any]:
        """Summarize training metrics"""
        if not metrics_data:
            return {}

        try:
            losses = [float(row[5]) for row in metrics_data if row[5] is not None]
            grad_norms = [float(row[7]) for row in metrics_data if row[7] is not None]

            return {
                "total_steps": len(metrics_data),
                "loss_trend": {
                    "initial": losses[0] if losses else None,
                    "final": losses[-1] if losses else None,
                    "min": min(losses) if losses else None,
                    "max": max(losses) if losses else None,
                    "variance": float(np.var(losses)) if losses else None
                },
                "gradient_norms": {
                    "mean": float(np.mean(grad_norms)) if grad_norms else None,
                    "max": max(grad_norms) if grad_norms else None,
                    "min": min(grad_norms) if grad_norms else None,
                    "std": float(np.std(grad_norms)) if grad_norms else None
                }
            }
        except Exception as e:
            logger.error(f"Error summarizing metrics: {e}")
            return {}

    def summarize_anomalies(self, anomalies_data: List) -> Dict[str, Any]:
        """Summarize detected anomalies"""
        if not anomalies_data:
            return {"total": 0, "by_type": {}, "by_severity": {}}

        try:
            by_type = {}
            by_severity = {}

            for row in anomalies_data:
                anomaly_type = row[3]
                severity = row[4]

                by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1
                by_severity[severity] = by_severity.get(severity, 0) + 1

            return {
                "total": len(anomalies_data),
                "by_type": by_type,
                "by_severity": by_severity
            }
        except Exception as e:
            logger.error(f"Error summarizing anomalies: {e}")
            return {}

    def generate_recommendations(self, matched_patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations"""
        all_recommendations = []

        for pattern in matched_patterns[:2]:  # Top 2 patterns
            all_recommendations.extend(pattern['recommendations'])

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations[:5]  # Top 5 recommendations

    def suggest_preventive_measures(self, matched_patterns: List[Dict[str, Any]]) -> List[str]:
        """Suggest preventive measures for future training"""
        preventive_measures = [
            "Set up gradient monitoring and clipping from the start",
            "Use learning rate scheduling (e.g., ReduceLROnPlateau)",
            "Implement early stopping based on validation metrics",
            "Add comprehensive logging and monitoring",
            "Use proper weight initialization techniques",
            "Implement data validation checks before training",
            "Set up automated model checkpointing",
            "Monitor GPU memory usage during training"
        ]

        return preventive_measures[:4]  # Return top 4

    def get_session_history(self) -> Dict[str, Any]:
        """Get RCA session history"""
        try:
            with sqlite3.connect(self.rca_db_path) as conn:
                sessions = conn.execute('''
                    SELECT * FROM rca_sessions
                    ORDER BY start_time DESC
                    LIMIT 10
                ''').fetchall()

                session_list = []
                for session in sessions:
                    session_dict = {
                        "session_id": session[1],
                        "start_time": session[2],
                        "end_time": session[3],
                        "status": session[4],
                        "failure_type": session[5],
                        "root_cause": session[6],
                        "confidence": session[7]
                    }
                    session_list.append(session_dict)

                return {
                    "status": "success",
                    "sessions": session_list
                }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def export_analysis_report(self, session_id: str) -> Dict[str, Any]:
        """Export detailed analysis report"""
        try:
            with sqlite3.connect(self.rca_db_path) as conn:
                session_data = conn.execute('''
                    SELECT * FROM rca_sessions WHERE session_id = ?
                ''', (session_id,)).fetchone()

                if not session_data:
                    return {"status": "error", "message": "Session not found"}

                metrics_data = conn.execute('''
                    SELECT * FROM training_metrics WHERE session_id = ?
                ''', (session_id,)).fetchall()

                anomalies_data = conn.execute('''
                    SELECT * FROM anomaly_detections WHERE session_id = ?
                ''', (session_id,)).fetchall()

            report = {
                "session_info": {
                    "session_id": session_data[1],
                    "start_time": session_data[2],
                    "end_time": session_data[3],
                    "status": session_data[4],
                    "failure_type": session_data[5],
                    "root_cause": session_data[6],
                    "confidence": session_data[7]
                },
                "metrics_analysis": self.summarize_metrics(metrics_data),
                "anomalies_analysis": self.summarize_anomalies(anomalies_data),
                "recommendations": json.loads(session_data[8]) if session_data[8] else [],
                "raw_data": {
                    "metrics_count": len(metrics_data),
                    "anomalies_count": len(anomalies_data)
                }
            }

            # Save report to file
            report_path = os.path.join(self.workspace_path, f"rca_report_{session_id}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            return {
                "status": "success",
                "report": report,
                "report_path": report_path
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

def main():
    """Main entry point for RCA Engine"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Command required"}))
        return

    command = sys.argv[1]
    workspace_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        rca_engine = RCAEngine(workspace_path)

        if command == "start_session":
            session_id = sys.argv[3] if len(sys.argv) > 3 else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result = rca_engine.start_session(session_id)

        elif command == "log_step":
            session_id = sys.argv[3]
            epoch = int(sys.argv[4])
            step = int(sys.argv[5])
            metrics = json.loads(sys.argv[6])
            result = rca_engine.log_training_step(session_id, epoch, step, metrics)

        elif command == "analyze_failure":
            session_id = sys.argv[3]
            failure_info = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
            result = rca_engine.analyze_failure(session_id, failure_info)

        elif command == "get_history":
            result = rca_engine.get_session_history()

        elif command == "export_report":
            session_id = sys.argv[3]
            result = rca_engine.export_analysis_report(session_id)

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