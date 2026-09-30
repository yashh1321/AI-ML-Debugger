#!/usr/bin/env python3
"""
Privacy-Aware Training
Differential privacy training with DP-SGD, privacy budget tracking, and utility analysis.
"""

import json
import sys
import os
import numpy as np
import time
import sqlite3

from typing import Dict, List, Any, Tuple
import math
from collections import defaultdict

class PrivacyAccountant:
    """Track privacy budget and compute privacy guarantees"""

    def __init__(self):
        self.privacy_ledger = []
        self.total_epsilon = 0.0
        self.total_delta = 0.0

    def add_privacy_cost(self, epsilon: float, delta: float, mechanism: str, step: int):
        """Add privacy cost to the ledger"""
        self.privacy_ledger.append({
            "step": step,
            "epsilon": epsilon,
            "delta": delta,
            "mechanism": mechanism,
            "timestamp": time.time()
        })
        self.total_epsilon += epsilon
        self.total_delta += delta

    def compute_rdp_privacy(self, noise_multiplier: float, batch_size: int, dataset_size: int, steps: int, orders: List[float] = None) -> Dict[str, float]:
        """Compute privacy using Renyi Differential Privacy (RDP)"""
        if orders is None:
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        sampling_probability = batch_size / dataset_size
        rdp_orders = []
        rdp_eps = []

        for order in orders:
            if order == 1:
                continue

            # RDP for Gaussian mechanism
            rdp = (sampling_probability ** 2) * steps / (2 * noise_multiplier ** 2)
            rdp_orders.append(order)
            rdp_eps.append(rdp)

        # Convert RDP to (ε, δ)-DP
        target_delta = 1e-5
        eps_values = []

        for order, rdp_epsilon in zip(rdp_orders, rdp_eps):
            if order > 1:
                eps = rdp_epsilon + math.log(target_delta) / (order - 1)
                eps_values.append(max(0, eps))

        final_epsilon = min(eps_values) if eps_values else float('inf')

        return {
            "epsilon": final_epsilon,
            "delta": target_delta,
            "rdp_orders": rdp_orders,
            "rdp_eps": rdp_eps,
            "steps": steps,
            "noise_multiplier": noise_multiplier
        }

    def get_privacy_budget_remaining(self, epsilon_budget: float, delta_budget: float) -> Dict[str, float]:
        """Calculate remaining privacy budget"""
        return {
            "epsilon_remaining": max(0, epsilon_budget - self.total_epsilon),
            "delta_remaining": max(0, delta_budget - self.total_delta),
            "epsilon_used": self.total_epsilon,
            "delta_used": self.total_delta,
            "budget_exhausted": self.total_epsilon >= epsilon_budget or self.total_delta >= delta_budget
        }

class NoiseGenerator:
    """Generate and track noise for differential privacy"""

    def __init__(self, noise_multiplier: float, l2_norm_clip: float):
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.noise_history = []

    def add_gaussian_noise(self, gradients: np.ndarray, step: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Add Gaussian noise to gradients for DP-SGD"""
        # Clip gradients
        grad_norm = np.linalg.norm(gradients)
        clipping_factor = min(1.0, self.l2_norm_clip / (grad_norm + 1e-10))
        clipped_gradients = gradients * clipping_factor

        # Add noise
        noise_scale = self.noise_multiplier * self.l2_norm_clip
        noise = np.random.normal(0, noise_scale, gradients.shape)
        noisy_gradients = clipped_gradients + noise

        # Track noise statistics
        noise_stats = {
            "step": step,
            "original_norm": float(grad_norm),
            "clipped_norm": float(np.linalg.norm(clipped_gradients)),
            "noise_norm": float(np.linalg.norm(noise)),
            "final_norm": float(np.linalg.norm(noisy_gradients)),
            "clipping_factor": float(clipping_factor),
            "noise_scale": float(noise_scale)
        }

        self.noise_history.append(noise_stats)

        return noisy_gradients, noise_stats

    def get_noise_statistics(self) -> Dict[str, Any]:
        """Get comprehensive noise statistics"""
        if not self.noise_history:
            return {"error": "No noise history available"}

        clipping_factors = [entry["clipping_factor"] for entry in self.noise_history]
        noise_norms = [entry["noise_norm"] for entry in self.noise_history]

        return {
            "average_clipping_factor": np.mean(clipping_factors),
            "clipping_frequency": sum(1 for cf in clipping_factors if cf < 1.0) / len(clipping_factors),
            "average_noise_norm": np.mean(noise_norms),
            "noise_variance": np.var(noise_norms),
            "steps_recorded": len(self.noise_history),
            "recent_history": self.noise_history[-10:]  # Last 10 steps
        }

class UtilityPrivacyAnalyzer:
    """Analyze privacy-utility tradeoffs"""

    def __init__(self):
        self.experiment_results = []

    def analyze_tradeoff(self, privacy_params: Dict[str, float], performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze privacy-utility tradeoff for given parameters"""
        epsilon = privacy_params.get("epsilon", 0)

        accuracy = performance_metrics.get("accuracy", 0)
        loss = performance_metrics.get("loss", float('inf'))

        # Calculate utility score (higher is better)
        utility_score = accuracy - (loss / 10.0)  # Simple utility metric

        # Calculate privacy score (lower epsilon is better privacy)
        privacy_score = 1.0 / (1.0 + epsilon) if epsilon > 0 else 1.0

        # Combined score
        combined_score = 0.6 * utility_score + 0.4 * privacy_score

        analysis = {
            "privacy_params": privacy_params,
            "performance_metrics": performance_metrics,
            "utility_score": utility_score,
            "privacy_score": privacy_score,
            "combined_score": combined_score,
            "tradeoff_analysis": self.generate_tradeoff_insights(privacy_params, performance_metrics)
        }

        self.experiment_results.append(analysis)
        return analysis

    def generate_tradeoff_insights(self, privacy_params: Dict[str, float], performance_metrics: Dict[str, float]) -> List[str]:
        """Generate insights about privacy-utility tradeoff"""
        insights = []

        epsilon = privacy_params.get("epsilon", 0)
        noise_multiplier = privacy_params.get("noise_multiplier", 0)
        accuracy = performance_metrics.get("accuracy", 0)

        if epsilon < 1.0:
            insights.append("Strong privacy guarantee (ε < 1), may impact utility")
        elif epsilon < 5.0:
            insights.append("Moderate privacy guarantee, good utility-privacy balance")
        else:
            insights.append("Weak privacy guarantee (ε > 5), high utility expected")

        if noise_multiplier > 2.0:
            insights.append("High noise level may significantly impact convergence")
        elif noise_multiplier < 0.5:
            insights.append("Low noise level, privacy guarantees may be weak")

        if accuracy < 0.5:
            insights.append("Very low accuracy, consider reducing noise or increasing budget")
        elif accuracy > 0.9:
            insights.append("High accuracy maintained despite privacy constraints")

        return insights

    def recommend_privacy_parameters(self, target_accuracy: float, max_epsilon: float) -> Dict[str, Any]:
        """Recommend privacy parameters for target performance"""
        # Simulate parameter search
        candidate_params = []

        for noise_mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for clip_norm in [0.5, 1.0, 2.0, 5.0]:
                # Estimate epsilon for these parameters
                estimated_epsilon = min(max_epsilon, 10.0 / noise_mult)  # Simplified estimation

                # Estimate accuracy impact
                accuracy_penalty = noise_mult * 0.05  # More noise = lower accuracy
                estimated_accuracy = max(0.0, 1.0 - accuracy_penalty)

                if estimated_accuracy >= target_accuracy and estimated_epsilon <= max_epsilon:
                    candidate_params.append({
                        "noise_multiplier": noise_mult,
                        "l2_norm_clip": clip_norm,
                        "estimated_epsilon": estimated_epsilon,
                        "estimated_accuracy": estimated_accuracy,
                        "privacy_score": 1.0 / (1.0 + estimated_epsilon)
                    })

        # Sort by combined score
        candidate_params.sort(key=lambda x: x["privacy_score"] + x["estimated_accuracy"], reverse=True)

        return {
            "target_accuracy": target_accuracy,
            "max_epsilon": max_epsilon,
            "recommended_params": candidate_params[:5],  # Top 5 recommendations
            "feasible": len(candidate_params) > 0
        }

class PrivacyAuditLogger:
    """Log and audit privacy-related events"""

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.audit_log = []

    def log_privacy_event(self, event_type: str, details: Dict[str, Any], step: int = None):
        """Log a privacy-related event"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "step": step,
            "details": details,
            "id": len(self.audit_log)
        }
        self.audit_log.append(event)

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of privacy audit"""
        event_types = defaultdict(int)
        for event in self.audit_log:
            event_types[event["event_type"]] += 1

        return {
            "total_events": len(self.audit_log),
            "event_types": dict(event_types),
            "first_event": self.audit_log[0]["timestamp"] if self.audit_log else None,
            "last_event": self.audit_log[-1]["timestamp"] if self.audit_log else None,
            "recent_events": self.audit_log[-10:]  # Last 10 events
        }

    def export_audit_log(self, filepath: str) -> bool:
        """Export audit log to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
            return True
        except Exception:
            return False

class PrivacyAwareTrainer:
    """Main privacy-aware training coordinator"""

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.db_path = os.path.join(workspace_path, '.vscode', 'ai_debugger_privacy.db')
        self.accountant = PrivacyAccountant()
        self.analyzer = UtilityPrivacyAnalyzer()
        self.audit_logger = PrivacyAuditLogger(workspace_path)
        self.noise_generator = None
        self.init_database()

    def init_database(self):
        """Initialize privacy tracking database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS privacy_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    session_name TEXT,
                    privacy_params TEXT,
                    final_epsilon REAL,
                    final_delta REAL,
                    performance_metrics TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS privacy_budget_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    step INTEGER,
                    epsilon_spent REAL,
                    delta_spent REAL,
                    mechanism TEXT,
                    FOREIGN KEY (session_id) REFERENCES privacy_sessions (id)
                )
            ''')

    def start_private_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize private training session"""
        noise_multiplier = config.get('noise_multiplier', 1.0)
        l2_norm_clip = config.get('l2_norm_clip', 1.0)
        epsilon_budget = config.get('epsilon_budget', 10.0)
        delta_budget = config.get('delta_budget', 1e-5)

        # Initialize noise generator
        self.noise_generator = NoiseGenerator(noise_multiplier, l2_norm_clip)

        # Create session
        session_id = self.create_privacy_session(config)

        # Log session start
        self.audit_logger.log_privacy_event("session_start", {
            "session_id": session_id,
            "privacy_params": config
        })

        return {
            "status": "success",
            "session_id": session_id,
            "privacy_params": {
                "noise_multiplier": noise_multiplier,
                "l2_norm_clip": l2_norm_clip,
                "epsilon_budget": epsilon_budget,
                "delta_budget": delta_budget
            },
            "recommendations": self.generate_training_recommendations(config)
        }

    def compute_privacy_loss(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Compute privacy loss for training configuration"""
        batch_size = training_config.get('batch_size', 32)
        dataset_size = training_config.get('dataset_size', 50000)
        epochs = training_config.get('epochs', 10)
        noise_multiplier = training_config.get('noise_multiplier', 1.0)

        steps_per_epoch = dataset_size // batch_size
        total_steps = steps_per_epoch * epochs

        # Compute RDP-based privacy
        privacy_analysis = self.accountant.compute_rdp_privacy(
            noise_multiplier, batch_size, dataset_size, total_steps
        )

        # Update accountant
        self.accountant.add_privacy_cost(
            privacy_analysis["epsilon"],
            privacy_analysis["delta"],
            "dp_sgd",
            total_steps
        )

        return {
            "status": "success",
            "privacy_analysis": privacy_analysis,
            "training_config": training_config,
            "privacy_budget_status": self.accountant.get_privacy_budget_remaining(
                training_config.get('epsilon_budget', 10.0),
                training_config.get('delta_budget', 1e-5)
            )
        }

    def simulate_training_step(self, step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a private training step"""
        if not self.noise_generator:
            return {"error": "No active training session"}

        step = step_config.get('step', 0)
        gradient_shape = step_config.get('gradient_shape', [100])

        # Generate synthetic gradients
        gradients = np.random.randn(*gradient_shape)

        # Apply DP-SGD
        noisy_gradients, noise_stats = self.noise_generator.add_gaussian_noise(gradients, step)

        # Log the step
        self.audit_logger.log_privacy_event("training_step", {
            "step": step,
            "noise_stats": noise_stats
        }, step)

        return {
            "status": "success",
            "step": step,
            "noise_stats": noise_stats,
            "gradient_info": {
                "original_shape": gradient_shape,
                "original_norm": float(np.linalg.norm(gradients)),
                "noisy_norm": float(np.linalg.norm(noisy_gradients))
            }
        }

    def analyze_privacy_utility_tradeoff(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze privacy-utility tradeoff"""
        privacy_params = experiment_config.get('privacy_params', {})
        performance_metrics = experiment_config.get('performance_metrics', {})

        analysis = self.analyzer.analyze_tradeoff(privacy_params, performance_metrics)

        return {
            "status": "success",
            "tradeoff_analysis": analysis,
            "visualization_data": self.prepare_tradeoff_visualization(analysis),
            "recommendations": self.generate_optimization_recommendations(analysis)
        }

    def get_privacy_recommendations(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Get privacy parameter recommendations"""
        target_accuracy = requirements.get('target_accuracy', 0.8)
        max_epsilon = requirements.get('max_epsilon', 5.0)

        recommendations = self.analyzer.recommend_privacy_parameters(target_accuracy, max_epsilon)

        return {
            "status": "success",
            "recommendations": recommendations,
            "parameter_guidance": self.generate_parameter_guidance(recommendations)
        }

    def get_noise_visualization_data(self) -> Dict[str, Any]:
        """Get data for noise visualization"""
        if not self.noise_generator:
            return {"error": "No active training session"}

        noise_stats = self.noise_generator.get_noise_statistics()

        # Prepare visualization data
        steps = [entry["step"] for entry in self.noise_generator.noise_history]
        clipping_factors = [entry["clipping_factor"] for entry in self.noise_generator.noise_history]
        noise_norms = [entry["noise_norm"] for entry in self.noise_generator.noise_history]

        return {
            "status": "success",
            "noise_statistics": noise_stats,
            "visualization_data": {
                "steps": steps,
                "clipping_factors": clipping_factors,
                "noise_norms": noise_norms,
                "noise_multiplier": self.noise_generator.noise_multiplier,
                "l2_norm_clip": self.noise_generator.l2_norm_clip
            }
        }

    def get_privacy_audit(self) -> Dict[str, Any]:
        """Get privacy audit summary"""
        audit_summary = self.audit_logger.get_audit_summary()
        budget_status = self.accountant.get_privacy_budget_remaining(10.0, 1e-5)  # Default budgets

        return {
            "status": "success",
            "audit_summary": audit_summary,
            "privacy_ledger": self.accountant.privacy_ledger[-20:],  # Last 20 entries
            "budget_status": budget_status
        }

    def generate_training_recommendations(self, config: Dict[str, Any]) -> List[str]:
        """Generate recommendations for private training"""
        recommendations = []

        noise_multiplier = config.get('noise_multiplier', 1.0)
        l2_norm_clip = config.get('l2_norm_clip', 1.0)

        if noise_multiplier < 0.5:
            recommendations.append("Low noise multiplier may provide weak privacy guarantees")
        elif noise_multiplier > 3.0:
            recommendations.append("High noise multiplier may severely impact convergence")

        if l2_norm_clip < 0.1:
            recommendations.append("Very small clipping norm may cause frequent clipping")
        elif l2_norm_clip > 10.0:
            recommendations.append("Large clipping norm may reduce effectiveness of DP")

        recommendations.extend([
            "Monitor clipping frequency to ensure meaningful privacy",
            "Track privacy budget consumption throughout training",
            "Consider using learning rate schedules adapted for DP"
        ])

        return recommendations

    def prepare_tradeoff_visualization(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for privacy-utility tradeoff visualization"""
        return {
            "privacy_score": analysis.get("privacy_score", 0),
            "utility_score": analysis.get("utility_score", 0),
            "combined_score": analysis.get("combined_score", 0),
            "epsilon": analysis.get("privacy_params", {}).get("epsilon", 0),
            "accuracy": analysis.get("performance_metrics", {}).get("accuracy", 0)
        }

    def generate_optimization_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        utility_score = analysis.get("utility_score", 0)
        privacy_score = analysis.get("privacy_score", 0)

        if utility_score < 0.5:
            recommendations.append("Consider reducing noise or increasing privacy budget")
        if privacy_score < 0.5:
            recommendations.append("Consider increasing noise for better privacy")

        return recommendations

    def generate_parameter_guidance(self, recommendations: Dict[str, Any]) -> List[str]:
        """Generate guidance for parameter selection"""
        guidance = []

        if recommendations.get("feasible", False):
            guidance.append("Feasible parameter combinations found")
            guidance.append("Start with the top recommended parameters")
        else:
            guidance.append("No feasible parameters for given constraints")
            guidance.append("Consider relaxing accuracy or privacy requirements")

        return guidance

    def create_privacy_session(self, config: Dict[str, Any]) -> int:
        """Create new privacy session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO privacy_sessions (timestamp, session_name, privacy_params, final_epsilon, final_delta, performance_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                config.get('session_name', 'privacy_session'),
                json.dumps(config),
                0.0,
                0.0,
                '{}'
            ))
            return cursor.lastrowid

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python privacy_aware_training.py <command> <workspace_path> [args...]"}))
        return

    command = sys.argv[1]
    workspace_path = sys.argv[2]

    try:
        trainer = PrivacyAwareTrainer(workspace_path)

        if command == "start_private_training":
            config = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            result = trainer.start_private_training(config)

        elif command == "compute_privacy_loss":
            config = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            result = trainer.compute_privacy_loss(config)

        elif command == "simulate_training_step":
            config = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            result = trainer.simulate_training_step(config)

        elif command == "analyze_privacy_utility_tradeoff":
            config = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            result = trainer.analyze_privacy_utility_tradeoff(config)

        elif command == "get_privacy_recommendations":
            config = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            result = trainer.get_privacy_recommendations(config)

        elif command == "get_noise_visualization_data":
            result = trainer.get_noise_visualization_data()

        elif command == "get_privacy_audit":
            result = trainer.get_privacy_audit()

        else:
            result = {"error": f"Unknown command: {command}"}

        print(json.dumps(result, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e), "type": "exception"}))

if __name__ == "__main__":
    main()