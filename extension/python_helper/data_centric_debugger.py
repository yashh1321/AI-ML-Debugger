#!/usr/bin/env python3
"""
Data-Centric Debugger
Advanced data quality analysis with heatmaps, drift detection, and sample influence tracking.
"""

import json
import sys
import os
import numpy as np
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from collections import defaultdict, Counter
import pandas as pd

class LabelAnalyzer:
    """Analyze label distribution and quality issues"""

    def __init__(self):
        self.label_stats = {}
        self.noise_detections = []

    def analyze_label_distribution(self, labels: List[Any]) -> Dict[str, Any]:
        """Analyze label distribution for imbalance and patterns"""
        label_counts = Counter(labels)
        total_samples = len(labels)

        # Calculate statistics
        most_common = label_counts.most_common()
        least_freq_count = most_common[-1][1] if most_common else 0
        most_freq_count = most_common[0][1] if most_common else 0

        imbalance_ratio = most_freq_count / least_freq_count if least_freq_count > 0 else float('inf')

        # Gini coefficient for inequality
        sorted_counts = sorted(label_counts.values())
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * cumsum[-1]) - (n + 1) / n

        return {
            "total_samples": total_samples,
            "unique_labels": len(label_counts),
            "label_counts": dict(label_counts),
            "imbalance_ratio": imbalance_ratio,
            "gini_coefficient": gini,
            "distribution_type": self.classify_distribution(label_counts),
            "recommendations": self.generate_balance_recommendations(label_counts, imbalance_ratio)
        }

    def detect_label_noise(self, features: np.ndarray, labels: List[Any]) -> Dict[str, Any]:
        """Detect potential label noise using various heuristics"""
        noise_candidates = []

        # Method 1: k-NN consistency check
        knn_inconsistencies = self.knn_label_consistency(features, labels)

        # Method 2: Isolation forest for outliers
        outlier_indices = self.find_feature_outliers(features)

        # Method 3: Entropy-based detection
        entropy_anomalies = self.entropy_based_noise_detection(features, labels)

        # Combine methods
        all_suspects = set(knn_inconsistencies + outlier_indices + entropy_anomalies)

        noise_details = []
        for idx in all_suspects:
            confidence = 0
            reasons = []

            if idx in knn_inconsistencies:
                confidence += 0.4
                reasons.append("k-NN inconsistency")
            if idx in outlier_indices:
                confidence += 0.3
                reasons.append("feature outlier")
            if idx in entropy_anomalies:
                confidence += 0.3
                reasons.append("entropy anomaly")

            noise_details.append({
                "sample_index": int(idx),
                "confidence": confidence,
                "reasons": reasons,
                "label": labels[idx] if idx < len(labels) else None
            })

        return {
            "total_noise_candidates": len(noise_details),
            "noise_candidates": sorted(noise_details, key=lambda x: x["confidence"], reverse=True)[:20],
            "noise_percentage": len(noise_details) / len(labels) * 100,
            "detection_methods": {
                "knn_inconsistencies": len(knn_inconsistencies),
                "feature_outliers": len(outlier_indices),
                "entropy_anomalies": len(entropy_anomalies)
            }
        }

    def knn_label_consistency(self, features: np.ndarray, labels: List[Any], k: int = 5) -> List[int]:
        """Find samples whose labels disagree with k nearest neighbors"""
        inconsistencies = []

        if len(features) < k:
            return inconsistencies

        # Simplified k-NN check (using random sampling for efficiency)
        sample_indices = np.random.choice(len(features), min(1000, len(features)), replace=False)

        for i in sample_indices:
            # Calculate distances to all other points
            distances = np.sum((features - features[i]) ** 2, axis=1)
            nearest_indices = np.argsort(distances)[1:k+1]  # Exclude self

            # Check label consistency
            nearest_labels = [labels[idx] for idx in nearest_indices]
            majority_label = Counter(nearest_labels).most_common(1)[0][0]

            if labels[i] != majority_label:
                inconsistencies.append(i)

        return inconsistencies

    def find_feature_outliers(self, features: np.ndarray, contamination: float = 0.1) -> List[int]:
        """Find feature space outliers using isolation forest concept"""
        # Simplified outlier detection using statistical methods
        outliers = []

        for feature_idx in range(features.shape[1]):
            feature_values = features[:, feature_idx]
            q1, q3 = np.percentile(feature_values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (feature_values < lower_bound) | (feature_values > upper_bound)
            outliers.extend(np.where(outlier_mask)[0].tolist())

        return list(set(outliers))

    def entropy_based_noise_detection(self, features: np.ndarray, labels: List[Any]) -> List[int]:
        """Detect noise using local entropy analysis"""
        anomalies = []

        # Compute local neighborhoods and their label entropy
        for i in range(min(500, len(features))):  # Sample for efficiency
            # Find local neighborhood
            distances = np.sum((features - features[i]) ** 2, axis=1)
            neighbors_idx = np.argsort(distances)[:20]  # 20 nearest neighbors

            neighbor_labels = [labels[idx] for idx in neighbors_idx]
            label_counts = Counter(neighbor_labels)

            # Calculate entropy
            probs = np.array(list(label_counts.values())) / len(neighbor_labels)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))

            # High entropy indicates mixed labels (potential noise)
            if entropy > 1.5:  # Threshold for suspicion
                anomalies.append(i)

        return anomalies

    def classify_distribution(self, label_counts: Counter) -> str:
        """Classify the type of label distribution"""
        values = list(label_counts.values())
        max_val = max(values)
        min_val = min(values)

        ratio = max_val / min_val if min_val > 0 else float('inf')

        if ratio == 1:
            return "perfectly_balanced"
        elif ratio <= 2:
            return "well_balanced"
        elif ratio <= 10:
            return "moderately_imbalanced"
        elif ratio <= 100:
            return "highly_imbalanced"
        else:
            return "extremely_imbalanced"

    def generate_balance_recommendations(self, label_counts: Counter, ratio: float) -> List[str]:
        """Generate recommendations for handling label imbalance"""
        recommendations = []

        if ratio > 10:
            recommendations.append("Consider oversampling minority classes (SMOTE, ADASYN)")
            recommendations.append("Use stratified sampling for train/validation splits")
            recommendations.append("Apply class weights in loss function")

        if ratio > 100:
            recommendations.append("Consider undersampling majority class")
            recommendations.append("Use ensemble methods with balanced subsets")

        if len(label_counts) > 50:
            recommendations.append("Consider hierarchical classification")
            recommendations.append("Group similar classes together")

        return recommendations

class DataDriftDetector:
    """Detect data drift across time periods or datasets"""

    def __init__(self):
        self.baseline_stats = None
        self.drift_history = []

    def detect_feature_drift(self, baseline_data: np.ndarray, current_data: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """Detect drift in feature distributions"""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(baseline_data.shape[1])]

        drift_results = {}

        for i, feature_name in enumerate(feature_names):
            baseline_feature = baseline_data[:, i]
            current_feature = current_data[:, i]

            # Statistical tests for drift
            drift_results[feature_name] = {
                "ks_statistic": self.kolmogorov_smirnov_test(baseline_feature, current_feature),
                "psi": self.population_stability_index(baseline_feature, current_feature),
                "mean_shift": float(np.mean(current_feature) - np.mean(baseline_feature)),
                "std_ratio": float(np.std(current_feature) / (np.std(baseline_feature) + 1e-10)),
                "drift_magnitude": self.calculate_drift_magnitude(baseline_feature, current_feature)
            }

        # Overall drift assessment
        overall_drift = self.assess_overall_drift(drift_results)

        return {
            "feature_drift": drift_results,
            "overall_assessment": overall_drift,
            "drift_heatmap": self.generate_drift_heatmap(drift_results),
            "recommendations": self.generate_drift_recommendations(overall_drift)
        }

    def kolmogorov_smirnov_test(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Simplified KS test for distribution comparison"""
        # Sort both arrays
        baseline_sorted = np.sort(baseline)
        current_sorted = np.sort(current)

        # Create CDFs
        n1, n2 = len(baseline), len(current)
        all_values = np.concatenate([baseline_sorted, current_sorted])
        all_values = np.sort(np.unique(all_values))

        cdf1 = np.searchsorted(baseline_sorted, all_values, side='right') / n1
        cdf2 = np.searchsorted(current_sorted, all_values, side='right') / n2

        # Calculate KS statistic
        ks_statistic = np.max(np.abs(cdf1 - cdf2))

        return float(ks_statistic)

    def population_stability_index(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        # Create bins based on baseline data
        bin_edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Calculate distributions
        baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
        current_counts = np.histogram(current, bins=bin_edges)[0]

        # Convert to proportions
        baseline_props = baseline_counts / len(baseline)
        current_props = current_counts / len(current)

        # Calculate PSI
        psi = 0
        for i in range(len(baseline_props)):
            if baseline_props[i] > 0 and current_props[i] > 0:
                psi += (current_props[i] - baseline_props[i]) * np.log(current_props[i] / baseline_props[i])

        return float(psi)

    def calculate_drift_magnitude(self, baseline: np.ndarray, current: np.ndarray) -> str:
        """Calculate overall drift magnitude"""
        ks_stat = self.kolmogorov_smirnov_test(baseline, current)
        psi = self.population_stability_index(baseline, current)

        if ks_stat < 0.1 and psi < 0.1:
            return "no_drift"
        elif ks_stat < 0.2 and psi < 0.2:
            return "slight_drift"
        elif ks_stat < 0.3 and psi < 0.25:
            return "moderate_drift"
        else:
            return "severe_drift"

    def assess_overall_drift(self, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall drift across all features"""
        drift_magnitudes = [feature["drift_magnitude"] for feature in drift_results.values()]
        severe_count = sum(1 for mag in drift_magnitudes if mag == "severe_drift")
        moderate_count = sum(1 for mag in drift_magnitudes if mag == "moderate_drift")

        total_features = len(drift_magnitudes)
        severe_ratio = severe_count / total_features
        moderate_ratio = moderate_count / total_features

        if severe_ratio > 0.3:
            overall_status = "critical"
        elif severe_ratio > 0.1 or moderate_ratio > 0.5:
            overall_status = "concerning"
        elif moderate_ratio > 0.2:
            overall_status = "moderate"
        else:
            overall_status = "stable"

        return {
            "status": overall_status,
            "severe_drift_features": severe_count,
            "moderate_drift_features": moderate_count,
            "stable_features": total_features - severe_count - moderate_count,
            "drift_percentage": (severe_count + moderate_count) / total_features * 100
        }

    def generate_drift_heatmap(self, drift_results: Dict[str, Any]) -> List[List[float]]:
        """Generate heatmap data for drift visualization"""
        features = list(drift_results.keys())
        metrics = ["ks_statistic", "psi", "mean_shift", "std_ratio"]

        heatmap_data = []
        for metric in metrics:
            row = []
            for feature in features:
                value = drift_results[feature][metric]
                # Normalize values for heatmap (0-1 scale)
                if metric in ["ks_statistic", "psi"]:
                    normalized = min(value / 0.5, 1.0)  # Cap at 0.5 for visualization
                elif metric == "mean_shift":
                    normalized = min(abs(value) / 2.0, 1.0)  # Cap at 2 std devs
                elif metric == "std_ratio":
                    normalized = min(abs(value - 1.0), 1.0)  # Distance from 1.0
                row.append(normalized)
            heatmap_data.append(row)

        return heatmap_data

    def generate_drift_recommendations(self, overall_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on drift analysis"""
        recommendations = []
        status = overall_assessment["status"]

        if status == "critical":
            recommendations.extend([
                "Immediate retraining recommended",
                "Investigate data collection process changes",
                "Consider domain adaptation techniques",
                "Implement real-time monitoring"
            ])
        elif status == "concerning":
            recommendations.extend([
                "Schedule model retraining soon",
                "Monitor performance metrics closely",
                "Consider incremental learning approaches"
            ])
        elif status == "moderate":
            recommendations.extend([
                "Continue monitoring",
                "Consider feature engineering updates",
                "Plan for model refresh in next cycle"
            ])

        return recommendations

class SampleInfluenceTracker:
    """Track which samples contribute most to model behavior"""

    def __init__(self):
        self.influence_cache = {}

    def calculate_sample_influence(self, losses: np.ndarray, gradients: np.ndarray = None) -> Dict[str, Any]:
        """Calculate influence of individual samples"""
        n_samples = len(losses)

        # Method 1: Loss-based influence
        loss_influence = self.loss_based_influence(losses)

        # Method 2: Gradient-based influence (if available)
        gradient_influence = self.gradient_based_influence(gradients) if gradients is not None else None

        # Method 3: Cook's distance approximation
        cooks_distance = self.approximate_cooks_distance(losses)

        # Combine influences
        combined_influence = self.combine_influence_scores(loss_influence, gradient_influence, cooks_distance)

        return {
            "sample_influences": combined_influence,
            "top_influential_samples": self.get_top_influential(combined_influence, top_k=20),
            "influence_distribution": self.analyze_influence_distribution(combined_influence),
            "problematic_samples": self.identify_problematic_samples(losses, combined_influence)
        }

    def loss_based_influence(self, losses: np.ndarray) -> np.ndarray:
        """Calculate influence based on loss values"""
        # Higher loss = higher influence
        normalized_losses = (losses - np.min(losses)) / (np.max(losses) - np.min(losses) + 1e-10)
        return normalized_losses

    def gradient_based_influence(self, gradients: np.ndarray) -> np.ndarray:
        """Calculate influence based on gradient magnitudes"""
        # Higher gradient magnitude = higher influence
        gradient_norms = np.linalg.norm(gradients, axis=1)
        normalized_grads = (gradient_norms - np.min(gradient_norms)) / (np.max(gradient_norms) - np.min(gradient_norms) + 1e-10)
        return normalized_grads

    def approximate_cooks_distance(self, losses: np.ndarray) -> np.ndarray:
        """Approximate Cook's distance for outlier detection"""
        # Simple approximation based on loss deviation
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        cooks_d = np.abs(losses - mean_loss) / (std_loss + 1e-10)
        return cooks_d / np.max(cooks_d)  # Normalize

    def combine_influence_scores(self, loss_influence: np.ndarray, gradient_influence: Optional[np.ndarray], cooks_distance: np.ndarray) -> np.ndarray:
        """Combine different influence measures"""
        combined = 0.5 * loss_influence + 0.5 * cooks_distance

        if gradient_influence is not None:
            combined = 0.4 * loss_influence + 0.3 * gradient_influence + 0.3 * cooks_distance

        return combined

    def get_top_influential(self, influences: np.ndarray, top_k: int = 20) -> List[Dict[str, Any]]:
        """Get top influential samples"""
        top_indices = np.argsort(influences)[-top_k:][::-1]

        return [
            {
                "sample_index": int(idx),
                "influence_score": float(influences[idx]),
                "rank": rank + 1
            }
            for rank, idx in enumerate(top_indices)
        ]

    def analyze_influence_distribution(self, influences: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of influence scores"""
        return {
            "mean_influence": float(np.mean(influences)),
            "std_influence": float(np.std(influences)),
            "skewness": float(self.calculate_skewness(influences)),
            "high_influence_threshold": float(np.percentile(influences, 95)),
            "high_influence_count": int(np.sum(influences > np.percentile(influences, 95)))
        }

    def calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of influence distribution"""
        mean = np.mean(data)
        std = np.std(data)
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness

    def identify_problematic_samples(self, losses: np.ndarray, influences: np.ndarray) -> List[Dict[str, Any]]:
        """Identify samples that might be problematic"""
        high_loss_threshold = np.percentile(losses, 95)
        high_influence_threshold = np.percentile(influences, 95)

        problematic = []
        for i, (loss, influence) in enumerate(zip(losses, influences)):
            if loss > high_loss_threshold and influence > high_influence_threshold:
                problematic.append({
                    "sample_index": int(i),
                    "loss": float(loss),
                    "influence": float(influence),
                    "problem_type": "high_loss_high_influence"
                })

        return problematic[:10]  # Return top 10

class DataCentricDebugger:
    """Main data-centric debugging interface"""

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.db_path = os.path.join(workspace_path, '.vscode', 'ai_debugger_data.db')
        self.label_analyzer = LabelAnalyzer()
        self.drift_detector = DataDriftDetector()
        self.influence_tracker = SampleInfluenceTracker()
        self.init_database()

    def init_database(self):
        """Initialize data analysis database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_analysis_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    analysis_type TEXT,
                    dataset_name TEXT,
                    results TEXT,
                    metadata TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS drift_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    baseline_hash TEXT,
                    current_hash TEXT,
                    drift_score REAL,
                    features_affected TEXT,
                    FOREIGN KEY (session_id) REFERENCES data_analysis_sessions (id)
                )
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_drift_detections_session
                ON drift_detections (session_id)
            ''')

    def analyze_dataset_quality(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive dataset quality analysis"""
        # Simulate dataset loading and analysis
        n_samples = dataset_config.get('n_samples', 1000)
        n_features = dataset_config.get('n_features', 10)
        n_classes = dataset_config.get('n_classes', 5)

        # Generate synthetic data for demonstration
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, n_classes, n_samples)

        # Add some imbalance
        labels[int(n_samples * 0.8):] = 0  # Make class 0 dominant

        # Perform analyses
        label_analysis = self.label_analyzer.analyze_label_distribution(labels.tolist())
        noise_analysis = self.label_analyzer.detect_label_noise(features, labels.tolist())

        # Store results
        session_id = self.create_analysis_session("dataset_quality", dataset_config.get('name', 'unknown'))

        results = {
            "session_id": session_id,
            "dataset_info": {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes
            },
            "label_analysis": label_analysis,
            "noise_analysis": noise_analysis,
            "quality_score": self.calculate_quality_score(label_analysis, noise_analysis),
            "recommendations": self.generate_quality_recommendations(label_analysis, noise_analysis)
        }

        return {"status": "success", "results": results}

    def detect_data_drift(self, drift_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect data drift between datasets"""
        baseline_config = drift_config.get('baseline', {})
        current_config = drift_config.get('current', {})

        # Generate synthetic baseline and current data
        n_features = drift_config.get('n_features', 10)
        baseline_data = np.random.randn(1000, n_features)

        # Add drift to current data
        drift_strength = drift_config.get('drift_strength', 0.5)
        current_data = np.random.randn(1000, n_features) + drift_strength

        feature_names = drift_config.get('feature_names', [f"feature_{i}" for i in range(n_features)])

        drift_results = self.drift_detector.detect_feature_drift(baseline_data, current_data, feature_names)

        session_id = self.create_analysis_session("drift_detection", drift_config.get('name', 'drift_analysis'))

        return {
            "status": "success",
            "session_id": session_id,
            "drift_analysis": drift_results,
            "visualization_data": self.prepare_drift_visualization(drift_results)
        }

    def track_sample_influence(self, influence_config: Dict[str, Any]) -> Dict[str, Any]:
        """Track influential samples in training"""
        n_samples = influence_config.get('n_samples', 1000)

        # Generate synthetic loss and gradient data
        losses = np.random.exponential(1.0, n_samples)  # Exponential distribution for losses
        gradients = np.random.randn(n_samples, 10)  # Random gradients

        influence_results = self.influence_tracker.calculate_sample_influence(losses, gradients)

        session_id = self.create_analysis_session("sample_influence", influence_config.get('name', 'influence_analysis'))

        return {
            "status": "success",
            "session_id": session_id,
            "influence_analysis": influence_results,
            "visualization_data": self.prepare_influence_visualization(influence_results)
        }

    def calculate_quality_score(self, label_analysis: Dict[str, Any], noise_analysis: Dict[str, Any]) -> float:
        """Calculate overall dataset quality score (0-1)"""
        score = 1.0

        # Penalize for imbalance
        imbalance_ratio = label_analysis.get('imbalance_ratio', 1)
        if imbalance_ratio > 10:
            score -= 0.3
        elif imbalance_ratio > 5:
            score -= 0.1

        # Penalize for noise
        noise_percentage = noise_analysis.get('noise_percentage', 0)
        score -= min(noise_percentage / 100 * 0.5, 0.4)

        return max(0.0, score)

    def generate_quality_recommendations(self, label_analysis: Dict[str, Any], noise_analysis: Dict[str, Any]) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []

        # Add label-specific recommendations
        recommendations.extend(label_analysis.get('recommendations', []))

        # Add noise-specific recommendations
        if noise_analysis.get('noise_percentage', 0) > 5:
            recommendations.extend([
                "Review and clean detected noisy samples",
                "Consider data validation pipelines",
                "Implement label verification process"
            ])

        return recommendations

    def prepare_drift_visualization(self, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for drift visualization"""
        return {
            "heatmap_data": drift_results.get('drift_heatmap', []),
            "feature_names": list(drift_results.get('feature_drift', {}).keys()),
            "metrics": ["KS Statistic", "PSI", "Mean Shift", "Std Ratio"],
            "overall_status": drift_results.get('overall_assessment', {}).get('status', 'unknown')
        }

    def prepare_influence_visualization(self, influence_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for influence visualization"""
        influences = influence_results.get('sample_influences', [])

        return {
            "influence_histogram": np.histogram(influences, bins=20)[0].tolist(),
            "histogram_bins": np.histogram(influences, bins=20)[1].tolist(),
            "top_samples": influence_results.get('top_influential_samples', []),
            "distribution_stats": influence_results.get('influence_distribution', {})
        }

    def create_analysis_session(self, analysis_type: str, dataset_name: str) -> int:
        """Create new analysis session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO data_analysis_sessions (timestamp, analysis_type, dataset_name, results, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                analysis_type,
                dataset_name,
                '{}',
                '{}'
            ))
            return cursor.lastrowid

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python data_centric_debugger.py <command> <workspace_path> [args...]"}))
        return

    command = sys.argv[1]
    workspace_path = sys.argv[2]

    try:
        debugger = DataCentricDebugger(workspace_path)

        if command == "analyze_dataset_quality":
            config = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            result = debugger.analyze_dataset_quality(config)

        elif command == "detect_data_drift":
            config = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            result = debugger.detect_data_drift(config)

        elif command == "track_sample_influence":
            config = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
            result = debugger.track_sample_influence(config)

        else:
            result = {"error": f"Unknown command: {command}"}

        print(json.dumps(result, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e), "type": "exception"}))

if __name__ == "__main__":
    main()