import json
import os
import sys
import numpy as np
import base64
from typing import Dict, List
import io

class ExplainabilityTools:
    """Explainability and attribution tools for ML models"""

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.supported_methods = ['shap', 'lime', 'integrated_gradients', 'grad_cam', 'attention']
        self.model_types = ['vision', 'nlp', 'tabular', 'time_series']

    def detect_model_type(self, model_info: Dict) -> str:
        """Detect the type of ML model for appropriate explanation method"""
        try:
            # Try to infer from model architecture or input shape
            if 'input_shape' in model_info:
                shape = model_info['input_shape']
                if len(shape) == 4:  # (batch, channels, height, width)
                    return 'vision'
                elif len(shape) == 3:  # (batch, sequence, features)
                    return 'nlp'
                elif len(shape) == 2:  # (batch, features)
                    return 'tabular'

            # Try to infer from model class name
            if 'model_class' in model_info:
                model_class = model_info['model_class'].lower()
                if any(keyword in model_class for keyword in ['conv', 'resnet', 'efficientnet', 'vit']):
                    return 'vision'
                elif any(keyword in model_class for keyword in ['bert', 'gpt', 'transformer', 'lstm', 'rnn']):
                    return 'nlp'
                elif any(keyword in model_class for keyword in ['linear', 'mlp', 'dense']):
                    return 'tabular'

            return 'unknown'
        except Exception:
            return 'unknown'

    def get_recommended_methods(self, model_type: str) -> List[str]:
        """Get recommended explanation methods for a model type"""
        recommendations = {
            'vision': ['grad_cam', 'integrated_gradients', 'shap'],
            'nlp': ['attention', 'integrated_gradients', 'lime', 'shap'],
            'tabular': ['shap', 'lime', 'permutation_importance'],
            'time_series': ['shap', 'integrated_gradients'],
            'unknown': ['shap', 'lime']
        }
        return recommendations.get(model_type, ['shap', 'lime'])

    def generate_shap_explanation(self, model_path: str, data_path: str,
                                sample_indices: List[int] = None) -> Dict:
        """Generate SHAP explanations"""
        try:
            # This would require actual model loading and SHAP integration
            # For demonstration, we'll create a mock explanation

            explanation = {
                'method': 'shap',
                'feature_importance': {
                    'feature_names': [f'feature_{i}' for i in range(10)],
                    'importance_values': np.random.randn(10).tolist(),
                    'base_value': 0.5
                },
                'sample_explanations': []
            }

            # Generate mock sample explanations
            if sample_indices is None:
                sample_indices = [0, 1, 2]

            for idx in sample_indices:
                sample_explanation = {
                    'sample_index': idx,
                    'prediction': np.random.random(),
                    'feature_contributions': np.random.randn(10).tolist(),
                    'visualization_data': self._create_mock_visualization('shap')
                }
                explanation['sample_explanations'].append(sample_explanation)

            return {
                'success': True,
                'explanation': explanation
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def generate_lime_explanation(self, model_path: str, data_path: str,
                                sample_indices: List[int] = None) -> Dict:
        """Generate LIME explanations"""
        try:
            explanation = {
                'method': 'lime',
                'sample_explanations': []
            }

            if sample_indices is None:
                sample_indices = [0]

            for idx in sample_indices:
                sample_explanation = {
                    'sample_index': idx,
                    'prediction': np.random.random(),
                    'local_explanation': {
                        'feature_names': [f'feature_{i}' for i in range(10)],
                        'feature_weights': np.random.randn(10).tolist()
                    },
                    'visualization_data': self._create_mock_visualization('lime')
                }
                explanation['sample_explanations'].append(sample_explanation)

            return {
                'success': True,
                'explanation': explanation
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def generate_integrated_gradients(self, model_path: str, data_path: str,
                                    sample_indices: List[int] = None) -> Dict:
        """Generate Integrated Gradients explanations"""
        try:
            explanation = {
                'method': 'integrated_gradients',
                'sample_explanations': []
            }

            if sample_indices is None:
                sample_indices = [0]

            for idx in sample_indices:
                sample_explanation = {
                    'sample_index': idx,
                    'prediction': np.random.random(),
                    'attributions': np.random.randn(224, 224, 3).tolist(),  # Mock image attributions
                    'visualization_data': self._create_mock_visualization('integrated_gradients')
                }
                explanation['sample_explanations'].append(sample_explanation)

            return {
                'success': True,
                'explanation': explanation
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def generate_grad_cam(self, model_path: str, data_path: str,
                         sample_indices: List[int] = None, layer_name: str = None) -> Dict:
        """Generate Grad-CAM explanations"""
        try:
            explanation = {
                'method': 'grad_cam',
                'layer_name': layer_name or 'last_conv_layer',
                'sample_explanations': []
            }

            if sample_indices is None:
                sample_indices = [0]

            for idx in sample_indices:
                # Mock Grad-CAM heatmap
                heatmap = np.random.rand(224, 224)

                sample_explanation = {
                    'sample_index': idx,
                    'prediction': np.random.random(),
                    'heatmap': heatmap.tolist(),
                    'overlaid_image': self._create_mock_visualization('grad_cam'),
                    'confidence': np.random.random()
                }
                explanation['sample_explanations'].append(sample_explanation)

            return {
                'success': True,
                'explanation': explanation
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def generate_attention_visualization(self, model_path: str, data_path: str,
                                       sample_indices: List[int] = None) -> Dict:
        """Generate attention visualization for transformer models"""
        try:
            explanation = {
                'method': 'attention',
                'sample_explanations': []
            }

            if sample_indices is None:
                sample_indices = [0]

            for idx in sample_indices:
                # Mock attention weights
                seq_length = 128
                num_heads = 12
                attention_weights = np.random.rand(num_heads, seq_length, seq_length)

                sample_explanation = {
                    'sample_index': idx,
                    'tokens': [f'token_{i}' for i in range(seq_length)],
                    'attention_weights': attention_weights.tolist(),
                    'head_weights': np.random.rand(num_heads).tolist(),
                    'visualization_data': self._create_mock_visualization('attention')
                }
                explanation['sample_explanations'].append(sample_explanation)

            return {
                'success': True,
                'explanation': explanation
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _create_mock_visualization(self, method: str) -> str:
        """Create mock visualization data (base64 encoded image)"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))

            if method == 'shap':
                # Mock SHAP waterfall plot
                features = [f'Feature {i}' for i in range(10)]
                values = np.random.randn(10)
                colors = ['red' if v < 0 else 'blue' for v in values]
                ax.barh(features, values, color=colors)
                ax.set_title('SHAP Feature Importance')
                ax.set_xlabel('SHAP Value')

            elif method == 'lime':
                # Mock LIME explanation
                features = [f'Feature {i}' for i in range(8)]
                weights = np.random.randn(8)
                colors = ['red' if w < 0 else 'green' for w in weights]
                ax.barh(features, weights, color=colors)
                ax.set_title('LIME Local Explanation')
                ax.set_xlabel('Feature Weight')

            elif method == 'integrated_gradients':
                # Mock heatmap
                data = np.random.rand(10, 10)
                im = ax.imshow(data, cmap='RdYlBu_r')
                ax.set_title('Integrated Gradients Attribution')
                plt.colorbar(im)

            elif method == 'grad_cam':
                # Mock Grad-CAM overlay
                image = np.random.rand(224, 224, 3)
                heatmap = np.random.rand(224, 224)
                ax.imshow(image)
                ax.imshow(heatmap, alpha=0.5, cmap='jet')
                ax.set_title('Grad-CAM Visualization')
                ax.axis('off')

            elif method == 'attention':
                # Mock attention heatmap
                attention_matrix = np.random.rand(20, 20)
                im = ax.imshow(attention_matrix, cmap='Blues')
                ax.set_title('Attention Weights')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                plt.colorbar(im)

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return image_base64

        except ImportError:
            return "mock_visualization_data"
        except Exception:
            return "error_generating_visualization"

    def compare_explanations(self, explanations: List[Dict]) -> Dict:
        """Compare multiple explanations side by side"""
        try:
            if not explanations:
                return {'success': False, 'error': 'No explanations provided'}

            comparison = {
                'methods': [exp.get('method', 'unknown') for exp in explanations],
                'sample_count': len(explanations[0].get('sample_explanations', [])),
                'feature_importance_comparison': {},
                'prediction_consistency': {},
                'visualization_grid': []
            }

            # Compare feature importance if available
            for i, exp in enumerate(explanations):
                method = exp.get('method', f'method_{i}')
                if 'feature_importance' in exp:
                    comparison['feature_importance_comparison'][method] = exp['feature_importance']

            # Compare predictions across methods
            for i in range(comparison['sample_count']):
                sample_predictions = {}
                for exp in explanations:
                    method = exp.get('method', 'unknown')
                    if i < len(exp.get('sample_explanations', [])):
                        pred = exp['sample_explanations'][i].get('prediction')
                        sample_predictions[method] = pred

                comparison['prediction_consistency'][f'sample_{i}'] = sample_predictions

            return {
                'success': True,
                'comparison': comparison
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def export_explanation(self, explanation: Dict, format: str = 'html',
                          output_path: str = None) -> Dict:
        """Export explanation in various formats"""
        try:
            if output_path is None:
                output_path = os.path.join(
                    self.workspace_path,
                    f"explanation_{explanation.get('method', 'unknown')}.{format}"
                )

            if format == 'html':
                html_content = self._generate_html_report(explanation)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

            elif format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(explanation, f, indent=2)

            elif format == 'md':
                markdown_content = self._generate_markdown_report(explanation)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

            else:
                return {'success': False, 'error': f'Unsupported format: {format}'}

            return {
                'success': True,
                'output_path': output_path,
                'format': format
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_html_report(self, explanation: Dict) -> str:
        """Generate HTML report for explanation"""
        method = explanation.get('method', 'Unknown')

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{method.upper()} Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .sample {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; }}
                .visualization {{ text-align: center; margin: 10px 0; }}
                .feature-importance {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{method.upper()} Explanation Report</h1>
                <p>Generated by AI-ML Debugger</p>
            </div>
        """

        # Add feature importance if available
        if 'feature_importance' in explanation:
            html += "<div class='feature-importance'><h2>Feature Importance</h2>"
            for name, value in zip(
                explanation['feature_importance']['feature_names'],
                explanation['feature_importance']['importance_values']
            ):
                html += f"<p>{name}: {value:.4f}</p>"
            html += "</div>"

        # Add sample explanations
        for sample in explanation.get('sample_explanations', []):
            html += f"""
            <div class="sample">
                <h3>Sample {sample['sample_index']}</h3>
                <p>Prediction: {sample.get('prediction', 'N/A'):.4f}</p>
                <div class="visualization">
                    <img src="data:image/png;base64,{sample.get('visualization_data', '')}"
                         alt="Visualization" style="max-width: 100%; height: auto;">
                </div>
            </div>
            """

        html += "</body></html>"
        return html

    def _generate_markdown_report(self, explanation: Dict) -> str:
        """Generate Markdown report for explanation"""
        method = explanation.get('method', 'Unknown')

        md = f"# {method.upper()} Explanation Report\n\n"
        md += "Generated by AI-ML Debugger\n\n"

        if 'feature_importance' in explanation:
            md += "## Feature Importance\n\n"
            for name, value in zip(
                explanation['feature_importance']['feature_names'],
                explanation['feature_importance']['importance_values']
            ):
                md += f"- {name}: {value:.4f}\n"
            md += "\n"

        md += "## Sample Explanations\n\n"
        for sample in explanation.get('sample_explanations', []):
            md += f"### Sample {sample['sample_index']}\n\n"
            md += f"Prediction: {sample.get('prediction', 'N/A'):.4f}\n\n"

        return md

def main():
    if len(sys.argv) < 2:
        print("Usage: python explainability_tools.py <command> [args...]")
        return

    command = sys.argv[1]
    workspace_path = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()

    explainer = ExplainabilityTools(workspace_path)

    if command == "detect_model_type":
        model_info = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        model_type = explainer.detect_model_type(model_info)
        print(json.dumps({'success': True, 'model_type': model_type}))

    elif command == "get_recommendations":
        model_type = sys.argv[3] if len(sys.argv) > 3 else 'unknown'
        methods = explainer.get_recommended_methods(model_type)
        print(json.dumps({'success': True, 'recommended_methods': methods}))

    elif command == "generate_shap":
        model_path = sys.argv[3]
        data_path = sys.argv[4]
        sample_indices = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
        result = explainer.generate_shap_explanation(model_path, data_path, sample_indices)
        print(json.dumps(result))

    elif command == "generate_lime":
        model_path = sys.argv[3]
        data_path = sys.argv[4]
        sample_indices = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
        result = explainer.generate_lime_explanation(model_path, data_path, sample_indices)
        print(json.dumps(result))

    elif command == "generate_integrated_gradients":
        model_path = sys.argv[3]
        data_path = sys.argv[4]
        sample_indices = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
        result = explainer.generate_integrated_gradients(model_path, data_path, sample_indices)
        print(json.dumps(result))

    elif command == "generate_grad_cam":
        model_path = sys.argv[3]
        data_path = sys.argv[4]
        sample_indices = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
        layer_name = sys.argv[6] if len(sys.argv) > 6 else None
        result = explainer.generate_grad_cam(model_path, data_path, sample_indices, layer_name)
        print(json.dumps(result))

    elif command == "generate_attention":
        model_path = sys.argv[3]
        data_path = sys.argv[4]
        sample_indices = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
        result = explainer.generate_attention_visualization(model_path, data_path, sample_indices)
        print(json.dumps(result))

    elif command == "compare_explanations":
        explanations = json.loads(sys.argv[3])
        result = explainer.compare_explanations(explanations)
        print(json.dumps(result))

    elif command == "export_explanation":
        explanation = json.loads(sys.argv[3])
        format_type = sys.argv[4] if len(sys.argv) > 4 else 'html'
        output_path = sys.argv[5] if len(sys.argv) > 5 else None
        result = explainer.export_explanation(explanation, format_type, output_path)
        print(json.dumps(result))

    else:
        print(json.dumps({'success': False, 'error': f'Unknown command: {command}'}))

if __name__ == "__main__":
    main()