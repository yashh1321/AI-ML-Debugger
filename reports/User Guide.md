# VS Code AI/ML Debugger Extension User Guide

This comprehensive guide will help you get the most out of the VS Code AI/ML Debugger extension.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Model Explorer](#model-explorer)
3. [Tensor Inspector](#tensor-inspector)
4. [Training Console](#training-console)
5. [Metrics Dashboard](#metrics-dashboard)
6. [Gradient & Activation Visualizer](#gradient--activation-visualizer)
7. [Model Export & Compilation](#model-export--compilation)
8. [Framework Support & Configuration](#framework-support--configuration)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

## Installation and Setup

### Prerequisites

Before installing the extension, ensure you have:

- Visual Studio Code 1.80.0 or higher
- Python 3.7 or higher
- One of the supported ML frameworks:
  - PyTorch 1.8+
  - TensorFlow 2.4+
  - JAX/Flax 0.3.0+
  - ONNX Runtime 1.13+

### Installation Steps

1. **Install the extension**:
   - Open VS Code
   - Go to the Extensions view (Ctrl+Shift+X)
   - Search for "AI/ML Debugger"
   - Click "Install"
   - Alternatively, download the VSIX file and install via "Install from VSIX..."

2. **Python dependencies**:
   
   The extension now features automatic dependency management. When you first use the extension:
   - It detects your Python environment
   - Checks for required dependencies
   - Offers to install any missing packages
   - Creates a dedicated virtual environment if needed

   You can also manually install dependencies using the "AI Debugger: Install Dependencies" command from the command palette.

3. **Verify Installation**:
   - Open the command palette (Ctrl+Shift+P)
   - Run "AI Debugger: Ping Python Helper"
   - You should see a success message

### Initial Configuration

1. Open VS Code settings (Ctrl+,)
2. Search for "AI/ML Debugger"
3. Configure the settings according to your preferences
4. Save the settings

## Model Explorer

The Model Explorer provides an interactive visualization of your model's architecture.

### Opening the Model Explorer

1. Click on the AI/ML Debugger icon in the Activity Bar
2. Select the "Model Architecture" tab

### Loading a Model

1. Click the "Load Model" button
2. Select your model file:
   - PyTorch: `.pt`, `.pth`
   - TensorFlow: `.h5`, `.keras`, `.pb`
   - JAX: `.pkl`
   - ONNX: `.onnx`
3. The model architecture will be displayed in the explorer

### Navigating the Model

- **Zoom**: Use the mouse wheel or pinch gesture
- **Pan**: Click and drag the background
- **Select Layer**: Click on a layer to view its details
- **Expand/Collapse**: Click on the expand/collapse icon for nested layers
- **Search**: Use the search box to find specific layers by name or type

### Layer Details

When you select a layer, the details panel shows:

- Layer name and type
- Input and output shapes
- Parameter count
- Configuration details
- Performance metrics (if available)

### Exporting the Model Visualization

1. Click the "Export" button
2. Choose the export format (PNG, SVG, or PDF)
3. Select the destination folder
4. Click "Save"

## Tensor Inspector

The Tensor Inspector allows you to visualize and analyze tensor values during model execution.

### Opening the Tensor Inspector

1. Click on the AI/ML Debugger icon in the Activity Bar
2. Select the "Tensor Inspector" tab

### Connecting to a Running Model

1. Start your model training or inference with the debugger enabled
2. The Tensor Inspector will automatically detect available tensors

### Browsing Tensors

- Use the tensor browser to navigate through available tensors
- Filter tensors by name, shape, or type
- Sort tensors by various properties
- Group tensors by layer or type

### Visualizing Tensor Data

1. Select a tensor from the list
2. Choose a visualization type:
   - **Histogram**: Shows the distribution of values
   - **Heatmap**: Visualizes 2D slices with color mapping
   - **Line Plot**: Shows values along a dimension
   - **3D Plot**: Visualizes 3D tensors (for compatible shapes)

### Tensor Statistics

The statistics panel shows:

- Minimum, maximum, mean, and standard deviation
- Sparsity (percentage of zero values)
- Gradient information (if available)
- Shape and data type

### Navigating Multi-dimensional Tensors

- Use dimension sliders to navigate through tensor dimensions
- Select specific indices to view
- Transpose dimensions for different views

## Training Console

The Training Console provides control over the training process.

### Opening the Training Console

1. Click on the AI/ML Debugger icon in the Activity Bar
2. Select the "Training Console" tab

### Connecting to a Training Session

1. Start your training script with the debugger enabled
2. The Training Console will automatically connect

### Training Controls

- **Start/Pause**: Control the training process
- **Step Batch**: Advance one batch at a time
- **Step Epoch**: Advance one epoch at a time
- **Run to Loss**: Run until a specific loss value is reached
- **Run to Accuracy**: Run until a specific accuracy is reached

### Setting Breakpoints

1. Click the "Add Breakpoint" button
2. Choose a breakpoint type:
   - **Loss Threshold**: Break when loss reaches a value
   - **Accuracy Threshold**: Break when accuracy reaches a value
   - **Gradient Threshold**: Break when gradients exceed a value
   - **Epoch Count**: Break after a specific number of epochs
   - **Custom Condition**: Define a custom breakpoint condition

### Batch Inspection

When training is paused:

1. View the current batch data
2. Inspect input samples
3. View model predictions
4. Compare with ground truth
5. Analyze gradients and activations

## Metrics Dashboard

The Metrics Dashboard visualizes training metrics in real-time.

### Opening the Metrics Dashboard

1. Click on the AI/ML Debugger icon in the Activity Bar
2. Select the "Metrics Dashboard" tab

### Connecting to a Training Session

1. Start your training script with the debugger enabled
2. The Metrics Dashboard will automatically connect and display metrics

### Customizing the Dashboard

1. Click the "Customize" button
2. Add, remove, or rearrange charts
3. Configure chart types and properties
4. Save your custom layout

### Available Chart Types

- **Line Chart**: Track metrics over time
- **Bar Chart**: Compare values across categories
- **Scatter Plot**: Visualize relationships between metrics
- **Heatmap**: Show patterns in 2D data
- **Histogram**: Visualize distributions

### Adding Annotations

1. Right-click on a point in a chart
2. Select "Add Annotation"
3. Enter annotation text
4. Click "Save"

### Exporting Metrics

1. Click the "Export" button
2. Choose the export format (CSV, JSON, or Excel)
3. Select the metrics to export
4. Click "Save"

## Gradient & Activation Visualizer

The Gradient & Activation Visualizer helps you monitor and analyze gradients and activations during training.

### Opening the Gradient Visualizer

1. Click on the AI/ML Debugger icon in the Activity Bar
2. Select the "Gradient & Activation Visualizer" tab

### Monitoring Gradients

1. Start your training script with the debugger enabled
2. The Gradient Visualizer will automatically connect
3. Select layers to monitor from the layer selector
4. View gradients and activations in real-time

### Visualization Options

- **Heatmap**: View gradients as a heatmap
- **Histogram**: See gradient distribution
- **Line Graph**: Track gradient changes over time
- **3D Visualization**: View complex gradient patterns

### Detecting Gradient Issues

The visualizer automatically detects common gradient issues:

- **Vanishing Gradients**: Highlighted when gradients become too small
- **Exploding Gradients**: Warned when gradients exceed thresholds
- **Dead Neurons**: Identified when activations consistently remain at zero

### Customizing Detection Thresholds

1. Open VS Code settings
2. Search for "AI Debugger: Gradients"
3. Adjust the following settings:
   - **Vanishing Threshold**: Default 1e-7
   - **Exploding Threshold**: Default 1000
   - **Update Frequency**: How often to refresh visualizations
   - **Sampling Rate**: How much gradient data to collect

## Model Export & Compilation

### Opening the Model Export Panel

1. Click on the AI/ML Debugger icon in the Activity Bar
2. Click on "Model Export" in the command palette

### Exporting to ONNX

1. Load a model in your Python session
2. In the Model Export panel, select "ONNX" as the format
3. Configure export options:
   - Input shape (e.g., [1, 3, 224, 224])
   - ONNX opset version (9-17, default is 14)
   - Dynamic axes (optional)
4. Click "Export"
5. Choose a destination file

### Exporting to TorchScript

1. Load a PyTorch model in your Python session
2. In the Model Export panel, select "TorchScript" as the format
3. Choose export method:
   - Tracing (for models without control flow)
   - Scripting (for models with control flow)
4. Configure input shape
5. Click "Export"
6. Choose a destination file

### Exporting TensorFlow Models

1. Load a TensorFlow model in your Python session
2. In the Model Export panel, select the desired format:
   - SavedModel
   - TensorFlow Lite
   - ONNX (requires tf2onnx)
3. Configure export options
4. Click "Export"
5. Choose a destination file

### Exporting JAX Models

1. Load a JAX model in your Python session
2. In the Model Export panel, select the desired format:
   - Pickle (default for JAX)
   - ONNX (experimental)
3. Configure export options
4. Click "Export"
5. Choose a destination file

### AOT Compilation

1. Load a PyTorch model (requires PyTorch 2.0+)
2. In the Model Export panel, select "AOT Compilation"
3. Configure optimization options:
   - **Optimization Level**: 
     - none: No optimizations
     - basic: Basic optimizations (default)
     - extended: More aggressive optimizations
     - full: All available optimizations
   - **Target Hardware**:
     - CPU: For CPU deployment (most compatible)
     - CUDA: For NVIDIA GPU acceleration
     - Vulkan: For cross-platform GPU support
     - MPS: For Apple Metal Performance Shaders
4. Click "Compile"
5. Choose a destination file

### Comparing Model Performance

1. Export your model using different methods
2. Use the benchmarking tools in the extension:
   ```
   # Through VS Code command palette
   > AI Debugger: Benchmark Exported Models
   ```
3. View performance comparison in the Results panel:
   - Inference speed
   - Memory usage
   - Model size
   - Compatibility reports

## Framework Support & Configuration

The extension supports multiple ML frameworks with specific features for each. See [FRAMEWORK_SUPPORT.md](FRAMEWORK_SUPPORT.md) for detailed information.

### PyTorch Support

- Model architecture visualization
- Tensor inspection and visualization
- Training control and monitoring
- Export to ONNX, TorchScript, and other formats
- AOT compilation for deployment with multiple optimization levels

### TensorFlow/Keras Support

- Model architecture visualization
- Tensor inspection and visualization
- Training control and monitoring
- Export to ONNX, SavedModel, and TFLite formats

### JAX/Flax Support

- Model architecture visualization
- Parameter inspection and visualization
- Training monitoring
- Export to pickle and ONNX formats

### ONNX Support

- Model architecture visualization
- Import and export of ONNX models
- Conversion between frameworks via ONNX

## Troubleshooting

For detailed troubleshooting information, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Common Issues

#### Extension Not Connecting to Model

1. Verify that the model is running with debugging enabled
2. Check that the required Python packages are installed
3. Ensure the model is compatible with the extension
4. Check the Output panel for error messages

#### Slow Performance

1. Reduce the sampling rate in settings
2. Disable auto-refresh for visualizations
3. Limit the number of tensors being monitored
4. Use a more powerful machine for very large models

#### Framework Detection Issues

1. Manually specify the framework in settings
2. Ensure the framework is properly installed
3. Check for version compatibility
4. Restart VS Code after installing frameworks

### Getting Help

- Check the [GitHub repository](https://github.com/yashh1321/AI-ML-Debugger) for known issues
- Submit a new issue with detailed information

## Advanced Usage

### Custom Visualizations

You can create custom visualizations for your tensors:

1. Click the "Custom Visualization" button in the Tensor Inspector
2. Write a custom visualization script
3. Configure visualization parameters
4. Click "Apply"

### Remote Debugging

To debug models running on remote machines:

1. Configure SSH connection in VS Code
2. Install the extension on the remote machine
3. Connect to the remote session
4. Use the extension as normal

### Integration with Other Extensions

The AI/ML Debugger extension works well with:

- Python extension for VS Code
- Jupyter Notebooks extension
- Remote Development extension
- GitHub Copilot

### Performance Optimization

For large models or datasets:

1. Enable the "Performance Mode" in settings
2. Reduce the sampling rate
3. Disable auto-refresh for visualizations
4. Use selective tensor monitoring