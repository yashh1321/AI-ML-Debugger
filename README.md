# VS Code AI Debugger

A powerful Visual Studio Code extension for debugging and visualizing machine learning models, with support for PyTorch, TensorFlow, and JAX.

## Features

- **Multi-Framework Support**: Works with PyTorch, TensorFlow, and JAX
- **Automatic Framework Detection**: Identifies which ML framework your project uses
- **Zero-Config Setup**: Automatically installs missing dependencies when needed
- **Model Explorer**: Visualize and navigate model architecture
- **Tensor Inspector**: Examine tensor values and shapes during training
- **Metrics Dashboard**: Track and visualize training progress
- **Gradient Visualization**: See gradient flow through your model
- **Model Export**: One-click export to ONNX or framework-specific formats

## What's New in 1.3.1

- Added support for TensorFlow models with automatic detection
- Added support for JAX models with automatic detection
- Improved dependency management with automatic installation of required frameworks
- Fixed extension icon loading issue
- Enhanced error handling for framework detection

## Installation

1. Download the VSIX file from the releases section
2. In VS Code, go to Extensions view
3. Click "..." in the top-right corner
4. Select "Install from VSIX..."
5. Choose the downloaded file

## Requirements

- VS Code 1.60.0 or higher
- Python 3.8 or higher

## Usage

1. Open a Python file containing your ML model
2. Run the command "Start AI Debugger" from the command palette
3. The extension will automatically detect and install required frameworks
4. Use the sidebar tools to explore and debug your model

## License

MIT