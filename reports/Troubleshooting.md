# VS Code AI/ML Debugger - Troubleshooting Guide

> Updated: May 23, 2025 with icon fixes, improved JAX framework detection, and comprehensive testing

> This guide provides solutions for common issues encountered when using the VS Code AI/ML Debugger extension. If you encounter any problems not covered here, please check our GitHub repository or submit an issue.

## Automatic Dependency Management

The extension features a zero-configuration setup that automatically handles dependencies. When you first run the extension:

1. It creates a dedicated Python virtual environment in the extension directory
2. Automatically installs all required packages
3. Starts the helper process with the proper environment

This process happens automatically without requiring any manual steps.

### Troubleshooting Automatic Setup

If you encounter issues with the automatic setup:

1. Check the status bar indicator at the bottom of VS Code
2. Click on it to test the Python helper process connection
3. If it shows an error, try using the "AI Debugger: Restart Python Helper" command
4. For dependency issues, use the "AI Debugger: Install Dependencies" command

## Common Issues and Solutions

### Activity Bar Icon Not Showing

**Issue:**
The AI/ML Debugger icon doesn't appear in the activity bar after installation.

**Solution:**
This issue has been fixed in version 1.3.3. If you're using an older version, please update to the latest version. If the issue persists:

1. Try reloading VS Code window (Ctrl+Shift+P -> "Developer: Reload Window")
2. Check if the extension is enabled in the Extensions view
3. Reinstall the extension if needed

### Framework Detection Issues

#### PyTorch Not Detected

**Symptoms:**
- PyTorch is installed but not detected by the extension
- Error message: "'module' object has no attribute 'TensorBase'"
- Only TensorFlow appears in the framework list

**Solutions:**
1. Ensure you have PyTorch installed in your Python environment:
   ```bash
   pip install torch torchvision
   ```

2. Check your PyTorch installation:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

3. If using a newer PyTorch version (2.0+), make sure the extension is updated to the latest version

#### TensorFlow Not Detected

**Symptoms:**
- TensorFlow is installed but not detected by the extension
- Only PyTorch appears in the framework list

**Solutions:**
1. Ensure you have TensorFlow installed in your Python environment:
   ```bash
   pip install tensorflow
   ```

2. Check your TensorFlow installation:
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

#### JAX/Flax Not Detected

**Symptoms:**
- JAX is installed but not detected by the extension
- Only PyTorch or TensorFlow appears in the framework list

**Solutions:**
1. Ensure you have JAX and Flax installed in your Python environment:
   ```bash
   pip install jax jaxlib flax
   ```

2. Check your JAX installation:
   ```bash
   python -c "import jax; print(jax.__version__)"
   ```

3. On Windows, you might need to use CPU-only JAX version:
   ```bash
   pip install --upgrade "jax[cpu]"
   ```

### Model Export Issues

#### LSTM Model ONNX Export Warning

**Symptoms:**
- Warning message when exporting LSTM models to ONNX: "Exporting a model to ONNX with a batch_size other than 1..."

**Solutions:**
1. Use batch size of 1 for ONNX export:
   ```python
   # Instead of using variable batch size
   dummy_input = torch.randn(batch_size, seq_len, input_size)
   
   # Use fixed batch size of 1 for export
   dummy_input = torch.randn(1, seq_len, input_size)
   ```

2. Create a new LSTM model with batch_first=True specifically for export:
   ```python
   class ExportReadyLSTM(nn.Module):
       def __init__(self, input_size, hidden_size, num_layers, output_size):
           super().__init__()
           self.lstm = nn.LSTM(
               input_size, 
               hidden_size, 
               num_layers, 
               batch_first=True  # Important for ONNX export
           )
           self.fc = nn.Linear(hidden_size, output_size)
           
       def forward(self, x):
           # Initialize states inside the forward method
           h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
           c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
           out, _ = self.lstm(x, (h0, c0))
           out = self.fc(out[:, -1, :])
           return out
   
   # Create export-ready model
   export_model = ExportReadyLSTM(input_size, hidden_size, num_layers, output_size)
   ```

#### TensorFlow to ONNX Conversion Failing

**Symptoms:**
- Error message: "tf2onnx not available. Skipping TensorFlow to ONNX conversion."

**Solutions:**
1. Install the tf2onnx package:
   ```bash
   pip install tf2onnx
   ```

2. Make sure your TensorFlow model is compatible with ONNX conversion.

#### JAX Model Export Issues

**Symptoms:**
- Error when trying to export JAX models
- JAX models not correctly recognized for export

**Solutions:**
1. Ensure you're using the latest extension version (1.3.3+) which has improved JAX support
2. Use the correct export format - JAX models work best with pickle format:
   ```python
   import pickle
   with open("jax_model.pkl", "wb") as f:
       pickle.dump({"params": params, "model_def": model}, f)
   ```

### Performance Issues

#### Slow Model Visualization

**Symptoms:**
- Model architecture viewer is slow to render for large models
- Browser becomes unresponsive when viewing complex architectures

**Solutions:**
1. Use simplified view option in the model explorer
2. Reduce the number of layers displayed at once
3. Use the "Export as SVG" option and view the exported file separately

#### High Memory Usage

**Symptoms:**
- Extension crashes or becomes unresponsive when working with large models or tensors
- Out of memory errors

**Solutions:**
1. Reduce the sample size when visualizing tensors
2. Disable auto-refresh of visualizations
3. Close unused panels and views
4. Increase your system's memory allocation for VS Code

## Debugging the Extension

### Enable Verbose Logging

1. Open VS Code settings (`Ctrl+,`)
2. Search for "AI Debugger log level"
3. Set the log level to "Verbose"
4. Check the Output panel (View -> Output) and select "AI Debugger" from the dropdown

### Check Python Helper Process

1. Run the "AI Debugger: Ping Python Helper" command
2. Verify that the Python helper process is running
3. Check if the correct Python environment is being used

### Manual Installation of Dependencies

If automatic dependency installation fails, you can manually install the required packages:

```bash
# Navigate to the extension directory
cd ~/.vscode/extensions/yashh130021.vscode-ai-debugger-x.x.x/python_helper

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install tf2onnx
```

## Recent Fixes (May 2025)

### Fixed: Activity Bar Icon Not Showing

**Issue:**
- AI/ML Debugger icon missing from the VS Code activity bar
- Icon path incorrect in package.json configuration

**Fix:**
- Updated icon path to correctly point to existing SVG file
- Improved icon contrast for better visibility in both light and dark themes

### Fixed: JAX/Flax Framework Detection

**Issue:**
- JAX framework not properly detected even when installed
- Detection logic only looking for PyTorch and TensorFlow

**Fix:**
- Added explicit JAX and Flax detection in framework discovery
- Updated dependency checking to include JAX-specific packages
- Added JAX-specific model loading and export functionality

### Fixed: Dependency Management Issues

**Issue:**
- Optional dependencies not clearly indicated
- Some installations failing due to missing prerequisites

**Fix:**
- Improved dependency checking and auto-installation
- Better error messages for missing or incompatible packages
- Added framework-specific installation commands

## Still Having Issues?

If you're still experiencing problems after trying the solutions above, please:

1. Check our GitHub repository for known issues
2. Submit a detailed bug report including:
   - VS Code version
   - Extension version
   - Python version
   - ML framework version
   - Steps to reproduce the issue
   - Error messages or screenshots

We're committed to improving the extension and appreciate your feedback!