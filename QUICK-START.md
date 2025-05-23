# 🚀 AI/ML Debugger - Quick Start Guide

Get up and running with the most comprehensive ML debugging extension for VS Code in just 5 minutes!

## ⚡ Installation (1 minute)

### Method 1: From VSIX File
```bash
# Download the latest VSIX and install
code --install-extension vscode-ai-debugger-1.7.1.vsix
```

### Method 2: From Extensions Marketplace
1. Open VS Code Extensions view (`Ctrl+Shift+X`)
2. Search for "AI/ML Debugger"
3. Click **Install**

### Method 3: From Command Line
```bash
# Clone and build from source
git clone https://github.com/yashh1321/AI-ML-Debugger.git
cd AI-ML-Debugger
npm install && npm run package
code --install-extension vscode-ai-debugger-*.vsix
```

## 🎯 First Launch (2 minutes)

### 1. Open the Dashboard
Press `Ctrl+Alt+D` (or `Cmd+Alt+D` on Mac) to launch the unified dashboard.

### 2. Run Setup Wizard
- Click **Quick Start** or press `Ctrl+Alt+Q`
- The wizard will:
  - Detect your Python environment
  - Check ML framework installations
  - Set up optimal configurations
  - Install missing dependencies automatically

### 3. Auto-Detect Your Models
- Press `Ctrl+Alt+A` or click **Auto-Detect**
- The extension will:
  - Scan your workspace for ML code
  - Identify model architectures
  - Set up debugging configurations
  - Activate relevant debugging views

## 🏗️ Basic Usage (2 minutes)

### Your First Model Debugging Session

1. **Open a Python file** with ML model code:
```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))  # Set breakpoint here
        return self.linear2(x)

model = SimpleModel()
```

2. **Set a breakpoint** on the forward pass line

3. **Open Model Explorer** - Click the model icon in the activity bar or use Command Palette

4. **View Architecture** - See your model structure visualized instantly

5. **Start Debugging** - Run your code and explore tensors when breakpoint hits

## 📊 Essential Dashboard Features

### **Core Debugging Views** (Left Activity Bar)
- 🏗️ **Model Architecture** - Interactive model visualization
- 🔍 **Tensor Inspector** - Deep tensor analysis  
- 📊 **Metrics Dashboard** - Real-time training metrics
- ⏯️ **Training Console** - Step-through training control

### **Quick Actions** (Dashboard Center)
- **Smart LR Test** - Find optimal learning rates
- **Data Health Check** - Analyze dataset quality
- **Privacy Wizard** - Set up differential privacy
- **Model Benchmark** - Compare model performance

### **Advanced Tools** (Bottom Panel)  
- 🧪 **Experiment Tracker** - MLflow/W&B integration
- ⚡ **Performance Profiler** - CPU/GPU profiling
- 🔬 **Explainability Tools** - SHAP/LIME analysis
- ☁️ **Remote Debugger** - Cloud platform integration

## ⌨️ Essential Keyboard Shortcuts

| Shortcut | Action | When to Use |
|----------|--------|-------------|
| `Ctrl+Alt+D` | Open Dashboard | Start any debugging session |
| `Ctrl+Alt+P` | Command Palette | Quick access to 124 commands |
| `Ctrl+Alt+Q` | Quick Start | Setup new projects |
| `Ctrl+Alt+A` | Auto-Detect | Find models automatically |

## 🎯 Common Workflows

### **Debugging Training Issues**
1. Open **Error Detection Panel**
2. Set breakpoints in training loop
3. Use **Gradient Visualizer** to check gradients
4. Monitor metrics in **Metrics Dashboard**
5. Get AI suggestions from **LLM Copilot**

### **Model Architecture Analysis**
1. Open **Model Explorer** 
2. Load your model code
3. View layer-by-layer details
4. Check tensor shapes and connections
5. Export architecture diagrams

### **Performance Optimization**
1. Use **Auto-Tuning Optimizer** for learning rates
2. Run **Performance Profiler** for bottlenecks
3. Check **Data Pipeline Debugger** for data loading
4. Compare results with **Cross-Model Comparison**

### **Data Quality Analysis**
1. Open **Data-Centric Debugger**
2. Run **Data Health Check**
3. Detect drift with **Data Drift Detection**
4. Identify noise with **Label Noise Detection**

## 🛠️ Framework-Specific Setup

### **PyTorch Projects**
```python
# The extension auto-detects PyTorch models
import torch
import torch.nn as nn

# Your model will appear in Model Explorer automatically
class MyModel(nn.Module):
    # ... model definition
    pass
```

### **TensorFlow/Keras Projects**
```python
# Works with tf.keras models out of the box
import tensorflow as tf

# Model structure will be visualized automatically  
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```

### **JAX/Flax Projects**
```python
# JAX models are supported with Flax integration
import jax
import flax.linen as nn

# Model architecture will be detected and visualized
class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(10)(x)
```

## 🚀 Advanced Setup (Optional)

### **Cloud Integration**
1. **AWS SageMaker**: Use `connectToSageMaker` command
2. **Google Vertex AI**: Use `connectToVertexAI` command  
3. **Azure ML**: Use `connectToAzureML` command
4. **SSH Remote**: Use `connectViaSSH` command

### **Experiment Tracking**
1. **MLflow**: Configure endpoint in settings
2. **Weights & Biases**: Set API key in configuration
3. **Neptune**: Add project credentials
4. **Built-in**: Works automatically with local storage

### **Privacy-Aware Training**
1. Enable **Differential Privacy** in settings
2. Configure **Privacy Budget** parameters
3. Use **Privacy Wizard** for guided setup
4. Monitor with **Privacy Timeline**

## 🔧 Troubleshooting

### **Common Issues & Solutions**

**❌ Extension not loading**
- Solution: Check Python installation and restart VS Code

**❌ Models not detected**  
- Solution: Use `Ctrl+Alt+A` to force auto-detection

**❌ Python dependencies missing**
- Solution: Use `installDependencies` command for automatic setup

**❌ Performance issues**
- Solution: Adjust sampling rate in settings (`aiDebugger.samplingRate`)

### **Get Help**
- **Command Palette**: Search "AI Debugger" for all available commands
- **Tutorials**: Use `showTutorialsHub` for interactive guides
- **Documentation**: Access comprehensive guides from dashboard
- **Community**: Join discussions and get support

## 🎉 You're Ready!

**Congratulations!** You now have the most powerful ML debugging extension installed and configured. 

### **Next Steps:**
1. **Explore Features** - Try different debugging views and tools
2. **Customize Layout** - Save your preferred debugging setup  
3. **Install Plugins** - Extend functionality with custom plugins
4. **Share Setup** - Export configurations for team collaboration

### **Pro Tips:**
- 💡 Use the **LLM Copilot** for debugging advice
- 💡 Set up **Custom Layouts** for different project types
- 💡 Enable **Smart Alerts** for proactive issue detection
- 💡 Use **Performance Timeline** for production debugging

**Happy ML Debugging! 🚀🤖**

---

## 📚 What's Next?

- **[Complete Features Guide](FEATURES_OVERVIEW.md)** - Explore all 124 commands and 25 views
- **[User Guide](USER_GUIDE.md)** - Comprehensive usage documentation  
- **[Framework Support](FRAMEWORK_SUPPORT.md)** - Detailed framework compatibility
- **[Performance Guide](PERFORMANCE_OPTIMIZATION.md)** - Optimization best practices