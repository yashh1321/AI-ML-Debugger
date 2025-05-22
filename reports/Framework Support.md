# Framework Support

The AI/ML Debugger extension supports multiple deep learning frameworks with varying levels of functionality.

## PyTorch

**Support Level**: Full

### Features

- Complete model architecture visualization
- Tensor inspection for all tensor types
- Real-time training metrics visualization
- Gradient inspection and visualization
- Export to ONNX and TorchScript
- Model editing and code generation

### Requirements

- PyTorch 1.7.0 or higher
- torchvision (optional, for vision models)

### Example

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleModel()
input_tensor = torch.randn(1, 10)
output = model(input_tensor)

# Set a breakpoint here to inspect the model
```

## TensorFlow

**Support Level**: Advanced

### Features

- Model architecture visualization
- Tensor inspection
- Training metrics visualization
- Export to SavedModel and TFLite
- Limited gradient visualization

### Requirements

- TensorFlow 2.0.0 or higher
- Keras (included with TensorFlow)

### Example

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1)
])

input_tensor = tf.random.normal((1, 10))
output = model(input_tensor)

# Set a breakpoint here to inspect the model
```

## JAX/Flax

**Support Level**: Basic

### Features

- Model architecture visualization
- Basic tensor inspection
- Limited training metrics visualization

### Requirements

- JAX 0.3.0 or higher
- Flax 0.4.0 or higher
- Optax (optional, for optimizers)

### Example

```python
import jax
import jax.numpy as jnp
import flax.linen as nn

class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(50)(x)
        x = nn.relu(x)
        x = nn.Dense(20)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

model = SimpleModel()
key = jax.random.PRNGKey(0)
variables = model.init(key, jnp.ones((1, 10)))
input_tensor = jnp.ones((1, 10))
output = model.apply(variables, input_tensor)

# Set a breakpoint here to inspect the model
```

## Framework Auto-Detection

The extension automatically detects which framework you're using based on the imports and objects in your code. You can also manually select your preferred framework in the extension settings:

1. Open VS Code settings (File > Preferences > Settings)
2. Search for "AI Debugger"
3. Set your preferred framework under "AI Debugger: Framework: Preferred Framework"

## Framework-Specific Configuration

You can customize framework-specific settings in the extension settings panel under the "AI Debugger" section.