#!/usr/bin/env python3
"""
Tutorials & Community Hub Manager for AI/ML Debugger
Handles interactive tutorials and community content
"""

import json
import os
import sys
from typing import Dict, Any
from pathlib import Path

class TutorialsManager:
    """Manages tutorials and community content"""

    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path or os.getcwd()
        self.tutorials_dir = os.path.join(self.workspace_path, ".vscode", "ai-debugger-tutorials")
        self.templates_dir = os.path.join(self.tutorials_dir, "templates")
        self.community_dir = os.path.join(self.tutorials_dir, "community")
        self.ensure_directories()

        # Built-in tutorials
        self.builtin_tutorials = {
            "pytorch_basics": {
                "id": "pytorch_basics",
                "title": "PyTorch Basics with Debugging",
                "description": "Learn PyTorch fundamentals with integrated debugging",
                "framework": "pytorch",
                "difficulty": "beginner",
                "duration": "15 minutes",
                "tags": ["pytorch", "neural-networks", "basics"],
                "author": "AI Debugger Team",
                "steps": [
                    {
                        "title": "Create Your First Neural Network",
                        "description": "Define a simple feedforward network",
                        "code_template": "pytorch_basic_nn.py",
                        "debugging_points": ["model_architecture", "forward_pass"]
                    },
                    {
                        "title": "Train with Debugging",
                        "description": "Training loop with tensor inspection",
                        "code_template": "pytorch_training_loop.py",
                        "debugging_points": ["loss_tracking", "gradient_flow"]
                    }
                ]
            },
            "cnn_vision": {
                "id": "cnn_vision",
                "title": "Computer Vision with CNNs",
                "description": "Build and debug convolutional neural networks",
                "framework": "pytorch",
                "difficulty": "intermediate",
                "duration": "30 minutes",
                "tags": ["pytorch", "cnn", "computer-vision"],
                "author": "AI Debugger Team",
                "steps": [
                    {
                        "title": "CNN Architecture",
                        "description": "Design a CNN for image classification",
                        "code_template": "cnn_architecture.py",
                        "debugging_points": ["conv_layers", "feature_maps"]
                    },
                    {
                        "title": "Data Pipeline",
                        "description": "Set up data loading and augmentation",
                        "code_template": "vision_dataloader.py",
                        "debugging_points": ["batch_inspection", "augmentation_preview"]
                    }
                ]
            },
            "transformer_nlp": {
                "id": "transformer_nlp",
                "title": "Transformer Models for NLP",
                "description": "Debug attention mechanisms and sequence models",
                "framework": "pytorch",
                "difficulty": "advanced",
                "duration": "45 minutes",
                "tags": ["pytorch", "transformers", "nlp", "attention"],
                "author": "AI Debugger Team",
                "steps": [
                    {
                        "title": "Attention Mechanisms",
                        "description": "Implement and visualize attention",
                        "code_template": "attention_mechanism.py",
                        "debugging_points": ["attention_weights", "sequence_processing"]
                    },
                    {
                        "title": "Training Dynamics",
                        "description": "Monitor transformer training",
                        "code_template": "transformer_training.py",
                        "debugging_points": ["gradient_flow", "attention_patterns"]
                    }
                ]
            }
        }

        # Project templates
        self.project_templates = {
            "image_classification": {
                "id": "image_classification",
                "name": "Image Classification (CIFAR-10)",
                "description": "Complete image classification project with ResNet",
                "framework": "pytorch",
                "category": "computer_vision",
                "files": [
                    "model.py",
                    "dataset.py",
                    "train.py",
                    "utils.py",
                    "config.py"
                ],
                "debugging_presets": ["gradient_monitoring", "activation_inspection"]
            },
            "sentiment_analysis": {
                "id": "sentiment_analysis",
                "name": "Sentiment Analysis (BERT)",
                "description": "NLP sentiment analysis with BERT fine-tuning",
                "framework": "pytorch",
                "category": "nlp",
                "files": [
                    "bert_model.py",
                    "data_preprocessing.py",
                    "train.py",
                    "inference.py",
                    "config.py"
                ],
                "debugging_presets": ["embedding_inspection", "attention_heatmaps"]
            },
            "gan_training": {
                "id": "gan_training",
                "name": "GAN Training (DCGAN)",
                "description": "Generative Adversarial Network implementation",
                "framework": "pytorch",
                "category": "generative_ai",
                "files": [
                    "generator.py",
                    "discriminator.py",
                    "train_gan.py",
                    "utils.py",
                    "config.py"
                ],
                "debugging_presets": ["gan_stability", "mode_collapse_detection"]
            }
        }

    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        os.makedirs(self.tutorials_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.community_dir, exist_ok=True)

    def list_tutorials(self) -> Dict[str, Any]:
        """List all available tutorials"""
        try:
            tutorials = {}

            # Add built-in tutorials
            for tutorial_id, tutorial in self.builtin_tutorials.items():
                tutorials[tutorial_id] = {
                    **tutorial,
                    "source": "builtin",
                    "installed": True
                }

            # Add community tutorials
            community_file = os.path.join(self.community_dir, "tutorials.json")
            if os.path.exists(community_file):
                with open(community_file, 'r') as f:
                    community_tutorials = json.load(f)
                    for tutorial_id, tutorial in community_tutorials.items():
                        tutorials[tutorial_id] = {
                            **tutorial,
                            "source": "community",
                            "installed": os.path.exists(os.path.join(self.community_dir, tutorial_id))
                        }

            return {
                "status": "success",
                "tutorials": tutorials
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to list tutorials: {str(e)}"
            }

    def start_tutorial(self, tutorial_id: str, project_path: str = None) -> Dict[str, Any]:
        """Start an interactive tutorial"""
        try:
            # Get tutorial info
            if tutorial_id in self.builtin_tutorials:
                tutorial = self.builtin_tutorials[tutorial_id]
                source = "builtin"
            else:
                # Try community tutorials
                community_file = os.path.join(self.community_dir, "tutorials.json")
                if os.path.exists(community_file):
                    with open(community_file, 'r') as f:
                        community_tutorials = json.load(f)
                        if tutorial_id in community_tutorials:
                            tutorial = community_tutorials[tutorial_id]
                            source = "community"
                        else:
                            return {"status": "error", "message": f"Tutorial '{tutorial_id}' not found"}
                else:
                    return {"status": "error", "message": f"Tutorial '{tutorial_id}' not found"}

            # Create project directory
            if not project_path:
                project_path = os.path.join(self.workspace_path, f"tutorial_{tutorial_id}")

            os.makedirs(project_path, exist_ok=True)

            # Generate tutorial files
            self._generate_tutorial_files(tutorial, project_path, source)

            # Create tutorial progress file
            progress_file = os.path.join(project_path, ".tutorial_progress.json")
            progress = {
                "tutorial_id": tutorial_id,
                "current_step": 0,
                "completed_steps": [],
                "started_at": str(Path().resolve())
            }

            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

            return {
                "status": "success",
                "message": f"Tutorial '{tutorial['title']}' started",
                "project_path": project_path,
                "tutorial": tutorial
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to start tutorial: {str(e)}"
            }

    def _generate_tutorial_files(self, tutorial: Dict, project_path: str, source: str):
        """Generate tutorial code files"""
        # Create main tutorial file
        tutorial_file = os.path.join(project_path, "tutorial_guide.md")
        with open(tutorial_file, 'w') as f:
            f.write(f"# {tutorial['title']}\n\n")
            f.write(f"**Description:** {tutorial['description']}\n")
            f.write(f"**Framework:** {tutorial['framework']}\n")
            f.write(f"**Difficulty:** {tutorial['difficulty']}\n")
            f.write(f"**Duration:** {tutorial['duration']}\n\n")

            f.write("## Steps\n\n")
            for i, step in enumerate(tutorial['steps'], 1):
                f.write(f"### Step {i}: {step['title']}\n\n")
                f.write(f"{step['description']}\n\n")
                f.write("**Debugging Points:**\n")
                for point in step.get('debugging_points', []):
                    f.write(f"- {point}\n")
                f.write("\n")

        # Generate code templates for each step
        for i, step in enumerate(tutorial['steps']):
            template_name = step.get('code_template', f"step_{i+1}.py")
            self._create_code_template(tutorial['framework'], template_name, project_path, step)

    def _create_code_template(self, framework: str, template_name: str, project_path: str, step: Dict):
        """Create code template for tutorial step"""
        template_path = os.path.join(project_path, template_name)

        if framework == "pytorch" and "basic_nn" in template_name:
            code = '''import torch
import torch.nn as nn
import torch.optim as optim

# [TUTORIAL STEP 1]: Define your neural network
# Set a breakpoint here to explore the model architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        # [DEBUG TIP]: Use the model explorer to visualize this architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # [DEBUG POINT]: Set breakpoint here to inspect tensor shapes
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model instance
model = SimpleNN()
print(f"Model architecture: {model}")

# [TUTORIAL TASK]: Set a breakpoint on the next line and explore the model
# Use the tensor inspector to examine model parameters
for name, param in model.named_parameters():
    print(f"Layer: {name}, Shape: {param.shape}")
'''
        elif framework == "pytorch" and "training_loop" in template_name:
            code = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# [TUTORIAL STEP 2]: Training with debugging
def train_model():
    # Create dummy data for tutorial
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(dataloader):
            # [DEBUG POINT]: Set breakpoint here to inspect batch data
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # [DEBUG POINT]: Set breakpoint here to examine loss and gradients
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                # [TUTORIAL TIP]: Use the metrics dashboard to track this loss

if __name__ == "__main__":
    train_model()
'''
        else:
            # Generic template
            code = f'''# {step["title"]}
# {step["description"]}

# [TUTORIAL]: This is a placeholder template
# Add your {framework} code here

print("Tutorial step: {step['title']}")
'''

        with open(template_path, 'w') as f:
            f.write(code)

    def list_project_templates(self) -> Dict[str, Any]:
        """List all available project templates"""
        try:
            return {
                "status": "success",
                "templates": self.project_templates
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to list templates: {str(e)}"
            }

    def create_project_from_template(self, template_id: str, project_name: str, project_path: str = None) -> Dict[str, Any]:
        """Create a new project from template"""
        try:
            if template_id not in self.project_templates:
                return {"status": "error", "message": f"Template '{template_id}' not found"}

            template = self.project_templates[template_id]

            # Create project directory
            if not project_path:
                project_path = os.path.join(self.workspace_path, project_name)

            os.makedirs(project_path, exist_ok=True)

            # Generate project files
            self._generate_project_files(template, project_path, project_name)

            return {
                "status": "success",
                "message": f"Project '{project_name}' created from template '{template['name']}'",
                "project_path": project_path,
                "template": template
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create project: {str(e)}"
            }

    def _generate_project_files(self, template: Dict, project_path: str, project_name: str):
        """Generate project files from template"""
        # Create README
        readme_path = os.path.join(project_path, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"# {project_name}\n\n")
            f.write(f"**Template:** {template['name']}\n")
            f.write(f"**Description:** {template['description']}\n")
            f.write(f"**Framework:** {template['framework']}\n")
            f.write(f"**Category:** {template['category']}\n\n")
            f.write("## Debugging Presets\n\n")
            for preset in template.get('debugging_presets', []):
                f.write(f"- {preset}\n")
            f.write("\n## Files\n\n")
            for file in template['files']:
                f.write(f"- `{file}`: Generated template file\n")

        # Create project files based on template
        if template['id'] == 'image_classification':
            self._create_image_classification_files(project_path)
        elif template['id'] == 'sentiment_analysis':
            self._create_sentiment_analysis_files(project_path)
        elif template['id'] == 'gan_training':
            self._create_gan_training_files(project_path)

    def _create_image_classification_files(self, project_path: str):
        """Create image classification project files"""
        # model.py
        model_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # [DEBUG POINT]: Set breakpoint here to inspect feature maps
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # [DEBUG POINT]: Monitor activation patterns through the network
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
'''

        with open(os.path.join(project_path, "model.py"), 'w') as f:
            f.write(model_code)

        # train.py
        train_code = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model import ResNet

def train():
    # [DEBUG TIP]: Use the error detection panel to monitor training issues
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # Model, loss, optimizer
    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # [DEBUG POINT]: Set breakpoint here to inspect batch data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # [DEBUG TIP]: Monitor gradients for vanishing/exploding issues
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        scheduler.step()
        print(f'Epoch {epoch} completed, Average Loss: {running_loss/len(trainloader):.4f}')

if __name__ == "__main__":
    train()
'''

        with open(os.path.join(project_path, "train.py"), 'w') as f:
            f.write(train_code)

def main():
    """Main function for tutorials manager"""
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"error": "No command provided"}))
            return

        command = sys.argv[1]
        workspace_path = sys.argv[2] if len(sys.argv) > 2 else None
        manager = TutorialsManager(workspace_path)

        if command == "list_tutorials":
            result = manager.list_tutorials()
            print(json.dumps(result))

        elif command == "start_tutorial":
            tutorial_id = sys.argv[3]
            project_path = sys.argv[4] if len(sys.argv) > 4 else None
            result = manager.start_tutorial(tutorial_id, project_path)
            print(json.dumps(result))

        elif command == "list_templates":
            result = manager.list_project_templates()
            print(json.dumps(result))

        elif command == "create_project":
            template_id = sys.argv[3]
            project_name = sys.argv[4]
            project_path = sys.argv[5] if len(sys.argv) > 5 else None
            result = manager.create_project_from_template(template_id, project_name, project_path)
            print(json.dumps(result))

        else:
            print(json.dumps({"error": f"Unknown command: {command}"}))

    except Exception as e:
        error_result = {
            "error": str(e),
            "command": command if 'command' in locals() else "unknown"
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    main()