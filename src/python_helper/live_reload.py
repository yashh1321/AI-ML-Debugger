#!/usr/bin/env python3
"""
Live Code Reload + Hot Swapping for AI/ML Debugger
Enable hot-swapping of model components without restarting training
"""

import json
import sys
import traceback
import os
import importlib
import inspect
import time
import shutil
from typing import Dict, List, Any
import logging
from datetime import datetime
import sqlite3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional ML framework imports
try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf  # noqa: F401
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import jax  # noqa: F401
    import flax  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

class LiveReloadManager:
    """Manages live code reloading and hot swapping for ML training"""

    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path or os.getcwd()
        self.reload_db_path = os.path.join(self.workspace_path, ".vscode", "live-reload.db")
        self.backup_dir = os.path.join(self.workspace_path, ".vscode", "reload-backups")
        self.ensure_database()
        self.ensure_backup_dir()

        # Reload state
        self.active_sessions = {}
        self.watched_files = set()
        self.file_observer = None
        self.reload_handlers = {}

        # Hot-swap registry
        self.registered_models = {}
        self.registered_optimizers = {}
        self.registered_loss_functions = {}
        self.registered_data_loaders = {}

        # Code modification tracking
        self.original_modules = {}
        self.modification_history = []

        # Safety checks
        self.max_reload_attempts = 3
        self.reload_cooldown = 2.0  # seconds
        self.last_reload_time = {}

    def ensure_database(self):
        """Create reload tracking database"""
        os.makedirs(os.path.dirname(self.reload_db_path), exist_ok=True)

        with sqlite3.connect(self.reload_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS reload_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_reloads INTEGER,
                    successful_reloads INTEGER,
                    failed_reloads INTEGER,
                    metadata TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS reload_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TIMESTAMP,
                    file_path TEXT,
                    component_type TEXT,
                    component_name TEXT,
                    reload_type TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    backup_path TEXT,
                    FOREIGN KEY (session_id) REFERENCES reload_sessions (session_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS component_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    component_id TEXT,
                    component_type TEXT,
                    component_name TEXT,
                    file_path TEXT,
                    class_name TEXT,
                    registration_time TIMESTAMP,
                    last_modified TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES reload_sessions (session_id)
                )
            ''')

    def ensure_backup_dir(self):
        """Create backup directory for code snapshots"""
        os.makedirs(self.backup_dir, exist_ok=True)

    def start_session(self, session_id: str, watch_patterns: List[str] = None) -> Dict[str, Any]:
        """Start a live reload session"""
        try:
            if session_id in self.active_sessions:
                return {"status": "error", "message": "Session already active"}

            # Default watch patterns
            if watch_patterns is None:
                watch_patterns = ["*.py", "models/*.py", "src/*.py"]

            # Create session record
            with sqlite3.connect(self.reload_db_path) as conn:
                conn.execute('''
                    INSERT INTO reload_sessions
                    (session_id, start_time, total_reloads, successful_reloads, failed_reloads, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (session_id, datetime.now(), 0, 0, 0, json.dumps({"watch_patterns": watch_patterns})))

            # Initialize session state
            self.active_sessions[session_id] = {
                "start_time": datetime.now(),
                "watch_patterns": watch_patterns,
                "total_reloads": 0,
                "successful_reloads": 0,
                "failed_reloads": 0,
                "registered_components": {}
            }

            # Start file watching
            self.start_file_watching(session_id, watch_patterns)

            return {
                "status": "success",
                "session_id": session_id,
                "message": "Live reload session started",
                "watch_patterns": watch_patterns
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def register_component(self, session_id: str, component_id: str,
                          component: Any, component_type: str,
                          file_path: str = None) -> Dict[str, Any]:
        """Register a component for hot swapping"""
        try:
            if session_id not in self.active_sessions:
                return {"status": "error", "message": "Session not active"}

            # Determine component details
            component_name = getattr(component, '__name__', str(component))
            class_name = component.__class__.__name__

            if file_path is None:
                # Try to get file path from component
                try:
                    file_path = inspect.getfile(component.__class__)
                except (TypeError, AttributeError):
                    file_path = "unknown"

            # Store original state for rollback
            if component_type == "model" and hasattr(component, 'state_dict'):
                # PyTorch model
                original_state = component.state_dict().copy()
                self.registered_models[component_id] = {
                    "component": component,
                    "original_state": original_state,
                    "file_path": file_path,
                    "session_id": session_id
                }
            elif component_type == "optimizer":
                # Optimizer
                original_state = component.state_dict().copy() if hasattr(component, 'state_dict') else None
                self.registered_optimizers[component_id] = {
                    "component": component,
                    "original_state": original_state,
                    "file_path": file_path,
                    "session_id": session_id
                }
            elif component_type == "loss_function":
                # Loss function
                self.registered_loss_functions[component_id] = {
                    "component": component,
                    "file_path": file_path,
                    "session_id": session_id
                }
            elif component_type == "data_loader":
                # Data loader
                self.registered_data_loaders[component_id] = {
                    "component": component,
                    "file_path": file_path,
                    "session_id": session_id
                }

            # Store in database
            with sqlite3.connect(self.reload_db_path) as conn:
                conn.execute('''
                    INSERT INTO component_registry
                    (session_id, component_id, component_type, component_name,
                     file_path, class_name, registration_time, last_modified, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    component_id,
                    component_type,
                    component_name,
                    file_path,
                    class_name,
                    datetime.now(),
                    datetime.now(),
                    json.dumps({"registered": True})
                ))

            # Add to session
            self.active_sessions[session_id]["registered_components"][component_id] = {
                "component_type": component_type,
                "component_name": component_name,
                "file_path": file_path
            }

            return {
                "status": "success",
                "component_id": component_id,
                "message": f"Component '{component_name}' registered for hot swapping"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_file_watching(self, session_id: str, watch_patterns: List[str]):
        """Start watching files for changes"""
        try:
            if self.file_observer is None:
                self.file_observer = Observer()
                event_handler = ReloadEventHandler(self, session_id)

                # Watch workspace directory
                self.file_observer.schedule(event_handler, self.workspace_path, recursive=True)
                self.file_observer.start()

                logger.info(f"Started file watching for session {session_id}")

        except Exception as e:
            logger.error(f"Error starting file watching: {e}")

    def handle_file_change(self, session_id: str, file_path: str) -> Dict[str, Any]:
        """Handle file change event"""
        try:
            if session_id not in self.active_sessions:
                return {"status": "error", "message": "Session not active"}

            # Check cooldown
            current_time = time.time()
            if file_path in self.last_reload_time:
                if current_time - self.last_reload_time[file_path] < self.reload_cooldown:
                    return {"status": "info", "message": "Reload cooldown active"}

            self.last_reload_time[file_path] = current_time

            # Create backup
            backup_path = self.create_backup(file_path)

            # Find components affected by this file
            affected_components = self.find_affected_components(session_id, file_path)

            results = []
            for component_id, component_info in affected_components.items():
                reload_result = self.reload_component(session_id, component_id, component_info, file_path, backup_path)
                results.append(reload_result)

            # Update session stats
            session = self.active_sessions[session_id]
            session["total_reloads"] += len(results)
            session["successful_reloads"] += sum(1 for r in results if r.get("success", False))
            session["failed_reloads"] += sum(1 for r in results if not r.get("success", False))

            # Update database
            with sqlite3.connect(self.reload_db_path) as conn:
                conn.execute('''
                    UPDATE reload_sessions
                    SET total_reloads = ?, successful_reloads = ?, failed_reloads = ?
                    WHERE session_id = ?
                ''', (
                    session["total_reloads"],
                    session["successful_reloads"],
                    session["failed_reloads"],
                    session_id
                ))

            return {
                "status": "success",
                "file_path": file_path,
                "affected_components": len(affected_components),
                "reload_results": results,
                "backup_path": backup_path
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def find_affected_components(self, session_id: str, file_path: str) -> Dict[str, Dict[str, Any]]:
        """Find components affected by file change"""
        affected = {}

        # Check registered models
        for comp_id, comp_info in self.registered_models.items():
            if comp_info["session_id"] == session_id and comp_info["file_path"] in file_path:
                affected[comp_id] = {"type": "model", "info": comp_info}

        # Check registered optimizers
        for comp_id, comp_info in self.registered_optimizers.items():
            if comp_info["session_id"] == session_id and comp_info["file_path"] in file_path:
                affected[comp_id] = {"type": "optimizer", "info": comp_info}

        # Check registered loss functions
        for comp_id, comp_info in self.registered_loss_functions.items():
            if comp_info["session_id"] == session_id and comp_info["file_path"] in file_path:
                affected[comp_id] = {"type": "loss_function", "info": comp_info}

        # Check registered data loaders
        for comp_id, comp_info in self.registered_data_loaders.items():
            if comp_info["session_id"] == session_id and comp_info["file_path"] in file_path:
                affected[comp_id] = {"type": "data_loader", "info": comp_info}

        return affected

    def reload_component(self, session_id: str, component_id: str,
                        component_info: Dict[str, Any], file_path: str,
                        backup_path: str) -> Dict[str, Any]:
        """Reload a specific component"""
        try:
            component_type = component_info["type"]
            comp_info = component_info["info"]
            original_component = comp_info["component"]

            # Log reload attempt
            with sqlite3.connect(self.reload_db_path) as conn:
                conn.execute('''
                    INSERT INTO reload_events
                    (session_id, timestamp, file_path, component_type, component_name,
                     reload_type, success, backup_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    datetime.now(),
                    file_path,
                    component_type,
                    component_id,
                    "auto_reload",
                    False,  # Will update if successful
                    backup_path
                ))

            if component_type == "model":
                return self.reload_model(session_id, component_id, original_component, file_path)
            elif component_type == "optimizer":
                return self.reload_optimizer(session_id, component_id, original_component, file_path)
            elif component_type == "loss_function":
                return self.reload_loss_function(session_id, component_id, original_component, file_path)
            elif component_type == "data_loader":
                return self.reload_data_loader(session_id, component_id, original_component, file_path)
            else:
                return {"status": "error", "message": f"Unknown component type: {component_type}"}

        except Exception as e:
            error_msg = str(e)

            # Update database with error
            with sqlite3.connect(self.reload_db_path) as conn:
                conn.execute('''
                    UPDATE reload_events
                    SET success = ?, error_message = ?
                    WHERE session_id = ? AND component_name = ? AND timestamp = (
                        SELECT MAX(timestamp) FROM reload_events
                        WHERE session_id = ? AND component_name = ?
                    )
                ''', (False, error_msg, session_id, component_id, session_id, component_id))

            return {"status": "error", "component_id": component_id, "message": error_msg}

    def reload_model(self, session_id: str, component_id: str,
                    original_model: Any, file_path: str) -> Dict[str, Any]:
        """Reload a model component"""
        try:
            if not TORCH_AVAILABLE:
                return {"status": "error", "message": "PyTorch not available"}

            # Get original state
            original_state = original_model.state_dict().copy()
            original_class = original_model.__class__

            # Reload the module
            module_name = self.get_module_name_from_path(file_path)
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[module_name] = module

            # Find the model class in the reloaded module
            reloaded_module = sys.modules[module_name]
            new_model_class = None

            for name, obj in inspect.getmembers(reloaded_module):
                if (inspect.isclass(obj) and
                    issubclass(obj, nn.Module) and
                    obj.__name__ == original_class.__name__):
                    new_model_class = obj
                    break

            if new_model_class is None:
                return {"status": "error", "message": f"Could not find model class {original_class.__name__}"}

            # Create new model instance with same parameters
            try:
                # Try to preserve original initialization arguments
                new_model = new_model_class()

                # Try to load original state dict
                try:
                    new_model.load_state_dict(original_state)
                    state_loaded = True
                except Exception as e:
                    logger.warning(f"Could not load original state: {e}")
                    state_loaded = False

                # Update the original model's class and methods
                original_model.__class__ = new_model_class

                # Copy over new methods
                for attr_name in dir(new_model):
                    if not attr_name.startswith('_') and callable(getattr(new_model, attr_name)):
                        setattr(original_model, attr_name, getattr(new_model, attr_name))

                # Update component registry
                self.registered_models[component_id]["component"] = original_model

                # Mark as successful
                with sqlite3.connect(self.reload_db_path) as conn:
                    conn.execute('''
                        UPDATE reload_events
                        SET success = ?
                        WHERE session_id = ? AND component_name = ? AND timestamp = (
                            SELECT MAX(timestamp) FROM reload_events
                            WHERE session_id = ? AND component_name = ?
                        )
                    ''', (True, session_id, component_id, session_id, component_id))

                return {
                    "status": "success",
                    "component_id": component_id,
                    "message": "Model reloaded successfully",
                    "state_preserved": state_loaded
                }

            except Exception as e:
                return {"status": "error", "message": f"Error creating new model instance: {str(e)}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def reload_optimizer(self, session_id: str, component_id: str,
                        original_optimizer: Any, file_path: str) -> Dict[str, Any]:
        """Reload an optimizer component"""
        try:
            # For optimizers, we typically can't hot-swap easily
            # But we can reload optimizer-related code
            logger.info(f"Optimizer hot-swap not fully supported yet: {component_id}")

            return {
                "status": "info",
                "component_id": component_id,
                "message": "Optimizer hot-swap requires manual restart"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def reload_loss_function(self, session_id: str, component_id: str,
                           original_loss: Any, file_path: str) -> Dict[str, Any]:
        """Reload a loss function component"""
        try:
            # Reload the module containing the loss function
            module_name = self.get_module_name_from_path(file_path)
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])

                # Update the registered loss function
                reloaded_module = sys.modules[module_name]
                loss_class_name = original_loss.__class__.__name__

                if hasattr(reloaded_module, loss_class_name):
                    new_loss_class = getattr(reloaded_module, loss_class_name)
                    original_loss.__class__ = new_loss_class

                    # Update component registry
                    self.registered_loss_functions[component_id]["component"] = original_loss

                    return {
                        "status": "success",
                        "component_id": component_id,
                        "message": "Loss function reloaded successfully"
                    }
                else:
                    return {"status": "error", "message": f"Loss function class {loss_class_name} not found"}
            else:
                return {"status": "error", "message": f"Module {module_name} not found"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def reload_data_loader(self, session_id: str, component_id: str,
                          original_loader: Any, file_path: str) -> Dict[str, Any]:
        """Reload a data loader component"""
        try:
            # Data loaders are complex to hot-swap, usually require restart
            return {
                "status": "info",
                "component_id": component_id,
                "message": "Data loader hot-swap requires manual restart"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_backup(self, file_path: str) -> str:
        """Create backup of file before reload"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(file_path)
            backup_name = f"{filename}.{timestamp}.backup"
            backup_path = os.path.join(self.backup_dir, backup_name)

            shutil.copy2(file_path, backup_path)
            return backup_path

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return ""

    def get_module_name_from_path(self, file_path: str) -> str:
        """Convert file path to module name"""
        rel_path = os.path.relpath(file_path, self.workspace_path)
        module_name = rel_path.replace(os.sep, '.').replace('.py', '')
        return module_name

    def rollback_component(self, session_id: str, component_id: str) -> Dict[str, Any]:
        """Rollback a component to its previous state"""
        try:
            if session_id not in self.active_sessions:
                return {"status": "error", "message": "Session not active"}

            # Find component in registry
            component_info = None
            registry = None

            if component_id in self.registered_models:
                component_info = self.registered_models[component_id]
                registry = "models"
            elif component_id in self.registered_optimizers:
                component_info = self.registered_optimizers[component_id]
                registry = "optimizers"
            elif component_id in self.registered_loss_functions:
                component_info = self.registered_loss_functions[component_id]
                registry = "loss_functions"
            elif component_id in self.registered_data_loaders:
                component_info = self.registered_data_loaders[component_id]
                registry = "data_loaders"

            if component_info is None:
                return {"status": "error", "message": "Component not found"}

            # Rollback based on component type
            if registry == "models" and "original_state" in component_info:
                component = component_info["component"]
                original_state = component_info["original_state"]

                try:
                    component.load_state_dict(original_state)
                    return {
                        "status": "success",
                        "component_id": component_id,
                        "message": "Model rolled back successfully"
                    }
                except Exception as e:
                    return {"status": "error", "message": f"Rollback failed: {str(e)}"}

            elif registry == "optimizers" and "original_state" in component_info:
                component = component_info["component"]
                original_state = component_info["original_state"]

                if original_state:
                    try:
                        component.load_state_dict(original_state)
                        return {
                            "status": "success",
                            "component_id": component_id,
                            "message": "Optimizer rolled back successfully"
                        }
                    except Exception as e:
                        return {"status": "error", "message": f"Rollback failed: {str(e)}"}
                else:
                    return {"status": "info", "message": "No rollback state available for optimizer"}

            else:
                return {"status": "info", "message": f"Rollback not supported for {registry}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def manual_reload(self, session_id: str, component_id: str,
                     file_path: str = None) -> Dict[str, Any]:
        """Manually trigger a component reload"""
        try:
            if session_id not in self.active_sessions:
                return {"status": "error", "message": "Session not active"}

            # Find component
            component_info = None
            if component_id in self.registered_models:
                component_info = {"type": "model", "info": self.registered_models[component_id]}
            elif component_id in self.registered_optimizers:
                component_info = {"type": "optimizer", "info": self.registered_optimizers[component_id]}
            elif component_id in self.registered_loss_functions:
                component_info = {"type": "loss_function", "info": self.registered_loss_functions[component_id]}
            elif component_id in self.registered_data_loaders:
                component_info = {"type": "data_loader", "info": self.registered_data_loaders[component_id]}

            if component_info is None:
                return {"status": "error", "message": "Component not found"}

            # Use provided file path or component's file path
            target_file_path = file_path or component_info["info"]["file_path"]

            # Create backup
            backup_path = self.create_backup(target_file_path)

            # Perform reload
            result = self.reload_component(session_id, component_id, component_info, target_file_path, backup_path)

            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a live reload session"""
        try:
            if session_id not in self.active_sessions:
                return {"status": "error", "message": "Session not found"}

            session = self.active_sessions[session_id]

            # Get recent reload events
            with sqlite3.connect(self.reload_db_path) as conn:
                recent_events = conn.execute('''
                    SELECT timestamp, component_name, reload_type, success, error_message
                    FROM reload_events
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (session_id,)).fetchall()

                registered_components = conn.execute('''
                    SELECT component_id, component_type, component_name, file_path
                    FROM component_registry
                    WHERE session_id = ?
                ''', (session_id,)).fetchall()

            return {
                "status": "success",
                "session_id": session_id,
                "session_info": {
                    "start_time": session["start_time"].isoformat(),
                    "total_reloads": session["total_reloads"],
                    "successful_reloads": session["successful_reloads"],
                    "failed_reloads": session["failed_reloads"],
                    "success_rate": (session["successful_reloads"] / max(session["total_reloads"], 1)) * 100
                },
                "registered_components": [
                    {
                        "component_id": comp[0],
                        "component_type": comp[1],
                        "component_name": comp[2],
                        "file_path": comp[3]
                    }
                    for comp in registered_components
                ],
                "recent_events": [
                    {
                        "timestamp": event[0],
                        "component_name": event[1],
                        "reload_type": event[2],
                        "success": bool(event[3]),
                        "error_message": event[4]
                    }
                    for event in recent_events
                ]
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Stop a live reload session"""
        try:
            if session_id not in self.active_sessions:
                return {"status": "error", "message": "Session not found"}

            # Stop file watching
            if self.file_observer:
                self.file_observer.stop()
                self.file_observer.join()
                self.file_observer = None

            # Update database
            session = self.active_sessions[session_id]
            with sqlite3.connect(self.reload_db_path) as conn:
                conn.execute('''
                    UPDATE reload_sessions
                    SET end_time = ?, total_reloads = ?, successful_reloads = ?, failed_reloads = ?
                    WHERE session_id = ?
                ''', (
                    datetime.now(),
                    session["total_reloads"],
                    session["successful_reloads"],
                    session["failed_reloads"],
                    session_id
                ))

            # Clean up registered components for this session
            components_to_remove = []
            for comp_id, comp_info in self.registered_models.items():
                if comp_info["session_id"] == session_id:
                    components_to_remove.append(("models", comp_id))

            for comp_id, comp_info in self.registered_optimizers.items():
                if comp_info["session_id"] == session_id:
                    components_to_remove.append(("optimizers", comp_id))

            for comp_id, comp_info in self.registered_loss_functions.items():
                if comp_info["session_id"] == session_id:
                    components_to_remove.append(("loss_functions", comp_id))

            for comp_id, comp_info in self.registered_data_loaders.items():
                if comp_info["session_id"] == session_id:
                    components_to_remove.append(("data_loaders", comp_id))

            # Remove components
            for registry, comp_id in components_to_remove:
                if registry == "models":
                    del self.registered_models[comp_id]
                elif registry == "optimizers":
                    del self.registered_optimizers[comp_id]
                elif registry == "loss_functions":
                    del self.registered_loss_functions[comp_id]
                elif registry == "data_loaders":
                    del self.registered_data_loaders[comp_id]

            # Remove session
            del self.active_sessions[session_id]

            return {
                "status": "success",
                "session_id": session_id,
                "message": "Live reload session stopped",
                "final_stats": {
                    "total_reloads": session["total_reloads"],
                    "successful_reloads": session["successful_reloads"],
                    "failed_reloads": session["failed_reloads"]
                }
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

class ReloadEventHandler(FileSystemEventHandler):
    """File system event handler for live reload"""

    def __init__(self, reload_manager: LiveReloadManager, session_id: str):
        self.reload_manager = reload_manager
        self.session_id = session_id

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            logger.info(f"File modified: {event.src_path}")
            self.reload_manager.handle_file_change(self.session_id, event.src_path)

def main():
    """Main entry point for Live Reload Manager"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Command required"}))
        return

    command = sys.argv[1]
    workspace_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        reload_manager = LiveReloadManager(workspace_path)

        if command == "start_session":
            session_id = sys.argv[3] if len(sys.argv) > 3 else f"reload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            watch_patterns = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
            result = reload_manager.start_session(session_id, watch_patterns)

        elif command == "register_component":
            session_id = sys.argv[3]
            component_id = sys.argv[4]
            component_type = sys.argv[5]
            file_path = sys.argv[6] if len(sys.argv) > 6 else None
            # Note: Actual component object would need to be passed differently in real usage
            result = {"status": "info", "message": "Component registration requires object reference"}

        elif command == "manual_reload":
            session_id = sys.argv[3]
            component_id = sys.argv[4]
            file_path = sys.argv[5] if len(sys.argv) > 5 else None
            result = reload_manager.manual_reload(session_id, component_id, file_path)

        elif command == "rollback_component":
            session_id = sys.argv[3]
            component_id = sys.argv[4]
            result = reload_manager.rollback_component(session_id, component_id)

        elif command == "get_status":
            session_id = sys.argv[3]
            result = reload_manager.get_session_status(session_id)

        elif command == "stop_session":
            session_id = sys.argv[3]
            result = reload_manager.stop_session(session_id)

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