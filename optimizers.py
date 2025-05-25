import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

class Hyperparameter:
    """Class representing a hyperparameter with validation and UI info"""
    def __init__(self, name, default_value, value_range=None, description="", widget_type="entry"):
        self.name = name
        self.default_value = default_value
        self.current_value = default_value
        self.value_range = value_range
        self.description = description
        self.widget_type = widget_type
    
    def validate(self, value):
        """Validate a proposed value for this hyperparameter"""
        if self.value_range is None:
            return True
            
        if isinstance(self.value_range, (list, tuple)):
            if isinstance(self.value_range[0], (int, float)):
                try:
                    val = float(value)
                    return self.value_range[0] <= val <= self.value_range[1]
                except:
                    return False
            else:
                return value in self.value_range
                
        return True

class OptimizerBase(ABC):
    """Base class for optimizers"""
    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self.hyperparameters = {}
    
    def add_hyperparameter(self, hyperparameter):
        """Add a hyperparameter to this optimizer"""
        self.hyperparameters[hyperparameter.name] = hyperparameter
    
    @abstractmethod
    def optimize(self, weights, gradients, learning_rate):
        """Optimize weights using gradients and learning rate"""
        pass

class OptimizerRegistry:
    """Registry for managing available optimizers"""
    def __init__(self):
        self.optimizers = {}
        self.custom_optimizers = {}
        self.custom_optimizer_path = "custom_optimizers"
    
    def register(self, optimizer_class):
        """Register a new optimizer class"""
        optimizer = optimizer_class()
        self.optimizers[optimizer.name] = optimizer_class
    
    def register_custom(self, optimizer_class):
        """Register a custom optimizer class"""
        optimizer = optimizer_class()
        self.custom_optimizers[optimizer.name] = optimizer_class
    
    def get_optimizer(self, name):
        """Get an instance of an optimizer by name"""
        if name in self.optimizers:
            return self.optimizers[name]()
        elif name in self.custom_optimizers:
            return self.custom_optimizers[name]()
        return None
    
    def get_optimizer_names(self):
        """Get list of registered optimizer names"""
        return list(self.optimizers.keys()) + list(self.custom_optimizers.keys())
    
    def load_custom_optimizers(self):
        """Load custom optimizers from the custom_optimizers directory"""
        import os
        import importlib.util
        import sys
        
        # Create custom optimizers directory if it doesn't exist
        os.makedirs(self.custom_optimizer_path, exist_ok=True)
        
        # Add custom_optimizers directory to Python path
        if self.custom_optimizer_path not in sys.path:
            sys.path.append(self.custom_optimizer_path)
        
        # Clear existing custom optimizers
        self.custom_optimizers.clear()
        
        # Load each Python file in the custom_optimizers directory
        for filename in os.listdir(self.custom_optimizer_path):
            if filename.endswith('.py') and filename != '__init__.py' and filename != 'optimizer_template.py':
                try:
                    # Get module name without .py extension
                    module_name = os.path.splitext(filename)[0]
                    
                    # Import the module
                    spec = importlib.util.spec_from_file_location(
                        module_name,
                        os.path.join(self.custom_optimizer_path, filename)
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find optimizer classes in the module
                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if (isinstance(item, type) and 
                            issubclass(item, OptimizerBase) and 
                            item != OptimizerBase):
                            self.register_custom(item)
                            
                except Exception as e:
                    print(f"Error loading custom optimizer from {filename}: {str(e)}")
    
    def reload_custom_optimizers(self):
        """Reload all custom optimizers"""
        self.load_custom_optimizers()
        
    def get_optimizer_info(self, name):
        """Get detailed information about an optimizer"""
        optimizer_class = self.optimizers.get(name) or self.custom_optimizers.get(name)
        if optimizer_class:
            optimizer = optimizer_class()
            return {
                'name': optimizer.name,
                'description': optimizer.description,
                'hyperparameters': {
                    name: {
                        'default_value': hp.default_value,
                        'value_range': hp.value_range,
                        'description': hp.description,
                        'widget_type': hp.widget_type
                    }
                    for name, hp in optimizer.hyperparameters.items()
                },
                'is_custom': name in self.custom_optimizers
            }
        return None

class AdaBeliefOptimizer(OptimizerBase):
    """AdaBelief Optimizer with Rectified Updates"""
    def __init__(self):
        super().__init__(name="AdaBelief", 
                        description="Adaptive Belief Optimizer with Rectified Updates")
        
        self.add_hyperparameter(Hyperparameter(
            name="beta1",
            default_value=0.9,
            value_range=(0.0, 1.0),
            description="Exponential decay rate for first moment estimates",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="beta2",
            default_value=0.999,
            value_range=(0.0, 1.0),
            description="Exponential decay rate for second moment estimates",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="epsilon",
            default_value=1e-8,
            value_range=(1e-10, 1e-2),
            description="Small constant for numerical stability",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="rectify",
            default_value="True",
            value_range=["True", "False"],
            description="Whether to use rectified updates",
            widget_type="combobox"
        ))
        
        # Initialize optimizer state
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0    # Timestep

    def optimize(self, weights, gradients, learning_rate):
        """Implement AdaBelief optimization logic"""
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)
        
        self.t += 1
        
        # Get hyperparameters
        beta1 = float(self.hyperparameters["beta1"].current_value)
        beta2 = float(self.hyperparameters["beta2"].current_value)
        epsilon = float(self.hyperparameters["epsilon"].current_value)
        rectify = self.hyperparameters["rectify"].current_value == "True"
        
        # Update moment estimates
        self.m = beta1 * self.m + (1 - beta1) * gradients
        self.v = beta2 * self.v + (1 - beta2) * (gradients - self.m) ** 2
        
        # Bias correction
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        
        # Compute update
        if rectify:
            # Rectified version
            update = np.where(
                m_hat * gradients > 0,
                learning_rate * m_hat / (np.sqrt(v_hat) + epsilon),
                0.0
            )
        else:
            # Standard version
            update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return weights - update

class LionOptimizer(OptimizerBase):
    """Lion Optimizer with Evolving Learning Rates"""
    def __init__(self):
        super().__init__(name="Lion", 
                        description="Lion Optimizer with Evolving Learning Rates")
        
        self.add_hyperparameter(Hyperparameter(
            name="pride_size",
            default_value=3,
            value_range=(1, 10),
            description="Number of lions in the pride (ensemble size)",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="territory_range",
            default_value=0.1,
            value_range=(0.01, 1.0),
            description="Range of territory exploration",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="evolution_rate",
            default_value=0.01,
            value_range=(0.001, 0.1),
            description="Rate of learning rate evolution",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="strategy",
            default_value="cooperative",
            value_range=["cooperative", "competitive", "hybrid"],
            description="Pride hunting strategy",
            widget_type="combobox"
        ))
        
        # Initialize optimizer state
        self.pride_weights = None
        self.pride_scores = None
        self.best_weights = None
        self.learning_rates = None

    def optimize(self, weights, gradients, learning_rate):
        """Implement Lion optimization logic"""
        # Initialize pride if needed
        if self.pride_weights is None:
            pride_size = int(self.hyperparameters["pride_size"].current_value)
            self.pride_weights = [np.copy(weights) for _ in range(pride_size)]
            self.pride_scores = np.zeros(pride_size)
            self.best_weights = np.copy(weights)
            self.learning_rates = np.ones(pride_size) * learning_rate
        
        # Get hyperparameters
        territory_range = float(self.hyperparameters["territory_range"].current_value)
        evolution_rate = float(self.hyperparameters["evolution_rate"].current_value)
        strategy = self.hyperparameters["strategy"].current_value
        
        # Update each lion in the pride
        for i in range(len(self.pride_weights)):
            if strategy == "cooperative":
                # Cooperative hunting: lions share information
                update = gradients + territory_range * np.random.randn(*weights.shape)
                self.pride_weights[i] -= self.learning_rates[i] * update
                
                # Evolve learning rate based on gradient consistency
                gradient_consistency = np.abs(np.mean(gradients))
                self.learning_rates[i] *= (1 + evolution_rate * gradient_consistency)
                
            elif strategy == "competitive":
                # Competitive hunting: lions explore independently
                local_gradients = gradients + territory_range * np.random.randn(*weights.shape)
                self.pride_weights[i] -= self.learning_rates[i] * local_gradients
                
                # Evolve learning rate based on relative performance
                self.learning_rates[i] *= (1 + evolution_rate * (i + 1) / len(self.pride_weights))
                
            else:  # hybrid
                # Hybrid strategy: combine cooperative and competitive approaches
                shared_info = np.mean([w - weights for w in self.pride_weights], axis=0)
                local_gradients = gradients + territory_range * np.random.randn(*weights.shape)
                update = 0.5 * (shared_info + local_gradients)
                self.pride_weights[i] -= self.learning_rates[i] * update
                
                # Evolve learning rate using both factors
                gradient_consistency = np.abs(np.mean(gradients))
                relative_rank = (i + 1) / len(self.pride_weights)
                self.learning_rates[i] *= (1 + evolution_rate * (gradient_consistency + relative_rank) / 2)
        
        # Clip learning rates to reasonable range
        self.learning_rates = np.clip(self.learning_rates, learning_rate * 0.1, learning_rate * 10.0)
        
        # Update best weights based on gradient magnitude
        gradient_norms = [np.linalg.norm(weights - w) for w in self.pride_weights]
        best_idx = np.argmin(gradient_norms)
        self.best_weights = np.copy(self.pride_weights[best_idx])
        
        # Return the best weights found
        return self.best_weights

class RAMSOptimizer(OptimizerBase):
    """Regularized Adaptive Momentum Scaling Optimizer"""
    def __init__(self):
        super().__init__(name="RAMS", 
                        description="Regularized Adaptive Momentum Scaling Optimizer")
        
        self.add_hyperparameter(Hyperparameter(
            name="momentum_decay",
            default_value=0.9,
            value_range=(0.1, 0.999),
            description="Decay rate for momentum",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="scaling_factor",
            default_value=0.01,
            value_range=(0.0001, 0.1),
            description="Factor for adaptive scaling",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="regularization",
            default_value=0.001,
            value_range=(0.0, 0.1),
            description="Regularization strength",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="adapt_method",
            default_value="rmsprop",
            value_range=["rmsprop", "adadelta", "custom"],
            description="Method for adaptation",
            widget_type="combobox"
        ))
        
        # Initialize optimizer state
        self.momentum = None
        self.scaling = None
        self.squared_grad = None

    def optimize(self, weights, gradients, learning_rate):
        """Implement RAMS optimization logic"""
        if self.momentum is None:
            self.momentum = np.zeros_like(weights)
            self.scaling = np.ones_like(weights)
            self.squared_grad = np.zeros_like(weights)
        
        # Get hyperparameters
        momentum_decay = float(self.hyperparameters["momentum_decay"].current_value)
        scaling_factor = float(self.hyperparameters["scaling_factor"].current_value)
        regularization = float(self.hyperparameters["regularization"].current_value)
        adapt_method = self.hyperparameters["adapt_method"].current_value
        
        # Update squared gradient history
        self.squared_grad = 0.9 * self.squared_grad + 0.1 * gradients ** 2
        
        # Add regularization to gradients
        regularized_gradients = gradients + regularization * weights
        
        # Update momentum
        self.momentum = momentum_decay * self.momentum + (1 - momentum_decay) * regularized_gradients
        
        # Update scaling based on adaptation method
        if adapt_method == "rmsprop":
            self.scaling = np.sqrt(self.squared_grad + 1e-8)
        elif adapt_method == "adadelta":
            self.scaling = np.sqrt(np.maximum(self.squared_grad, self.momentum ** 2) + 1e-8)
        else:  # custom
            self.scaling = scaling_factor * (np.abs(self.momentum) + 1e-8)
        
        # Compute update
        update = learning_rate * self.momentum / self.scaling
        
        return weights - update

class HESMOptimizer(OptimizerBase):
    """Hybrid Entropy Scaling Method Optimizer"""
    def __init__(self):
        super().__init__(name="HESM", 
                        description="Hybrid Entropy Scaling Method Optimizer")
        
        self.add_hyperparameter(Hyperparameter(
            name="ensemble_size",
            default_value=4,
            value_range=(2, 8),
            description="Number of models in ensemble",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="entropy_threshold",
            default_value=0.1,
            value_range=(0.01, 1.0),
            description="Threshold for entropy-based scaling",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="adaptation_rate",
            default_value=0.01,
            value_range=(0.001, 0.1),
            description="Rate of model adaptation",
            widget_type="entry"
        ))
        self.add_hyperparameter(Hyperparameter(
            name="fusion_method",
            default_value="weighted",
            value_range=["weighted", "selective", "adaptive"],
            description="Method for combining model predictions",
            widget_type="combobox"
        ))
        
        # Initialize optimizer state
        self.ensemble = None
        self.model_weights = None
        self.entropy_history = None
        self.adaptation_history = None

    def get_performance_metrics(self):
        """Get performance metrics for analysis"""
        if not self.adaptation_history:
            return None
            
        metrics = {
            'entropy_mean': np.mean([h['entropy'] for h in self.adaptation_history]),
            'entropy_std': np.std([h['entropy'] for h in self.adaptation_history]),
            'exploration_ratio': np.mean([
                h['entropy'] > float(self.hyperparameters['entropy_threshold'].current_value)
                for h in self.adaptation_history
            ]),
            'model_weights_final': self.model_weights,
            'model_weights_std': np.std([h['model_weights'] for h in self.adaptation_history], axis=0),
            'learning_rates_mean': np.mean([h['learning_rates'] for h in self.adaptation_history], axis=0),
            'learning_rates_std': np.std([h['learning_rates'] for h in self.adaptation_history], axis=0)
        }
        
        return metrics
    
    def get_convergence_analysis(self):
        """Analyze convergence behavior"""
        if not self.adaptation_history:
            return None
            
        # Calculate entropy trend
        entropy_values = np.array([h['entropy'] for h in self.adaptation_history])
        entropy_trend = np.polyfit(np.arange(len(entropy_values)), entropy_values, 1)[0]
        
        # Calculate weight stability
        weight_changes = np.diff([h['model_weights'] for h in self.adaptation_history], axis=0)
        weight_stability = np.mean(np.abs(weight_changes))
        
        # Analyze exploration vs exploitation phases
        threshold = float(self.hyperparameters['entropy_threshold'].current_value)
        exploration_phases = np.where(np.diff(entropy_values > threshold))[0] + 1
        
        analysis = {
            'entropy_trend': entropy_trend,
            'weight_stability': weight_stability,
            'phase_changes': len(exploration_phases),
            'phase_change_points': exploration_phases.tolist(),
            'is_converged': weight_stability < 0.01 and entropy_trend < 0,
            'convergence_iteration': self._find_convergence_point()
        }
        
        return analysis
    
    def _find_convergence_point(self):
        """Find the iteration where the optimizer converged"""
        if not self.adaptation_history:
            return None
            
        # Use weight stability as convergence criterion
        weight_changes = np.diff([h['model_weights'] for h in self.adaptation_history], axis=0)
        stability_metric = np.mean(np.abs(weight_changes), axis=1)
        
        # Find point where stability is consistently below threshold
        threshold = 0.01
        window_size = 10
        
        for i in range(len(stability_metric) - window_size):
            if np.all(stability_metric[i:i+window_size] < threshold):
                return i
        
        return None
    
    def get_model_contributions(self):
        """Analyze individual model contributions"""
        if not self.adaptation_history:
            return None
            
        n_models = len(self.model_weights)
        contributions = []
        
        for i in range(n_models):
            model_stats = {
                'model_id': i,
                'weight_mean': np.mean([h['model_weights'][i] for h in self.adaptation_history]),
                'weight_std': np.std([h['model_weights'][i] for h in self.adaptation_history]),
                'learning_rate_mean': np.mean([h['learning_rates'][i] for h in self.adaptation_history]),
                'learning_rate_std': np.std([h['learning_rates'][i] for h in self.adaptation_history]),
                'selection_frequency': np.mean([
                    np.argmax(h['model_weights']) == i 
                    for h in self.adaptation_history
                ])
            }
            contributions.append(model_stats)
        
        return contributions

    def optimize(self, weights, gradients, learning_rate):
        """Implement HESM optimization logic"""
        # Initialize ensemble if needed
        if self.ensemble is None:
            ensemble_size = int(self.hyperparameters["ensemble_size"].current_value)
            self.ensemble = [np.copy(weights) for _ in range(ensemble_size)]
            self.model_weights = np.ones(ensemble_size) / ensemble_size
            self.entropy_history = []
            self.adaptation_history = []
        
        # Get hyperparameters
        entropy_threshold = float(self.hyperparameters["entropy_threshold"].current_value)
        adaptation_rate = float(self.hyperparameters["adaptation_rate"].current_value)
        fusion_method = self.hyperparameters["fusion_method"].current_value
        
        # Calculate entropy of gradients
        gradient_magnitudes = np.abs(gradients)
        gradient_probs = gradient_magnitudes / (np.sum(gradient_magnitudes) + 1e-8)
        entropy = -np.sum(gradient_probs * np.log(gradient_probs + 1e-8))
        self.entropy_history.append(entropy)
        
        # Update each model in ensemble
        for i in range(len(self.ensemble)):
            # Calculate model-specific learning rate
            model_lr = learning_rate * (1 + adaptation_rate * i)
            
            # Apply entropy-based scaling
            if entropy > entropy_threshold:
                # High entropy: More exploration
                update = gradients + 0.1 * np.random.randn(*weights.shape)
            else:
                # Low entropy: More exploitation
                update = gradients
            
            # Update model weights
            self.ensemble[i] -= model_lr * update
            
            # Update model importance based on gradient consistency
            grad_consistency = np.abs(np.mean(gradients))
            self.model_weights[i] *= (1 + adaptation_rate * grad_consistency)
        
        # Normalize model weights
        self.model_weights /= np.sum(self.model_weights)
        
        # Combine model predictions based on fusion method
        if fusion_method == "weighted":
            # Weighted average of all models
            final_weights = np.sum([w * m for w, m in zip(self.model_weights, self.ensemble)], axis=0)
        elif fusion_method == "selective":
            # Select best performing model
            best_model_idx = np.argmax(self.model_weights)
            final_weights = self.ensemble[best_model_idx]
        else:  # adaptive
            # Adaptive combination based on entropy
            if entropy > entropy_threshold:
                # High uncertainty: Use weighted average
                final_weights = np.sum([w * m for w, m in zip(self.model_weights, self.ensemble)], axis=0)
            else:
                # Low uncertainty: Use best model
                best_model_idx = np.argmax(self.model_weights)
                final_weights = self.ensemble[best_model_idx]
        
        # Store adaptation history
        self.adaptation_history.append({
            'entropy': entropy,
            'model_weights': self.model_weights.copy(),
            'learning_rates': [learning_rate * (1 + adaptation_rate * i) for i in range(len(self.ensemble))]
        })
        
        return final_weights

# Create and initialize the optimizer registry
optimizer_registry = OptimizerRegistry()
optimizer_registry.register(AdaBeliefOptimizer)
optimizer_registry.register(LionOptimizer)
optimizer_registry.register(RAMSOptimizer)
optimizer_registry.register(HESMOptimizer) 