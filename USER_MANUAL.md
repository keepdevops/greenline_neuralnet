# Stock Grok Analysis - User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Data Management](#data-management)
4. [Visualization Features](#visualization-features)
5. [Model Training](#model-training)
6. [Grid View](#grid-view)
7. [Theme System](#theme-system)
8. [File Formats](#file-formats)
9. [Keyboard Shortcuts](#keyboard-shortcuts)
10. [Troubleshooting](#troubleshooting)

## Introduction
Stock Grok Analysis is a powerful GUI application for analyzing stock market data, training machine learning models, and visualizing market trends. The application combines modern machine learning techniques with an intuitive interface for both basic and advanced analysis.

## Getting Started

### System Requirements
- Python 3.9 or higher
- Required packages: numpy, pandas, scikit-learn, tensorflow, matplotlib, and others (use `pip install` command)
- Sufficient memory for large datasets (recommended: 8GB+)

### Launch Application
```bash
python3 grok_stock.py
```

## Data Management

### Supported File Formats
- CSV (.csv)
- JSON (.json)
- Parquet (.parquet)
- Feather (.feather)
- HDF5 (.h5, .hdf5)
- DuckDB (.duckdb)
- Arrow (.arrow)
- Pickle (.pkl, .pickle)

### Loading Data
1. Click "Open File" in the File Controls section
2. Select your data file
3. The application automatically detects the format and loads the data
4. Data statistics will be displayed in the info panel

### Data Requirements
Your data should include these columns:
- timestamp: Date/time of the trading data
- open: Opening price
- high: Highest price
- low: Lowest price
- close: Closing price
- vol: Trading volume
- (optional) ticker: Stock symbol

## Visualization Features

### Available Plot Types
1. **Stock Price**
   - Shows price movement over time
   - Displays actual vs predicted prices
   - Includes confidence bands

2. **Returns Distribution**
   - Histogram of returns
   - Normal distribution overlay
   - Statistical metrics

3. **Volatility**
   - Rolling volatility window
   - Volatility clustering analysis
   - Risk metrics

4. **Prediction vs Actual**
   - Scatter plot of predicted vs actual values
   - Regression line
   - R-squared metrics

5. **Residuals**
   - Residual analysis plot
   - Error distribution
   - Outlier detection

6. **Learning Curves**
   - Model training progress
   - Loss function evolution
   - Convergence analysis

7. **Correlation Matrix**
   - Feature correlations
   - Heat map visualization
   - Relationship strength indicators

8. **Feature Importance**
   - Bar chart of feature impacts
   - Relative importance scores
   - Feature ranking

9. **Uncertainty**
   - Confidence intervals
   - Prediction bands
   - Risk assessment

10. **Trading Signals**
    - Buy/Sell indicators
    - Signal strength
    - Entry/exit points

11. **Cumulative Returns**
    - Performance over time
    - Strategy comparison
    - Risk-adjusted returns

### Plot Controls
- Float/Dock button: Detach plot window for better viewing
- Plot Type dropdown: Select visualization type
- Auto-update on data changes
- Interactive zooming and panning
- Save plot functionality

## Model Training

### Available Optimizers

#### 1. AMDS (Adaptive Momentum Descent with Scaling)
The Adaptive Momentum Descent with Scaling optimizer combines momentum-based optimization with adaptive learning rates and automatic scaling.

**Key Features:**
- Dynamic learning rate adjustment based on gradient history
- Momentum-based updates for faster convergence
- Automatic feature scaling during optimization
- Gradient clipping to prevent exploding gradients

**Best Used For:**
- General-purpose optimization
- Noisy or sparse data
- Problems with varying feature scales
- Long training sequences

**Parameters:**
- Base learning rate: 0.01 (default)
- Momentum: 0.9 (adaptive)
- Scaling factor: Auto-adjusted
- Gradient clip threshold: 1.0

#### 2. AMDS+ (Enhanced AMDS)
An enhanced version of AMDS that incorporates additional optimization techniques for improved performance.

**Key Features:**
- Second-order momentum adaptation
- Nesterov accelerated gradients
- Dynamic momentum scheduling
- Adaptive gradient scaling
- Improved convergence guarantees

**Best Used For:**
- Complex optimization landscapes
- Deep neural networks
- Time-series data
- High-dimensional feature spaces

**Advanced Features:**
- Automatic learning rate warmup
- Gradient noise injection
- Momentum cycle scheduling
- Adaptive batch sizing

#### 3. CIPO (Constrained Interior Point Optimizer)
A specialized optimizer that handles constrained optimization problems using interior point methods.

**Key Features:**
- Barrier function optimization
- Constraint satisfaction guarantees
- Automatic step size adjustment
- Feasible region maintenance

**Best Used For:**
- Constrained optimization problems
- Portfolio optimization
- Risk-bounded trading
- Bounded parameter spaces

**Constraints Handling:**
- Linear constraints
- Nonlinear constraints
- Equality constraints
- Inequality constraints

#### 4. BCIPO (Bounded CIPO)
Extension of CIPO specifically designed for problems with box constraints and improved numerical stability.

**Key Features:**
- Box constraint handling
- Improved numerical stability
- Adaptive barrier parameters
- Efficient bound projection

**Best Used For:**
- Problems with parameter bounds
- Stability-critical applications
- Large-scale optimization
- Real-time trading systems

**Stability Features:**
- Condition number monitoring
- Adaptive regularization
- Stable matrix operations
- Bound feasibility maintenance

#### 5. BCIPO-Dropout
A variant of BCIPO that incorporates dropout regularization for improved generalization.

**Key Features:**
- Integrated dropout regularization
- Adaptive dropout rates
- Uncertainty estimation
- Improved generalization

**Best Used For:**
- Preventing overfitting
- Uncertainty quantification
- Ensemble-like behavior
- Robust model training

**Dropout Features:**
- Layer-wise dropout rates
- Adaptive mask generation
- Dropout scheduling
- Monte Carlo dropout inference

#### 6. BCIPO-HESM (Hybrid Entropy Scaling Method)
The most advanced optimizer combining BCIPO with entropy-based scaling and hybrid optimization techniques.

**Key Features:**
- Entropy-based feature scaling
- Hybrid optimization strategy
- Advanced regularization
- Multi-objective optimization

**Best Used For:**
- Complex market dynamics
- Multi-asset trading
- High-frequency trading
- Risk-aware optimization

**Advanced Capabilities:**
- Automatic feature importance weighting
- Dynamic entropy estimation
- Multi-scale optimization
- Adaptive regularization scheduling

### AdaBelief Optimizer
An adaptive optimizer that combines the benefits of AdaGrad and Adam with belief in the gradients.

**Hyperparameters:**
- `beta1` (0.0-1.0): Exponential decay rate for first moment estimates
- `beta2` (0.0-1.0): Exponential decay rate for second moment estimates
- `epsilon` (1e-10-1e-2): Small constant for numerical stability
- `rectify` (True/False): Whether to use rectified updates

Best for: Models with noisy gradients or when training stability is important.

### Lion Optimizer
A novel optimizer that uses an ensemble approach with evolving learning rates.

**Hyperparameters:**
- `pride_size` (1-10): Number of lions in the pride (ensemble size)
- `territory_range` (0.01-1.0): Range of territory exploration
- `evolution_rate` (0.001-0.1): Rate of learning rate evolution
- `strategy` (cooperative/competitive/hybrid): Pride hunting strategy

Best for: Complex optimization landscapes where traditional methods might get stuck.

### RAMS (Regularized Adaptive Momentum Scaling)
An optimizer that combines momentum with adaptive scaling and regularization.

**Hyperparameters:**
- `momentum_decay` (0.1-0.999): Decay rate for momentum
- `scaling_factor` (0.0001-0.1): Factor for adaptive scaling
- `regularization` (0.0-0.1): Regularization strength
- `adapt_method` (rmsprop/adadelta/custom): Method for adaptation

Best for: Problems requiring both momentum and regularization.

### HESM (Hybrid Entropy Scaling Method) Optimizer
A sophisticated optimizer that combines ensemble learning with entropy-based adaptation.

**Key Features:**
- Ensemble-based optimization with multiple models
- Entropy-guided exploration vs exploitation
- Adaptive model weighting
- Multiple fusion strategies for model combination

**Hyperparameters:**
- `ensemble_size` (2-8): Number of models in the ensemble
- `entropy_threshold` (0.01-1.0): Threshold for entropy-based scaling
- `adaptation_rate` (0.001-0.1): Rate of model adaptation
- `fusion_method` (weighted/selective/adaptive): Method for combining model predictions

**Best Used For:**
- Complex optimization problems
- Noisy or uncertain data
- Multi-modal optimization landscapes
- Problems requiring robust convergence

**Advanced Features:**

1. **Entropy-Based Scaling**
   - Automatically adjusts exploration vs exploitation
   - Higher entropy leads to more exploration
   - Lower entropy focuses on exploitation
   - Adaptive noise injection based on uncertainty

2. **Ensemble Management**
   - Dynamic model weighting based on performance
   - Individual learning rates for each model
   - Gradient consistency monitoring
   - Automatic model selection

3. **Fusion Strategies**
   - Weighted averaging of model predictions
   - Selective best model choice
   - Adaptive switching based on uncertainty
   - Performance-based weighting

4. **Adaptation Mechanisms**
   - Learning rate adaptation per model
   - Weight importance updating
   - Entropy history tracking
   - Model contribution analysis

**Implementation Details:**

```python
# Entropy calculation
gradient_magnitudes = np.abs(gradients)
gradient_probs = gradient_magnitudes / (np.sum(gradient_magnitudes) + 1e-8)
entropy = -np.sum(gradient_probs * np.log(gradient_probs + 1e-8))

# Model update with entropy-based scaling
if entropy > entropy_threshold:
    update = gradients + 0.1 * np.random.randn(*weights.shape)  # More exploration
else:
    update = gradients  # More exploitation

# Model weight adaptation
grad_consistency = np.abs(np.mean(gradients))
model_weights *= (1 + adaptation_rate * grad_consistency)
```

**Usage Guidelines:**

1. **Ensemble Size Selection**
   - Start with 4 models (default)
   - Increase for more complex problems
   - Decrease if computational resources are limited
   - Monitor ensemble diversity

2. **Entropy Threshold Tuning**
   - Default (0.1) works well for most cases
   - Lower for more stable convergence
   - Higher for more exploration
   - Adjust based on gradient statistics

3. **Adaptation Rate Configuration**
   - Start with default (0.01)
   - Increase for faster adaptation
   - Decrease for more stability
   - Monitor convergence behavior

4. **Fusion Method Selection**
   - "weighted": Best for stable problems
   - "selective": Best for multi-modal problems
   - "adaptive": Best for uncertain problems
   - Monitor performance metrics

**Performance Monitoring:**

The optimizer provides rich monitoring capabilities through its adaptation history:
```python
adaptation_history = {
    'entropy': current_entropy,
    'model_weights': weights_distribution,
    'learning_rates': per_model_learning_rates
}
```

Use this information to:
- Track entropy evolution
- Monitor model contributions
- Analyze learning dynamics
- Adjust hyperparameters

### Optimizer Selection Guidelines

1. **For Standard Training:**
   - Start with AMDS
   - Monitor convergence
   - Switch to AMDS+ if needed

2. **For Constrained Problems:**
   - Use CIPO for general constraints
   - Use BCIPO for box constraints
   - Consider BCIPO-Dropout for regularization

3. **For Advanced Applications:**
   - BCIPO-HESM for complex problems
   - AMDS+ for deep networks
   - BCIPO-Dropout for uncertainty

4. **Performance Considerations:**
   - Memory usage increases with optimizer complexity
   - Computational cost varies by optimizer
   - Consider hardware limitations
   - Monitor resource utilization

### Optimizer Hyperparameter Interaction

The optimizers interact with the hyperparameter controls in the following ways:

1. **Weight Initialization:**
   - AMDS/AMDS+: Works best with Xavier/He initialization
   - CIPO/BCIPO: Less sensitive to initialization
   - BCIPO-HESM: Automatically adapts to initialization

2. **Regularization:**
   - AMDS: Manual L1/L2 regularization
   - BCIPO-Dropout: Integrated regularization
   - BCIPO-HESM: Advanced adaptive regularization

3. **Learning Rate:**
   - AMDS: Base learning rate important
   - AMDS+: Adaptive learning rate
   - CIPO/BCIPO: Interior point specific rates
   - BCIPO-HESM: Multi-scale learning rates

### Monitoring and Debugging

Each optimizer provides specific metrics for monitoring:

1. **AMDS/AMDS+:**
   - Gradient norms
   - Momentum values
   - Learning rate adaptation

2. **CIPO/BCIPO:**
   - Constraint violations
   - Barrier parameters
   - Optimality conditions

3. **BCIPO-Dropout/HESM:**
   - Dropout statistics
   - Entropy measures
   - Regularization effects

### Best Practices

1. **Optimizer Selection:**
   - Start simple (AMDS)
   - Upgrade based on needs
   - Monitor performance metrics
   - Consider problem constraints

2. **Parameter Tuning:**
   - Use default values first
   - Adjust gradually
   - Monitor convergence
   - Watch for instabilities

3. **Performance Optimization:**
   - Balance speed vs accuracy
   - Monitor memory usage
   - Consider batch sizes
   - Use appropriate precision

### Training Parameters
- Learning Rate: Controls step size
- Iterations: Number of training steps
- Model Architecture: Automatically adapted to data

### Hyperparameters In-Depth

#### Mathematical Foundation and Implementation

1. **Weight Initialization Methods**

   a) **Xavier (Glorot) Initialization**
   ```python
   weights = np.random.randn(n_features) * np.sqrt(2.0 / (n_features))
   ```
   - Mathematically designed to maintain variance across layers
   - Optimal for tanh activation functions
   - Prevents vanishing/exploding gradients
   - Scale factor `sqrt(2/n)` where n is number of inputs
   - Best for: Deep networks with tanh activations

   b) **He Initialization**
   ```python
   weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
   ```
   - Variant of Xavier designed for ReLU activations
   - Accounts for ReLU's asymmetric nature
   - Maintains variance for positive-only activations
   - Best for: Deep networks with ReLU activations

2. **Bias Initialization**
   ```python
   bias = float(bias_init)  # Default: 0.0
   ```
   - Initial offset for neuron activation
   - Affects decision boundary position
   - Zero initialization prevents initial bias
   - Non-zero values can break symmetry
   - Best practice: Start with 0.0 unless you have specific reason

3. **Weight Decay (L2 Regularization)**
   ```python
   loss += weight_decay * np.sum(weights ** 2)
   ```
   - Penalizes large weights quadratically
   - Equivalent to Gaussian prior on weights
   - Prevents overfitting by constraining weights
   - Default: 0.001
   - Adjustment range: 0.0001 to 0.1

4. **L1 Regularization**
   ```

### Custom Optimizers

Stock Grok Analysis supports creating and using custom optimizers. This allows you to implement your own optimization algorithms and seamlessly integrate them with the application.

#### Creating a Custom Optimizer

1. Click the "+" button next to the optimizer selection dropdown
2. Enter a name for your optimizer (use CamelCase, e.g., "MyCustomOptimizer")
3. A new Python file will be created in the `custom_optimizers` directory
4. Edit the file to implement your optimization logic

#### Template Structure

The custom optimizer template provides a basic structure:

```python
class MyCustomOptimizer(OptimizerBase):
    def __init__(self):
        super().__init__(
            name="MyCustomOptimizer",
            description="Description of your optimizer"
        )
        
        # Add hyperparameters
        self.add_hyperparameter(Hyperparameter(
            name="param_name",
            default_value=0.01,
            value_range=(0.0, 1.0),
            description="Parameter description",
            widget_type="entry"  # or "combobox"
        ))
        
        # Initialize state variables
        self.state = None
    
    def optimize(self, weights, gradients, learning_rate):
        # Implement optimization logic here
        return updated_weights
```

#### Key Components

1. **Initialization**
   - Set optimizer name and description
   - Define hyperparameters with ranges and descriptions
   - Initialize any state variables needed for optimization

2. **Hyperparameters**
   - Use `add_hyperparameter()` to define configurable parameters
   - Supported widget types: "entry" (numeric input) or "combobox" (selection)
   - Provide value ranges and descriptions for GUI integration

3. **Optimization Logic**
   - Implement the `optimize()` method
   - Input: current weights, gradients, and learning rate
   - Output: updated weights
   - Maintain state between iterations if needed

#### Loading and Reloading

- Custom optimizers are automatically loaded at startup
- Click the "↻" button to reload after making changes
- Your optimizer will appear in the dropdown menu
- Hyperparameters will be automatically added to the GUI

#### Example Implementation

Here's an example of a custom optimizer combining momentum with RMSProp:

```python
class MomentumRMSProp(OptimizerBase):
    def __init__(self):
        super().__init__(
            name="MomentumRMSProp",
            description="Combines momentum with RMSProp"
        )
        
        self.add_hyperparameter(Hyperparameter(
            name="momentum",
            default_value=0.9,
            value_range=(0.0, 1.0),
            description="Momentum coefficient"
        ))
        
        self.add_hyperparameter(Hyperparameter(
            name="decay_rate",
            default_value=0.99,
            value_range=(0.0, 1.0),
            description="Decay rate for squared gradients"
        ))
        
        self.momentum_buffer = None
        self.squared_grad_buffer = None
    
    def optimize(self, weights, gradients, learning_rate):
        # Initialize state if needed
        if self.momentum_buffer is None:
            self.momentum_buffer = np.zeros_like(weights)
            self.squared_grad_buffer = np.zeros_like(weights)
        
        # Get hyperparameters
        momentum = float(self.hyperparameters["momentum"].current_value)
        decay_rate = float(self.hyperparameters["decay_rate"].current_value)
        
        # Update squared gradient buffer
        self.squared_grad_buffer = (
            decay_rate * self.squared_grad_buffer + 
            (1 - decay_rate) * gradients ** 2
        )
        
        # Compute adaptive learning rates
        adaptive_lr = learning_rate / np.sqrt(self.squared_grad_buffer + 1e-8)
        
        # Update momentum buffer
        self.momentum_buffer = (
            momentum * self.momentum_buffer + 
            adaptive_lr * gradients
        )
        
        return weights - self.momentum_buffer
```

#### Best Practices

1. **State Management**
   - Initialize state variables in `__init__()`
   - Check and initialize state in `optimize()` if needed
   - Use numpy arrays for numerical operations

2. **Hyperparameter Handling**
   - Provide meaningful default values
   - Set appropriate value ranges
   - Add descriptive tooltips
   - Validate parameter types

3. **Numerical Stability**
   - Add small epsilon values to denominators
   - Use np.clip() for bounded values
   - Handle edge cases and potential NaN values

4. **Performance**
   - Use vectorized operations when possible
   - Minimize memory allocations
   - Consider computational efficiency

5. **Documentation**
   - Add docstrings to your optimizer class
   - Document hyperparameters and their effects
   - Include usage examples if needed

#### Debugging Tips

1. Print intermediate values using logging:
   ```python
   import logging
   logging.info(f"Gradient norm: {np.linalg.norm(gradients)}")
   ```

2. Monitor state evolution:
   ```python
   logging.debug(f"State shape: {self.state.shape}")
   logging.debug(f"State range: [{np.min(self.state)}, {np.max(self.state)}]")
   ```

3. Check for numerical issues:
   ```python
   if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
       logging.warning("Detected NaN or Inf in weights")
   ```

#### File Organization

The custom optimizers are stored in the `custom_optimizers` directory:

```
custom_optimizers/
  ├── optimizer_template.py
  ├── my_optimizer.py
  ├── another_optimizer.py
  └── ...
```

- Each optimizer should be in its own file
- Filename should match class name (lowercase)
- Don't modify optimizer_template.py
- Keep related optimizers in separate files