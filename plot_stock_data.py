import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
import pandas as pd
import logging
import sys
from datetime import datetime
import psutil  # For memory monitoring
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def print_error(message):
    """Print error messages in red with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\033[91m{timestamp} - ERROR: {message}\033[0m", file=sys.stderr)

def print_warning(message):
    """Print warning messages in yellow with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\033[93m{timestamp} - WARNING: {message}\033[0m")

def print_info(message):
    """Print info messages in green with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\033[92m{timestamp} - INFO: {message}\033[0m")

def plot_stock_price(timestamps, prices, title="Stock Price Over Time"):
    """Plot basic stock price time series."""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, prices, 'b-', label='Price')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        plt.savefig('time_series_plot.png')
        plt.close()
    except Exception as e:
        print_error(f"Error plotting stock price: {str(e)}")
        print_info(f"Timestamps shape: {np.asarray(timestamps).shape}")
        print_info(f"Prices shape: {np.asarray(prices).shape}")

def plot_returns_distribution(returns, title="Returns Distribution"):
    """Plot distribution of returns."""
    try:
        # Ensure returns is a numpy array
        returns = np.asarray(returns)
        
        # Check if returns is a single float, if so, create a proper returns array
        if returns.ndim == 0 or (returns.ndim == 1 and len(returns) == 1):
            print_warning("Single return value provided. Cannot plot distribution.")
            return
            
        # Remove any infinite or NaN values
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            print_error("No valid returns to plot after filtering.")
            return
            
        plt.figure(figsize=(10, 6))
        sns.histplot(returns, kde=True, stat='density')
        plt.title(title)
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.grid(True)
        plt.savefig('returns_distribution.png')
        plt.close()
        
        # Print some statistics
        print_info(f"Returns statistics:")
        print_info(f"Mean: {np.mean(returns):.4f}")
        print_info(f"Std: {np.std(returns):.4f}")
        print_info(f"Min: {np.min(returns):.4f}")
        print_info(f"Max: {np.max(returns):.4f}")
        
    except Exception as e:
        print_error(f"Error plotting returns distribution: {str(e)}")
        print_info(f"Returns shape: {np.asarray(returns).shape}")
        print_info(f"Returns type: {type(returns)}")

def calculate_returns(prices):
    """Calculate returns from price series."""
    try:
        # Convert to numpy array if not already
        prices = np.asarray(prices)
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Remove any infinite or NaN values
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            print_error("No valid returns could be calculated.")
            return np.array([0.0])
            
        return returns
        
    except Exception as e:
        print_error(f"Error calculating returns: {str(e)}")
        return np.array([0.0])

def calculate_volatility(prices, window=20):
    """Calculate volatility in a memory-efficient way"""
    # Calculate returns first
    returns = np.diff(prices) / prices[:-1]
    
    # Initialize volatility array with the same length as returns
    volatility = np.zeros(len(returns))
    
    # Calculate rolling standard deviation
    for i in range(len(returns)):
        start_idx = max(0, i - window + 1)
        volatility[i] = np.std(returns[start_idx:(i + 1)])
    
    return volatility

def plot_volatility(timestamps, volatility, window=20, title="Rolling Volatility"):
    """Plot rolling volatility."""
    # Adjust timestamps to match volatility length
    timestamps_adj = timestamps[window-1:]  # Remove first (window-1) elements
    if len(timestamps_adj) != len(volatility):
        # Further trim if needed
        min_len = min(len(timestamps_adj), len(volatility))
        timestamps_adj = timestamps_adj[:min_len]
        volatility = volatility[:min_len]

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps_adj, volatility, 'r-', label=f'{window}-day Volatility')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.legend()
    plt.savefig('volatility_plot.png')
    plt.close()

def plot_prediction_vs_actual(y_true, y_pred, title="Predicted vs Actual Values"):
    """Plot predicted vs actual values."""
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig('pred_vs_actual.png')
    plt.close()

def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """Plot residuals."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.savefig('residual_plot.png')
    plt.close()

def plot_learning_curves(loss_history, title="Learning Curves"):
    """Plot learning curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', label='Training Loss')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('learning_curves.png')
    plt.close()

def plot_correlation_matrix(data, title="Feature Correlation Matrix"):
    """Plot correlation matrix of features."""
    corr = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.savefig('correlation_heatmap.png')
    plt.close()

def plot_feature_importance(feature_names, importances, title="Feature Importance"):
    """Plot feature importance."""
    importance_df = pd.DataFrame({'features': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['features'], importance_df['importance'])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_uncertainty(timestamps, predictions, uncertainties, actual=None, title="Predictions with Uncertainty"):
    """Plot predictions with uncertainty bands."""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, predictions, 'b-', label='Prediction')
    plt.fill_between(timestamps, 
                    predictions - 2*uncertainties,
                    predictions + 2*uncertainties,
                    color='b', alpha=0.2, label='95% Confidence')
    if actual is not None:
        plt.plot(timestamps, actual, 'r--', label='Actual')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.savefig('uncertainty_bands.png')
    plt.close()

def plot_trading_signals(timestamps, prices, buy_signals, sell_signals, title="Trading Signals"):
    """Plot trading signals."""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, prices, 'b-', label='Price')
    plt.scatter(timestamps[buy_signals], prices[buy_signals], color='g', marker='^', label='Buy Signal')
    plt.scatter(timestamps[sell_signals], prices[sell_signals], color='r', marker='v', label='Sell Signal')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.savefig('trading_signals.png')
    plt.close()

def plot_cumulative_returns(timestamps, returns, title="Cumulative Returns"):
    """Plot cumulative returns."""
    cumulative_returns = np.cumprod(1 + returns) - 1
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, cumulative_returns * 100, 'g-')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True)
    plt.savefig('cumulative_returns.png')
    plt.close()

def validate_columns(df):
    """Validate DataFrame columns and add missing ones with default values."""
    if df is None or df.empty:
        return True  # Return True without warnings when no data is loaded
    
    required_columns = ['ticker', 'timestamp', 'open', 'high', 'low', 'close']
    optional_columns = ['vol', 'volume', 'openint']
    
    # Check for required columns
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        print_error(f"Missing required columns: {', '.join(missing_required)}")
        print_info(f"Required columns: {', '.join(required_columns)}")
        print_info(f"Found columns: {', '.join(df.columns)}")
        return False
    
    # Handle volume column
    if 'vol' not in df.columns and 'volume' not in df.columns:
        print_warning("No volume column found. Adding default volume column.")
        # Create synthetic volume based on price movement
        price_change = np.abs(df['close'] - df['open'])
        df['vol'] = (price_change * 1000).astype(int)  # Synthetic volume
    elif 'volume' in df.columns and 'vol' not in df.columns:
        print_info("Renaming 'volume' column to 'vol'")
        df['vol'] = df['volume']
    
    # Add openint if missing
    if 'openint' not in df.columns:
        print_warning("No openint column found. Adding default values.")
        df['openint'] = 0
    
    print_info("Data validation complete. All necessary columns are present.")
    return True

def prepare_stock_data(df):
    """Prepare stock data by ensuring all necessary columns and proper data types."""
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Validate and fix columns
        if not validate_columns(df):
            return None
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                print_error(f"Failed to convert timestamp column: {str(e)}")
                return None
        
        # Convert numeric columns to float32 for memory efficiency
        numeric_columns = ['open', 'high', 'low', 'close', 'vol']
        for col in numeric_columns:
            try:
                df[col] = df[col].astype('float32')
            except Exception as e:
                print_error(f"Failed to convert {col} to float: {str(e)}")
                return None
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate additional features
        df['returns'] = calculate_returns(df['close'])
        df['volatility'] = calculate_volatility(df['close'])
        
        print_info("Data preparation completed successfully")
        return df
        
    except Exception as e:
        print_error(f"Error preparing stock data: {str(e)}")
        return None

def get_optimal_chunk_size(df):
    """Calculate optimal chunk size based on available memory"""
    available_memory = psutil.virtual_memory().available
    row_size = df.memory_usage(deep=True).sum() / len(df)
    safe_memory = available_memory * 0.1
    optimal_chunk = int(safe_memory / row_size)
    return max(100, min(1000, optimal_chunk))

def plot_stock_metrics(df, ticker=None, chunk_size=None):
    """Plot stock metrics with proper data preparation."""
    if df is None or df.empty:
        print_info("No data to plot. Please load data first.")
        return
        
    print_info("Starting to plot stock metrics")
    try:
        # Prepare data first
        df = prepare_stock_data(df)
        if df is None:
            print_error("Failed to prepare data for plotting")
            return
        
        if ticker:
            df = df[df['ticker'] == ticker]
            print_info(f"Plotting data for ticker: {ticker}")
        
        # Validate data
        if df.empty:
            print_error("No data to plot")
            return
            
        # Determine chunk size
        if chunk_size is None:
            chunk_size = get_optimal_chunk_size(df)
        print_info(f"Using chunk size: {chunk_size}")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        
        # Plot 1: Price Data (OHLC)
        for i, start in enumerate(chunks, 1):
            end = min(start + chunk_size, total_rows)
            chunk = df.iloc[start:end]
            
            if i % 10 == 0:
                print_info(f"Processing chunk {i}/{total_chunks}")
            
            ax1.plot(chunk['timestamp'], chunk['close'], label='Close' if start == 0 else "", color='blue')
            ax1.plot(chunk['timestamp'], chunk['high'], label='High' if start == 0 else "", color='green', alpha=0.5)
            ax1.plot(chunk['timestamp'], chunk['low'], label='Low' if start == 0 else "", color='red', alpha=0.5)
            ax1.plot(chunk['timestamp'], chunk['open'], label='Open' if start == 0 else "", color='orange', alpha=0.5)
        
        ax1.set_title('Stock Price Over Time (OHLC)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Volume
        print_info("Generating volume plot")
        volume_chunk_size = chunk_size // 2
        volume_chunks = range(0, total_rows, volume_chunk_size)
        
        for start in volume_chunks:
            end = min(start + volume_chunk_size, total_rows)
            chunk = df.iloc[start:end]
            ax2.bar(chunk['timestamp'], chunk['vol'], color='gray', alpha=0.7)
            
        ax2.set_title('Trading Volume')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
        # Calculate basic statistics
        print_info("Calculating metrics...")
        
        metrics = [
            f"Ticker: {df['ticker'].iloc[0]}",
            f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}",
            f"Trading days: {len(df)}",
            f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}",
            f"Average closing price: {df['close'].mean():.2f}",
            f"Average volume: {df['vol'].mean():,.0f}",
            f"Max volume: {df['vol'].max():,.0f}"
        ]
        
        print_info("Trading metrics:")
        for metric in metrics:
            print_info(metric)
        
    except Exception as e:
        print_error(f"Failed to plot stock metrics: {str(e)}")
        print_error(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

def plot_optimization_results(df, predictions, optimizer_name, chunk_size=None):
    print_info(f"Plotting optimization results for {optimizer_name}")
    try:
        # Convert to float32 for memory efficiency
        df['close'] = df['close'].astype('float32')
        predictions = predictions.astype('float32')
        
        if chunk_size is None:
            chunk_size = get_optimal_chunk_size(df)
        print_info(f"Using chunk size: {chunk_size}")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        total_rows = len(df)
        chunks = range(0, total_rows, chunk_size)
        total_chunks = len(range(0, total_rows, chunk_size))
        
        print_info(f"Processing {total_rows} rows in {total_chunks} chunks")
        
        # Plot actual vs predicted
        for i, start in enumerate(chunks, 1):
            end = min(start + chunk_size, total_rows)
            chunk_df = df.iloc[start:end]
            chunk_pred = predictions[start:end]
            
            if i % 10 == 0:
                print_info(f"Processing chunk {i}/{total_chunks}")
            
            ax1.plot(chunk_df['timestamp'], chunk_df['close'], 
                    label='Actual' if start == 0 else "", color='blue')
            ax1.plot(chunk_df['timestamp'], chunk_pred, 
                    label='Predicted' if start == 0 else "", color='red', linestyle='--')
            
        ax1.set_title(f'{optimizer_name} - Actual vs Predicted Prices')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Calculate and plot error
        print_info("Calculating error metrics...")
        mae_sum = 0
        mse_sum = 0
        count = 0
        
        error_chunk_size = chunk_size // 2
        error_chunks = range(0, total_rows, error_chunk_size)
        
        for start in error_chunks:
            end = min(start + error_chunk_size, total_rows)
            chunk_df = df.iloc[start:end]
            chunk_pred = predictions[start:end]
            
            error = chunk_df['close'] - chunk_pred
            ax2.plot(chunk_df['timestamp'], error, color='green')
            
            mae_sum += np.abs(error).sum()
            mse_sum += (error ** 2).sum()
            count += len(chunk_df)
            
        ax2.set_title('Prediction Error')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Error')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate final metrics
        mae = mae_sum / count
        mse = mse_sum / count
        rmse = np.sqrt(mse)
        
        print_info("Error metrics:")
        print_info(f"Mean Absolute Error: {mae:.4f}")
        print_info(f"Mean Squared Error: {mse:.4f}")
        print_info(f"Root Mean Squared Error: {rmse:.4f}")
        
    except Exception as e:
        print_error(f"Failed to plot optimization results: {str(e)}")
        print_error(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

def align_arrays(timestamps, prices, predictions, window=20):
    """Ensure all arrays are properly aligned and of the same length."""
    # Calculate minimum valid length considering window size
    min_len = min(len(timestamps), len(prices), len(predictions)) - window
    
    if min_len <= 0:
        raise ValueError("Not enough data points after considering window size")
    
    # Trim all arrays to the same length
    timestamps = timestamps[window:min_len+window]
    prices = prices[window:min_len+window]
    predictions = predictions[window:min_len+window]
    
    return timestamps, prices, predictions

def plot_after_optimization(timestamps, prices, predictions, optimizer_name="AMDS"):
    try:
        # 1. Align arrays using the new function
        timestamps, prices, predictions = align_arrays(timestamps, prices, predictions)

        # 2. Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Actual vs Predicted
        ax1.plot(timestamps, prices, 'b-', label='Actual', alpha=0.7)
        ax1.plot(timestamps, predictions, 'r--', label='Predicted', alpha=0.7)
        ax1.set_title(f'{optimizer_name} - Actual vs Predicted Prices')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Error
        error = prices - predictions
        ax2.plot(timestamps, error, 'g-', label='Error')
        ax2.set_title('Prediction Error')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Error')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{optimizer_name}_results.png')
        plt.close()
        
        # Print some metrics
        mae = np.mean(np.abs(error))
        mse = np.mean(error ** 2)
        rmse = np.sqrt(mse)
        print(f"\nOptimizer: {optimizer_name}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
    except Exception as e:
        print_error(f"Error plotting optimization results: {str(e)}")
        print_info("Debug info:")
        print_info(f"Original lengths:")
        print_info(f"Timestamps length: {len(timestamps)}")
        print_info(f"Prices length: {len(prices)}")
        print_info(f"Predictions length: {len(predictions)}")

class StockDataPlotter:
    def __init__(self):
        self.df = None
        self.current_file = None
        self.default_columns = {
            'ticker': None,
            'timestamp': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'vol': None,
            'openint': None
        }
        self.initialize_empty_data()
    
    def initialize_empty_data(self):
        """Initialize empty DataFrame with default columns."""
        self.df = pd.DataFrame(columns=list(self.default_columns.keys()))
        return self.df
    
    def load_data(self, file_path):
        """Load and validate data from file."""
        try:
            self.current_file = file_path
            df = pd.read_csv(file_path)
            prepared_df = prepare_stock_data(df)
            if prepared_df is not None:
                self.df = prepared_df
                return True
            return False
        except Exception as e:
            print_error(f"Error loading data: {str(e)}")
            return False

class StockPlotterGUI:
    def __init__(self, root):
        self.root = root
        self._init_variables()        # Initialize all variables first
        self._init_gui_structure()    # Setup basic GUI structure
        self._init_gui_components()   # Setup all GUI components (including status_label)
        self._init_data_structures()  # Initialize data structures
        self.restore_application_state()  # Restore previous state
        
    def _init_variables(self):
        self.df = None
        self.current_file = None
        self.default_columns = {
            'ticker': None,
            'timestamp': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'vol': None,
            'openint': None
        }
        self.model_state = {
            'weights': None,
            'bias': None,
            'scaler': None,
            'optimizer': None,
            'hyperparameters': {},
            'training_history': {}
        }
    
    def _init_gui_structure(self):
        self.root.title("Stock Data Plotter")
        self.root.geometry("1200x800")
        
        # Initialize data
        self.data = None
        self.current_file = None
        
        # Create main container with grid configuration
        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create paned window for resizable sections
        self.paned = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, sticky="nsew")
        
        # Configure main container grid
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Create control panel
        self.control_frame = ttk.Frame(self.paned, padding="5")
        self.plot_frame = ttk.Frame(self.paned, padding="5")
        
        # Add frames to paned window
        self.paned.add(self.control_frame, weight=1)
        self.paned.add(self.plot_frame, weight=3)
        
        self.setup_controls()
        self.setup_plot_area()
        
    def _init_gui_components(self):
        # File controls
        file_frame = ttk.LabelFrame(self.control_frame, text="File Controls", padding="5")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load Data", command=self.load_data).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Reload", command=self.reload_data).pack(fill=tk.X, pady=2)
        
        # Plot controls
        plot_frame = ttk.LabelFrame(self.control_frame, text="Plot Controls", padding="5")
        plot_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Plot type selection
        ttk.Label(plot_frame, text="Plot Type:").pack(fill=tk.X, pady=2)
        self.plot_type = tk.StringVar(value="Stock Price")
        plot_types = [
            "Stock Price",
            "Returns Distribution",
            "Volatility",
            "Correlation Matrix",
            "Feature Importance",
            "Trading Signals",
            "Volume Analysis"
        ]
        self.plot_combo = ttk.Combobox(plot_frame, textvariable=self.plot_type, values=plot_types)
        self.plot_combo.pack(fill=tk.X, pady=2)
        self.plot_combo.bind('<<ComboboxSelected>>', self.update_plot)
        
        # Time range controls
        range_frame = ttk.LabelFrame(plot_frame, text="Time Range", padding="5")
        range_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(range_frame, text="Start Date:").pack(fill=tk.X, pady=2)
        self.start_date = ttk.Entry(range_frame)
        self.start_date.pack(fill=tk.X, pady=2)
        
        ttk.Label(range_frame, text="End Date:").pack(fill=tk.X, pady=2)
        self.end_date = ttk.Entry(range_frame)
        self.end_date.pack(fill=tk.X, pady=2)
        
        # Plot options
        options_frame = ttk.LabelFrame(self.control_frame, text="Plot Options", padding="5")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Moving average
        self.ma_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Show Moving Average", 
                       variable=self.ma_var, 
                       command=self.update_plot).pack(fill=tk.X, pady=2)
        
        ttk.Label(options_frame, text="MA Period:").pack(fill=tk.X, pady=2)
        self.ma_period = ttk.Entry(options_frame)
        self.ma_period.insert(0, "20")
        self.ma_period.pack(fill=tk.X, pady=2)
        
        # Volume subplot
        self.volume_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Volume", 
                       variable=self.volume_var, 
                       command=self.update_plot).pack(fill=tk.X, pady=2)
        
        # Style controls
        style_frame = ttk.LabelFrame(self.control_frame, text="Style", padding="5")
        style_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.style_var = tk.StringVar(value="default")
        styles = ["default", "seaborn", "grayscale", "dark_background"]
        ttk.Label(style_frame, text="Plot Style:").pack(fill=tk.X, pady=2)
        style_combo = ttk.Combobox(style_frame, textvariable=self.style_var, values=styles)
        style_combo.pack(fill=tk.X, pady=2)
        style_combo.bind('<<ComboboxSelected>>', self.update_plot)
        
        # Action buttons
        ttk.Button(self.control_frame, text="Update Plot", 
                  command=self.update_plot).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Save Plot", 
                  command=self.save_plot).pack(fill=tk.X, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.control_frame, textvariable=self.status_var, 
                               wraplength=200)
        status_label.pack(fill=tk.X, padx=5, pady=5)
        
    def setup_plot_area(self):
        # Create figure and canvas
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
    def load_data(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select Data File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                self.data = pd.read_csv(file_path)
                self.current_file = file_path
                
                # Convert timestamp if present
                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(
                        self.data['timestamp'].astype(str).str.slice(0, 8), 
                        format='%Y%m%d'
                    )
                
                self.status_var.set(f"Loaded {len(self.data)} rows from {file_path}")
                self.update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Error loading data")
    
    def reload_data(self):
        if self.current_file:
            try:
                self.data = pd.read_csv(self.current_file)
                self.status_var.set(f"Reloaded {len(self.data)} rows")
                self.update_plot()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reload data: {str(e)}")
                self.status_var.set("Error reloading data")
        else:
            messagebox.showwarning("Warning", "No file loaded yet")
    
    def update_plot(self, event=None):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        try:
            # Clear figure
            self.fig.clear()
            
            # Set style
            plt.style.use(self.style_var.get())
            
            plot_type = self.plot_type.get()
            
            if plot_type == "Stock Price":
                self.plot_stock_price()
            elif plot_type == "Returns Distribution":
                self.plot_returns_distribution()
            elif plot_type == "Volatility":
                self.plot_volatility()
            elif plot_type == "Correlation Matrix":
                self.plot_correlation_matrix()
            elif plot_type == "Feature Importance":
                self.plot_feature_importance()
            elif plot_type == "Trading Signals":
                self.plot_trading_signals()
            elif plot_type == "Volume Analysis":
                self.plot_volume_analysis()
            
            self.canvas.draw()
            self.status_var.set(f"Updated {plot_type} plot")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update plot: {str(e)}")
            self.status_var.set("Error updating plot")
    
    def plot_stock_price(self):
        ax1 = self.fig.add_subplot(211 if self.volume_var.get() else 111)
        
        # Plot price
        ax1.plot(self.data['timestamp'], self.data['close'], label='Close Price')
        
        # Add moving average if requested
        if self.ma_var.get():
            period = int(self.ma_period.get())
            ma = self.data['close'].rolling(window=period).mean()
            ax1.plot(self.data['timestamp'], ma, 
                    label=f'{period}-day Moving Average', 
                    linestyle='--')
        
        ax1.set_title('Stock Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Add volume subplot if requested
        if self.volume_var.get() and 'vol' in self.data.columns:
            ax2 = self.fig.add_subplot(212)
            ax2.bar(self.data['timestamp'], self.data['vol'], alpha=0.5)
            ax2.set_title('Volume')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Volume')
            ax2.grid(True)
        
        self.fig.tight_layout()
    
    def plot_returns_distribution(self):
        # Calculate returns
        returns = self.data['close'].pct_change().dropna()
        
        ax = self.fig.add_subplot(111)
        ax.hist(returns, bins=50, density=True, alpha=0.7)
        
        # Add kernel density estimate
        sns.kdeplot(data=returns, ax=ax, color='red', label='KDE')
        
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Returns')
        ax.set_ylabel('Density')
        ax.grid(True)
        ax.legend()
        
        self.fig.tight_layout()
    
    def plot_volatility(self):
    # Calculate volatility
        window = int(self.ma_period.get())
        returns = self.data['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        
        ax = self.fig.add_subplot(111)
        ax.plot(self.data['timestamp'], volatility, label=f'{window}-day Volatility')
        
        ax.set_title('Rolling Volatility')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.grid(True)
        ax.legend()
        
        self.fig.tight_layout()
    
    def plot_correlation_matrix(self):
        # Select numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        
        ax = self.fig.add_subplot(111)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        
        ax.set_title('Correlation Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        self.fig.tight_layout()
    
    def plot_feature_importance(self):
        # Calculate simple feature importance based on correlation with close price
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'close']
        
        importance = {}
        for col in numeric_cols:
            corr = abs(self.data[col].corr(self.data['close']))
            importance[col] = corr
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        ax = self.fig.add_subplot(111)
        ax.barh(list(importance.keys()), list(importance.values()))
        
        ax.set_title('Feature Importance')
        ax.set_xlabel('Absolute Correlation with Close Price')
        
        self.fig.tight_layout()
    
    def plot_trading_signals(self):
        # Calculate simple moving average crossover signals
        short_period = 20
        long_period = 50
        
        short_ma = self.data['close'].rolling(window=short_period).mean()
        long_ma = self.data['close'].rolling(window=long_period).mean()
        
        # Generate signals
        buy_signals = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        sell_signals = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        
        # Plot
        ax1 = self.fig.add_subplot(211)
        ax1.plot(self.data['timestamp'], self.data['close'], label='Price')
        ax1.plot(self.data['timestamp'], short_ma, label=f'{short_period}-day MA')
        ax1.plot(self.data['timestamp'], long_ma, label=f'{long_period}-day MA')
        
        # Plot signals
        ax1.scatter(self.data['timestamp'][buy_signals], 
                   self.data['close'][buy_signals], 
                   marker='^', color='g', label='Buy Signal')
        ax1.scatter(self.data['timestamp'][sell_signals], 
                   self.data['close'][sell_signals], 
                   marker='v', color='r', label='Sell Signal')
        
        ax1.set_title('Trading Signals')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Add volume subplot
        if self.volume_var.get() and 'vol' in self.data.columns:
            ax2 = self.fig.add_subplot(212)
            ax2.bar(self.data['timestamp'], self.data['vol'], alpha=0.5)
            ax2.set_title('Volume')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Volume')
            ax2.grid(True)
        
        self.fig.tight_layout()
    
    def plot_volume_analysis(self):
        if 'vol' not in self.data.columns:
            messagebox.showwarning("Warning", "Volume data not available")
            return
        
        # Create subplots
        fig = self.fig
        gs = fig.add_gridspec(2, 2)
        
        # Volume over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.bar(self.data['timestamp'], self.data['vol'], alpha=0.5)
        ax1.set_title('Volume Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Volume')
        ax1.grid(True)
        
        # Volume distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.data['vol'], bins=50, density=True, alpha=0.7)
        ax2.set_title('Volume Distribution')
        ax2.set_xlabel('Volume')
        ax2.set_ylabel('Density')
        ax2.grid(True)
        
        # Price-Volume correlation
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(self.data['vol'], self.data['close'], alpha=0.5)
        ax3.set_title('Price vs Volume')
        ax3.set_xlabel('Volume')
        ax3.set_ylabel('Price')
        ax3.grid(True)
        
        self.fig.tight_layout()
    
    def save_plot(self):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), 
                          ("All files", "*.*")]
            )
            if file_path:
                self.fig.savefig(file_path)
                self.status_var.set(f"Saved plot to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
            self.status_var.set("Error saving plot")

def main():
    root = tk.Tk()
    app = StockPlotterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 