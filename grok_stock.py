import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from sklearn.utils.validation import check_is_fitted
from scipy.special import expit
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import logging
from datetime import datetime
import psutil
import json
import duckdb
import pyarrow.feather as feather
import pyarrow.parquet as pq
import polars as pl
import h5py
import pyarrow as pa
import gc
import signal
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

# Import our optimizers
from optimizers import optimizer_registry, OptimizerBase, Hyperparameter
from custom_optimizers.hesm import HESMOptimizer  # Add this import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Utility functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class PlotWindow(tk.Toplevel):
    """A floating window for plots that can be detached from the main window"""
    def __init__(self, parent):
        # Initialize with parent's root window
        super().__init__(parent.root)
        self.title("Plot View")
        self.geometry("800x600")
        
        # Configure window
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.state('normal')
        
        # Store reference to parent
        self.parent = parent
        
        # Create matplotlib figure and canvas
        self.fig = parent.fig
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        
    def on_closing(self):
        """Handle window closing by reattaching plot to main window"""
        self.parent.reattach_plot()
        self.destroy()

class DataGridView(ttk.Frame):
    """A grid view for displaying stock data"""
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_grid()
        
    def setup_grid(self):
        # Create treeview
        self.tree = ttk.Treeview(self)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Configure tree columns
        self.tree["columns"] = []
        self.tree["show"] = "headings"
        
    def update_data(self, df):
        """Update the grid with new data"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if df is None or df.empty:
            return
            
        # Configure columns
        columns = list(df.columns)
        self.tree["columns"] = columns
        
        # Set column headings
        for col in columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_column(c))
            # Adjust column width based on content
            max_width = max(
                len(str(col)),
                df[col].astype(str).str.len().max() if len(df) > 0 else 0
            )
            self.tree.column(col, width=min(max_width * 10, 200))
        
        # Add data
        for idx, row in df.iterrows():
            values = [str(row[col]) for col in columns]
            self.tree.insert("", "end", values=values)
            
        # Limit displayed rows for performance
        if len(df) > 1000:
            self.tree.delete(*self.tree.get_children()[1000:])
            
    def sort_column(self, col):
        """Sort tree contents when a column header is clicked"""
        l = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        l.sort()
        # Rearrange items in sorted positions
        for index, (val, k) in enumerate(l):
            self.tree.move(k, "", index)

class ManualWindow(tk.Toplevel):
    """A window for displaying the user manual"""
    def __init__(self, parent, style_manager=None):
        super().__init__(parent.root)  # Use parent.root as the parent window
        self.parent = parent
        self.style_manager = style_manager
        self.title("Stock Grok - User Manual")
        self.geometry("800x600")
        
        # Create text widget with scrollbar
        self.text_frame = ttk.Frame(self)
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbars
        self.vsb = ttk.Scrollbar(self.text_frame, orient="vertical")
        self.hsb = ttk.Scrollbar(self.text_frame, orient="horizontal")
        self.text = tk.Text(self.text_frame, wrap=tk.NONE, 
                           yscrollcommand=self.vsb.set,
                           xscrollcommand=self.hsb.set)
        
        # Configure scrollbars
        self.vsb.config(command=self.text.yview)
        self.hsb.config(command=self.text.xview)
        
        # Grid layout
        self.text.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        self.text_frame.grid_rowconfigure(0, weight=1)
        self.text_frame.grid_columnconfigure(0, weight=1)
        
        # Load and display manual content
        try:
            with open('USER_MANUAL.md', 'r') as f:
                content = f.read()
            self.text.insert('1.0', content)
            self.text.config(state='disabled')  # Make read-only
        except Exception as e:
            self.text.insert('1.0', f"Error loading manual: {str(e)}")
            self.text.config(state='disabled')
        
        # Configure tags for markdown-like formatting
        self.text.tag_configure("heading1", font=("TkDefaultFont", 14, "bold"))
        self.text.tag_configure("heading2", font=("TkDefaultFont", 12, "bold"))
        self.text.tag_configure("heading3", font=("TkDefaultFont", 10, "bold"))
        self.text.tag_configure("code", font=("Courier", 10))
        
        # Apply theme
        self.update_theme()
        
        # Apply basic markdown formatting
        self.apply_markdown_formatting()
    
    def update_theme(self):
        """Update the window theme"""
        if self.style_manager:
            colors = self.style_manager.colors[self.style_manager.current_theme]
            
            # Update text widget colors
            self.text.configure(
                background=colors['bg'],
                foreground=colors['fg'],
                insertbackground=colors['fg'],
                selectbackground=colors['select_bg'],
                selectforeground=colors['select_fg']
            )
            
            # Update text tags
            self.text.tag_configure("heading1", foreground=colors['fg'])
            self.text.tag_configure("heading2", foreground=colors['fg'])
            self.text.tag_configure("heading3", foreground=colors['fg'])
            self.text.tag_configure("code", 
                                  background=colors['button'],
                                  foreground=colors['fg'])
            
            # Update frame colors
            self.text_frame.configure(style='TFrame')
            
            # Force redraw
            self.update_idletasks()
    
    def apply_markdown_formatting(self):
        """Apply basic markdown formatting to the text"""
        self.text.config(state='normal')
        
        # Find and format headers
        content = self.text.get("1.0", tk.END)
        lines = content.split('\n')
        pos = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            if line.startswith('# '):
                self.text.tag_add("heading1", f"1.{pos}", f"1.{pos + line_length}")
            elif line.startswith('## '):
                self.text.tag_add("heading2", f"1.{pos}", f"1.{pos + line_length}")
            elif line.startswith('### '):
                self.text.tag_add("heading3", f"1.{pos}", f"1.{pos + line_length}")
            elif line.startswith('```'):
                code_start = pos + line_length
                try:
                    code_end = content.index('```', code_start)
                    self.text.tag_add("code", f"1.{code_start}", f"1.{code_end}")
                except ValueError:
                    pass
            
            pos += line_length
        
        self.text.config(state='disabled')

class StyleManager:
    """Manages application-wide styling and themes"""
    def __init__(self):
        self.style = ttk.Style()
        self._current_theme = 'light'
        
        # Get available themes
        self.available_themes = self.style.theme_names()
        
        # Define custom colors
        self.colors = {
            'light': {
                'bg': '#ffffff',
                'fg': '#333333',
                'select_bg': '#0078d7',
                'select_fg': '#ffffff',
                'button': '#f0f0f0',
                'button_pressed': '#e1e1e1',
                'border': '#cccccc',
                'input_bg': '#ffffff',
                'input_fg': '#333333',
                'disabled_bg': '#f5f5f5',
                'disabled_fg': '#a0a0a0'
            },
            'dark': {
                'bg': '#2d2d2d',
                'fg': '#ffffff',
                'select_bg': '#0078d7',
                'select_fg': '#ffffff',
                'button': '#3d3d3d',
                'button_pressed': '#4d4d4d',
                'border': '#555555',
                'input_bg': '#3d3d3d',
                'input_fg': '#ffffff',
                'disabled_bg': '#404040',
                'disabled_fg': '#808080'
            }
        }
        
        # Initialize with light theme
        self.apply_theme('light')
    
    def apply_theme(self, theme_name):
        """Apply the specified theme"""
        self._current_theme = theme_name
        colors = self.colors[theme_name]
        
        # Try to use a suitable built-in theme as base
        if theme_name == 'dark' and 'clam' in self.available_themes:
            self.style.theme_use('clam')
        elif 'vista' in self.available_themes:
            self.style.theme_use('vista')
        elif 'clam' in self.available_themes:
            self.style.theme_use('clam')
        
        # Configure ttk styles
        self.style.configure('TFrame', background=colors['bg'])
        self.style.configure('TLabel', 
                           background=colors['bg'],
                           foreground=colors['fg'])
        self.style.configure('TButton',
                           background=colors['button'],
                           foreground=colors['fg'],
                           bordercolor=colors['border'])
        self.style.map('TButton',
                      background=[('pressed', colors['button_pressed']),
                                ('active', colors['button'])],
                      foreground=[('disabled', colors['disabled_fg'])])
        
        self.style.configure('TEntry',
                           fieldbackground=colors['input_bg'],
                           foreground=colors['input_fg'],
                           bordercolor=colors['border'])
        
        self.style.configure('Treeview',
                           background=colors['bg'],
                           foreground=colors['fg'],
                           fieldbackground=colors['bg'])
        self.style.map('Treeview',
                      background=[('selected', colors['select_bg'])],
                      foreground=[('selected', colors['select_fg'])])
        
        # Configure special styles
        self.style.configure('Header.TLabel',
                           font=('TkDefaultFont', 12, 'bold'),
                           background=colors['bg'],
                           foreground=colors['fg'])
        
        self.style.configure('Title.TLabel',
                           font=('TkDefaultFont', 14, 'bold'),
                           background=colors['bg'],
                           foreground=colors['fg'])
        
        # Return colors for matplotlib plots
        return {
            'primary': colors['fg'],
            'secondary': colors['disabled_fg'],
            'background': colors['bg'],
            'grid': colors['border']
        }
    
    @property
    def current_theme(self):
        return self._current_theme
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        new_theme = 'dark' if self._current_theme == 'light' else 'light'
        return self.apply_theme(new_theme)

class StockGrokGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Grok Analysis")
        
        # Initialize style manager
        self.style_manager = StyleManager()
        
        # Configure main window
        self.root.geometry("1200x800")
        
        # Add shutdown flag
        self.is_shutting_down = False
        
        # Configure shutdown protocol
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        
        # Initialize data variables
        self.data = None
        self.file_path = None
        self.n_samples = 0
        self.timestamps = np.array([], dtype=np.int64)
        self.prices = np.array([], dtype=np.float64)
        self.returns = np.array([], dtype=np.float64)
        self.volatility = np.array([], dtype=np.float64)
        self.predictions = np.array([], dtype=np.float64)
        self.uncertainties = np.array([], dtype=np.float64)
        self.buy_signals = np.array([], dtype=bool)
        self.sell_signals = np.array([], dtype=bool)
        
        # Initialize optimizer registry and load custom optimizers
        self.optimizer_registry = optimizer_registry
        self.optimizer_registry.load_custom_optimizers()
        self.current_optimizer = None
        
        # Initialize GUI variables
        self.optimizer_var = tk.StringVar(value=self.optimizer_registry.get_optimizer_names()[0])
        self.lr_var = tk.StringVar(value="0.01")
        self.iterations_var = tk.StringVar(value="1000")
        self.plot_type_var = tk.StringVar(value="Stock Price")
        self.file_path_var = tk.StringVar()
        self.data_info_var = tk.StringVar(value="No data loaded")
        
        # Initialize hyperparameter variables
        self.hyperparameter_vars = {}
        
        # Create tooltip dictionary
        self.tooltips = {}
        
        # Add model state variables
        self.current_model = None
        self.model_state = {
            'weights': None,
            'bias': None,
            'scaler': None,
            'optimizer': None,
            'hyperparameters': {},
            'training_history': {}
        }
        
        # Add plot caching
        self.current_plot_type = None
        self.last_plot_update = None
        self.plot_needs_update = True
        
        # Initialize plot window
        self.plot_window = None
        self.is_plot_detached = False
        
        # Add cleanup on keyboard interrupt
        self.root.bind('<Control-c>', lambda e: self.shutdown())
        self.root.bind('<Control-q>', lambda e: self.shutdown())
        
        # Restore previous state
        self.restore_application_state()
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure main window grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create horizontal paned window for controls and content
        self.h_paned = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.h_paned.grid(row=0, column=0, sticky="nsew")
        
        # Configure main container grid
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Create frames
        self.control_frame = ttk.Frame(self.h_paned, padding="5")
        self.content_paned = ttk.PanedWindow(self.h_paned, orient=tk.VERTICAL)
        
        # Add frames to horizontal paned window
        self.h_paned.add(self.control_frame, weight=1)
        self.h_paned.add(self.content_paned, weight=4)
        
        self.setup_controls()
        self.setup_plot_area()
        
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def enter(event):
            # Create tooltip window
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)  # Remove window decorations
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            # Create tooltip label
            label = ttk.Label(tooltip, text=text, justify=tk.LEFT,
                            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                            wraplength=250)
            label.pack()
            
            # Store tooltip reference
            self.tooltips[widget] = tooltip
        
        def leave(event):
            # Destroy tooltip if it exists
            if widget in self.tooltips:
                self.tooltips[widget].destroy()
                del self.tooltips[widget]
        
        def motion(event):
            # Update tooltip position
            if widget in self.tooltips:
                self.tooltips[widget].wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        
        # Bind events
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
        widget.bind('<Motion>', motion)

    def setup_controls(self):
        # File controls frame
        file_frame = ttk.LabelFrame(self.control_frame, text="File Controls", padding="5")
        file_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.file_path_label = ttk.Label(file_frame, textvariable=self.file_path_var, wraplength=200)
        self.file_path_label.grid(row=0, column=0, columnspan=3, pady=5)
        
        # Button frame for better organization
        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        ttk.Button(button_frame, text="Open File", command=self.open_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Reload Data", command=self.reload_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Manual", command=self.show_manual).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Toggle Theme", command=self.toggle_theme).pack(side=tk.LEFT, padx=2)
        
        # Data info display
        self.data_info_label = ttk.Label(file_frame, textvariable=self.data_info_var, 
                                       wraplength=200, justify=tk.LEFT)
        self.data_info_label.grid(row=2, column=0, columnspan=3, pady=5, padx=5, sticky="w")
        
        # Plot controls frame
        plot_frame = ttk.LabelFrame(self.control_frame, text="Plot Controls", padding="5")
        plot_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Add float/dock button
        self.float_button = ttk.Button(plot_frame, text="Float Plot", command=self.toggle_plot_window)
        self.float_button.grid(row=0, column=0, columnspan=2, pady=5, padx=5, sticky="ew")
        
        ttk.Label(plot_frame, text="Plot Type:").grid(row=1, column=0, pady=5, padx=5)
        self.plot_type_combo = ttk.Combobox(plot_frame, textvariable=self.plot_type_var, 
            values=["Stock Price", "Returns Distribution", "Volatility", "Prediction vs Actual", 
                   "Residuals", "Learning Curves", "Correlation Matrix", "Feature Importance", 
                   "Uncertainty", "Trading Signals", "Cumulative Returns", "HESM Analysis"])
        self.plot_type_combo.grid(row=1, column=1, pady=5, padx=5, sticky="ew")
        self.plot_type_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        
        # Model controls frame
        model_frame = ttk.LabelFrame(self.control_frame, text="Model Controls", padding="5")
        model_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Optimizer selection and management
        optimizer_frame = ttk.Frame(model_frame)
        optimizer_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(optimizer_frame, text="Optimizer:").pack(side=tk.LEFT, padx=5)
        self.optimizer_combo = ttk.Combobox(optimizer_frame, textvariable=self.optimizer_var,
                                          values=self.optimizer_registry.get_optimizer_names())
        self.optimizer_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Add reload button for custom optimizers
        ttk.Button(optimizer_frame, text="↻", width=3,
                  command=self.reload_custom_optimizers).pack(side=tk.LEFT, padx=2)
        
        # Add button to create new custom optimizer
        ttk.Button(optimizer_frame, text="+", width=3,
                  command=self.create_custom_optimizer).pack(side=tk.LEFT, padx=2)
        
        self.optimizer_combo.bind('<<ComboboxSelected>>', self.update_hyperparameters)
        
        # Basic training parameters
        ttk.Label(model_frame, text="Iterations:").grid(row=1, column=0, pady=5, padx=5)
        self.iterations_entry = ttk.Entry(model_frame, textvariable=self.iterations_var)
        self.iterations_entry.grid(row=1, column=1, pady=5, padx=5, sticky="ew")
        
        ttk.Label(model_frame, text="Learning Rate:").grid(row=2, column=0, pady=5, padx=5)
        self.lr_entry = ttk.Entry(model_frame, textvariable=self.lr_var)
        self.lr_entry.grid(row=2, column=1, pady=5, padx=5, sticky="ew")
        
        # Hyperparameter frame
        self.hyper_frame = ttk.LabelFrame(self.control_frame, text="Optimizer Hyperparameters", padding="5")
        self.hyper_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Initialize hyperparameters for default optimizer
        self.update_hyperparameters()
        
        # Add model save/load frame
        model_io_frame = ttk.LabelFrame(self.control_frame, text="Model Save/Load", padding="5")
        model_io_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        # Add save/load buttons
        ttk.Button(model_io_frame, text="Save Model", command=self.save_model).grid(row=0, column=0, pady=5, padx=5)
        ttk.Button(model_io_frame, text="Load Model", command=self.load_model).grid(row=0, column=1, pady=5, padx=5)
        
        # Action buttons
        ttk.Button(self.control_frame, text="Train Model", command=self.train_model).grid(row=5, column=0, pady=10, padx=5, sticky="ew")
        ttk.Button(self.control_frame, text="Update Plot", command=self.update_plot).grid(row=6, column=0, pady=5, padx=5, sticky="ew")
        
        # Status
        self.status_label = ttk.Label(self.control_frame, textvariable=self.data_info_var, wraplength=200)
        self.status_label.grid(row=7, column=0, pady=10, padx=5)
        
    def update_hyperparameters(self, event=None):
        """Update hyperparameter controls based on selected optimizer"""
        # Clear existing hyperparameter widgets
        for widget in self.hyper_frame.winfo_children():
            widget.destroy()
        
        # Clear existing hyperparameter variables
        self.hyperparameter_vars.clear()
        
        # Get selected optimizer
        optimizer_name = self.optimizer_var.get()
        optimizer = self.optimizer_registry.get_optimizer(optimizer_name)
        
        if optimizer:
            # Create widgets for each hyperparameter
            for row, (name, hp) in enumerate(optimizer.hyperparameters.items()):
                # Create label
                ttk.Label(self.hyper_frame, text=f"{name}:").grid(
                    row=row, column=0, pady=2, padx=5, sticky="w"
                )
                
                # Create variable and widget based on type
                if hp.widget_type == "entry":
                    var = tk.StringVar(value=str(hp.current_value))
                    widget = ttk.Entry(self.hyper_frame, textvariable=var, width=10)
                elif hp.widget_type == "combobox":
                    var = tk.StringVar(value=str(hp.current_value))
                    widget = ttk.Combobox(self.hyper_frame, textvariable=var,
                                        values=hp.value_range, width=10)
                else:
                    continue
                
                widget.grid(row=row, column=1, pady=2, padx=5, sticky="ew")
                self.hyperparameter_vars[name] = var
                
                # Add tooltip
                self.create_tooltip(widget, hp.description)
            
            # Add reset button
            ttk.Button(self.hyper_frame, text="Reset to Defaults",
                      command=self.reset_hyperparameters).grid(
                row=len(optimizer.hyperparameters), column=0, columnspan=2,
                pady=5, padx=5, sticky="ew"
            )
        
        # Update current optimizer
        self.current_optimizer = optimizer

    def reset_hyperparameters(self):
        """Reset hyperparameters to default values"""
        if self.current_optimizer:
            for name, hp in self.current_optimizer.hyperparameters.items():
                if name in self.hyperparameter_vars:
                    self.hyperparameter_vars[name].set(str(hp.default_value))
                    hp.current_value = hp.default_value

    def setup_plot_area(self):
        """Setup the plot area in the main window"""
        # Create plot container but don't show it initially
        self.plot_container = ttk.Frame(self.main_container, padding="5")
        
        # Create figure
        self.fig = Figure(figsize=(10, 6))
        
        # Create canvas but don't pack it initially
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        
        # Create toolbar but don't pack it initially
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_container)
        
        # Create initial empty plot
        ax = self.fig.add_subplot(111)
        ax.set_title("No data loaded")
        ax.grid(True)
        self.canvas.draw()
    
    def toggle_plot_window(self):
        """Toggle between floating and docked plot window"""
        if self.is_plot_detached:
            self.reattach_plot()
        else:
            self.detach_plot()
            # Force a plot update after detaching
            self.plot_needs_update = True
            self.update_plot()
    
    def detach_plot(self):
        """Create a floating window for the plot"""
        if not self.is_plot_detached:
            # Create new plot window
            self.plot_window = PlotWindow(self)
            
            # Update button text
            self.float_button.configure(text="Close Plot")
            self.is_plot_detached = True
            
            # Force a redraw in the new window
            self.plot_window.canvas.draw()
    
    def reattach_plot(self):
        """Handle closing the plot window"""
        if self.is_plot_detached:
            # Update button text
            self.float_button.configure(text="Float Plot")
            self.is_plot_detached = False
            
            # Close the floating window if it exists
            if self.plot_window:
                self.plot_window.destroy()
                self.plot_window = None
    
    def open_csv(self):
        """Open file dialog to select data file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Data File",
                filetypes=[
                    ("All Supported Files", "*.csv *.json *.parquet *.feather *.h5 *.hdf5 *.duckdb *.arrow *.pkl *.pickle"),
                    ("CSV files", "*.csv"),
                    ("JSON files", "*.json"),
                    ("Parquet files", "*.parquet"),
                    ("Feather files", "*.feather"),
                    ("HDF5 files", "*.h5 *.hdf5"),
                    ("DuckDB files", "*.duckdb"),
                    ("Arrow files", "*.arrow"),
                    ("Pickle files", "*.pkl *.pickle"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.file_path = file_path
                self.file_path_var.set(f"File: {os.path.basename(file_path)}")
                self.load_data()
        except Exception as e:
            messagebox.showerror("Error", f"Error opening file: {str(e)}")
    
    def load_data(self):
        """Load and process the data file"""
        try:
            # Load the data file
            print(f"Loading data from: {self.file_path}")
            self.data = self.load_data_from_file(self.file_path)
            print(f"Raw data shape: {self.data.shape}")
            print(f"Columns found: {self.data.columns.tolist()}")
            
            # Handle timestamp format if present
            if 'timestamp' in self.data.columns:
                print("Processing timestamps...")
                try:
                    # Try different timestamp formats
                    if self.data['timestamp'].dtype == 'object':
                        # Try parsing as string format YYYYMMDD
                        self.data['timestamp'] = pd.to_datetime(
                            self.data['timestamp'].astype(str).str.slice(0, 8), 
                            format='%Y%m%d',
                            errors='coerce'
                        )
                    else:
                        # Try parsing as numeric timestamp
                        self.data['timestamp'] = pd.to_datetime(
                            self.data['timestamp'],
                            errors='coerce'
                        )
                    print(f"First few timestamps after conversion: {self.data['timestamp'].head()}")
                except Exception as e:
                    print(f"Warning: Could not parse timestamps: {str(e)}")
            
            # Handle volume/vol column
            if 'volume' in self.data.columns and 'vol' not in self.data.columns:
                self.data['vol'] = self.data['volume']
                self.data.drop('volume', axis=1, inplace=True)
            
            # Basic data validation
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                messagebox.showwarning("Warning", 
                    f"Missing columns: {', '.join(missing_columns)}\n"
                    "Some features may not work properly.")
            
            # Convert numeric columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'vol']
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Drop any rows with NaN values
            original_len = len(self.data)
            self.data = self.data.dropna(subset=['timestamp', 'close'])
            if len(self.data) < original_len:
                print(f"Dropped {original_len - len(self.data)} rows with NaN values")
            
            # Sort by timestamp and reset index
            if 'timestamp' in self.data.columns:
                self.data = self.data.sort_values('timestamp').reset_index(drop=True)
            
            # Process the data
            self.initialize_data()
            
            # Update info with more detailed statistics
            info_text = (
                f"Rows: {len(self.data):,}\n"
                f"Columns: {', '.join(self.data.columns)}\n"
                f"Memory Usage: {self.data.memory_usage().sum() / 1024 / 1024:.1f} MB\n"
            )
            
            if 'timestamp' in self.data.columns:
                info_text += (
                    f"Date Range: {self.data['timestamp'].min().strftime('%Y-%m-%d')} to "
                    f"{self.data['timestamp'].max().strftime('%Y-%m-%d')}\n"
                )
            
            if 'ticker' in self.data.columns:
                n_tickers = len(self.data['ticker'].unique())
                info_text += f"Unique Tickers: {n_tickers}"
            
            self.data_info_var.set(info_text)
            
            # Update status
            self.status_label.config(text="Data loaded successfully")
            
            # Initial plot
            self.update_plot()
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            print(error_msg)
            print("Debug info:")
            if hasattr(self, 'data') and self.data is not None:
                print(f"Data shape: {self.data.shape}")
                print(f"Columns: {self.data.columns.tolist()}")
                print("First few rows of timestamp column:")
                if 'timestamp' in self.data.columns:
                    print(self.data['timestamp'].head())
            messagebox.showerror("Error", error_msg)
            self.data_info_var.set("Error loading data")
    
    def reload_data(self):
        """Reload data from the current file"""
        if self.file_path:
            self.load_data()
        else:
            messagebox.showwarning("Warning", "No file loaded yet")
    
    def initialize_data(self):
        """Initialize data and model results"""
        if self.data is None:
            return
        
        try:
            self.n_samples = len(self.data)
            print(f"Initializing data with {self.n_samples} samples")
            
            # Convert timestamp to numerical index for plotting
            self.timestamps = np.arange(self.n_samples)
            self.prices = np.array(self.data['close'].values, dtype=float)
            
            # Calculate returns (avoiding division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.returns = np.diff(self.prices) / self.prices[:-1]
                self.returns = np.nan_to_num(self.returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate volatility with proper window handling
            window = 20
            self.volatility = np.zeros(self.n_samples)  # Initialize with same length as prices
            
            # Calculate rolling volatility
            for i in range(window, self.n_samples):
                self.volatility[i] = np.std(self.returns[max(0, i-window):i])
            
            # Fill initial window period with the first calculated volatility value
            first_valid_vol = self.volatility[window]
            self.volatility[:window] = first_valid_vol
            
            # Initialize predictions and signals with proper lengths
            self.predictions = np.zeros_like(self.prices)
            self.uncertainties = np.ones_like(self.prices) * np.std(self.returns)
            self.buy_signals = np.zeros_like(self.prices, dtype=bool)
            self.sell_signals = np.zeros_like(self.prices, dtype=bool)
            
            # Create features DataFrame with proper alignment
            features_dict = {
                'Price': self.prices,
                'Predicted': self.predictions,
                'Returns': np.pad(self.returns, (1, 0), mode='edge'),
                'Volatility': self.volatility
            }
            
            # Add volume if available
            if 'vol' in self.data.columns:
                features_dict['Volume'] = self.data['vol'].values
            
            # Create features DataFrame
            self.features = pd.DataFrame(
                features_dict,
                index=self.data['timestamp']  # Use actual timestamps as index
            )
            
            # Verify all arrays have the same length
            array_lengths = {
                'timestamps': len(self.timestamps),
                'prices': len(self.prices),
                'returns (padded)': len(self.features['Returns']),
                'volatility': len(self.volatility),
                'predictions': len(self.predictions),
                'uncertainties': len(self.uncertainties),
                'features': len(self.features)
            }
            
            print("Array lengths after initialization:")
            for name, length in array_lengths.items():
                print(f"{name}: {length}")
            
            if not all(length == self.n_samples for length in array_lengths.values()):
                raise ValueError("Array length mismatch detected")
            
            self.status_label.config(text="Data initialized successfully")
            
        except Exception as e:
            print(f"Error in initialize_data: {str(e)}")
            print("Debug info:")
            if hasattr(self, 'data') and self.data is not None:
                print(f"Data shape: {self.data.shape}")
                print(f"Columns: {self.data.columns.tolist()}")
                print("Sample of data:")
                print(self.data.head())
            self.status_label.config(text=f"Error initializing data: {str(e)}")
            # Initialize empty arrays in case of error
            self.timestamps = np.array([])
            self.prices = np.array([])
            self.returns = np.array([])
            self.volatility = np.array([])
            self.predictions = np.array([])
            self.features = pd.DataFrame()
    
    def train_model(self):
        """Train the selected model with hyperparameters"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        try:
            # Get selected optimizer
            optimizer_name = self.optimizer_var.get()
            optimizer = self.optimizer_registry.get_optimizer(optimizer_name)
            
            if not optimizer:
                raise ValueError(f"Optimizer {optimizer_name} not found")
            
            # Update optimizer hyperparameters from GUI
            for name, var in self.hyperparameter_vars.items():
                if name in optimizer.hyperparameters:
                    value = var.get()
                    if not optimizer.hyperparameters[name].validate(value):
                        raise ValueError(f"Invalid value for {name}: {value}")
                    optimizer.hyperparameters[name].current_value = value
            
            # Get basic training parameters
            iterations = int(self.iterations_var.get())
            learning_rate = float(self.lr_var.get())
            
            self.status_label.config(text=f"Training {optimizer_name}...")
            
            # Prepare the data for training
            x = self.data[['open', 'high', 'low', 'vol']].values
            y = self.data['close'].values
            
            # Ensure all arrays have the same length
            min_len = min(len(self.timestamps), len(y))
            x = x[:min_len]
            y = y[:min_len]
            self.timestamps = self.timestamps[:min_len]
            self.prices = self.prices[:min_len]
            
            # Create and fit scaler
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
            
            # Initialize weights randomly
            n_features = x.shape[1]
            weights = np.random.randn(n_features) * 0.01
            bias = 0.0
            
            # Training loop
            loss_history = []
            for i in range(iterations):
                # Forward pass
                predictions = np.dot(x, weights) + bias
                
                # Compute gradients
                error = predictions - y
                weight_gradients = np.dot(x.T, error) / len(y)
                bias_gradient = np.mean(error)
                
                # Update weights using optimizer
                weights = optimizer.optimize(weights, weight_gradients, learning_rate)
                bias -= learning_rate * bias_gradient
                
                # Record loss
                loss = np.mean(error ** 2)
                loss_history.append(loss)
                
                # Print progress occasionally
                if (i + 1) % 100 == 0:
                    self.status_label.config(text=f"Iteration {i+1}/{iterations}, Loss: {loss:.6f}")
                    self.root.update_idletasks()
            
            # Store final predictions
            self.predictions = np.dot(x, weights) + bias
            
            # Store model state
            self.current_model = True
            
            # Get additional metrics for HESM optimizer
            optimizer_metrics = {}
            if isinstance(optimizer, HESMOptimizer):
                optimizer_metrics.update({
                    'performance_metrics': optimizer.get_performance_metrics(),
                    'convergence_analysis': optimizer.get_convergence_analysis(),
                    'model_contributions': optimizer.get_model_contributions()
                })
            
            self.model_state.update({
                'weights': weights,
                'bias': bias,
                'scaler': scaler,
                'optimizer': optimizer_name,
                'hyperparameters': {
                    name: hp.current_value
                    for name, hp in optimizer.hyperparameters.items()
                },
                'training_history': {
                    'loss': loss_history,
                    'optimizer_metrics': optimizer_metrics
                }
            })
            
            # Mark plot for update
            self.plot_needs_update = True
            
            # Update the plot immediately after training
            self.plot_type_var.set("Stock Price")  # Switch to stock price view
            self.update_plot()
            
            # Save the training results
            np.savetxt('predictions.csv', self.predictions, delimiter=',')
            
            # Print detailed metrics for HESM
            if isinstance(optimizer, HESMOptimizer):
                metrics = optimizer_metrics['performance_metrics']
                analysis = optimizer_metrics['convergence_analysis']
                contributions = optimizer_metrics['model_contributions']
                
                print("\nHESM Optimizer Metrics:")
                print(f"Mean Entropy: {metrics['entropy_mean']:.4f} ± {metrics['entropy_std']:.4f}")
                print(f"Exploration Ratio: {metrics['exploration_ratio']:.2%}")
                print(f"\nConvergence Analysis:")
                print(f"Entropy Trend: {analysis['entropy_trend']:.6f}")
                print(f"Weight Stability: {analysis['weight_stability']:.6f}")
                print(f"Phase Changes: {analysis['phase_changes']}")
                if analysis['is_converged']:
                    print(f"Converged at iteration: {analysis['convergence_iteration']}")
                else:
                    print("Not fully converged")
                
                print("\nModel Contributions:")
                for model in contributions:
                    print(f"\nModel {model['model_id']+1}:")
                    print(f"  Weight: {model['weight_mean']:.4f} ± {model['weight_std']:.4f}")
                    print(f"  Learning Rate: {model['learning_rate_mean']:.4f} ± {model['learning_rate_std']:.4f}")
                    print(f"  Selection Frequency: {model['selection_frequency']:.2%}")
            
            self.status_label.config(text=f"Training completed: {optimizer_name}")
            print(f"Training completed with {optimizer_name}")
            print(f"Final shapes - X: {x.shape}, y: {y.shape}, predictions: {self.predictions.shape}")
            
        except ValueError as ve:
            error_msg = f"Invalid parameter value: {str(ve)}"
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg)
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            print(error_msg)
            print(f"Debug info:")
            print(f"Optimizer: {self.optimizer_var.get()}")
            print(f"Iterations: {self.iterations_var.get()}")
            print(f"Learning rate: {self.lr_var.get()}")
            if hasattr(self, 'timestamps'):
                print(f"Timestamps shape: {self.timestamps.shape}")
            if hasattr(self, 'prices'):
                print(f"Prices shape: {self.prices.shape}")
            if hasattr(self, 'predictions'):
                print(f"Predictions shape: {self.predictions.shape}")
            self.status_label.config(text=error_msg)

    def get_chunk_size(self):
        """Calculate optimal chunk size based on available memory"""
        try:
            available_memory = psutil.virtual_memory().available
            # Estimate memory needed per data point (timestamps + prices + predictions)
            bytes_per_point = 8 * 3  # 8 bytes per float64 * 3 arrays
            # Use 10% of available memory
            safe_chunk_size = int((available_memory * 0.1) / bytes_per_point)
            # Ensure chunk size is reasonable
            return max(1000, min(safe_chunk_size, 100000))
        except:
            return 10000  # Default chunk size if memory check fails
    
    def update_plot(self):
        """Update the plot based on selected type"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        if not self.is_plot_detached:
            return  # Only update plot if it's detached
            
        try:
            plot_type = self.plot_type_var.get()
            
            # Check if we need to update the plot
            if not self.plot_needs_update and plot_type == self.current_plot_type:
                return
            
            print(f"\nUpdating plot: {plot_type}")
            self.fig.clear()
            
            # Set style to grayscale
            plt.style.use('grayscale')
            
            ax = self.fig.add_subplot(111)
            
            # Get optimal chunk size
            chunk_size = self.get_chunk_size()
            print(f"Using chunk size: {chunk_size}")
            
            # Ensure all base arrays are the same length
            min_len = min(len(self.timestamps), len(self.prices), len(self.predictions))
            timestamps = self.timestamps[:min_len]
            prices = self.prices[:min_len]
            predictions = self.predictions[:min_len]
            
            # Define grayscale colors
            colors = {
                'primary': '#333333',      # Dark gray for main lines
                'secondary': '#666666',    # Medium gray for secondary lines
                'tertiary': '#999999',     # Light gray for tertiary elements
                'background': '#f0f0f0',   # Very light gray for backgrounds
                'highlight': '#000000',    # Black for highlights
                'positive': '#404040',     # Dark gray for positive signals
                'negative': '#808080',     # Medium gray for negative signals
                'grid': '#cccccc'          # Light gray for grid lines
            }
            
            if plot_type == "Stock Price":
                try:
                    # Plot in chunks
                    for i in range(0, min_len, chunk_size):
                        end_idx = min(i + chunk_size, min_len)
                        chunk_timestamps = timestamps[i:end_idx]
                        chunk_prices = prices[i:end_idx]
                        chunk_predictions = predictions[i:end_idx]
                        
                        ax.plot(chunk_timestamps, chunk_prices, '-', 
                               color=colors['primary'], 
                               label='Actual Price' if i == 0 else "", 
                               alpha=0.8)
                        ax.plot(chunk_timestamps, chunk_predictions, '--', 
                               color=colors['secondary'], 
                               label='Predicted Price' if i == 0 else "", 
                               alpha=0.6)
                    
                    ax.set_title('Stock Price Over Time')
                    ax.set_xlabel('Time Index')
                    ax.set_ylabel('Price')
                    ax.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
                    ax.legend()
                    
                except Exception as e:
                    print(f"Error in Stock Price plotting: {str(e)}")
                    raise
            
            elif plot_type == "Returns Distribution":
                if len(self.returns) > 0:
                    ax.hist(self.returns, bins=50, density=True, 
                           color=colors['primary'], alpha=0.7, 
                           edgecolor=colors['highlight'])
                    ax.set_title('Returns Distribution')
                    ax.set_xlabel('Returns')
                    ax.set_ylabel('Density')
                    ax.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
            
            elif plot_type == "Volatility":
                if len(self.volatility) > 0:
                    vol_timestamps = timestamps[20:]  # Adjust for volatility window
                    vol_len = min(len(vol_timestamps), len(self.volatility))
                    
                    ax.plot(vol_timestamps[:vol_len], self.volatility[:vol_len], '-',
                           color=colors['primary'], label='Volatility')
                    ax.set_title('Rolling Volatility')
                    ax.set_xlabel('Time Index')
                    ax.set_ylabel('Volatility')
                    ax.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
                    ax.legend()
            
            elif plot_type == "Prediction vs Actual":
                ax.scatter(prices, predictions, alpha=0.5, 
                          color=colors['primary'])
                min_val = min(prices.min(), predictions.min())
                max_val = max(prices.max(), predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], '--',
                       color=colors['secondary'], lw=2)
                ax.set_title('Predicted vs Actual Values')
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
            
            elif plot_type == "Residuals":
                residuals = prices - predictions
                ax.scatter(predictions, residuals, alpha=0.5,
                          color=colors['primary'])
                ax.axhline(y=0, color=colors['secondary'], linestyle='--')
                ax.set_title('Residual Plot')
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
            
            elif plot_type == "Learning Curves":
                # Generate mock learning curve for visualization
                iterations = np.arange(100)
                loss = 1.0 / (1.0 + 0.1 * iterations) + 0.1 * np.random.randn(100)
                ax.plot(iterations, loss, '-', color=colors['primary'],
                       label='Training Loss')
                ax.set_title('Learning Curves')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                ax.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
                ax.legend()
            
            elif plot_type == "Correlation Matrix":
                try:
                    # Use the features DataFrame directly for correlation
                    corr_matrix = self.features.corr()
                    
                    # Plot correlation matrix with grayscale colormap
                    sns.heatmap(corr_matrix, 
                              annot=True, 
                              cmap='gray', 
                              center=0, 
                              ax=ax, 
                              fmt='.2f',
                              square=True)
                    
                    ax.set_title('Feature Correlation Matrix')
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    
                    # Adjust layout to prevent label cutoff
                    plt.tight_layout()
                    
                except Exception as e:
                    print(f"Error in correlation matrix plotting: {str(e)}")
                    raise
            
            elif plot_type == "Feature Importance":
                try:
                    if 'ticker' in self.data.columns:
                        ticker_data = self.data.copy()
                        # Convert numeric columns to float32 for efficiency
                        for col in ticker_data.select_dtypes(include=[np.number]).columns:
                            ticker_data[col] = ticker_data[col].astype('float32')
                        
                        # Use consistent feature names that match the mapping
                        features = ['Open', 'High', 'Low', 'Close', 'Volume']
                        importances = self._calculate_feature_importance(ticker_data, features)
                        
                        # Sort features by importance
                        sorted_idx = np.argsort(importances)[::-1]  # Reverse sort for highest first
                        pos = np.arange(len(features)) + .5
                        
                        # Create the horizontal bar plot
                        ax.barh(pos, importances[sorted_idx], align='center',
                               color=colors['primary'], alpha=0.7,
                               edgecolor=colors['highlight'])
                        
                        # Customize the plot
                        ax.set_yticks(pos)
                        ax.set_yticklabels(np.array(features)[sorted_idx])
                        ax.set_xlabel('Relative Importance')
                        ax.set_title('Feature Importance in Stock Price Analysis\n' + 
                                   'Shows how each feature contributes to:\n' +
                                   '• Current price correlation (30%)\n' +
                                   '• Future price prediction (30%)\n' +
                                   '• Volatility impact (20%)\n' +
                                   '• Volume relationship (20%)')
                        
                        # Add percentage labels on bars
                        for i, v in enumerate(importances[sorted_idx]):
                            ax.text(v, i, f' {v:.1%}', va='center',
                                   color=colors['highlight'])
                        
                        # Add grid and set it behind the bars
                        ax.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
                        ax.set_axisbelow(True)
                    
                except Exception as e:
                    print(f"Error in Feature Importance plotting: {str(e)}")
                    raise
            
            elif plot_type == "Uncertainty":
                for i in range(0, min_len, chunk_size):
                    end_idx = min(i + chunk_size, min_len)
                    chunk_timestamps = timestamps[i:end_idx]
                    chunk_predictions = predictions[i:end_idx]
                    chunk_uncertainties = self.uncertainties[i:end_idx]
                    chunk_prices = prices[i:end_idx]
                    
                    ax.plot(chunk_timestamps, chunk_predictions, '-',
                           color=colors['primary'],
                           label='Prediction' if i == 0 else "")
                    ax.fill_between(chunk_timestamps,
                                  chunk_predictions - 2*chunk_uncertainties,
                                  chunk_predictions + 2*chunk_uncertainties,
                                  color=colors['tertiary'],
                                  alpha=0.2,
                                  label='95% Confidence' if i == 0 else "")
                    ax.plot(chunk_timestamps, chunk_prices, '--',
                           color=colors['secondary'],
                           label='Actual' if i == 0 else "")
                
                ax.set_title('Predictions with Uncertainty')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
                ax.legend()
            
            elif plot_type == "Trading Signals":
                for i in range(0, min_len, chunk_size):
                    end_idx = min(i + chunk_size, min_len)
                    chunk_timestamps = timestamps[i:end_idx]
                    chunk_prices = prices[i:end_idx]
                    chunk_buy = self.buy_signals[i:end_idx]
                    chunk_sell = self.sell_signals[i:end_idx]
                    
                    ax.plot(chunk_timestamps, chunk_prices, '-',
                           color=colors['primary'],
                           label='Price' if i == 0 else "")
                    if np.any(chunk_buy):
                        ax.scatter(chunk_timestamps[chunk_buy],
                                 chunk_prices[chunk_buy],
                                 color=colors['positive'],
                                 marker='^',
                                 label='Buy Signal' if i == 0 else "")
                    if np.any(chunk_sell):
                        ax.scatter(chunk_timestamps[chunk_sell],
                                 chunk_prices[chunk_sell],
                                 color=colors['negative'],
                                 marker='v',
                                 label='Sell Signal' if i == 0 else "")
                
                ax.set_title('Trading Signals')
                ax.set_xlabel('Time')
                ax.set_ylabel('Price')
                ax.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
                ax.legend()
            
            elif plot_type == "HESM Analysis":
                try:
                    # Get the current optimizer
                    optimizer = self.optimizer_registry.get_optimizer(self.optimizer_var.get())
                    
                    if isinstance(optimizer, HESMOptimizer) and optimizer.adaptation_history:
                        # Create subplots for HESM analysis
                        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                        
                        # Plot entropy evolution
                        ax1 = self.fig.add_subplot(gs[0, 0])
                        entropy_values = [h['entropy'] for h in optimizer.adaptation_history]
                        ax1.plot(entropy_values, color=colors['primary'], label='Entropy')
                        ax1.axhline(y=float(optimizer.hyperparameters['entropy_threshold'].current_value),
                                  color=colors['secondary'], linestyle='--', label='Threshold')
                        ax1.set_title('Entropy Evolution')
                        ax1.set_xlabel('Iteration')
                        ax1.set_ylabel('Entropy')
                        ax1.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
                        ax1.legend()
                        
                        # Plot model weights evolution
                        ax2 = self.fig.add_subplot(gs[0, 1])
                        model_weights = np.array([h['model_weights'] for h in optimizer.adaptation_history])
                        for i in range(model_weights.shape[1]):
                            ax2.plot(model_weights[:, i], 
                                   label=f'Model {i+1}',
                                   alpha=0.7)
                        ax2.set_title('Model Weights Evolution')
                        ax2.set_xlabel('Iteration')
                        ax2.set_ylabel('Weight')
                        ax2.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
                        ax2.legend()
                        
                        # Plot learning rates
                        ax3 = self.fig.add_subplot(gs[1, 0])
                        learning_rates = np.array([h['learning_rates'] for h in optimizer.adaptation_history])
                        for i in range(learning_rates.shape[1]):
                            ax3.plot(learning_rates[:, i],
                                   label=f'Model {i+1}',
                                   alpha=0.7)
                        ax3.set_title('Learning Rates Evolution')
                        ax3.set_xlabel('Iteration')
                        ax3.set_ylabel('Learning Rate')
                        ax3.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
                        ax3.legend()
                        
                        # Plot exploration vs exploitation
                        ax4 = self.fig.add_subplot(gs[1, 1])
                        threshold = float(optimizer.hyperparameters['entropy_threshold'].current_value)
                        exploration = np.array(entropy_values) > threshold
                        exploitation = ~exploration
                        iterations = np.arange(len(entropy_values))
                        
                        ax4.fill_between(iterations[exploration], 0, 1, 
                                       color=colors['primary'], alpha=0.3,
                                       label='Exploration')
                        ax4.fill_between(iterations[exploitation], 0, 1,
                                       color=colors['secondary'], alpha=0.3,
                                       label='Exploitation')
                        ax4.set_title('Exploration vs Exploitation')
                        ax4.set_xlabel('Iteration')
                        ax4.set_ylim(0, 1)
                        ax4.grid(True, color=colors['grid'], linestyle='--', alpha=0.5)
                        ax4.legend()
                        
                        self.fig.suptitle('HESM Optimizer Analysis', y=1.02)
                    else:
                        ax = self.fig.add_subplot(111)
                        ax.text(0.5, 0.5, 'HESM analysis only available\nwhen using HESM optimizer',
                               ha='center', va='center')
                        ax.set_axis_off()
                    
                except Exception as e:
                    print(f"Error in HESM analysis plotting: {str(e)}")
                    raise
            
            # Update plot cache info
            self.current_plot_type = plot_type
            self.plot_needs_update = False
            
            self.fig.tight_layout()
            
            # Update only the floating window
            if self.is_plot_detached and self.plot_window:
                self.plot_window.canvas.draw()
            
            self.status_label.config(text=f"Updated plot: {plot_type}")
            
        except Exception as e:
            error_msg = f"Plot error: {str(e)}"
            print(error_msg)
            self.status_label.config(text=error_msg)

    def _calculate_feature_importance(self, data, features):
        """
        Calculate feature importance scores based on multiple factors:
        1. Price correlation: How strongly each feature correlates with price movements
        2. Lead-lag relationship: How well each feature predicts future price changes
        3. Volatility impact: How much each feature contributes to price volatility
        4. Trading volume relationship: How features relate to trading activity
        """
        try:
            importance_scores = []
            
            # Get only numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            numeric_data = data[numeric_columns].copy()  # Make a copy to avoid warnings
            
            # Create feature name mapping (both ways)
            feature_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'vol'
            }
            
            # Handle NaN and inf values
            numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
            numeric_data = numeric_data.ffill().bfill()
            
            # 1. Calculate price correlations (both current and future)
            with np.errstate(divide='ignore', invalid='ignore'):
                current_correlations = np.abs(numeric_data.corr()['close'])
                current_correlations = current_correlations.fillna(0)
            
            # Calculate 1-day future returns
            future_returns = numeric_data['close'].pct_change().shift(-1)
            with np.errstate(divide='ignore', invalid='ignore'):
                future_correlations = np.abs(numeric_data.corrwith(future_returns))
                future_correlations = future_correlations.fillna(0)
            
            # 2. Calculate volatility impact using 20-day rolling window
            price_volatility = numeric_data['close'].rolling(window=20, min_periods=1).std()
            with np.errstate(divide='ignore', invalid='ignore'):
                volatility_impact = np.abs(numeric_data.corrwith(price_volatility))
                volatility_impact = volatility_impact.fillna(0)
            
            # 3. Calculate volume relationship if volume data exists
            if 'vol' in numeric_data.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    volume_relationship = np.abs(numeric_data.corrwith(numeric_data['vol']))
                    volume_relationship = volume_relationship.fillna(0)
            else:
                volume_relationship = pd.Series(0, index=numeric_data.columns)
            
            for feature in features:
                # Get the corresponding column name from mapping
                data_col = feature_mapping.get(feature)
                
                if data_col is not None and data_col in numeric_data.columns:
                    # Weight different factors for overall importance
                    importance = (
                        0.3 * current_correlations.get(data_col, 0) +  # Current price correlation
                        0.3 * future_correlations.get(data_col, 0) +   # Future price prediction
                        0.2 * volatility_impact.get(data_col, 0) +     # Volatility impact
                        0.2 * volume_relationship.get(data_col, 0)     # Volume relationship
                    )
                else:
                    print(f"Warning: Feature '{feature}' (mapped to '{data_col}') not found in columns: {list(numeric_data.columns)}")
                    importance = 0.0
                
                importance_scores.append(importance)
            
            # Normalize scores to percentages
            importance_scores = np.array(importance_scores)
            importance_scores = np.abs(importance_scores)
            total_importance = importance_scores.sum()
            if total_importance > 0:
                importance_scores = importance_scores / total_importance
            
            return importance_scores
            
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            print("Available numeric columns:", list(data.select_dtypes(include=[np.number]).columns))
            print("All columns:", list(data.columns))
            return np.ones(len(features)) / len(features)

    def load_data_from_file(self, file_path):
        """Load data from various file formats into a pandas DataFrame."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext == '.json':
                return pd.read_json(file_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(file_path)
            elif file_ext == '.feather':
                return pd.DataFrame(feather.read_feather(file_path))
            elif file_ext == '.h5' or file_ext == '.hdf5':
                # Try to load as Keras model first
                try:
                    model = tf.keras.models.load_model(file_path)
                    # Extract weights and create a DataFrame
                    weights = []
                    for layer in model.layers:
                        layer_weights = layer.get_weights()
                        if layer_weights:
                            weights.extend(layer_weights)
                    return pd.DataFrame({'weights': weights})
                except:
                    # If not a Keras model, try as regular HDF5
                    with h5py.File(file_path, 'r') as f:
                        # Convert HDF5 dataset to DataFrame
                        data = {}
                        def visit_func(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                data[name] = obj[:]
                        f.visititems(visit_func)
                        return pd.DataFrame(data)
            elif file_ext == '.duckdb':
                # Connect to DuckDB database
                conn = duckdb.connect(file_path)
                # Get list of tables
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if not tables:
                    raise ValueError("No tables found in DuckDB database")
                # Read first table by default
                table_name = tables[0][0]
                df = conn.execute(f"SELECT * FROM {table_name}").df()
                conn.close()
                return df
            elif file_ext == '.arrow':
                return pd.DataFrame(pl.read_ipc(file_path).to_pandas())
            elif file_ext == '.pkl' or file_ext == '.pickle':
                return pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
        except Exception as e:
            raise Exception(f"Error loading {file_ext} file: {str(e)}")

    def save_model(self):
        """Save the current model state to a file"""
        if self.current_model is None:
            messagebox.showwarning("Warning", "No trained model to save")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".h5",
                filetypes=[
                    ("HDF5 files", "*.h5"),
                    ("NumPy files", "*.npy"),
                    ("JSON files", "*.json"),
                    ("Parquet files", "*.parquet"),
                    ("Feather files", "*.feather"),
                    ("DuckDB files", "*.duckdb"),
                    ("Arrow files", "*.arrow"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ],
                title="Save Model"
            )
            
            if not file_path:
                return

            # Create a dictionary with all model data
            model_data = {
                'weights': self.model_state['weights'],
                'bias': self.model_state['bias'],
                'optimizer': self.optimizer_var.get(),
                'hyperparameters': {
                    'learning_rate': float(self.lr_var.get()),
                    'iterations': int(self.iterations_var.get())
                },
                'training_history': self.model_state['training_history'],
                'feature_scaler': {
                    'mean': self.model_state['scaler'].mean_ if self.model_state['scaler'] is not None else None,
                    'scale': self.model_state['scaler'].scale_ if self.model_state['scaler'] is not None else None
                } if self.model_state['scaler'] is not None else None
            }

            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.h5':
                # Save using h5py
                with h5py.File(file_path, 'w') as f:
                    # Save numpy arrays
                    f.create_dataset('weights', data=model_data['weights'])
                    f.create_dataset('bias', data=model_data['bias'])
                    
                    # Save scalar values
                    f.attrs['optimizer'] = model_data['optimizer']
                    f.attrs['learning_rate'] = model_data['hyperparameters']['learning_rate']
                    f.attrs['iterations'] = model_data['hyperparameters']['iterations']
                    
                    # Save training history
                    history_grp = f.create_group('training_history')
                    for key, value in model_data['training_history'].items():
                        if isinstance(value, (list, np.ndarray)):
                            history_grp.create_dataset(key, data=np.array(value))
                    
                    # Save scaler if it exists
                    if model_data['feature_scaler'] is not None:
                        scaler_grp = f.create_group('scaler')
                        scaler_grp.attrs['mean'] = model_data['feature_scaler']['mean']
                        scaler_grp.attrs['scale'] = model_data['feature_scaler']['scale']
            
            elif file_ext == '.npy':
                # Save as NumPy array
                np.save(file_path, {
                    'weights': model_data['weights'],
                    'bias': model_data['bias'],
                    'scaler_mean': model_data['feature_scaler']['mean'] if model_data['feature_scaler'] else None,
                    'scaler_scale': model_data['feature_scaler']['scale'] if model_data['feature_scaler'] else None
                })
                
                # Save metadata separately
                meta_path = file_path.replace('.npy', '_meta.json')
                with open(meta_path, 'w') as f:
                    json.dump({
                        'optimizer': model_data['optimizer'],
                        'hyperparameters': model_data['hyperparameters'],
                        'training_history': {
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in model_data['training_history'].items()
                        }
                    }, f)
            
            elif file_ext == '.json':
                # Convert numpy arrays to lists for JSON serialization
                json_data = {
                    'weights': model_data['weights'].tolist(),
                    'bias': float(model_data['bias']),
                    'optimizer': model_data['optimizer'],
                    'hyperparameters': model_data['hyperparameters'],
                    'training_history': {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in model_data['training_history'].items()
                    },
                    'feature_scaler': {
                        'mean': model_data['feature_scaler']['mean'].tolist() if model_data['feature_scaler'] else None,
                        'scale': model_data['feature_scaler']['scale'].tolist() if model_data['feature_scaler'] else None
                    } if model_data['feature_scaler'] else None
                }
                with open(file_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
            
            elif file_ext == '.parquet':
                # Convert to DataFrame and save as Parquet
                df_data = {
                    'weights': [model_data['weights'].tobytes()],
                    'bias': [float(model_data['bias'])],
                    'optimizer': [model_data['optimizer']],
                    'learning_rate': [model_data['hyperparameters']['learning_rate']],
                    'iterations': [model_data['hyperparameters']['iterations']]
                }
                if model_data['feature_scaler']:
                    df_data.update({
                        'scaler_mean': [model_data['feature_scaler']['mean'].tobytes()],
                        'scaler_scale': [model_data['feature_scaler']['scale'].tobytes()]
                    })
                df = pd.DataFrame(df_data)
                df.to_parquet(file_path)
            
            elif file_ext == '.feather':
                # Convert to DataFrame and save as Feather
                df_data = {
                    'weights': [model_data['weights'].tobytes()],
                    'bias': [float(model_data['bias'])],
                    'optimizer': [model_data['optimizer']],
                    'learning_rate': [model_data['hyperparameters']['learning_rate']],
                    'iterations': [model_data['hyperparameters']['iterations']]
                }
                if model_data['feature_scaler']:
                    df_data.update({
                        'scaler_mean': [model_data['feature_scaler']['mean'].tobytes()],
                        'scaler_scale': [model_data['feature_scaler']['scale'].tobytes()]
                    })
                df = pd.DataFrame(df_data)
                df.to_feather(file_path)
            
            elif file_ext == '.duckdb':
                # Save to DuckDB
                conn = duckdb.connect(file_path)
                df_data = {
                    'weights': [model_data['weights'].tobytes()],
                    'bias': [float(model_data['bias'])],
                    'optimizer': [model_data['optimizer']],
                    'learning_rate': [model_data['hyperparameters']['learning_rate']],
                    'iterations': [model_data['hyperparameters']['iterations']]
                }
                if model_data['feature_scaler']:
                    df_data.update({
                        'scaler_mean': [model_data['feature_scaler']['mean'].tobytes()],
                        'scaler_scale': [model_data['feature_scaler']['scale'].tobytes()]
                    })
                df = pd.DataFrame(df_data)
                conn.execute("CREATE TABLE IF NOT EXISTS model_data AS SELECT * FROM df")
                conn.close()
            
            elif file_ext == '.arrow':
                # Save as Arrow/IPC format
                df_data = {
                    'weights': [model_data['weights'].tobytes()],
                    'bias': [float(model_data['bias'])],
                    'optimizer': [model_data['optimizer']],
                    'learning_rate': [model_data['hyperparameters']['learning_rate']],
                    'iterations': [model_data['hyperparameters']['iterations']]
                }
                if model_data['feature_scaler']:
                    df_data.update({
                        'scaler_mean': [model_data['feature_scaler']['mean'].tobytes()],
                        'scaler_scale': [model_data['feature_scaler']['scale'].tobytes()]
                    })
                df = pd.DataFrame(df_data)
                table = pa.Table.from_pandas(df)
                with pa.OSFile(file_path, 'wb') as sink:
                    writer = pa.ipc.new_file(sink, table.schema)
                    writer.write_table(table)
                    writer.close()
            
            elif file_ext == '.csv':
                # Save as CSV with separate files for numpy arrays
                base_path = os.path.splitext(file_path)[0]
                # Save numpy arrays
                np.save(f"{base_path}_weights.npy", model_data['weights'])
                np.save(f"{base_path}_bias.npy", model_data['bias'])
                if model_data['feature_scaler']:
                    np.save(f"{base_path}_scaler_mean.npy", model_data['feature_scaler']['mean'])
                    np.save(f"{base_path}_scaler_scale.npy", model_data['feature_scaler']['scale'])
                
                # Save metadata as CSV
                df = pd.DataFrame({
                    'parameter': ['optimizer', 'learning_rate', 'iterations'],
                    'value': [
                        model_data['optimizer'],
                        model_data['hyperparameters']['learning_rate'],
                        model_data['hyperparameters']['iterations']
                    ]
                })
                df.to_csv(file_path, index=False)
            
            self.status_label.config(text=f"Model saved to {os.path.basename(file_path)}")
            
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg)

    def load_model(self):
        """Load a saved model from a file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[
                    ("All Supported Files", "*.h5 *.npy *.json *.parquet *.feather *.duckdb *.arrow *.csv"),
                    ("HDF5 files", "*.h5"),
                    ("NumPy files", "*.npy"),
                    ("JSON files", "*.json"),
                    ("Parquet files", "*.parquet"),
                    ("Feather files", "*.feather"),
                    ("DuckDB files", "*.duckdb"),
                    ("Arrow files", "*.arrow"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ],
                title="Load Model"
            )
            
            if not file_path:
                return

            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.h5':
                # Load from HDF5
                with h5py.File(file_path, 'r') as f:
                    self.model_state['weights'] = f['weights'][:]
                    self.model_state['bias'] = f['bias'][:]
                    
                    self.optimizer_var.set(f.attrs['optimizer'])
                    self.lr_var.set(str(f.attrs['learning_rate']))
                    self.iterations_var.set(str(f.attrs['iterations']))
                    
                    self.model_state['training_history'] = {}
                    if 'training_history' in f:
                        for key in f['training_history'].keys():
                            self.model_state['training_history'][key] = f['training_history'][key][:]
                    
                    if 'scaler' in f:
                        scaler = StandardScaler()
                        scaler.mean_ = f['scaler'].attrs['mean']
                        scaler.scale_ = f['scaler'].attrs['scale']
                        self.model_state['scaler'] = scaler
            
            elif file_ext == '.npy':
                # Load from NumPy
                data = np.load(file_path, allow_pickle=True).item()
                self.model_state['weights'] = data['weights']
                self.model_state['bias'] = data['bias']
                
                if 'scaler_mean' in data and 'scaler_scale' in data:
                    scaler = StandardScaler()
                    scaler.mean_ = data['scaler_mean']
                    scaler.scale_ = data['scaler_scale']
                    self.model_state['scaler'] = scaler
                
                # Load metadata from companion file
                meta_path = file_path.replace('.npy', '_meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                        self.optimizer_var.set(meta_data['optimizer'])
                        self.lr_var.set(str(meta_data['hyperparameters']['learning_rate']))
                        self.iterations_var.set(str(meta_data['hyperparameters']['iterations']))
                        self.model_state['training_history'] = meta_data['training_history']
            
            elif file_ext == '.json':
                # Load from JSON
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.model_state['weights'] = np.array(data['weights'])
                    self.model_state['bias'] = float(data['bias'])
                    self.optimizer_var.set(data['optimizer'])
                    self.lr_var.set(str(data['hyperparameters']['learning_rate']))
                    self.iterations_var.set(str(data['hyperparameters']['iterations']))
                    self.model_state['training_history'] = {
                        k: np.array(v) if isinstance(v, list) else v
                        for k, v in data['training_history'].items()
                    }
                    
                    if data['feature_scaler']:
                        scaler = StandardScaler()
                        scaler.mean_ = np.array(data['feature_scaler']['mean'])
                        scaler.scale_ = np.array(data['feature_scaler']['scale'])
                        self.model_state['scaler'] = scaler
            
            elif file_ext == '.parquet':
                # Load from Parquet
                df = pd.read_parquet(file_path)
                self.model_state['weights'] = np.frombuffer(df['weights'].iloc[0])
                self.model_state['bias'] = float(df['bias'].iloc[0])
                self.optimizer_var.set(df['optimizer'].iloc[0])
                self.lr_var.set(str(df['learning_rate'].iloc[0]))
                self.iterations_var.set(str(df['iterations'].iloc[0]))
                
                if 'scaler_mean' in df.columns:
                    scaler = StandardScaler()
                    scaler.mean_ = np.frombuffer(df['scaler_mean'].iloc[0])
                    scaler.scale_ = np.frombuffer(df['scaler_scale'].iloc[0])
                    self.model_state['scaler'] = scaler
            
            elif file_ext == '.feather':
                # Load from Feather
                df = pd.read_feather(file_path)
                self.model_state['weights'] = np.frombuffer(df['weights'].iloc[0])
                self.model_state['bias'] = float(df['bias'].iloc[0])
                self.optimizer_var.set(df['optimizer'].iloc[0])
                self.lr_var.set(str(df['learning_rate'].iloc[0]))
                self.iterations_var.set(str(df['iterations'].iloc[0]))
                
                if 'scaler_mean' in df.columns:
                    scaler = StandardScaler()
                    scaler.mean_ = np.frombuffer(df['scaler_mean'].iloc[0])
                    scaler.scale_ = np.frombuffer(df['scaler_scale'].iloc[0])
                    self.model_state['scaler'] = scaler
            
            elif file_ext == '.duckdb':
                # Load from DuckDB
                conn = duckdb.connect(file_path)
                df = conn.execute("SELECT * FROM model_data").df()
                conn.close()
                
                self.model_state['weights'] = np.frombuffer(df['weights'].iloc[0])
                self.model_state['bias'] = float(df['bias'].iloc[0])
                self.optimizer_var.set(df['optimizer'].iloc[0])
                self.lr_var.set(str(df['learning_rate'].iloc[0]))
                self.iterations_var.set(str(df['iterations'].iloc[0]))
                
                if 'scaler_mean' in df.columns:
                    scaler = StandardScaler()
                    scaler.mean_ = np.frombuffer(df['scaler_mean'].iloc[0])
                    scaler.scale_ = np.frombuffer(df['scaler_scale'].iloc[0])
                    self.model_state['scaler'] = scaler
            
            elif file_ext == '.arrow':
                # Load from Arrow/IPC
                with pa.ipc.open_file(file_path) as reader:
                    table = reader.read_all()
                    df = table.to_pandas()
                
                self.model_state['weights'] = np.frombuffer(df['weights'].iloc[0])
                self.model_state['bias'] = float(df['bias'].iloc[0])
                self.optimizer_var.set(df['optimizer'].iloc[0])
                self.lr_var.set(str(df['learning_rate'].iloc[0]))
                self.iterations_var.set(str(df['iterations'].iloc[0]))
                
                if 'scaler_mean' in df.columns:
                    scaler = StandardScaler()
                    scaler.mean_ = np.frombuffer(df['scaler_mean'].iloc[0])
                    scaler.scale_ = np.frombuffer(df['scaler_scale'].iloc[0])
                    self.model_state['scaler'] = scaler
            
            elif file_ext == '.csv':
                # Load from CSV and companion files
                base_path = os.path.splitext(file_path)[0]
                
                # Load numpy arrays
                self.model_state['weights'] = np.load(f"{base_path}_weights.npy")
                self.model_state['bias'] = np.load(f"{base_path}_bias.npy")
                
                # Try to load scaler if it exists
                try:
                    scaler = StandardScaler()
                    scaler.mean_ = np.load(f"{base_path}_scaler_mean.npy")
                    scaler.scale_ = np.load(f"{base_path}_scaler_scale.npy")
                    self.model_state['scaler'] = scaler
                except FileNotFoundError:
                    pass
                
                # Load metadata from CSV
                df = pd.read_csv(file_path)
                params = dict(zip(df['parameter'], df['value']))
                self.optimizer_var.set(params['optimizer'])
                self.lr_var.set(str(params['learning_rate']))
                self.iterations_var.set(str(params['iterations']))
            
            # Set current model flag
            self.current_model = True
            
            # Update predictions if data is loaded
            if self.data is not None:
                self.update_predictions()
            
            self.status_label.config(text=f"Model loaded from {os.path.basename(file_path)}")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg)

    def update_predictions(self):
        """Update predictions using the loaded model"""
        if self.current_model and self.data is not None:
            try:
                # Prepare features
                features = self.data[['open', 'high', 'low', 'vol']].values
                
                # Scale features if scaler exists
                if self.model_state['scaler'] is not None:
                    features = self.model_state['scaler'].transform(features)
                
                # Generate predictions
                self.predictions = np.dot(features, self.model_state['weights']) + self.model_state['bias']
                
                # Update plot
                self.plot_needs_update = True
                self.update_plot()
                
            except Exception as e:
                self.status_label.config(text=f"Error updating predictions: {str(e)}")

    def shutdown(self):
        """Gracefully shutdown the application"""
        try:
            self.is_shutting_down = True
            self.status_label.config(text="Shutting down...")
            
            # Save current application state
            self.save_application_state()
            
            # Close any open plot windows
            if self.plot_window:
                self.plot_window.destroy()
            
            # Clean up matplotlib resources
            plt.close('all')
            
            # Clean up any open file handles
            if hasattr(self, 'data') and isinstance(self.data, pd.DataFrame):
                self.data = None
            
            # Clean up any open database connections
            try:
                for conn in duckdb.connections:
                    conn.close()
            except:
                pass
            
            # Save current model if exists
            if self.current_model:
                try:
                    # Create 'autosave' directory if it doesn't exist
                    os.makedirs('autosave', exist_ok=True)
                    
                    # Save model with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    model_path = os.path.join('autosave', f'model_autosave_{timestamp}.h5')
                    
                    # Use existing save_model logic but with direct path
                    model_data = {
                        'weights': self.model_state['weights'],
                        'bias': self.model_state['bias'],
                        'optimizer': self.optimizer_var.get(),
                        'hyperparameters': {
                            'learning_rate': float(self.lr_var.get()),
                            'iterations': int(self.iterations_var.get())
                        },
                        'training_history': self.model_state['training_history'],
                        'feature_scaler': {
                            'mean': self.model_state['scaler'].mean_ if self.model_state['scaler'] is not None else None,
                            'scale': self.model_state['scaler'].scale_ if self.model_state['scaler'] is not None else None
                        } if self.model_state['scaler'] is not None else None
                    }
                    
                    with h5py.File(model_path, 'w') as f:
                        f.create_dataset('weights', data=model_data['weights'])
                        f.create_dataset('bias', data=model_data['bias'])
                        f.attrs['optimizer'] = model_data['optimizer']
                        f.attrs['learning_rate'] = model_data['hyperparameters']['learning_rate']
                        f.attrs['iterations'] = model_data['hyperparameters']['iterations']
                        
                        history_grp = f.create_group('training_history')
                        for key, value in model_data['training_history'].items():
                            if isinstance(value, (list, np.ndarray)):
                                history_grp.create_dataset(key, data=np.array(value))
                        
                        if model_data['feature_scaler'] is not None:
                            scaler_grp = f.create_group('scaler')
                            scaler_grp.attrs['mean'] = model_data['feature_scaler']['mean']
                            scaler_grp.attrs['scale'] = model_data['feature_scaler']['scale']
                    
                    print(f"Model auto-saved to {model_path}")
                except Exception as e:
                    print(f"Warning: Could not auto-save model: {str(e)}")
            
            # Destroy the main window
            self.root.destroy()
            
            # Perform final cleanup
            gc.collect()
            
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
            # Force destroy if graceful shutdown fails
            self.root.destroy()

    def save_application_state(self):
        """Save current application state"""
        try:
            # Create 'autosave' directory if it doesn't exist
            os.makedirs('autosave', exist_ok=True)
            
            # Save application state
            state = {
                'window_geometry': self.root.geometry(),
                'optimizer': self.optimizer_var.get(),
                'learning_rate': self.lr_var.get(),
                'iterations': self.iterations_var.get(),
                'plot_type': self.plot_type_var.get(),
                'is_plot_detached': self.is_plot_detached,
                'last_file_path': self.file_path if hasattr(self, 'file_path') else None,
                'theme': self.style_manager.current_theme,
                'hyperparameters': {
                    'weight_init': self.weight_init_var.get(),
                    'bias_init': self.bias_init_var.get(),
                    'weight_decay': self.weight_decay_var.get(),
                    'l1_reg': self.l1_reg_var.get(),
                    'l2_reg': self.l2_reg_var.get(),
                    'dropout_rate': self.dropout_rate_var.get()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Save state to JSON file
            state_path = os.path.join('autosave', 'last_session_state.json')
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"Application state saved to {state_path}")
            
        except Exception as e:
            print(f"Warning: Could not save application state: {str(e)}")

    def restore_application_state(self):
        """Restore previous application state"""
        try:
            state_path = os.path.join('autosave', 'last_session_state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                # Restore window geometry
                self.root.geometry(state['window_geometry'])
                
                # Restore variables
                self.optimizer_var.set(state['optimizer'])
                self.lr_var.set(state['learning_rate'])
                self.iterations_var.set(state['iterations'])
                self.plot_type_var.set(state['plot_type'])
                
                # Restore hyperparameters if they exist
                if 'hyperparameters' in state:
                    self.weight_init_var.set(state['hyperparameters'].get('weight_init', 'xavier'))
                    self.bias_init_var.set(state['hyperparameters'].get('bias_init', '0.0'))
                    self.weight_decay_var.set(state['hyperparameters'].get('weight_decay', '0.001'))
                    self.l1_reg_var.set(state['hyperparameters'].get('l1_reg', '0.0'))
                    self.l2_reg_var.set(state['hyperparameters'].get('l2_reg', '0.001'))
                    self.dropout_rate_var.set(state['hyperparameters'].get('dropout_rate', '0.1'))
                
                # Restore theme
                if 'theme' in state:
                    self.style_manager.apply_theme(state['theme'])
                
                # Restore last file if it exists
                if state['last_file_path'] and os.path.exists(state['last_file_path']):
                    self.file_path = state['last_file_path']
                    self.load_data()
                
                print(f"Application state restored from {state_path}")
                
        except Exception as e:
            print(f"Warning: Could not restore application state: {str(e)}")

    def show_manual(self):
        """Show the user manual window"""
        manual_window = ManualWindow(self, self.style_manager)
        manual_window.grab_set()  # Make the window modal
        self.manual_window = manual_window  # Store reference for theme updates

    def toggle_theme(self):
        """Toggle between light and dark themes"""
        plot_colors = self.style_manager.toggle_theme()
        
        # Update plot colors if plot exists
        if hasattr(self, 'fig') and self.fig:
            self.plot_needs_update = True
            self.update_plot()
        
        # Update manual window if it exists
        if hasattr(self, 'manual_window') and self.manual_window:
            self.manual_window.update_theme(self.style_manager.current_theme)
        
        # Save theme preference
        self.save_application_state()

    def reload_custom_optimizers(self):
        """Reload custom optimizers and update the GUI"""
        try:
            # Reload optimizers
            self.optimizer_registry.reload_custom_optimizers()
            
            # Update combobox values
            current_optimizer = self.optimizer_var.get()
            self.optimizer_combo['values'] = self.optimizer_registry.get_optimizer_names()
            
            # Try to keep the current selection if it still exists
            if current_optimizer in self.optimizer_registry.get_optimizer_names():
                self.optimizer_var.set(current_optimizer)
            else:
                self.optimizer_var.set(self.optimizer_registry.get_optimizer_names()[0])
            
            # Update hyperparameters for current selection
            self.update_hyperparameters()
            
            self.status_label.config(text="Custom optimizers reloaded")
            
        except Exception as e:
            error_msg = f"Error reloading custom optimizers: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg)
    
    def create_custom_optimizer(self):
        """Create a new custom optimizer from template"""
        try:
            # Ask for optimizer name
            name = simpledialog.askstring(
                "New Custom Optimizer",
                "Enter optimizer name (no spaces, CamelCase):",
                parent=self.root
            )
            
            if not name:
                return
            
            # Validate name
            if not name.isidentifier():
                raise ValueError("Invalid name. Use only letters, numbers, and underscores.")
            
            # Create filename
            filename = f"{name.lower()}.py"
            filepath = os.path.join("custom_optimizers", filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                if not messagebox.askyesno("Warning", 
                    f"File {filename} already exists. Overwrite?"):
                    return
            
            # Copy template content
            with open(os.path.join("custom_optimizers", "optimizer_template.py"), 'r') as f:
                template = f.read()
            
            # Create new optimizer file
            with open(filepath, 'w') as f:
                # Replace template class name with new name
                new_content = template.replace("CustomOptimizerTemplate", name)
                new_content = new_content.replace(
                    'name="CustomOptimizer"',
                    f'name="{name}"'
                )
                f.write(new_content)
            
            # Reload optimizers
            self.reload_custom_optimizers()
            
            # Show success message with instructions
            messagebox.showinfo("Success",
                f"Created {filename}\n\n"
                "To implement your optimizer:\n"
                "1. Edit the file in the custom_optimizers directory\n"
                "2. Implement your optimization logic\n"
                "3. Click the reload button (↻) to load your changes"
            )
            
        except Exception as e:
            error_msg = f"Error creating custom optimizer: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg)

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def enter(event):
            # Create tooltip window
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)  # Remove window decorations
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            # Create tooltip label
            label = ttk.Label(tooltip, text=text, justify=tk.LEFT,
                            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                            wraplength=250)
            label.pack()
            
            # Store tooltip reference
            self.tooltips[widget] = tooltip
        
        def leave(event):
            # Destroy tooltip if it exists
            if widget in self.tooltips:
                self.tooltips[widget].destroy()
                del self.tooltips[widget]
        
        def motion(event):
            # Update tooltip position
            if widget in self.tooltips:
                self.tooltips[widget].wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        
        # Bind events
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
        widget.bind('<Motion>', motion)

class GridStockGrokGUI(StockGrokGUI):
    """Extended GUI with grid view"""
    def __init__(self, root):
        # Store root reference
        self.root = root
        
        # Initialize all variables first
        self._init_variables()
        
        # Initialize the UI components that parent might need
        self._init_ui_components()
        
        # Call parent's __init__
        super().__init__(root)
        
        # Setup additional UI components
        self._setup_grid_components()
    
    def _init_variables(self):
        """Initialize all variables before parent initialization"""
        # Grid-specific variables
        self.grid_view = None
        self.grid_controls = None
        self.row_limit_var = tk.StringVar(value="1000")
        self.grid_container = None
        
        # Weight initialization variables
        self.weight_init_var = tk.StringVar(value="xavier")
        self.bias_init_var = tk.StringVar(value="0.0")
        self.weight_decay_var = tk.StringVar(value="0.001")
        self.l1_reg_var = tk.StringVar(value="0.0")
        self.l2_reg_var = tk.StringVar(value="0.001")
        self.dropout_rate_var = tk.StringVar(value="0.1")
        
        # Status variables
        self.data_info_var = tk.StringVar(value="No data loaded")
        self.status_label = None
        
        # Other required variables
        self.optimizer_var = tk.StringVar(value="AMDS")
        self.lr_var = tk.StringVar(value="0.01")
        self.iterations_var = tk.StringVar(value="1000")
        self.plot_type_var = tk.StringVar(value="Stock Price")
        self.file_path_var = tk.StringVar()
        
        # State flags
        self.is_shutting_down = False
        self.is_plot_detached = False
        self.plot_window = None
        self.current_optimizer = None
        self.current_plot_type = None
        self.plot_needs_update = True
        
        # Initialize data structures
        self.data = None
        self.file_path = None
        self.n_samples = 0
        self.timestamps = np.array([], dtype=np.int64)
        self.prices = np.array([], dtype=np.float64)
        self.returns = np.array([], dtype=np.float64)
        self.volatility = np.array([], dtype=np.float64)
        self.predictions = np.array([], dtype=np.float64)
        self.uncertainties = np.array([], dtype=np.float64)
        self.buy_signals = np.array([], dtype=bool)
        self.sell_signals = np.array([], dtype=bool)
        
        # Initialize model state
        self.current_model = None
        self.model_state = {
            'weights': None,
            'bias': None,
            'scaler': None,
            'optimizer': None,
            'hyperparameters': {},
            'training_history': {}
        }
        
        # Add progress variables
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text = tk.StringVar(value="")
        self.training_status = tk.StringVar(value="Not started")
        self.progress_bar = None
        self.training_status_label = None
    
    def _init_ui_components(self):
        """Initialize basic UI components needed by parent"""
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Create paned windows
        self.h_paned = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.h_paned.grid(row=0, column=0, sticky="nsew")
        
        # Create main frames
        self.control_frame = ttk.Frame(self.h_paned, padding="5")
        self.content_paned = ttk.PanedWindow(self.h_paned, orient=tk.VERTICAL)
        
        # Add frames to paned window
        self.h_paned.add(self.control_frame, weight=1)
        self.h_paned.add(self.content_paned, weight=4)
        
        # Create status label
        self.status_label = ttk.Label(self.control_frame, textvariable=self.data_info_var, wraplength=200)
        self.status_label.grid(row=7, column=0, pady=10, padx=5)
        
        # Add progress frame
        self.progress_frame = ttk.LabelFrame(self.control_frame, text="Training Progress", padding="5")
        self.progress_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        
        # Add progress bar
        self.progress_bar = ttk.Progressbar(self.progress_frame, 
                                          variable=self.progress_var,
                                          maximum=100.0)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        
        # Add status labels
        self.training_status_label = ttk.Label(self.progress_frame, 
                                             textvariable=self.training_status,
                                             wraplength=200)
        self.training_status_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        
        ttk.Label(self.progress_frame, textvariable=self.progress_text,
                 wraplength=200).grid(row=2, column=0, sticky="w", padx=5, pady=2)
    
    def _setup_grid_components(self):
        """Setup grid-specific UI components"""
        # Create grid container
        self.grid_container = ttk.Frame(self.content_paned, padding="5")
        self.content_paned.add(self.grid_container, weight=1)
        
        # Setup grid view
        self.grid_view = DataGridView(self.grid_container)
        self.grid_view.pack(fill=tk.BOTH, expand=True)
        
        # Add grid controls
        self.grid_controls = ttk.Frame(self.grid_container)
        self.grid_controls.pack(fill=tk.X, pady=5)
        
        # Add refresh button
        ttk.Button(self.grid_controls, text="Refresh Grid",
                  command=self.refresh_grid).pack(side=tk.LEFT, padx=5)
        
        # Add row limit entry
        ttk.Label(self.grid_controls, text="Max Rows:").pack(side=tk.LEFT, padx=5)
        self.row_limit_entry = ttk.Entry(self.grid_controls,
                                       textvariable=self.row_limit_var,
                                       width=10)
        self.row_limit_entry.pack(side=tk.LEFT, padx=5)
    
    def refresh_grid(self):
        """Refresh the grid view with current data"""
        if self.data is not None:
            try:
                max_rows = int(self.row_limit_var.get())
                self.grid_view.update_data(self.data.head(max_rows))
            except ValueError:
                messagebox.showerror("Error", "Invalid row limit")
                self.row_limit_var.set("1000")
                self.grid_view.update_data(self.data.head(1000))
    
    def load_data(self):
        """Override load_data to update grid view"""
        try:
            super().load_data()
            if self.data is not None:
                self.refresh_grid()
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            if hasattr(self, 'status_label'):
                self.status_label.config(text=error_msg)
            print(error_msg)
            if hasattr(self, 'data') and self.data is not None:
                print("Debug info:")
                print(f"Data shape: {self.data.shape}")
                print(f"Columns: {self.data.columns.tolist()}")
                print("First few rows of timestamp column:")
                if 'timestamp' in self.data.columns:
                    print(self.data['timestamp'].head())
    
    def initialize_data(self):
        """Override initialize_data to handle errors"""
        try:
            super().initialize_data()
        except Exception as e:
            error_msg = f"Error in initialize_data: {str(e)}"
            if hasattr(self, 'status_label'):
                self.status_label.config(text=error_msg)
            print(error_msg)
            if hasattr(self, 'data') and self.data is not None:
                print("Debug info:")
                print(f"Data shape: {self.data.shape}")
                print(f"Columns: {self.data.columns.tolist()}")
                print("Sample of data:")
                print(self.data.head())
    
    def restore_application_state(self):
        """Override restore_application_state to handle missing attributes"""
        try:
            super().restore_application_state()
        except Exception as e:
            print(f"Warning: Could not restore application state: {str(e)}")
            # Continue execution even if state restoration fails

def main():
    root = tk.Tk()
    
    try:
        app = GridStockGrokGUI(root)  # Use the grid-enabled version
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print("\nReceived shutdown signal")
            app.shutdown()
        
        # Register signal handlers
        if sys.platform != 'win32':  # Not on Windows
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGHUP, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        root.mainloop()
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        if 'app' in locals():
            app.shutdown()
        else:
            root.destroy()
        sys.exit(1)

if __name__ == "__main__":
    main() 