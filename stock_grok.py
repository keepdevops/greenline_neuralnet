import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from tkinter import filedialog, ttk, messagebox
import os
import sys
import logging
from datetime import datetime
from sklearn.utils.validation import check_is_fitted
from scipy.special import expit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Import plotting functions
from plot_stock_data import (
    plot_stock_price, plot_returns_distribution, plot_volatility,
    plot_prediction_vs_actual, plot_residuals, plot_learning_curves,
    plot_correlation_matrix, plot_feature_importance, plot_uncertainty,
    plot_trading_signals, plot_cumulative_returns, print_info,
    print_error, print_warning
)

# Define print functions in case import fails
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

# Utility functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def huber_loss(y, y_hat, delta=1.0, weights=None):
    error = y - y_hat
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
    loss = np.where(is_small_error, squared_loss, linear_loss)
    if weights is not None:
        if weights.shape != loss.shape:
            raise ValueError(f"Shape mismatch in huber_loss: loss shape {loss.shape}, weights shape {weights.shape}")
        loss = loss * weights
    return np.sum(loss) / np.sum(weights if weights is not None else 1)

def compute_gradients(x, y, y_hat, z, delta=1.0, activation='linear', per_sample=False):
    error = y_hat - y
    is_small_error = np.abs(error) <= delta
    dL_dyhat = np.where(is_small_error, error, delta * np.sign(error)) / len(y)
    dyhat_dz = 1 if activation == 'linear' else y_hat * (1 - y_hat)
    grad = (dL_dyhat * dyhat_dz)[:, np.newaxis] * x  # shape (n_samples, n_features)
    grad_b = dL_dyhat * dyhat_dz  # shape (n_samples,)
    if per_sample:
        return grad, grad_b
    return grad.sum(axis=0), grad_b.sum()

def compute_volatility(returns, window=20):
    return np.std(returns[-window:]) if len(returns) >= window else 1.0

def compute_ma_slope(y, window=5):
    if len(y) < window:
        return 0.0
    ma = np.convolve(y, np.ones(window)/window, mode='valid')
    return (ma[-1] - ma[-2]) / window if len(ma) > 1 else 0.0

def logistic_map(x, r=3.9):
    return r * x * (1 - x)

def expected_improvement(mu, sigma, best_loss, xi=0.01):
    z = (best_loss - mu - xi) / (sigma + 1e-9)
    return (best_loss - mu) * norm.cdf(z) + sigma * norm.pdf(z)

# Single Neuron Class
class SingleNeuron:
    def __init__(self, input_dim, activation='linear'):
        self.w = np.random.randn(input_dim) * 0.01
        self.b = 0.0
        self.activation = activation
    
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z if self.activation == 'linear' else sigmoid(z)
    
    def load_keras_model(self, file_path):
        try:
            model = tf.keras.models.load_model(file_path)
            weights = model.get_weights()
            if len(weights) >= 2:
                self.w = weights[0].flatten()[:len(self.w)]
                self.b = weights[1][0] if weights[1].size > 0 else 0.0
            return True
        except:
            return False

# Optimization Algorithms
def amds(x, y, timestamps, activation='linear', iterations=1000, neuron=None):
    w, b = neuron.w.copy(), neuron.b
    v_w, v_b = np.zeros_like(w), 0.0
    s_w, s_b = np.zeros_like(w), 0.0
    G_w, G_b = np.zeros_like(w), 0.0
    eta_0, beta_1, beta_2 = 0.01, 0.9, 0.999
    epsilon, theta = 1e-8, 1e-4
    prev_loss = float('inf')
    
    for t in range(iterations):
        z = np.dot(x, w) + b
        y_hat = z if activation == 'linear' else sigmoid(z)
        loss = huber_loss(y, y_hat, delta=1.0)
        delta_loss = prev_loss - loss
        
        g_w, g_b = compute_gradients(x, y, y_hat, z, delta=1.0, activation=activation)
        v_w = beta_1 * v_w + (1 - beta_1) * g_w
        v_b = beta_1 * v_b + (1 - beta_1) * g_b
        s_w = beta_2 * s_w + (1 - beta_2) * g_w ** 2
        s_b = beta_2 * s_b + (1 - beta_2) * g_b ** 2
        G_w += g_w ** 2
        G_b += g_b ** 2
        
        eta_t = eta_0 / (np.sqrt(abs(loss) + epsilon) * (1 + t / iterations))
        alpha_t = 1 if delta_loss < theta else 0
        delta_w = alpha_t * (eta_t * v_w / np.sqrt(s_w + epsilon)) + \
                  (1 - alpha_t) * (eta_t * v_w / np.sqrt(G_w + epsilon))
        delta_b = alpha_t * (eta_t * v_b / np.sqrt(s_b + epsilon)) + \
                  (1 - alpha_t) * (eta_t * v_b / np.sqrt(G_b + epsilon))
        
        w -= delta_w
        b -= delta_b
        prev_loss = loss
    
    neuron.w, neuron.b = w, b
    return w, b, loss

def amds_plus(x, y, timestamps, activation='linear', iterations=1000, neuron=None):
    w, b = neuron.w.copy(), neuron.b
    v_w, v_b = np.zeros_like(w), 0.0
    s_w, s_b = np.zeros_like(w), 0.0
    G_w, G_b = np.zeros_like(w), 0.0
    var_x = np.var(x, axis=0)
    mean_x = np.mean(x, axis=0)
    eta_0, beta_1, beta_2 = 0.005, 0.9, 0.999
    epsilon, C, lambda_1, lambda_2 = 1e-8, 0.5, 0.005, 0.01
    gamma, rho, delta = 0.01, 0.9, 1.0
    theta, sigma = 1e-4, 0.001
    prev_loss, stall_count = float('inf'), 0
    prev_g_w, prev_g_b = np.zeros_like(w), 0.0

    for t in range(iterations):
        z = np.dot(x, w) + b
        y_hat = z if activation == 'linear' else sigmoid(z)
        loss = huber_loss(y, y_hat, delta)
        delta_loss = prev_loss - loss
        if loss > 1.5 * prev_loss:
            stall_count = 0
        if delta_loss < 1e-6:
            stall_count += 1
            if stall_count >= 10:
                break
        else:
            stall_count = 0

        time_weight = np.exp(-gamma * (timestamps[-1] - timestamps))
        g_w_samples, g_b_samples = compute_gradients(x, y, y_hat, z, delta, activation, per_sample=True)
        g_w = (g_w_samples * time_weight[:, np.newaxis]).sum(axis=0)
        g_b = (g_b_samples * time_weight).sum()

        mean_x = rho * mean_x + (1 - rho) * np.mean(x, axis=0)
        var_x = rho * var_x + (1 - rho) * np.mean((x - mean_x) ** 2, axis=0)
        g_w = g_w / np.sqrt(var_x + epsilon)
        g_w = np.clip(g_w, -C, C)
        g_b = min(max(g_b, -C), C)

        v_w = beta_1 * v_w + (1 - beta_1) * g_w
        v_b = beta_1 * v_b + (1 - beta_1) * g_b
        v_w_hat = v_w / (1 - beta_1 ** (t + 1))
        v_b_hat = v_b / (1 - beta_1 ** (t + 1))
        s_w = beta_2 * s_w + (1 - beta_2) * g_w ** 2
        s_b = beta_2 * s_b + (1 - beta_2) * g_b ** 2
        s_w_hat = s_w / (1 - beta_2 ** (t + 1))
        s_b_hat = s_b / (1 - beta_2 ** (t + 1))
        G_w += g_w ** 2
        G_b += g_b ** 2

        sigma_ret = compute_volatility(np.diff(y) / y[:-1] if len(y) > 1 else [1.0])
        kappa_w = np.abs(g_w - prev_g_w)
        kappa_b = abs(g_b - prev_g_b)
        eta_t = eta_0 / (np.sqrt(abs(loss) + epsilon) * (1 + kappa_w + epsilon) * (1 + sigma_ret + epsilon))
        eta_t_b = eta_0 / (np.sqrt(abs(loss) + epsilon) * (1 + kappa_b + epsilon) * (1 + sigma_ret + epsilon))
        arg = np.clip(-(delta_loss - theta) / sigma, -50, 50)
        alpha_t = 1 / (1 + np.exp(arg))

        delta_w = alpha_t * (eta_t * v_w_hat / np.sqrt(s_w_hat + epsilon)) + \
                  (1 - alpha_t) * (eta_t * v_w_hat / np.sqrt(G_w + epsilon)) + \
                  lambda_1 * np.sign(w) + lambda_2 * w
        delta_b = alpha_t * (eta_t_b * v_b_hat / np.sqrt(s_b_hat + epsilon)) + \
                  (1 - alpha_t) * (eta_t_b * v_b_hat / np.sqrt(G_b + epsilon))

        w -= delta_w
        b -= delta_b
        prev_g_w, prev_g_b = g_w, g_b
        prev_loss = loss

    neuron.w, neuron.b = w, b
    return w, b, loss

def cipo(x, y, timestamps, activation='linear', iterations=1000, neuron=None):
    # print("DEBUG: timestamps shape in cipo:", timestamps.shape)
    w, b = neuron.w.copy(), neuron.b
    N, alpha_0, beta, tau = 5, 0.01, 0.1, 0.1
    gamma, delta, theta = 0.01, 1.0, 0.001
    prev_loss, stall_count = float('inf'), 0
    returns = np.diff(y) / y[:-1] if len(y) > 1 else [1.0]
    chaos_states = np.random.rand(N, x.shape[1] + 1)

    for t in range(iterations):
        sigma_vol = compute_volatility(returns)
        ma_slope = compute_ma_slope(y)
        time_weights = np.exp(-gamma * (timestamps[-1] - timestamps))
        # print("DEBUG: time_weights shape in cipo:", time_weights.shape, "y shape:", y.shape)
        if time_weights.shape != y.shape:
            raise ValueError(f"[cipo] time_weights shape {time_weights.shape} does not match y shape {y.shape}")

        candidates = []
        rewards = []
        for i in range(N):
            chaos_states[i] = logistic_map(chaos_states[i])
            delta_w = alpha_0 * (2 * chaos_states[i, :-1] - 1) * sigma_vol
            delta_b = alpha_0 * (2 * chaos_states[i, -1] - 1) * sigma_vol
            w_cand = w + delta_w
            b_cand = b + delta_b
            z = np.dot(x, w_cand) + b_cand
            y_hat = z if activation == 'linear' else sigmoid(z)
            loss = huber_loss(y, y_hat, delta, time_weights)
            trend_score = np.sign(y_hat[-1] - y[-2]) * ma_slope if len(y) > 1 else 0.0
            reward = -loss + beta * trend_score
            candidates.append((w_cand, b_cand))
            rewards.append(reward)
        
        rewards = np.array(rewards)
        exp_rewards = np.exp(rewards / tau)
        if not np.all(np.isfinite(exp_rewards)):
            exp_rewards = np.nan_to_num(exp_rewards, nan=0.0, posinf=0.0, neginf=0.0)
        sum_exp = np.sum(exp_rewards)
        if sum_exp == 0 or np.isnan(sum_exp):
            probs = np.ones_like(exp_rewards) / len(exp_rewards)
        else:
            probs = exp_rewards / sum_exp
        idx = np.random.choice(N, p=probs)
        w, b = candidates[idx]
        
        z = np.dot(x, w) + b
        y_hat = z if activation == 'linear' else sigmoid(z)
        loss = huber_loss(y, y_hat, delta, time_weights)
        delta_loss = prev_loss - loss
        alpha_t = alpha_0 * (1 + abs(delta_loss) / theta + sigma_vol)
        chaos_states = np.clip(chaos_states + alpha_t * (np.random.rand(N, x.shape[1] + 1) - 0.5), 0, 1)
        
        if loss > 1.5 * prev_loss:
            stall_count = 0
        if delta_loss < 1e-6:
            stall_count += 1
            if stall_count >= 10:
                break
        else:
            stall_count = 0
        
        prev_loss = loss
        returns = np.append(returns, (y[-1] - y[-2]) / y[-2] if t > 0 and y[-2] != 0 else returns[-1])
    
    neuron.w, neuron.b = w, b
    return w, b, loss

def bcipo(x, y, timestamps, activation='linear', iterations=1000, neuron=None):
    # print("DEBUG: timestamps shape in bcipo:", timestamps.shape)
    w, b = neuron.w.copy(), neuron.b
    N, alpha_0, beta_1, beta_2 = 5, 0.01, 0.1, 0.5
    tau, gamma, delta = 0.1, 0.01, 1.0
    theta = 0.001
    prev_loss, stall_count = float('inf'), 0
    returns = np.diff(y) / y[:-1] if len(y) > 1 else [1.0]
    chaos_states = np.random.rand(N, x.shape[1] + 1)
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), random_state=0)
    X_history = []
    y_history = []

    for t in range(iterations):
        sigma_vol = compute_volatility(returns)
        ma_slope = compute_ma_slope(y)
        time_weights = np.exp(-gamma * (timestamps[-1] - timestamps))
        # print("DEBUG: time_weights shape in bcipo:", time_weights.shape, "y shape:", y.shape)
        if time_weights.shape != y.shape:
            raise ValueError(f"[bcipo] time_weights shape {time_weights.shape} does not match y shape {y.shape}")

        candidates = []
        mu_losses = []
        sigma_losses = []
        rewards = []
        for i in range(N):
            chaos_states[i] = logistic_map(chaos_states[i])
            delta_w = alpha_0 * (2 * chaos_states[i, :-1] - 1) * sigma_vol
            delta_b = alpha_0 * (2 * chaos_states[i, -1] - 1) * sigma_vol
            w_cand = w + delta_w
            b_cand = b + delta_b
            params = np.append(w_cand, b_cand)
            if len(X_history) > 0:
                mu, sigma = gp.predict([params], return_std=True)
            else:
                mu, sigma = [0.0], [1.0]
            z = np.dot(x, w_cand) + b_cand
            y_hat = z if activation == 'linear' else sigmoid(z)
            loss = huber_loss(y, y_hat, delta, time_weights)
            trend_score = np.sign(y_hat[-1] - y[-2]) * ma_slope if len(y) > 1 else 0.0
            best_loss = min(y_history) if y_history else loss
            mu = np.atleast_1d(mu)
            sigma = np.atleast_1d(sigma)
            ei = expected_improvement(mu[0], sigma[0], best_loss)
            reward = -mu[0] + beta_1 * trend_score + beta_2 * ei
            candidates.append((w_cand, b_cand, loss))
            mu_losses.append(mu[0])
            sigma_losses.append(sigma[0])
            rewards.append(reward)
        
        rewards = np.array(rewards)
        exp_rewards = np.exp(rewards / tau)
        if not np.all(np.isfinite(exp_rewards)):
            exp_rewards = np.nan_to_num(exp_rewards, nan=0.0, posinf=0.0, neginf=0.0)
        sum_exp = np.sum(exp_rewards)
        if sum_exp == 0 or np.isnan(sum_exp):
            probs = np.ones_like(exp_rewards) / len(exp_rewards)
        else:
            probs = exp_rewards / sum_exp
        idx = np.random.choice(N, p=probs)
        w, b, actual_loss = candidates[idx]
        
        X_history.append(np.append(w, b))
        y_history.append(actual_loss)
        if len(X_history) > 1:
            gp.fit(np.array(X_history), np.array(y_history))
        
        alpha_t = alpha_0 * (1 + np.mean(sigma_losses) + sigma_vol)
        chaos_states = np.clip(chaos_states + alpha_t * (np.random.rand(N, x.shape[1] + 1) - 0.5), 0, 1)
        
        delta_loss = prev_loss - actual_loss
        if actual_loss > 1.5 * prev_loss:
            stall_count = 0
        if delta_loss < 1e-6:
            stall_count += 1
            if stall_count >= 10:
                break
        else:
            stall_count = 0
        
        prev_loss = actual_loss
        returns = np.append(returns, (y[-1] - y[-2]) / y[-2] if t > 0 and y[-2] != 0 else returns[-1])
    
    neuron.w, neuron.b = w, b
    return w, b, actual_loss

def bcipo_dropout(x, y, timestamps, activation='linear', iterations=1000, neuron=None):
    # print("DEBUG: timestamps shape in bcipo_dropout:", timestamps.shape)
    w, b = neuron.w.copy(), neuron.b
    N, alpha_0, beta_1, beta_2 = 5, 0.01, 0.1, 0.5
    tau, gamma, delta = 0.1, 0.01, 1.0
    theta, K = 0.001, 10
    prev_loss, stall_count = float('inf'), 0
    returns = np.diff(y) / y[:-1] if len(y) > 1 else [1.0]
    chaos_states = np.random.rand(N, x.shape[1] + 1)
    nn = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', 
                      solver='adam', learning_rate_init=0.01, 
                      max_iter=100, random_state=0)
    X_history = []
    y_history = []

    for t in range(iterations):
        sigma_vol = compute_volatility(returns)
        ma_slope = compute_ma_slope(y)
        time_weights = np.exp(-gamma * (timestamps[-1] - timestamps))
        # print("DEBUG: time_weights shape in bcipo_dropout:", time_weights.shape, "y shape:", y.shape)
        if time_weights.shape != y.shape:
            raise ValueError(f"[bcipo_dropout] time_weights shape {time_weights.shape} does not match y shape {y.shape}")

        candidates = []
        mu_losses = []
        sigma_losses = []
        rewards = []
        for i in range(N):
            chaos_states[i] = logistic_map(chaos_states[i])
            delta_w = alpha_0 * (2 * chaos_states[i, :-1] - 1) * sigma_vol
            delta_b = alpha_0 * (2 * chaos_states[i, -1] - 1) * sigma_vol
            w_cand = w + delta_w
            b_cand = b + delta_b
            params = np.append(w_cand, b_cand).reshape(1, -1)
            if len(X_history) > 0:
                try:
                    check_is_fitted(nn)
                    predictions = [nn.predict(params)[0] for _ in range(K)]
                    mu = np.mean(predictions)
                    sigma = np.std(predictions)
                except Exception:
                    mu = 0.0
                    sigma = 1.0
            else:
                mu, sigma = 0.0, 1.0
            z = np.dot(x, w_cand) + b_cand
            y_hat = z if activation == 'linear' else sigmoid(z)
            loss = huber_loss(y, y_hat, delta, time_weights)
            trend_score = np.sign(y_hat[-1] - y[-2]) * ma_slope if len(y) > 1 else 0.0
            best_loss = min(y_history) if y_history else loss
            mu = np.atleast_1d(mu)
            sigma = np.atleast_1d(sigma)
            if mu.shape != (params.shape[0],):
                mu = np.full(params.shape[0], mu.item() if mu.size == 1 else 0.0)
            if sigma.shape != (params.shape[0],):
                sigma = np.full(params.shape[0], sigma.item() if sigma.size == 1 else 1.0)
            ei = expected_improvement(mu[0], sigma[0], best_loss)
            reward = -mu[0] + beta_1 * trend_score + beta_2 * ei
            candidates.append((w_cand, b_cand, loss))
            mu_losses.append(mu[0])
            sigma_losses.append(sigma[0])
            rewards.append(reward)
        
        rewards = np.array(rewards)
        exp_rewards = np.exp(rewards / tau)
        if not np.all(np.isfinite(exp_rewards)):
            exp_rewards = np.nan_to_num(exp_rewards, nan=0.0, posinf=0.0, neginf=0.0)
        sum_exp = np.sum(exp_rewards)
        if sum_exp == 0 or np.isnan(sum_exp):
            probs = np.ones_like(exp_rewards) / len(exp_rewards)
        else:
            probs = exp_rewards / sum_exp
        idx = np.random.choice(N, p=probs)
        w, b, actual_loss = candidates[idx]
        
        X_history.append(np.append(w, b))
        y_history.append(actual_loss)
        if len(X_history) > 1:
            nn.fit(np.array(X_history), np.array(y_history))
        
        alpha_t = alpha_0 * (1 + np.mean(sigma_losses) + sigma_vol)
        chaos_states = np.clip(chaos_states + alpha_t * (np.random.rand(N, x.shape[1] + 1) - 0.5), 0, 1)
        
        delta_loss = prev_loss - actual_loss
        if actual_loss > 1.5 * prev_loss:
            stall_count = 0
        if delta_loss < 1e-6:
            stall_count += 1
            if stall_count >= 10:
                break
        else:
            stall_count = 0
        
        prev_loss = actual_loss
        returns = np.append(returns, (y[-1] - y[-2]) / y[-2] if t > 0 and y[-2] != 0 else returns[-1])
    
    neuron.w, neuron.b = w, b
    return w, b, actual_loss

def bcipo_hesm(x, y, timestamps, activation='linear', iterations=1000, neuron=None):
    # print("DEBUG: timestamps shape in bcipo_hesm:", timestamps.shape)
    w, b = neuron.w.copy(), neuron.b
    N, alpha_0, beta_1, beta_2 = 5, 0.01, 0.1, 0.5
    tau, gamma, delta = 0.1, 0.01, 1.0
    theta = 0.001
    prev_loss, stall_count = float('inf'), 0
    returns = np.diff(y) / y[:-1] if len(y) > 1 else [1.0]
    chaos_states = np.random.rand(N, x.shape[1] + 1)
    hesm = HESM()
    X_history = []
    y_history = []
    time_weights = []

    for t in range(iterations):
        sigma_vol = compute_volatility(returns)
        ma_slope = compute_ma_slope(y)
        time_weights_t = np.exp(-gamma * (timestamps[-1] - timestamps))
        # print("DEBUG: time_weights_t shape in bcipo_hesm:", time_weights_t.shape, "y shape:", y.shape)
        if time_weights_t.shape != y.shape:
            raise ValueError(f"[bcipo_hesm] time_weights_t shape {time_weights_t.shape} does not match y shape {y.shape}")

        candidates = []
        mu_losses = []
        sigma_losses = []
        rewards = []
        for i in range(N):
            chaos_states[i] = logistic_map(chaos_states[i])
            delta_w = alpha_0 * (2 * chaos_states[i, :-1] - 1) * sigma_vol
            delta_b = alpha_0 * (2 * chaos_states[i, -1] - 1) * sigma_vol
            w_cand = w + delta_w
            b_cand = b + delta_b
            params = np.append(w_cand, b_cand).reshape(1, -1)
            if len(X_history) > 0:
                mu, sigma = hesm.predict(params)
            else:
                mu, sigma = np.array([0.0]), np.array([1.0])
            mu = np.atleast_1d(mu)
            sigma = np.atleast_1d(sigma)
            if mu.shape != (params.shape[0],):
                mu = np.full(params.shape[0], mu.item() if mu.size == 1 else 0.0)
            if sigma.shape != (params.shape[0],):
                sigma = np.full(params.shape[0], sigma.item() if sigma.size == 1 else 1.0)
            z = np.dot(x, w_cand) + b_cand
            y_hat = z if activation == 'linear' else sigmoid(z)
            loss = huber_loss(y, y_hat, delta, time_weights_t)
            trend_score = np.sign(y_hat[-1] - y[-2]) * ma_slope if len(y) > 1 else 0.0
            best_loss = min(y_history) if y_history else loss
            ei = expected_improvement(mu[0], sigma[0], best_loss)
            reward = -mu[0] + beta_1 * trend_score + beta_2 * ei
            candidates.append((w_cand, b_cand, loss))
            mu_losses.append(mu[0])
            sigma_losses.append(sigma[0])
            rewards.append(reward)
        
        rewards = np.array(rewards)
        exp_rewards = np.exp(rewards / tau)
        if not np.all(np.isfinite(exp_rewards)):
            exp_rewards = np.nan_to_num(exp_rewards, nan=0.0, posinf=0.0, neginf=0.0)
        sum_exp = np.sum(exp_rewards)
        if sum_exp == 0 or np.isnan(sum_exp):
            probs = np.ones_like(exp_rewards) / len(exp_rewards)
        else:
            probs = exp_rewards / sum_exp
        idx = np.random.choice(N, p=probs)
        w, b, actual_loss = candidates[idx]
        
        X_history.append(np.append(w, b))
        y_history.append(actual_loss)
        time_weights.append(time_weights_t)
        if len(X_history) > 1:
            hesm.fit(np.array(X_history), np.array(y_history), np.array(time_weights[-1]))
            hesm.update_errors(np.array([X_history[-1]]), actual_loss)
        
        alpha_t = alpha_0 * (1 + np.mean(sigma_losses) + sigma_vol)
        chaos_states = np.clip(chaos_states + alpha_t * (np.random.rand(N, x.shape[1] + 1) - 0.5), 0, 1)
        
        delta_loss = prev_loss - actual_loss
        if actual_loss > 1.5 * prev_loss:
            stall_count = 0
        if delta_loss < 1e-6:
            stall_count += 1
            if stall_count >= 10:
                break
        else:
            stall_count = 0
        
        prev_loss = actual_loss
        returns = np.append(returns, (y[-1] - y[-2]) / y[-2] if t > 0 and y[-2] != 0 else returns[-1])
    
    neuron.w, neuron.b = w, b
    return w, b, actual_loss

class HESM:
    def __init__(self):
        self.gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0),
            random_state=0,
            optimizer="fmin_l_bfgs_b"
        )
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
        self.dropout_nn = MLPRegressor(
            hidden_layer_sizes=(10, 10),
            activation='relu',
            solver='adam',
            learning_rate_init=0.01,
            max_iter=500,
            random_state=0
        )
        self.poly = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('reg', LinearRegression())
        ])
        self.models = [self.gp, self.rf, self.dropout_nn, self.poly]
        self.errors = [1.0] * 4

    def fit(self, X, y, weights=None):
        if weights is not None:
            sample_weight = weights / np.sum(weights)
            for model in self.models:
                try:
                    model.fit(X, y, **({'sample_weight': sample_weight} if hasattr(model, 'fit') else {}))
                except:
                    model.fit(X, y)
        else:
            for model in self.models:
                model.fit(X, y)

    def predict(self, X):
        mu_preds, sigma_preds = [], []
        n_samples = X.shape[0]
        for i, model in enumerate(self.models):
            if model == self.dropout_nn:
                try:
                    check_is_fitted(model)
                    predictions = [model.predict(X) for _ in range(10)]
                    mu = np.mean(predictions, axis=0), np.std(predictions, axis=0)
                except Exception:
                    mu = np.zeros(n_samples)
                    sigma = np.ones(n_samples)
            elif model == self.poly:
                reg = model.named_steps['reg']
                if hasattr(reg, "coef_"):
                    mu = model.predict(X)
                    try:
                        bootstrap_preds = []
                        for _ in range(5):
                            idx = np.random.choice(len(self.X_history), len(self.X_history), replace=True)
                            model.fit(self.X_history[idx], self.y_history[idx])
                            bootstrap_preds.append(model.predict(X))
                        sigma = np.std(bootstrap_preds, axis=0)
                    except AttributeError:
                        sigma = np.ones_like(mu)
                else:
                    mu = np.zeros(n_samples)
                    sigma = np.ones(n_samples)
            elif isinstance(model, RandomForestRegressor):
                if hasattr(model, "estimators_") and len(model.estimators_) > 0:
                    all_preds = np.array([tree.predict(X) for tree in model.estimators_])
                    mu = np.mean(all_preds, axis=0)
                    sigma = np.std(all_preds, axis=0)
                else:
                    mu = np.zeros(n_samples)
                    sigma = np.ones(n_samples)
            else:
                mu, sigma = model.predict(X, return_std=True)
            mu = np.atleast_1d(mu)
            sigma = np.atleast_1d(sigma)
            if mu.shape != (n_samples,):
                mu = np.full(n_samples, mu.item() if mu.size == 1 else 0.0)
            if sigma.shape != (n_samples,):
                sigma = np.full(n_samples, sigma.item() if sigma.size == 1 else 1.0)
            mu_preds.append(mu)
            sigma_preds.append(sigma)

        weights = [1 / (s + 1e-8) / (e + 1e-8) for s, e in zip(sigma_preds, self.errors)]
        weights = np.array(weights) / np.sum(weights)
        mu = np.sum([w * m for w, m in zip(weights, mu_preds)], axis=0)
        sigma = np.sqrt(np.sum([w * (s**2 + (m - mu)**2) for w, m, s in zip(weights, mu_preds, sigma_preds)], axis=0))
        return mu, sigma

    def update_errors(self, X, true_loss):
        predictions = []
        for model in self.models:
            if model == self.dropout_nn:
                predictions.append(np.mean([model.predict(X) for _ in range(10)], axis=0))
            else:
                predictions.append(model.predict(X))
        self.errors = [np.mean(np.abs(p - true_loss)) for p in predictions]

# Tkinter GUI
class NeuralNetGUI:
    def __init__(self, root):
        print_info("Initializing Neural Network GUI")
        self.root = root
        self.root.title("Single Neuron Neural Network for Stock Prediction")
        self.neuron = None
        self.scaler = StandardScaler()
        self.optimizers = {
            'AMDS': amds,
            'AMDS+': amds_plus,
            'CIPO': cipo,
            'BCIPO': bcipo,
            'BCIPO-Dropout': bcipo_dropout,
            'BCIPO-HESM': bcipo_hesm
        }
        
        # Grid layout
        self.create_widgets()
    
    def create_widgets(self):
        # Labels
        tk.Label(self.root, text="Keras Model (.h5):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        tk.Label(self.root, text="Stock Data (CSV):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        tk.Label(self.root, text="Optimizer:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        tk.Label(self.root, text="Results:").grid(row=5, column=0, padx=5, pady=5, sticky='ne')
        
        # Entries and Buttons
        self.model_path = tk.StringVar()
        self.model_entry = tk.Entry(self.root, textvariable=self.model_path, width=30)
        self.model_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Browse", command=self.load_model).grid(row=0, column=2, padx=5, pady=5)
        
        self.data_path = tk.StringVar()
        self.data_entry = tk.Entry(self.root, textvariable=self.data_path, width=30)
        self.data_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Browse", command=self.load_data).grid(row=1, column=2, padx=5, pady=5)
        
        self.optimizer_var = tk.StringVar(value='BCIPO-HESM')
        self.optimizer_menu = ttk.Combobox(self.root, textvariable=self.optimizer_var, 
                                          values=list(self.optimizers.keys()), state='readonly')
        self.optimizer_menu.grid(row=2, column=1, padx=5, pady=5, columnspan=2, sticky='ew')
        
        tk.Button(self.root, text="Train", command=self.train).grid(row=3, column=1, padx=5, pady=5)
        
        self.result_text = tk.Text(self.root, height=5, width=40)
        self.result_text.grid(row=5, column=1, padx=5, pady=5, columnspan=2)
    
    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if file_path:
            self.model_path.set(file_path)
            if self.neuron and self.neuron.load_keras_model(file_path):
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Loaded Keras model: {os.path.basename(file_path)}\n")
            else:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Failed to load Keras model or neuron not initialized.\n")
    
    def load_data(self):
        """Load and process the CSV file"""
        try:
            # Load the CSV file
            self.data = pd.read_csv(self.data_path.get())
            
            # Handle volume/vol column
            if 'volume' in self.data.columns:
                self.data['vol'] = self.data['volume']
                self.data.drop('volume', axis=1, inplace=True)
            
            # Basic data validation
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'vol']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                # Special handling for volume/vol
                if 'vol' in missing_columns and 'volume' in self.data.columns:
                    missing_columns.remove('vol')
                
                if missing_columns:  # If there are still missing columns
                    messagebox.showwarning("Warning", 
                        f"Missing columns: {', '.join(missing_columns)}\n"
                        "Some features may not work properly.")
            
            # Process the data
            self.initialize_data()
            
            # Update info
            self.data_info_var.set(
                f"Rows: {len(self.data)}\n"
                f"Columns: {', '.join(self.data.columns)}\n"
                f"Date range: {self.data['timestamp'].iloc[0]} to {self.data['timestamp'].iloc[-1]}"
            )
            
            # Update status
            self.status_var.set("Data loaded successfully")
            
            # Initial plot
            self.update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
    
    def initialize_data(self):
        """Initialize data and model results"""
        if self.data is None:
            return
        
        try:
            self.n_samples = len(self.data)
            
            # Initialize basic data
            self.timestamps = np.arange(self.n_samples)
            self.prices = self.data['close'].values
            self.returns = np.diff(self.prices) / self.prices[:-1]
            
            # Handle volume/vol column
            volume_data = self.data['vol'] if 'vol' in self.data.columns else \
                         self.data['volume'] if 'volume' in self.data.columns else \
                         np.zeros(self.n_samples)
            
            # Calculate volatility
            window = 20
            self.volatility = np.array([np.std(self.returns[max(0, i-window):i]) 
                                      for i in range(window, len(self.returns))])
            
            # Initialize predictions and other metrics
            self.predictions = self.prices.copy()
            self.uncertainties = np.ones_like(self.prices) * np.std(self.returns)
            self.buy_signals = np.zeros_like(self.prices, dtype=bool)
            self.sell_signals = np.zeros_like(self.prices, dtype=bool)
            
            # Store features
            self.features = pd.DataFrame({
                'open': self.data['open'],
                'high': self.data['high'],
                'low': self.data['low'],
                'vol': volume_data,
                'price': self.prices,
                'returns': np.pad(self.returns, (1, 0)),
                'volatility': np.pad(self.volatility, (window-1, 0))
            })
            
            self.status_var.set("Data initialized successfully")
        except Exception as e:
            self.status_var.set(f"Error initializing data: {str(e)}")

    def train(self):
        data_path = self.data_path.get()
        optimizer_name = self.optimizer_var.get()
        
        if not data_path or not os.path.exists(data_path):
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please select a valid CSV file.\n")
            return
        
        try:
            df = pd.read_csv(data_path)
            required_columns = ['timestamp', 'close', 'open', 'high', 'low', 'vol']
            if not all(col in df.columns for col in required_columns):
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "CSV must contain 'timestamp', 'close', 'open', 'high', 'low', 'vol' columns.\n")
                return
            
            # Clean timestamp: keep only the part before the space
            df['timestamp'] = df['timestamp'].astype(str).str.split().str[0]
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'close', 'open', 'high', 'low', 'vol'])
            
            print("Rows remaining after cleaning:", len(df))
            if len(df) == 0:
                print("First 5 rows before cleaning:\n", pd.read_csv(data_path).head())
                print("First 5 rows after cleaning:\n", df.head())
                print("NaN counts per column:\n", df.isna().sum())
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "No data left after cleaning. Please check your CSV for missing or invalid values.\n")
                return
            
            timestamps = df['timestamp'].values
            # print("DEBUG: timestamps shape in train:", timestamps.shape)
            y = df['close'].values
            feature_columns = ['open', 'high', 'low', 'vol']
            x = df[feature_columns].values
            
            # Normalize features
            x = self.scaler.fit_transform(x)
            
            # Initialize neuron if not already done or input dimension changed
            if self.neuron is None or len(self.neuron.w) != x.shape[1]:
                self.neuron = SingleNeuron(input_dim=x.shape[1], activation='linear')
                # Reload Keras model if path exists
                model_path = self.model_path.get()
                if model_path and os.path.exists(model_path):
                    self.neuron.load_keras_model(model_path)
            
            optimizer = self.optimizers[optimizer_name]
            w, b, loss = optimizer(x, y, timestamps, activation='linear', iterations=1000, neuron=self.neuron)
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Optimizer: {optimizer_name}\n")
            self.result_text.insert(tk.END, f"Weights: {w}\n")
            self.result_text.insert(tk.END, f"Bias: {b}\n")
            self.result_text.insert(tk.END, f"Final Loss: {loss:.6f}\n")
        
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}\n")

    def plot_optimizer_results(self):
        """Plot comparison of optimizer results"""
        try:
            optimizer = self.optimizer_var.get()
            plot_type = self.plot_type_var.get()
            
            self.fig.clear()
            
            if plot_type == "Optimizer Comparison Detail":
                gs = self.fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
                
                # Loss curve
                ax1 = self.fig.add_subplot(gs[0, 0])
                ax1.plot(self.loss_history, label=optimizer)
                ax1.set_title('Loss Curve')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Loss')
                ax1.grid(True)
                ax1.legend()
                
                # Learning rate adaptation
                ax2 = self.fig.add_subplot(gs[0, 1])
                ax2.plot(self.learning_rates, label='Learning Rate')
                ax2.set_title('Learning Rate Adaptation')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Learning Rate')
                ax2.grid(True)
                
                # Prediction with uncertainty
                ax3 = self.fig.add_subplot(gs[1, 0])
                ax3.plot(self.timestamps, self.predictions, label='Prediction')
                ax3.fill_between(self.timestamps, 
                               self.predictions - 2*self.uncertainties,
                               self.predictions + 2*self.uncertainties,
                               alpha=0.2, label='Uncertainty')
                ax3.plot(self.timestamps, self.prices, 'r--', label='Actual')
                ax3.set_title('Predictions with Uncertainty')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Price')
                ax3.grid(True)
                ax3.legend()
                
                # Feature importance
                ax4 = self.fig.add_subplot(gs[1, 1])
                features = ['open', 'high', 'low', 'vol']
                importances = self.calculate_feature_importance()
                ax4.bar(features, importances)
                ax4.set_title('Feature Importance')
                ax4.set_xticklabels(features, rotation=45)
                ax4.grid(True)
                
                # Cumulative returns
                ax5 = self.fig.add_subplot(gs[2, 0])
                returns = np.diff(self.predictions) / self.predictions[:-1]
                cumulative_returns = np.cumprod(1 + returns) - 1
                ax5.plot(cumulative_returns * 100, label=optimizer)
                ax5.set_title('Cumulative Returns (%)')
                ax5.set_xlabel('Time')
                ax5.set_ylabel('Return (%)')
                ax5.grid(True)
                ax5.legend()
                
                # Error distribution
                ax6 = self.fig.add_subplot(gs[2, 1])
                errors = self.predictions - self.prices
                ax6.hist(errors, bins=50, density=True, alpha=0.7)
                ax6.set_title('Error Distribution')
                ax6.set_xlabel('Prediction Error')
                ax6.set_ylabel('Density')
                ax6.grid(True)
            
            self.fig.suptitle(f'{optimizer} Optimization Results', y=1.02)
            self.canvas.draw()
            self.status_var.set(f"Updated optimizer results plot: {plot_type}")
        
        except Exception as e:
            self.status_var.set(f"Error plotting optimizer results: {str(e)}")

    def calculate_feature_importance(self):
        """Calculate feature importance scores"""
        try:
            features = ['open', 'high', 'low', 'vol']
            importances = []
            
            for feature in features:
                if feature == 'vol' and feature not in self.data.columns and 'volume' in self.data.columns:
                    feature_data = self.data['volume']
                else:
                    feature_data = self.data.get(feature, np.zeros(len(self.prices)))
                
                corr = np.abs(np.corrcoef(feature_data, self.prices)[0, 1])
                importances.append(corr)
            
            return np.array(importances)
        except Exception as e:
            self.status_var.set(f"Error calculating feature importance: {str(e)}")
            return np.zeros(4)

    def update_plot(self):
        """Update the plot based on selected type"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
                
        try:
            plot_type = self.plot_type_var.get()
            
            if plot_type == "Correlation Matrix":
                # Handle volume/vol column
                volume_data = self.data['vol'] if 'vol' in self.data.columns else \
                             self.data['volume'] if 'volume' in self.data.columns else \
                             np.zeros(self.n_samples)
                
                features = pd.DataFrame({
                    'price': self.prices,
                    'returns': np.pad(self.returns, (1, 0)),
                    'volatility': np.pad(self.volatility, (19, 0)),
                    'vol': volume_data
                })
                plot_correlation_matrix(features)
            
            # ... (rest of the plotting code) ...
            
        except Exception as e:
            self.status_var.set(f"Plot error: {str(e)}")

class StockPredictor:
    def __init__(self, data_path='data.csv'):
        self.data = pd.read_csv(data_path)
        self.prepare_data()
        self.initialize_models()
        self.history = {
            'loss': [],
            'predictions': [],
            'uncertainties': [],
            'returns': [],
            'volatilities': [],
            'trading_signals': {'buy': [], 'sell': []}
        }

    def prepare_data(self):
        """Prepare the stock data for training and prediction"""
        # Convert timestamp to datetime if needed
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Calculate returns
        self.data['returns'] = self.data['close'].pct_change()
        
        # Calculate volatility
        self.data['volatility'] = self.data['returns'].rolling(window=20).std()
        
        # Prepare features (customize based on your needs)
        self.features = self.data[[
            'open', 'high', 'low', 'vol'
        ]].values
        
        self.timestamps = np.arange(len(self.data))
        self.prices = self.data['close'].values
        
        # Scale features
        self.feature_scaler = StandardScaler()
        self.features = self.feature_scaler.fit_transform(self.features)

    def initialize_models(self):
        """Initialize all optimization models"""
        self.models = {
            'amds': None,
            'amds_plus': None,
            'cipo': None,
            'bcipo': None,
            'bcipo_dropout': None,
            'bcipo_hesm': None
        }

    def train(self, model_name, **kwargs):
        """Train a specific model and update history"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Get the appropriate optimization function
        if model_name == 'amds':
            optimizer = amds
        elif model_name == 'amds_plus':
            optimizer = amds_plus
        elif model_name == 'cipo':
            optimizer = cipo
        elif model_name == 'bcipo':
            optimizer = bcipo
        elif model_name == 'bcipo_dropout':
            optimizer = bcipo_dropout
        elif model_name == 'bcipo_hesm':
            optimizer = bcipo_hesm
        
        # Train the model
        w, b, loss = optimizer(
            self.features,
            self.prices,
            self.timestamps,
            **kwargs
        )
        
        # Store the model
        self.models[model_name] = (w, b)
        
        # Update history
        self.history['loss'].append(loss)
        
        # Generate predictions
        predictions = np.dot(self.features, w) + b
        self.history['predictions'].append(predictions)
        
        # Calculate uncertainties (example method)
        uncertainties = np.abs(predictions - self.prices) * 0.1
        self.history['uncertainties'].append(uncertainties)
        
        # Generate trading signals (example method)
        buy_signals = predictions > self.prices
        sell_signals = predictions < self.prices
        self.history['trading_signals']['buy'].append(buy_signals)
        self.history['trading_signals']['sell'].append(sell_signals)
        
        # Create all plots
        self.create_plots(model_name)

    def create_plots(self, model_name):
        """Create all plots for the trained model"""
        # Basic price plot with predictions
        plot_stock_price(self.timestamps, self.prices, 
                        title=f"{model_name} - Stock Price and Predictions")
        
        # Returns distribution
        plot_returns_distribution(self.data['returns'].dropna(), 
                                title=f"{model_name} - Returns Distribution")
        
        # Volatility
        plot_volatility(self.timestamps[19:], 
                       self.data['volatility'].dropna().values,
                       title=f"{model_name} - Volatility")
        
        # Predictions vs Actual
        plot_prediction_vs_actual(self.prices, 
                                self.history['predictions'][-1],
                                title=f"{model_name} - Predicted vs Actual")
        
        # Residuals
        plot_residuals(self.prices, 
                      self.history['predictions'][-1],
                      title=f"{model_name} - Residuals")
        
        # Learning curves
        plot_learning_curves(self.history['loss'],
                           title=f"{model_name} - Learning Curve")
        
        # Feature correlation
        plot_correlation_matrix(self.data[['open', 'high', 'low', 'close', 'vol']],
                              title=f"{model_name} - Feature Correlation")
        
        # Feature importance (example calculation)
        w = self.models[model_name][0]
        importances = np.abs(w) / np.sum(np.abs(w))
        plot_feature_importance(['open', 'high', 'low', 'vol'],
                              importances,
                              title=f"{model_name} - Feature Importance")
        
        # Uncertainty
        plot_uncertainty(self.timestamps,
                        self.history['predictions'][-1],
                        self.history['uncertainties'][-1],
                        self.prices,
                        title=f"{model_name} - Predictions with Uncertainty")
        
        # Trading signals
        plot_trading_signals(self.timestamps,
                           self.prices,
                           self.history['trading_signals']['buy'][-1],
                           self.history['trading_signals']['sell'][-1],
                           title=f"{model_name} - Trading Signals")
        
        # Cumulative returns
        plot_cumulative_returns(self.timestamps[1:],
                              self.data['returns'].dropna().values,
                              title=f"{model_name} - Cumulative Returns")

def main():
    # Example usage
    predictor = StockPredictor('data.csv')
    
    # Train all models
    models = ['amds', 'amds_plus', 'cipo', 'bcipo', 'bcipo_dropout', 'bcipo_hesm']
    for model in models:
        print(f"Training {model}...")
        predictor.train(model, iterations=1000, activation='linear')
        print(f"Completed training {model} and generated plots.")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetGUI(root)
    root.mainloop()

def validate_columns(df):
    """Validate that DataFrame has all required columns"""
    expected_columns = ['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'vol', 'openint']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        print_error(f"Missing required columns: {', '.join(missing_columns)}")
        print_info(f"Expected columns: {', '.join(expected_columns)}")
        print_info(f"Found columns: {', '.join(df.columns)}")
        return False
    
    print_info("All required columns present")
    return True

def load_data(file_path):
    print_info(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print_info(f"Successfully loaded {len(df)} rows")
        
        # Validate columns
        if not validate_columns(df):
            return None
            
        # Display data info
        print_info(f"Data summary:")
        print_info(f"- Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print_info(f"- Number of tickers: {df['ticker'].nunique()}")
        print_info(f"- Trading volume range: {df['vol'].min():,.0f} to {df['vol'].max():,.0f}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print_warning("Missing values detected:")
            for col in missing_values[missing_values > 0].index:
                print_warning(f"- {col}: {missing_values[col]} missing values")
        
        return df
    except Exception as e:
        print_error(f"Failed to load data: {str(e)}")
        return None

def preprocess_data(df):
    print_info("Starting data preprocessing")
    try:
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Sort by timestamp
        df_processed = df_processed.sort_values('timestamp')
        print_info("Data sorted by timestamp")
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_processed['timestamp']):
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
            print_info("Converted timestamp to datetime format")
        
        # Check for and handle any zero or negative values in price columns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            invalid_prices = (df_processed[col] <= 0).sum()
            if invalid_prices > 0:
                print_warning(f"Found {invalid_prices} invalid {col} prices (<=0)")
        
        # Check for and handle any negative volume
        if (df_processed['vol'] < 0).any():
            print_warning("Found negative volume values")
        
        print_info("Data preprocessing completed")
        return df_processed
        
    except Exception as e:
        print_error(f"Preprocessing failed: {str(e)}")
        return None
