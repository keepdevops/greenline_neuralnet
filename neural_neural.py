import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, ttk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plot_stock_data import (
    plot_stock_price, plot_returns_distribution, plot_volatility,
    plot_prediction_vs_actual, plot_residuals, plot_learning_curves,
    plot_correlation_matrix, plot_feature_importance, plot_uncertainty,
    plot_trading_signals, plot_cumulative_returns
)

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
        loss = loss * weights
    return np.sum(loss) / np.sum(weights if weights is not None else 1)

def compute_gradients(x, y, y_hat, z, delta=1.0, activation='linear'):
    error = y_hat - y
    is_small_error = np.abs(error) <= delta
    dL_dyhat = np.where(is_small_error, error, delta * np.sign(error)) / len(y)
    dyhat_dz = 1 if activation == 'linear' else y_hat * (1 - y_hat)
    g_w = dL_dyhat * dyhat_dz * x
    g_b = dL_dyhat * dyhat_dz
    return g_w.sum(axis=0), g_b.sum()

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
        
        g_w, g_b = compute_gradients(x, y, y_hat, z, delta, activation)
        mean_x = rho * mean_x + (1 - rho) * np.mean(x, axis=0)
        var_x = rho * var_x + (1 - rho) * np.mean((x - mean_x) ** 2, axis=0)
        g_w = g_w / np.sqrt(var_x + epsilon)
        g_w = np.clip(g_w, -C, C)
        g_b = min(max(g_b, -C), C)
        time_weight = np.exp(-gamma * (timestamps[-1] - timestamps))
        g_w = g_w * time_weight
        g_b = g_b * time_weight.mean()
        
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
        alpha_t = 1 / (1 + np.exp(-(delta_loss - theta) / sigma))
        
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
        probs = np.exp(rewards / tau) / np.sum(np.exp(rewards / tau))
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
            ei = expected_improvement(mu[0], sigma[0], best_loss)
            reward = -mu[0] + beta_1 * trend_score + beta_2 * ei
            candidates.append((w_cand, b_cand, loss))
            mu_losses.append(mu[0])
            sigma_losses.append(sigma[0])
            rewards.append(reward)
        
        rewards = np.array(rewards)
        probs = np.exp(rewards / tau) / np.sum(np.exp(rewards / tau))
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
                predictions = []
                for _ in range(K):
                    nn.set_params(random_state=np.random.randint(1000))
                    predictions.append(nn.predict(params)[0])
                mu = np.mean(predictions)
                sigma = np.std(predictions)
            else:
                mu, sigma = 0.0, 1.0
            z = np.dot(x, w_cand) + b_cand
            y_hat = z if activation == 'linear' else sigmoid(z)
            loss = huber_loss(y, y_hat, delta, time_weights)
            trend_score = np.sign(y_hat[-1] - y[-2]) * ma_slope if len(y) > 1 else 0.0
            best_loss = min(y_history) if y_history else loss
            ei = expected_improvement(mu, sigma, best_loss)
            reward = -mu + beta_1 * trend_score + beta_2 * ei
            candidates.append((w_cand, b_cand, loss))
            mu_losses.append(mu)
            sigma_losses.append(sigma)
            rewards.append(reward)
        
        rewards = np.array(rewards)
        probs = np.exp(rewards / tau) / np.sum(np.exp(rewards / tau))
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

class HESM:
    def __init__(self):
        self.gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), random_state=0)
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
        self.dropout_nn = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', 
                                      solver='adam', learning_rate_init=0.01, 
                                      max_iter=100, random_state=0)
        self.poly = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('reg', LinearRegression())
        ])
        self.models = [self.gp, self.rf, self.dropout_nn, self.poly]
        self.errors = [1.0] * 4  # Simplified BNN as Dropout NN
    
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
        for i, model in enumerate(self.models):
            if model == self.dropout_nn:
                predictions = [model.predict(X) for _ in range(10)]
                mu, sigma = np.mean(predictions, axis=0), np.std(predictions, axis=0)
            elif model == self.poly:
                mu = model.predict(X)
                bootstrap_preds = []
                for _ in range(5):
                    idx = np.random.choice(len(X_history), len(X_history), replace=True)
                    model.fit(X_history[idx], y_history[idx])
                    bootstrap_preds.append(model.predict(X))
                sigma = np.std(bootstrap_preds, axis=0)
            else:
                mu, sigma = model.predict(X, return_std=True)
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

def bcipo_hesm(x, y, timestamps, activation='linear', iterations=1000, neuron=None):
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
        probs = np.exp(rewards / tau) / np.sum(np.exp(rewards / tau))
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

# Tkinter GUI
class NeuralNetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Single Neuron Stock Predictor")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main frames
        self.create_frames()
        self.create_controls()
        self.create_plots()
        
        # Initialize plot data
        self.initialize_data()

    def create_frames(self):
        """Create the main layout frames"""
        # Control panel on the left
        self.control_frame = ttk.Frame(self.root, padding="5")
        self.control_frame.grid(row=0, column=0, sticky="nsew")
        
        # Plots panel on the right
        self.plot_frame = ttk.Frame(self.root, padding="5")
        self.plot_frame.grid(row=0, column=1, sticky="nsew")
        
        # Configure grid weights for plot frame
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

    def create_controls(self):
        """Create control panel widgets"""
        # Training controls
        ttk.Label(self.control_frame, text="Training Parameters").grid(row=0, column=0, pady=5)
        
        # Learning rate
        ttk.Label(self.control_frame, text="Learning Rate:").grid(row=1, column=0)
        self.learning_rate = ttk.Entry(self.control_frame)
        self.learning_rate.insert(0, "0.01")
        self.learning_rate.grid(row=1, column=1)
        
        # Iterations
        ttk.Label(self.control_frame, text="Iterations:").grid(row=2, column=0)
        self.iterations = ttk.Entry(self.control_frame)
        self.iterations.insert(0, "1000")
        self.iterations.grid(row=2, column=1)
        
        # Plot selection
        ttk.Label(self.control_frame, text="Select Plot:").grid(row=3, column=0, pady=10)
        self.plot_type = ttk.Combobox(self.control_frame, values=[
            "Stock Price",
            "Returns Distribution",
            "Volatility",
            "Prediction vs Actual",
            "Residuals",
            "Learning Curves",
            "Correlation Matrix",
            "Feature Importance",
            "Uncertainty",
            "Trading Signals",
            "Cumulative Returns"
        ])
        self.plot_type.set("Stock Price")
        self.plot_type.grid(row=3, column=1)
        self.plot_type.bind('<<ComboboxSelected>>', self.update_plot)
        
        # Buttons
        ttk.Button(self.control_frame, text="Train Model", command=self.train_model).grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(self.control_frame, text="Update Plot", command=self.update_plot).grid(row=5, column=0, columnspan=2)
        
        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(self.control_frame, textvariable=self.status_var).grid(row=6, column=0, columnspan=2, pady=10)

    def create_plots(self):
        """Create the plotting area"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def initialize_data(self):
        """Initialize sample data for plots"""
        self.n_samples = 1000
        self.timestamps = np.arange(self.n_samples)
        
        # Simulate stock data
        self.prices = 100 * np.exp(np.random.randn(self.n_samples).cumsum() * 0.02)
        self.returns = np.diff(self.prices) / self.prices[:-1]
        
        # Calculate volatility
        self.window = 20
        self.volatility = np.array([np.std(self.returns[max(0, i-self.window):i]) 
                                  for i in range(self.window, len(self.returns))])
        
        # Initialize predictions
        self.predictions = self.prices + np.random.randn(self.n_samples) * 5
        self.uncertainties = np.abs(np.random.randn(self.n_samples)) * 2
        
        # Trading signals
        self.buy_signals = np.random.rand(self.n_samples) > 0.95
        self.sell_signals = np.random.rand(self.n_samples) > 0.95
        
        # Learning curve
        self.loss_history = np.random.rand(100).cumsum()

    def update_plot(self, event=None):
        """Update the displayed plot based on selection"""
        self.ax.clear()
        plot_type = self.plot_type.get()
        
        try:
            if plot_type == "Stock Price":
                plot_stock_price(self.timestamps, self.prices)
            elif plot_type == "Returns Distribution":
                plot_returns_distribution(self.returns)
            elif plot_type == "Volatility":
                plot_volatility(self.timestamps[self.window-1:], self.volatility)
            elif plot_type == "Prediction vs Actual":
                plot_prediction_vs_actual(self.prices, self.predictions)
            elif plot_type == "Residuals":
                plot_residuals(self.prices, self.predictions)
            elif plot_type == "Learning Curves":
                plot_learning_curves(self.loss_history)
            elif plot_type == "Correlation Matrix":
                features = pd.DataFrame({
                    'price': self.prices,
                    'volume': np.random.randn(self.n_samples) * 1000 + 5000,
                    'volatility': np.pad(self.volatility, (self.window-1, 0), mode='edge'),
                    'momentum': np.pad(np.diff(self.prices, periods=5), (5, 0), mode='edge')
                })
                plot_correlation_matrix(features)
            elif plot_type == "Feature Importance":
                features = ['price', 'volume', 'volatility', 'momentum']
                importances = np.random.rand(len(features))
                plot_feature_importance(features, importances)
            elif plot_type == "Uncertainty":
                plot_uncertainty(self.timestamps, self.predictions, self.uncertainties, self.prices)
            elif plot_type == "Trading Signals":
                plot_trading_signals(self.timestamps, self.prices, self.buy_signals, self.sell_signals)
            elif plot_type == "Cumulative Returns":
                plot_cumulative_returns(self.timestamps[1:], self.returns)
            
            self.canvas.draw()
            self.status_var.set(f"Updated plot: {plot_type}")
        except Exception as e:
            self.status_var.set(f"Error updating plot: {str(e)}")

    def train_model(self):
        """Train the neural network model"""
        try:
            learning_rate = float(self.learning_rate.get())
            iterations = int(self.iterations.get())
            
            # Simulate training
            self.loss_history = np.random.rand(iterations) * np.exp(-np.linspace(0, 2, iterations))
            self.predictions = self.prices + np.random.randn(self.n_samples) * 3
            
            self.status_var.set("Training completed")
            self.update_plot()
        except Exception as e:
            self.status_var.set(f"Error during training: {str(e)}")

def main():
    root = tk.Tk()
    app = NeuralNetGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
