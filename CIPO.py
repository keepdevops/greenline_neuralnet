import numpy as np
import pandas as pd

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

def logistic_map(x, r=3.9):
    return r * x * (1 - x)

def compute_volatility(returns, window=20):
    return np.std(returns[-window:]) if len(returns) >= window else 1.0

def compute_ma_slope(y, window=5):
    if len(y) < window:
        return 0.0
    ma = np.convolve(y, np.ones(window)/window, mode='valid')
    return (ma[-1] - ma[-2]) / window if len(ma) > 1 else 0.0

def cipo_stock(x, y, timestamps, activation='linear', iterations=1000):
    # Initialize parameters
    w = np.random.randn(x.shape[1]) * 0.01
    b = 0.0
    N, alpha_0, beta, tau = 5, 0.01, 0.1, 0.1
    gamma, delta, theta = 0.01, 1.0, 0.001
    prev_loss, stall_count = float('inf'), 0
    returns = np.diff(y) / y[:-1] if len(y) > 1 else [1.0]
    chaos_states = np.random.rand(N, x.shape[1] + 1)  # For weights and bias
    
    for t in range(iterations):
        # Compute stock-specific metrics
        sigma_vol = compute_volatility(returns)
        ma_slope = compute_ma_slope(y)
        time_weights = np.exp(-gamma * (timestamps[-1] - timestamps))
        
        # Generate candidate parameters
        candidates = []
        rewards = []
        for i in range(N):
            # Update chaotic states
            chaos_states[i] = logistic_map(chaos_states[i])
            delta_w = alpha_0 * (2 * chaos_states[i, :-1] - 1) * sigma_vol
            delta_b = alpha_0 * (2 * chaos_states[i, -1] - 1) * sigma_vol
            
            # Candidate parameters
            w_cand = w + delta_w
            b_cand = b + delta_b
            
            # Forward pass
            z = np.dot(x, w_cand) + b_cand
            y_hat = z if activation == 'linear' else sigmoid(z)
            
            # Compute reward
            loss = huber_loss(y, y_hat, delta, time_weights)
            trend_score = np.sign(y_hat[-1] - y[-2]) * ma_slope if len(y) > 1 else 0.0
            reward = -loss + beta * trend_score
            
            candidates.append((w_cand, b_cand))
            rewards.append(reward)
        
        # Probabilistic selection
        rewards = np.array(rewards)
        probs = np.exp(rewards / tau) / np.sum(np.exp(rewards / tau))
        idx = np.random.choice(N, p=probs)
        w, b = candidates[idx]
        
        # Compute loss for monitoring
        z = np.dot(x, w) + b
        y_hat = z if activation == 'linear' else sigmoid(z)
        loss = huber_loss(y, y_hat, delta, time_weights)
        
        # Adaptive exploration
        delta_loss = prev_loss - loss
        alpha_t = alpha_0 * (1 + abs(delta_loss) / theta + sigma_vol)
        chaos_states = np.clip(chaos_states + alpha_t * (np.random.rand(N, x.shape[1] + 1) - 0.5), 0, 1)
        
        # Early stopping
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
    
    return w, b, loss

# Example usage
if __name__ == "__main__":
    # Simulated stock data
    np.random.seed(0)
    n_samples, n_features = 100, 3
    x = np.random.randn(n_samples, n_features)  # Features (e.g., RSI, MACD, volume)
    timestamps = np.arange(n_samples)
    y = np.cumsum(0.1 * np.random.randn(n_samples))  # Simulated price
    w, b, final_loss = cipo_stock(x, y, timestamps, activation='linear')
    print(f"Final weights: {w}, Bias: {b}, Loss: {final_loss}")
