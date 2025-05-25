import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def huber_loss(y, y_hat, delta=1.0):
    error = y - y_hat
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

def compute_gradients(x, y, y_hat, z, delta=1.0, activation='linear', per_sample=False):
    """Compute gradients with proper array size handling."""
    # Ensure all arrays have the same length
    min_len = min(len(y), len(y_hat))
    y = y[:min_len]
    y_hat = y_hat[:min_len]
    x = x[:min_len]
    
    error = y_hat - y
    is_small_error = np.abs(error) <= delta
    dL_dyhat = np.where(is_small_error, error, delta * np.sign(error)) / len(y)
    dyhat_dz = 1 if activation == 'linear' else y_hat * (1 - y_hat)
    
    if per_sample:
        g_w = np.array([dL_dyhat[i] * dyhat_dz * x[i] for i in range(len(x))])
        g_b = dL_dyhat * dyhat_dz
        return g_w, g_b
    else:
        g_w = np.sum(dL_dyhat * dyhat_dz * x.T, axis=1)
        g_b = np.sum(dL_dyhat * dyhat_dz)
        return g_w, g_b

def compute_volatility(returns, window=20):
    return np.std(returns[-window:]) if len(returns) >= window else 1.0

def amds_plus_stock(x, y, timestamps, activation='linear', iterations=1000):
    # Initialize parameters
    w = np.random.randn(x.shape[1]) * 0.01
    b = 0.0
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
    
    # Ensure all input arrays have the same length
    min_len = min(len(x), len(y), len(timestamps))
    x = x[:min_len]
    y = y[:min_len]
    timestamps = timestamps[:min_len]
    returns = np.diff(y) / y[:-1] if len(y) > 1 else np.array([1.0])
    
    for t in range(iterations):
        # Forward pass
        z = np.dot(x, w) + b
        y_hat = z if activation == 'linear' else sigmoid(z)
        loss = huber_loss(y, y_hat, delta)
        
        # Adaptive early stopping
        delta_loss = prev_loss - loss
        if loss > 1.5 * prev_loss:
            stall_count = 0
        if delta_loss < 1e-6:
            stall_count += 1
            if stall_count >= 10:
                break
        else:
            stall_count = 0
        
        # Compute gradients
        g_w, g_b = compute_gradients(x, y, y_hat, z, delta, activation)
        
        # Update feature variance
        mean_x = rho * mean_x + (1 - rho) * np.mean(x, axis=0)
        var_x = rho * var_x + (1 - rho) * np.mean((x - mean_x) ** 2, axis=0)
        g_w = g_w / np.sqrt(var_x + epsilon)
        
        # Clip gradients
        g_w = np.clip(g_w, -C, C)
        g_b = min(max(g_b, -C), C)
        
        # Temporal weighting
        time_weight = np.exp(-gamma * (timestamps[-1] - timestamps))
        g_w = g_w * time_weight
        g_b = g_b * time_weight.mean()
        
        # Update momentum
        v_w = beta_1 * v_w + (1 - beta_1) * g_w
        v_b = beta_1 * v_b + (1 - beta_1) * g_b
        v_w_hat = v_w / (1 - beta_1 ** (t + 1))
        v_b_hat = v_b / (1 - beta_1 ** (t + 1))
        
        # Update second moment
        s_w = beta_2 * s_w + (1 - beta_2) * g_w ** 2
        s_b = beta_2 * s_b + (1 - beta_2) * g_b ** 2
        s_w_hat = s_w / (1 - beta_2 ** (t + 1))
        s_b_hat = s_b / (1 - beta_2 ** (t + 1))
        
        # Update cumulative squared gradients
        G_w += g_w ** 2
        G_b += g_b ** 2
        
        # Compute volatility
        sigma_ret = compute_volatility(returns)
        returns = np.append(returns, (y[-1] - y[-2]) / y[-2] if t > 0 else returns[-1])
        
        # Compute curvature
        kappa_w = np.abs(g_w - prev_g_w)
        kappa_b = abs(g_b - prev_g_b)
        
        # Dynamic learning rate
        eta_t = eta_0 / (np.sqrt(abs(loss) + epsilon) * (1 + kappa_w + epsilon) * (1 + sigma_ret + epsilon))
        eta_t_b = eta_0 / (np.sqrt(abs(loss) + epsilon) * (1 + kappa_b + epsilon) * (1 + sigma_ret + epsilon))
        
        # Blending factor
        alpha_t = 1 / (1 + np.exp(-(delta_loss - theta) / sigma))
        
        # Compute updates
        delta_w = alpha_t * (eta_t * v_w_hat / np.sqrt(s_w_hat + epsilon)) + \
                  (1 - alpha_t) * (eta_t * v_w_hat / np.sqrt(G_w + epsilon)) + \
                  lambda_1 * np.sign(w) + lambda_2 * w
        delta_b = alpha_t * (eta_t_b * v_b_hat / np.sqrt(s_b_hat + epsilon)) + \
                  (1 - alpha_t) * (eta_t_b * v_b_hat / np.sqrt(G_b + epsilon))
        
        # Update parameters
        w -= delta_w
        b -= delta_b
        
        # Store gradients
        prev_g_w, prev_g_b = g_w, g_b
        prev_loss = loss
    
    return w, b, loss

# Example usage
if __name__ == "__main__":
    # Simulated stock data
    np.random.seed(0)
    n_samples, n_features = 100, 3
    x = np.random.randn(n_samples, n_features)  # Features (e.g., RSI, MACD, volume)
    timestamps = np.arange(n_samples)
    y = np.cumsum(0.1 * np.random.randn(n_samples))  # Simulated price
    w, b, final_loss = amds_plus_stock(x, y, timestamps, activation='linear')
    print(f"Final weights: {w}, Bias: {b}, Loss: {final_loss}")
