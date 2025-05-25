import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

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

def expected_improvement(mu, sigma, best_loss, xi=0.01):
    z = (best_loss - mu - xi) / (sigma + 1e-9)
    return (best_loss - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def bcipo_stock(x, y, timestamps, activation='linear', iterations=1000):
    # Initialize parameters
    w = np.random.randn(x.shape[1]) * 0.01
    b = 0.0
    
    # Ensure all input arrays have the same length
    min_len = min(len(x), len(y), len(timestamps))
    x = x[:min_len]
    y = y[:min_len]
    timestamps = timestamps[:min_len]
    
    # Initialize GP and history
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), random_state=0)
    X_history = []
    y_history = []
    
    # Parameters for BCIPO
    N = 10  # Number of chaos states
    chaos_states = np.random.rand(N, x.shape[1] + 1)  # +1 for bias
    alpha_0 = 0.01
    beta_1, beta_2 = 0.5, 0.3
    tau = 0.1
    delta = 1.0
    
    # Calculate initial volatility
    returns = np.diff(y) / y[:-1] if len(y) > 1 else np.array([1.0])
    sigma_vol = np.std(returns)
    ma_window = 20
    ma = np.convolve(y, np.ones(ma_window)/ma_window, mode='valid')
    ma_slope = (ma[-1] - ma[-2]) / ma[-2] if len(ma) > 1 else 0.0
    
    prev_loss, stall_count = float('inf'), 0
    
    for t in range(iterations):
        # Compute stock metrics
        time_weights = np.exp(-0.01 * (timestamps[-1] - timestamps))
        
        # Generate candidate parameters
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
            
            # Predict loss with GP
            params = np.append(w_cand, b_cand)
            if len(X_history) > 0:
                mu, sigma = gp.predict([params], return_std=True)
            else:
                mu, sigma = [0.0], [1.0]
            
            # Compute actual loss and trend score
            z = np.dot(x, w_cand) + b_cand
            y_hat = z if activation == 'linear' else sigmoid(z)
            loss = huber_loss(y, y_hat, delta, time_weights)
            trend_score = np.sign(y_hat[-1] - y[-2]) * ma_slope if len(y) > 1 else 0.0
            
            # Compute EI
            best_loss = min(y_history) if y_history else loss
            ei = expected_improvement(mu[0], sigma[0], best_loss)
            
            # Compute reward
            reward = -mu[0] + beta_1 * trend_score + beta_2 * ei
            
            candidates.append((w_cand, b_cand, loss))
            mu_losses.append(mu[0])
            sigma_losses.append(sigma[0])
            rewards.append(reward)
        
        # Probabilistic selection
        rewards = np.array(rewards)
        probs = np.exp(rewards / tau) / np.sum(np.exp(rewards / tau))
        idx = np.random.choice(N, p=probs)
        w, b, actual_loss = candidates[idx]
        
        # Update GP
        X_history.append(np.append(w, b))
        y_history.append(actual_loss)
        if len(X_history) > 1:
            gp.fit(np.array(X_history), np.array(y_history))
        
        # Update exploration
        alpha_t = alpha_0 * (1 + np.mean(sigma_losses) + sigma_vol)
        chaos_states = np.clip(chaos_states + alpha_t * (np.random.rand(N, x.shape[1] + 1) - 0.5), 0, 1)
        
        # Early stopping
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
    
    return w, b, actual_loss

# Example usage
if __name__ == "__main__":
    # Simulated stock data
    np.random.seed(0)
    n_samples, n_features = 100, 3
    x = np.random.randn(n_samples, n_features)  # Features (e.g., RSI, MACD, volume)
    timestamps = np.arange(n_samples)
    y = np.cumsum(0.1 * np.random.randn(n_samples))  # Simulated price
    w, b, final_loss = bcipo_stock(x, y, timestamps, activation='linear')
    print(f"Final weights: {w}, Bias: {b}, Loss: {final_loss}")
