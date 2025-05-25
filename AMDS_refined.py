import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(y_hat, y):
    return np.mean((y_hat - y) ** 2)  # MSE for regression

def compute_gradients(x, y, y_hat, z, activation='linear'):
    if activation == 'linear':
        dL_dyhat = 2 * (y_hat - y) / len(y)
        dyhat_dz = 1
    elif activation == 'sigmoid':
        dL_dyhat = 2 * (y_hat - y) / len(y)
        dyhat_dz = y_hat * (1 - y_hat)
    g_w = dL_dyhat * dyhat_dz * x
    g_b = dL_dyhat * dyhat_dz
    return g_w, g_b

# AMDS+ Backpropagation
def amds_plus(x, y, activation='linear', iterations=1000):
    # Initialize parameters
    w = np.random.randn(x.shape[1])  # Weights
    b = 0.0  # Bias
    v_w, v_b = np.zeros_like(w), 0.0  # Momentum
    s_w, s_b = np.zeros_like(w), 0.0  # Second moment
    G_w, G_b = np.zeros_like(w), 0.0  # Cumulative squared gradients
    eta_0, beta_1, beta_2 = 0.01, 0.9, 0.999
    epsilon, C, lambda_reg = 1e-8, 1.0, 0.01
    theta, sigma = 1e-4, 0.001
    prev_loss, stall_count = float('inf'), 0
    prev_g_w, prev_g_b = np.zeros_like(w), 0.0
    
    for t in range(iterations):
        # Forward pass
        z = np.dot(x, w) + b
        y_hat = z if activation == 'linear' else sigmoid(z)
        loss = compute_loss(y_hat, y)
        
        # Early stopping
        delta_loss = prev_loss - loss
        if delta_loss < 1e-6:
            stall_count += 1
            if stall_count >= 10:
                break
        else:
            stall_count = 0
        
        # Compute gradients
        g_w, g_b = compute_gradients(x, y, y_hat, z, activation)
        
        # Clip gradients
        g_w = np.clip(g_w, -C, C)
        g_b = min(max(g_b, -C), C)
        
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
        
        # Compute curvature
        kappa_w = np.abs(g_w - prev_g_w)
        kappa_b = abs(g_b - prev_g_b)
        
        # Dynamic learning rate
        eta_t = eta_0 / (np.sqrt(abs(loss) + epsilon) * (1 + kappa_w + epsilon))
        eta_t_b = eta_0 / (np.sqrt(abs(loss) + epsilon) * (1 + kappa_b + epsilon))
        
        # Blending factor
        alpha_t = 1 / (1 + np.exp(-(delta_loss - theta) / sigma))
        
        # Compute updates
        delta_w = alpha_t * (eta_t * v_w_hat / np.sqrt(s_w_hat + epsilon)) + \
                  (1 - alpha_t) * (eta_t * v_w_hat / np.sqrt(G_w + epsilon)) + \
                  lambda_reg * w
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
    # Sample data (replace with actual data)
    np.random.seed(0)
    x = np.random.randn(100, 2)  # 100 samples, 2 features
    y = np.dot(x, np.array([1.0, -0.5])) + 0.1 * np.random.randn(100)  # Linear regression
    w, b, final_loss = amds_plus(x, y, activation='linear')
    print(f"Final weights: {w}, Bias: {b}, Loss: {final_loss}")
