# Initialize parameters
w, b = initialize_weights()
v_w, v_b = 0, 0  # Momentum
s_w, s_b = 0, 0  # Second moment
G_w, G_b = 0, 0  # Cumulative squared gradients
eta_0, beta_1, beta_2, epsilon = 0.01, 0.9, 0.999, 1e-8
theta, T = 1e-4, 1000
prev_loss = inf

for t in range(iterations):
    # Forward pass
    z = w * x + b
    y_hat = activation(z)
    loss = compute_loss(y_hat, y)
    
    # Compute gradients
    g_w = compute_gradient_w(loss, y_hat, x)
    g_b = compute_gradient_b(loss, y_hat)
    
    # Update momentum
    v_w = beta_1 * v_w + (1 - beta_1) * g_w
    v_b = beta_1 * v_b + (1 - beta_1) * g_b
    
    # Update second moment
    s_w = beta_2 * s_w + (1 - beta_2) * g_w**2
    s_b = beta_2 * s_b + (1 - beta_2) * g_b**2
    
    # Update cumulative squared gradients
    G_w += g_w**2
    G_b += g_b**2
    
    # Dynamic learning rate
    eta_t = eta_0 / (sqrt(abs(loss) + epsilon) * (1 + t / T))
    
    # Switch based on loss reduction
    delta_loss = prev_loss - loss
    if delta_loss < theta:
        delta_w = eta_t * v_w / sqrt(s_w + epsilon)
        delta_b = eta_t * v_b / sqrt(s_b + epsilon)
    else:
        delta_w = eta_t * v_w / sqrt(G_w + epsilon)
        delta_b = eta_t * v_b / sqrt(G_b + epsilon)
    
    # Update parameters
    w -= delta_w
    b -= delta_b
    prev_loss = loss
