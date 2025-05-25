v_w = 0  # Momentum
eta_0 = 0.01  # Base learning rate
beta = 0.9  # Momentum factor
for t in range(iterations):
    g_w = compute_gradient(loss, w)  # Gradient w.r.t. weight
    curvature = abs(g_w - prev_g_w)  # Approximate curvature
    eta_t = eta_0 / (sqrt(abs(loss) + 1e-8) * (curvature + 1e-8))  # Adaptive LR
    v_w = beta * v_w + (1 - beta) * g_w  # Momentum update
    w = w - eta_t * v_w  # Update weight
    prev_g_w = g_w
