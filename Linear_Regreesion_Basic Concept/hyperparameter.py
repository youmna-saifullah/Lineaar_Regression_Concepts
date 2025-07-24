# Sample data
X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

# 🔷 User Input for Hyperparameters
learning_rate = float(input("Enter learning rate (e.g., 0.01): "))
epochs = int(input("Enter number of epochs (e.g., 100): "))
batch_size = int(input("Enter batch size (e.g., 2 or 5): "))

# 🔷 Initial guess for parameters
m = 0
b = 0

# 🔁 Training loop
for epoch in range(epochs):
    total_loss = 0
    for batch_start in range(0, len(X), batch_size):
        m_grad_sum = 0
        b_grad_sum = 0
        batch_loss = 0
        count = 0

        for i in range(batch_start, min(batch_start + batch_size, len(X))):
            y_pred = m * X[i] + b
            error = Y[i] - y_pred

            # Accumulate gradients
            m_grad_sum += -2 * X[i] * error
            b_grad_sum += -2 * error
            batch_loss += error ** 2
            count += 1

        # Average gradients
        m_grad = m_grad_sum / count
        b_grad = b_grad_sum / count

        # Gradient descent update
        m -= learning_rate * m_grad
        b -= learning_rate * b_grad

        total_loss += batch_loss

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1} → Loss: {total_loss:.4f} | m: {m:.4f}, b: {b:.4f}")

# 🔚 Final Result
print("\nTraining complete!")
print(f"Final Model: y = {m:.2f}x + {b:.2f}")

# 🔎 Predict
new_x = float(input("Enter value of X to predict Y: "))
predicted_y = m * new_x + b
print(f"Predicted Y = {predicted_y:.2f}")
