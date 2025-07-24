# Sample data: X = input (e.g. study hours), Y = output (e.g. marks)
X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

# Function to calculate Mean Squared Error (Loss)
def compute_loss(X, Y, m, b):
    total_loss = 0
    for i in range(len(X)):
        y_pred = m * X[i] + b
        total_loss += (Y[i] - y_pred) ** 2
    return total_loss / len(X)

# Function to train the model using gradient descent
def train_model(m, b, learning_rate, epochs):
    n = len(X)
    for epoch in range(epochs):
        dm = 0
        db = 0
        for i in range(n):
            y_pred = m * X[i] + b
            error = Y[i] - y_pred
            dm += -2 * X[i] * error
            db += -2 * error

        m = m - (learning_rate * dm / n)
        b = b - (learning_rate * db / n)

        if epoch % 10 == 0:
            loss = compute_loss(X, Y, m, b)
            print(f"Epoch {epoch}: Loss = {loss:.4f} | m = {m:.2f}, b = {b:.2f}")
    return m, b

# Main loop for user interaction
while True:
    print("\n📌 Linear Regression Training")
    try:
        # Take user inputs
        m = float(input("Enter initial value of m (slope): "))
        b = float(input("Enter initial value of b (intercept): "))
        learning_rate = float(input("Enter learning rate (e.g. 0.01): "))
        epochs = int(input("Enter number of epochs (e.g. 100): "))

        # Train the model
        final_m, final_b = train_model(m, b, learning_rate, epochs)

        # Final model and loss
        final_loss = compute_loss(X, Y, final_m, final_b)
        print(f"\n✅ Final Line: Y = {round(final_m, 2)} * X + {round(final_b, 2)}")
        print(f"📉 Final Loss (MSE): {final_loss:.4f}")

        # Ask if user wants to try again
        again = input("\n🔁 Try again with different values? (yes/no): ").strip().lower()
        if again != 'yes':
            print("👋 Thanks for learning! Goodbye.")
            break

    except ValueError:
        print("❌ Invalid input. Please enter numbers only.")
