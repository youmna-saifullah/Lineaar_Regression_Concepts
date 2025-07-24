# Import libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # For loss
import matplotlib.pyplot as plt

# Training data (X = input, Y = actual output)
X = [[1], [2], [3], [4], [5]]
Y = [2, 4, 5, 4, 5]

# Step 1: Create and train the model
model = LinearRegression()
model.fit(X, Y)

# Step 2: Make predictions
predicted = model.predict(X)

# Step 3: Calculate Loss (Mean Squared Error)
loss = mean_squared_error(Y, predicted)
print("Loss (Mean Squared Error):", loss)

# Optional: Show prediction line
plt.scatter(X, Y, color='blue', label='Actual Points')
plt.plot(X, predicted, color='red', label='Predicted Line')
plt.title('Linear Regression with Loss')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

'''
MSE Value	Interpretation (for small Y like 2–10)
0 _ 1	    ✅ Very good
1 _ 5	    ⚠️ Acceptable, but can improve
> 5	        ❌ High error _ model not learnig Rate
'''