# Import libraries
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Sample training data (house size vs price)
X = [[500], [1000], [1500], [2000], [2500]]  # House sizes in square feet
Y = [100, 150, 200, 250, 300]  # Prices in thousands

# Step 2: Create and train the model
model = LinearRegression()
model.fit(X, Y)

# Step 3: Predict values using the trained model
predicted = model.predict(X)

# Step 4: Print slope and intercept
print("Model learned the line: Price = ", model.coef_[0], "* Size +", model.intercept_)

# Step 5: Plotting the data and line
plt.scatter(X, Y, color='blue', label='Actual Prices')
plt.plot(X, predicted, color='red', label='Predicted Line')
plt.title('House Price Prediction')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price (in thousands)')
plt.legend()
#plt.show()

# Step 6: Predict price from user input
def predict_price():
    size = float(input("Enter the size of the house in square feet: "))
    price = model.predict([[size]])
    print(f"Predicted price for {size} sq ft is approximately: ${price[0]:.2f}k")

# Step 7: Call the function
predict_price()
