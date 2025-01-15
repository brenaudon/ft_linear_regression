import csv
import numpy as np
import matplotlib.pyplot as plt

# Function to read data from CSV file
def read_data(file_path):
    mileage = []
    price = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mileage.append(float(row['km']))
            price.append(float(row['price']))

    # Convert lists to numpy arrays for easier manipulation
    mileage = np.array(mileage)
    price = np.array(price)

    return mileage, price

# Function to normalize the data
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Function to denormalize the data
def denormalize(data, original_min, original_max):
    return data * (original_max - original_min) + original_min

# Function to calculate the cost of the linear regression model
def cost(mileage, price, theta0, theta1):
    m = len(mileage)
    total_error = 0
    for i in range(m):
        estimate_price = theta0 + theta1 * mileage[i]
        error = estimate_price - price[i]
        total_error += error ** 2
    return total_error / (2 * m)

# Function to perform linear regression using gradient descent
def train_linear_regression(mileage, price, learning_rate=0.05, iterations=1000):
    theta0 = 0
    theta1 = 0
    m = len(mileage)
    cost_history = []

    # Normalize the data
    mileage = normalize(mileage)
    price = normalize(price)

    print(mileage)
    print(price)

    for _ in range(iterations):
        sum_errors_theta0 = 0
        sum_errors_theta1 = 0
        for i in range(m):
            estimate_price = theta0 + theta1 * mileage[i]
            error = estimate_price - price[i]
            sum_errors_theta0 += error
            sum_errors_theta1 += error * mileage[i]

        tmp_theta0 = learning_rate * (1/m) * sum_errors_theta0
        tmp_theta1 = learning_rate * (1/m) * sum_errors_theta1

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        cost_history.append(cost(mileage, price, theta0, theta1))

    return theta0, theta1, cost_history

# Function to save the trained theta values
def save_thetas(theta0, theta1, file_path='thetas.txt'):
    with open(file_path, 'w') as file:
        file.write(f'{theta0},{theta1}')

# Function to display the data points and the regression line
def display_regression_line(mileage, price, theta0, theta1):
    # Plot the data points
    plt.scatter(mileage, price, color='blue', label='Data points')

    normalized_mileage = normalize(mileage)

    # Calculate the regression line and denormalize it
    regression_line = theta0 + theta1 * normalized_mileage
    denormalized_regression_line = denormalize(regression_line, np.min(price), np.max(price))

    plt.plot(mileage, denormalized_regression_line, color='red', label='Regression line')

    # Add labels and title
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression')
    plt.legend()

    # Show the plot
    plt.show()

# Function to display the cost history
def display_cost_history(cost_history):
    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost History')
    plt.show()

# Main function
def main():
    mileage, price = read_data('data.csv')
    theta0, theta1, cost_history = train_linear_regression(mileage, price)
    save_thetas(theta0, theta1)
    print(f'Trained thetas: theta0 = {theta0}, theta1 = {theta1}')

    # Display the regression line
    display_regression_line(mileage, price, theta0, theta1)

    # Display the cost history
    display_cost_history(cost_history)

if __name__ == '__main__':
    main()