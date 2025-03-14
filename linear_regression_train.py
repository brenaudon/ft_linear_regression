"""
This script contains functions to train a linear regression model using gradient descent.
Results are displayed using matplotlib and saved to be used in the prediction script.

Dependencies:
    - numpy
    - matplotlib
    - scipy
    - csv
    - sys
"""

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def read_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read data from a CSV file.

    @param file_path: The path to the CSV file.
    @type  file_path: str

    @return: The mileage and price data.
    @rtype:  tuple of numpy arrays
    """
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


def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize the data.

    @param data: The data to be normalized.
    @type  data: numpy array

    @return: The normalized data.
    @rtype:  numpy array
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def denormalize(data: np.ndarray, original_min: float, original_max: float) -> np.ndarray:
    """Denormalize the data.

    @param data: The data to be denormalized.
    @type  data: numpy array
    @param original_min: The original minimum value.
    @type  original_min: number
    @param original_max: The original maximum value.
    @type  original_max: number

    @return: The denormalized data.
    @rtype:  numpy array
    """
    return data * (original_max - original_min) + original_min


def cost(mileage: np.ndarray, price: np.ndarray, theta0: float, theta1: float) -> float:
    """Calculate the cost of the linear regression model.

    @param mileage: The mileage data.
    @type  mileage: numpy array
    @param price: The price data.
    @type  price: numpy array
    @param theta0: The intercept of the regression line.
    @type  theta0: number
    @param theta1: The slope of the regression line.
    @type  theta1: number

    @return: The total cost.
    @rtype:  number
    """
    m = len(mileage)
    total_error = 0
    for i in range(m):
        estimate_price = theta0 + theta1 * mileage[i]
        error = estimate_price - price[i]
        total_error += error ** 2
    return total_error / (2 * m)


def train_linear_regression(mileage: np.ndarray, price: np.ndarray, initial_learning_rate: float = 0.1, iterations: int = 1000, decay_rate: float = 0.0) -> tuple[float, float, list[float]]:
    """Perform linear regression using gradient descent.

    @param mileage: The mileage data.
    @type  mileage: numpy array
    @param price: The price data.
    @type  price: numpy array
    @param initial_learning_rate: The learning rate for gradient descent.
    @type  initial_learning_rate: number
    @param iterations: The number of iterations for gradient descent.
    @type  iterations: int
    @param decay_rate: The rate at which the learning rate decays.
    @type  decay_rate: number

    @return: The intercept, slope, and cost history.
    @rtype:  tuple of (number, number, list of numbers)
    """
    theta0 = 0
    theta1 = 0
    m = len(mileage)
    cost_history = []

    # Normalize the data
    mileage = normalize(mileage)
    price = normalize(price)

    for i in range(iterations):
        learning_rate = initial_learning_rate / (1 + decay_rate * i)
        sum_errors_theta0 = 0
        sum_errors_theta1 = 0
        for j in range(m):
            estimate_price = theta0 + theta1 * mileage[j]
            error = estimate_price - price[j]
            sum_errors_theta0 += error
            sum_errors_theta1 += error * mileage[j]

        tmp_theta0 = learning_rate * (1/m) * sum_errors_theta0
        tmp_theta1 = learning_rate * (1/m) * sum_errors_theta1

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        cost_history.append(cost(mileage, price, theta0, theta1))

    return theta0, theta1, cost_history


def get_denormalized_thetas(price: np.ndarray, mileage: np.ndarray, theta0: float, theta1: float) -> tuple[float, float]:
    """Get the denormalized theta values.

    @param price: The price data.
    @type  price: numpy array
    @param mileage: The mileage data.
    @type  mileage: numpy array
    @param theta0: The intercept of the regression line.
    @type  theta0: number
    @param theta1: The slope of the regression line.
    @type  theta1: number

    @return: The denormalized intercept and slope.
    @rtype:  tuple of (number, number)
    """
    normalized_mileage = normalize(mileage)

    # Calculate the regression line and denormalize it
    regression_line = theta0 + theta1 * normalized_mileage
    denormalized_regression_line = denormalize(regression_line, np.min(price), np.max(price))

    linear_function_data = linregress(mileage, denormalized_regression_line)

    return linear_function_data.intercept, linear_function_data.slope #theta0, theta1


def save_thetas(theta0: float, theta1: float, file_path: str = 'thetas.txt'):
    """Save the trained theta values to a file.

    @param theta0: The intercept of the regression line.
    @type  theta0: number
    @param theta1: The slope of the regression line.
    @type  theta1: number
    @param file_path: The path to the file.
    @type  file_path: str
    """
    with open(file_path, 'w') as file:
        file.write(f'{theta0},{theta1}')


def display_regression_line(mileage: np.ndarray, price: np.ndarray, theta0: float, theta1: float):
    """Display the data points and the regression line.

    @param mileage: The mileage data.
    @type  mileage: numpy array
    @param price: The price data.
    @type  price: numpy array
    @param theta0: The intercept of the regression line.
    @type  theta0: number
    @param theta1: The slope of the regression line.
    @type  theta1: number
    """
    # Plot the data points
    plt.scatter(mileage, price, color='blue', label='Data points')

    regression_line = theta0 + theta1 * mileage

    plt.plot(mileage, regression_line, color='red', label='Regression line')

    # Add labels and title
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression')
    plt.legend()

    # Show the plot
    plt.show()


def display_cost_history(cost_history: list[float]):
    """Display the cost history.

    @param cost_history: The list of cost values for each iteration.
    @type  cost_history: list of numbers
    """
    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost History')
    plt.show()


# Main function
def main():
    """Main function to train the linear regression model and display results."""
    # Check if the dataset file is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python linear_regression_train.py <dataset_file>")
        return

    dataset_file = sys.argv[1]

    try:
        # Load dataset
        mileage, price = read_data(dataset_file)
        theta0, theta1, cost_history = train_linear_regression(mileage, price, decay_rate=0.001)
        theta0, theta1 = get_denormalized_thetas(price, mileage, theta0, theta1)
        save_thetas(theta0, theta1)
        print(f'Trained thetas: theta0 = {theta0}, theta1 = {theta1}')

        # Display the regression line
        display_regression_line(mileage, price, theta0, theta1)

        # Display the cost history
        display_cost_history(cost_history)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()