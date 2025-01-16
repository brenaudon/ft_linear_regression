import numpy as np
from linear_regression_train import read_data
from linear_regression_predict import load_thetas


def calculate_r_squared(mileage, price, theta0, theta1):
    """Calculate the R-squared value.

    @param mileage: The mileage data.
    @type  mileage: numpy array
    @param price: The price data.
    @type  price: numpy array
    @param theta0: The intercept of the regression line.
    @type  theta0: number
    @param theta1: The slope of the regression line.
    @type  theta1: number

    @return: The R-squared value.
    @rtype:  number
    """
    ss_total = np.sum((price - np.mean(price)) ** 2)
    ss_residual = np.sum((price - (theta0 + theta1 * mileage)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared


def main():
    """Main function to load data, load thetas, and calculate R-squared."""
    # Load the data
    mileage, price = read_data('data.csv')

    # Load the trained theta values
    theta0, theta1 = load_thetas()

    # Calculate R-squared
    r_squared = calculate_r_squared(mileage, price, theta0, theta1)

    # Display the R-squared value
    print(f"R-squared: {r_squared}")

if __name__ == '__main__':
    main()
