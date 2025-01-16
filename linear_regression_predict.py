"""This script contains functions to load trained parameters and predict the price of a car
 based on its mileage."""

import os

def load_thetas(file_path: str = 'thetas.txt') -> tuple[float, float]:
    """Load the trained theta values from a file.

    @param file_path: The path to the file.
    @type  file_path: str

    @return: The intercept and slope of the regression line.
    @rtype:  tuple of (number, number)
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            theta0, theta1 = map(float, file.read().split(','))
    else:
        theta0, theta1 = 0.0, 0.0
    return theta0, theta1


def predict_price(mileage: float, theta0: float, theta1: float) -> float:
    """Predict the price based on mileage.

    @param mileage: The mileage of the car.
    @type  mileage: number
    @param theta0: The intercept of the regression line.
    @type  theta0: number
    @param theta1: The slope of the regression line.
    @type  theta1: number

    @return: The estimated price.
    @rtype:  number
    """
    return theta0 + theta1 * mileage


def main():
    """Main function to load thetas, prompt user for mileage, and predict price."""
    # Load the trained theta values
    theta0, theta1 = load_thetas()

    # Prompt the user for mileage input
    mileage = float(input("Enter the mileage of the car: "))

    # Predict the price
    estimated_price = predict_price(mileage, theta0, theta1)

    # Display the estimated price
    print(f"The estimated price for a car with {mileage} km is: {estimated_price}")

if __name__ == '__main__':
    main()