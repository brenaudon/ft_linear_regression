import os

def load_thetas(file_path='thetas.txt'):
    """Load the trained theta values from a file.

    Keyword arguments:
    file_path -- the path to the file (default 'thetas.txt')

    Returns:
    theta0 -- the intercept of the regression line
    theta1 -- the slope of the regression line
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            theta0, theta1 = map(float, file.read().split(','))
    else:
        theta0, theta1 = 0.0, 0.0
    return theta0, theta1


def predict_price(mileage, theta0, theta1):
    """Predict the price based on mileage.

    Keyword arguments:
    mileage -- the mileage of the car
    theta0 -- the intercept of the regression line
    theta1 -- the slope of the regression line

    Returns:
    estimated price
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