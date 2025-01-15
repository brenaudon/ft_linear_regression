import os
import numpy as np
from linear_regression_train import read_data, denormalize

# Function to load the trained theta values from thetas.txt
def load_thetas(file_path='thetas.txt'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            theta0, theta1 = map(float, file.read().split(','))
    else:
        theta0, theta1 = 0.0, 0.0
    return theta0, theta1

# Function to predict the price based on mileage
def predict_price(mileage, theta0, theta1, original_mileage_min, original_mileage_max, original_price_min, original_price_max):
    # Normalize the mileage
    normalized_mileage = (mileage - original_mileage_min) / (original_mileage_max - original_mileage_min)

    # Predict the normalized price
    normalized_price = theta0 + (theta1 * normalized_mileage)

    # Denormalize the predicted price
    estimated_price = denormalize(normalized_price, original_price_min, original_price_max)

    return estimated_price

# Main function
def main():
    # Load the trained theta values
    theta0, theta1 = load_thetas()

    # Read the original data from data.csv
    original_mileage, original_price = read_data('data.csv')

    # Get the min and max values for mileage and price
    original_mileage_min = np.min(original_mileage)
    original_mileage_max = np.max(original_mileage)
    original_price_min = np.min(original_price)
    original_price_max = np.max(original_price)

    # Prompt the user for mileage input
    mileage = float(input("Enter the mileage of the car: "))

    # Predict the price
    estimated_price = predict_price(mileage, theta0, theta1, original_mileage_min, original_mileage_max, original_price_min, original_price_max)

    # Display the estimated price
    print(f"The estimated price for a car with {mileage} km is: {estimated_price}")

if __name__ == '__main__':
    main()