import os
import numpy as np

# Function to load the trained theta values from thetas.txt
def load_thetas(file_path='thetas.txt'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            theta0, theta1 = map(float, file.read().split(','))
    else:
        theta0, theta1 = 0.0, 0.0
    return theta0, theta1

# Function to predict the price based on mileage
def predict_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

# Main function
def main():
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