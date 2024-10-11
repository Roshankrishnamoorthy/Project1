import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def elastic_net_gradient_descent(X, y, alpha, l1_ratio, num_iterations, learning_rate):
    m, n = X.shape
    beta = np.zeros(n)
    for _ in range(num_iterations):
        predictions = X.dot(beta)
        errors = predictions - y
        gradient = (X.T.dot(errors) + alpha * ((1 - l1_ratio) * beta + l1_ratio * np.sign(beta))) / m
        beta -= learning_rate * gradient
    return beta

def visualize_elastic_net_coefficients(X, y, alpha_values):
    plt.figure(figsize=(12, 8))
    for alpha in alpha_values:
        beta = elastic_net_gradient_descent(X, y, alpha, l1_ratio=0.5,
                                            num_iterations=1000, learning_rate=0.001)
        plt.plot(beta, marker='o', label=f'alpha={alpha}')
    
    plt.title('ElasticNet Coefficients for Different Alpha Values')
    plt.xlabel('Features')
    plt.ylabel('Coefficients')
    plt.legend()
    plt.grid(True)
    plt.show()

def process_and_visualize(data, target_column):
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Convert target to NumPy array and flatten it
    y_train_np = y_train.to_numpy().ravel()

    # Define alpha values to visualize
    alpha_values = [0.01, 0.1, 1, 10]
    
    visualize_elastic_net_coefficients(X_train_scaled, y_train_np, alpha_values)

# Example usage with any dataset
# Load your dataset into a DataFrame `df` and specify the target column name.
# df = pd.read_csv('your_dataset.csv')
# process_and_visualize(df, 'target_column_name')

