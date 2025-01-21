import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from genbooster import RustBooster
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Load diabetes dataset
X, y = load_diabetes(return_X_y=True)
y = y.astype(np.float64)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

# Test different learning rates
learning_rates = 10**np.linspace(-6, -1, 10)
n_estimators_list = np.linspace(100, 1000, 10)
rmses_train = [[] for _ in range(len(n_estimators_list))]
rmses_test = [[] for _ in range(len(n_estimators_list))]

for i, n_estimators in tqdm(enumerate(n_estimators_list)):
    for j, lr in enumerate(learning_rates):
        print(f"\nTesting learning rate: {lr}")    
        model = RustBooster(
            base_estimator=Ridge(alpha=0.001),
            n_estimators=int(n_estimators),
            learning_rate=lr,
            n_hidden_features=5,
            direct_link=True
        )    
        # Train and evaluate
        model.fit(X_train, y_train, dropout=0.0, seed=42)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        training_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        print(f"Training RMSE: {training_rmse}")
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        print(f"Test RMSE: {test_rmse}")
        rmses_train[i].append(float(training_rmse))
        rmses_test[i].append(float(test_rmse))

print("Training RMSE for different learning rates:", rmses_train)
print("Test RMSE for different learning rates:", rmses_test)

# Create meshgrid for 2D visualization
X, Y = np.meshgrid(learning_rates, n_estimators_list)

# Create two subplots for training and test RMSE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot training RMSE
train_contour = ax1.contourf(X, Y, rmses_train, levels=20, cmap='viridis')
ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('Number of Estimators')
ax1.set_title('Training RMSE')
plt.colorbar(train_contour, ax=ax1, label='RMSE')

# Plot test RMSE
test_contour = ax2.contourf(X, Y, rmses_test, levels=20, cmap='viridis')
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Number of Estimators')
ax2.set_title('Test RMSE')
plt.colorbar(test_contour, ax=ax2, label='RMSE')

plt.tight_layout()
plt.show()

