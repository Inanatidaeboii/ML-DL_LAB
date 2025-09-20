#EASY
##problem_1
import numpy as np
### Dataset
X = np.array([1,2,3,4,5,6,7,8])
y = np.array([45,50,60,65,75,80,90,95])

X_mean = np.sum(X)//len(X)
y_mean = np.sum(y)//len(y)

print(f"X mean is : {X_mean}\nY mean is : {y_mean}")

numurator = 0
dominator = 0
for i in range(len(X)):
    numurator += (X[i]-X_mean) * (y[i] - y_mean)
    dominator += (X[i]-X_mean)**2

slope = numurator//dominator
intercept = y_mean - (slope*X_mean)

print(f"Intercept is : {intercept}\nSlope is : {slope}")

#Medium

def train_ols(X,y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    numerator = 0
    denominator = 0
    for i in range(len(X)):
        numerator += (X[i]-X_mean) * (y[i] - y_mean)
        denominator += (X[i]-X_mean)**2
    slope = numerator / denominator
    intercept = y_mean - (slope * X_mean)
    return slope, intercept

m,b = train_ols(X,y)
print(f"The calculated slope (m) is: {m:.4f}")
print(f"The calculated intercept (b) is: {b:.4f}")

def calculate_mse(y,y_pred):
    n = len(y)
    error = 0
    for i in range(n):
        error += (y[i] - y_pred[i])
        error = error ** 2
    mse_val = error / n
    return mse_val

y_true_example = [1, 2, 3, 4]
y_pred_example = [1, 2.5, 2.5, 4.2]
mse = calculate_mse(y_true_example, y_pred_example)
print(f"Example MSE: {mse:.4f}")
#Hard

def gradient_descent_step(X, y, current_m, current_b, learning_rate):
    n = len(X)

    y_pred = (current_m * X) - current_b

    m_gradient = (-2/n) * np.sum(X * (y - y_pred))
    
    b_gradient = (-2/n) * np.sum(y - y_pred)

    new_m = current_m - (learning_rate * m_gradient)

    new_b = current_b - (learning_rate * b_gradient)

    return new_m, new_b

m_initial = 0
b_initial = 0
lr = 0.01

m_new, b_new = gradient_descent_step(X, y, m_initial, b_initial, lr)
print(f"After one step:")
print(f"New Slope: {m_new:.4f}")
print(f"New Intercept: {b_new:.4f}")