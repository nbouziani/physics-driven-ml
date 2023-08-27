import numpy as np

# Loading
X_train = np.load("../../data/datasets/linear_elasticity/X.npy", allow_pickle=True)
y_train = np.load("../../data/datasets/linear_elasticity/y.npy", allow_pickle=True)
num_samples_to_display = 1

for i in range(num_samples_to_display):
    a11, a22, a12 = X_train[i]
    stress_xx, stress_yy, stress_xy = y_train[i][0], y_train[i][1], y_train[i][2]
    stress = np.array([[stress_xx, stress_xy], [stress_xy, stress_yy]])
    print(f"Sample {i + 1}:")
    print("-----------")
    print(f"Strain:\n[[{a11:.3f}, {a12:.3f}]\n [{a12:.3f}, {a22:.3f}]]")
    print(f"Stress:\n{stress}")
    print()
