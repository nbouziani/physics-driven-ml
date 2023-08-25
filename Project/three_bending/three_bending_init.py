import numpy as np
import matplotlib.pyplot as plt

# Material properties
Youngs_modulus = 210e9  # Pa (Pascals)
Poissons_ratio = 0.3

# Geometric properties
length = 0.1  # meters
width = 0.01  # meters
thickness = 0.005  # meters

# Applied force
force = 1000  # Newtons

# Calculate moment of inertia (assuming rectangular cross-section)
moment_of_inertia = (width * thickness ** 3) / 12

# Calculate displacement at the center using beam bending formula
displacement = (force * length ** 3) / (3 * Youngs_modulus * moment_of_inertia)

# Calculate strain and stress
strain = displacement / length
stress = Youngs_modulus * strain

print("Displacement at the center:", displacement)
print("Strain:", strain)
print("Stress:", stress)

# Plot stress-strain curve
strain_values = np.linspace(0, strain, 100)
stress_values = strain_values * Youngs_modulus
plt.plot(strain_values, stress_values)
plt.xlabel("Strain")
plt.ylabel("Stress (Pa)")
plt.title("Stress-Strain Curve")
plt.grid()
plt.show()
