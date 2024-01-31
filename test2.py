import numpy as np
import math

# Define the parameters
angle = 1.9768690590246878

# Solve for x
# x_values = np.arccos(3.141592653589793*2 - angle) / 31.41592653589793
x_values =0.13707430348215943

# Print the result
print("Possible values for x:", x_values)
s = 0.0780826346935732 *np.cos( 31.41592653589793 *x_values + 1.9768690590246878 )
print(s)

y = - angle / 31.41592653589793 + 0.2*4
print(y)

a=5.2644469875410635*np.cos(math.pi / 3)
print(a)


