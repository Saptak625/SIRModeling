# Import required modules
import numpy as np
import matplotlib.pyplot as plt

# 1D arrays
x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)

# Meshgrid
X,Y = np.meshgrid(x,y)

# Assign vector directions
Ex = (X + 1)/((X+1)**2 + Y**2) - (X - 1)/((X-1)**2 + Y**2)
Ey = Y/((X+1)**2 + Y**2) - Y/((X-1)**2 + Y**2)

# Depict illustration
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,Ex,Ey, density=1.4, linewidth=None, color='#A23BEC')
plt.plot(-1,0,'-or')
plt.plot(1,0,'-og')
plt.title('Electromagnetic Field')

# Show plot with grid
plt.grid()
plt.show()
