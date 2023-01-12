import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.75, 1./10 
# A grid of time points (in days)
t = np.linspace(0, 160, 160)

print(f'Beta: {beta}, Gamma: {gamma}')
print(f'R0: {beta/gamma}')
if (beta/gamma) > 1:
    print('Epidemic!')

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# R Plot
r = (beta*S)/(gamma*N)
plt.xlim(0, 150)
plt.ylim(0, 10)
plt.plot(t, r, label = "Reproductive Number")
plt.axhline(y = 1, linestyle = '--')
plt.legend(loc='upper right')
plt.show()

# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.xlim(0, 150)
plt.ylim(0, 1200)
plt.plot(t, S, label = "Susceptible")
plt.plot(t, I, label = "Infected")
plt.plot(t, R, label = "Recovered")
plt.legend(loc='upper right')
plt.show()


#Plot Vector Field
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.xlim(0, 20)
# plt.ylim(0, 1200)
# ax.axes.set_xlim3d(left=0, right=50) 
# ax.axes.set_ylim3d(bottom=0, top=50) 
# ax.axes.set_zlim3d(bottom=50, top=100) 
# plt.quiver(s, i, u, v)
# plt.streamplot(S, I, U, V)
# plt.stackplot(t, I, S, R, labels=['Infected','Susceptible','Recovered'])
# plt.legend(loc='upper right')
# plt.show()