import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
E0, I0, Q0, R0 = 0, 1, 0, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - E0 - I0 - Q0 - R0 
# Contact rate(beta), incubation period(phi, in 1/days), quarantine percentage(zeta), mean recovery rate in non-quarantine(gamma, in 1/days), mean recovery rate in quarantine(kappa, in 1/days), and immunity wearoff rate(mu, in 1/days).
beta, phi, zeta, gamma, kappa, mu = 0.75, 1./10, 0.25, 1./10, 1./10, 1./180
# A grid of time points (in days)
t = np.linspace(0, 160*10, 160*10)

print(f'Beta: {beta}, Gamma: {gamma}')
print(f'R0: {beta/gamma}')
if (beta/gamma) > 1:
    print('Epidemic!')

# The SEIQRS model differential equations.
def deriv(y, t, N, beta, phi, zeta, gamma, kappa, mu):
    S, E, I, Q, R = y
    dSdt = ((-beta * S * I) / N) + (mu * R)
    dEdt = ((beta * S * I) / N) - (phi * E)
    dIdt = (phi * E) - (gamma * I) - (zeta * I)
    dQdt = (zeta * I) - (kappa * Q)
    dRdt = (gamma * I) + (kappa * Q) - (mu * R)
    return dSdt, dEdt, dIdt, dQdt, dRdt

# Initial conditions vector
y0 = S0, E0, I0, Q0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, phi, zeta, gamma, kappa, mu))
S, E, I, Q, R = ret.T

#Plot Stacked Area Graph
plt.xlim(0, 160*10)
plt.ylim(0, 1200)
plt.stackplot(t, E, I, Q, S, R, labels=['Exposed', 'Infected', 'Quarantined','Susceptible','Recovered'])
plt.legend(loc='upper right')
plt.show()

#Plot Line Graph
# plt.xlim(0, 120)
# plt.ylim(0, 1200)
# plt.plot(t, S, label = "Susceptible")
# plt.plot(t, E, label = "Exposed")
# plt.plot(t, I, label = "Infected")
# plt.plot(t, Q, label = "Quarantined")
# plt.plot(t, R, label = "Recovered")
# plt.legend(loc='upper right')
# plt.show()