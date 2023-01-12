import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
from r_plot import r_plot

# Test Case
test_case = 'Lockdown'
print(f'Test Case: {test_case}')

# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
E0, I0, Q0, R0 = 0, 1, 0, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - E0 - I0 - Q0 - R0 
# Contact rate(beta), incubation period(phi, in 1/days), quarantine rate(zeta, in 1/days), mean recovery rate in non-quarantine(gamma, in 1/days), mean recovery rate in quarantine(kappa, in 1/days), and immunity wearoff rate(mu, in 1/days).
beta, phi, zeta, gamma, kappa, mu = 0.9*0.7, 1./10, (1./10), 1./10, 1./10, (1./60)
# A grid of time points (in days)
t = np.linspace(0, 160*10, 160*10)

print(f'Beta: {beta}, Gamma: {gamma}')
r0 = r_plot(S0, N, beta, gamma, kappa)
print(f'R0: {r0}')
if (r0) > 1:
    print('Epidemic!')

# The SEIQRS model differential equations.
def deriv(y, t, beta, phi, zeta, gamma, kappa, mu):
    S, E, I, Q, R = y
    dSdt = ((-beta * S * I)/N) + (mu * R)
    dEdt = ((beta * S * I)/N) - (phi * E)
    dIdt = (phi * E) - (gamma * I) - (zeta * I)
    dQdt = (zeta * I) - (kappa * Q)
    dRdt = (gamma * I) + (kappa * Q) - (mu * R)
    return dSdt, dEdt, dIdt, dQdt, dRdt

# Initial conditions vector
y0 = S0, E0, I0, Q0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(beta, phi, zeta, gamma, kappa, mu))
S, E, I, Q, R = ret.T

Se = (N*(gamma+zeta))/beta
Ie = (phi*kappa*mu*N*(beta-gamma-zeta))/(beta*((kappa*(gamma+zeta)*(mu+phi))+(phi*mu*(kappa+zeta))))
Ee = (beta*Se*Ie)/(N*phi)
Qe = (zeta*Ie)/(kappa)
Re = (Ie*(gamma+zeta))/mu

print(f'Se: {Se}, Ee: {Ee}, Ie: {Ie}, Qe: {Qe}, Re: {Re}')

# R Plot
r_vals = r_plot(S, N, beta, gamma, zeta)
plt.xlim(0, 160)
plt.ylim(0, 6)
plt.plot(t, r_vals, label = "Reproductive Number")
plt.axhline(y = 1, linestyle = '--')
plt.title(f'Reproductive Number over Time ({test_case})')
plt.xlabel('Time (Days)')
plt.ylabel('Reproductive Number')
plt.savefig(f'pics\{test_case} Case_R_plot.png')
plt.show()

#Plot Stacked Area Graph
plt.xlim(0, 150)
plt.ylim(0, 1200)
plt.stackplot(t, E, I, Q, S, R, labels=['Exposed', 'Infected', 'Quarantined','Susceptible','Recovered'])
plt.legend(loc='upper right')
plt.title(f'Stacked SEIQRS States over Time ({test_case} Case)')
plt.xlabel('Time (Days)')
plt.ylabel('Amount of Population (People)')
plt.savefig(f'pics\{test_case} Case_Stack.png')
plt.show()

#Plot Line Graph
plt.xlim(0, 150)
plt.ylim(0, 1200)
plt.plot(t, S, label = "Susceptible")
plt.plot(t, E, label = "Exposed")
plt.plot(t, I, label = "Infected")
plt.plot(t, Q, label = "Quarantined")
plt.plot(t, R, label = "Recovered")
plt.axhline(y = Se, linestyle = '--')
plt.axhline(y = Ee, linestyle = '--')
plt.axhline(y = Ie, linestyle = '--')
plt.axhline(y = Qe, linestyle = '--')
plt.axhline(y = Re, linestyle = '--')
plt.legend(loc='upper right')
plt.title(f'SEIQRS State over Time ({test_case} Case)')
plt.xlabel('Time (Days)')
plt.ylabel('Amount of Population (People)')
plt.savefig(f'pics\{test_case} Case.png')
plt.show()