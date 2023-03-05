import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
from r_plot import r_plot
from model import deriv

# Test Case
save = False
test_case = 'High Population'
print(f'Test Case: {test_case}')

# Total population, N.
N = 100000
# Initial number of infected and recovered individuals, I0 and R0.
E0, I0, Q0, R0, D0 = 0, 1, 0, 0, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - E0 - I0 - Q0 - R0 
# Contact rate(beta), incubation period(phi, in 1/days), quarantine rate(zeta, in 1/days), mean recovery rate in non-quarantine(gamma, in 1/days), mean recovery rate in quarantine(kappa, in 1/days), and immunity wearoff rate(mu, in 1/days).
beta, phi, zeta, gamma, kappa, mu, alpha, epsilon = 0.9, 1./10, (1./10), 1./10, 1./10, (1./60), 0.1, 0.1
# A grid of time points (in days)
t = np.linspace(0, 1600, 1600)

print(f'Beta: {beta}, Gamma: {gamma}')
r0 = r_plot(S0, N, beta, gamma, kappa, alpha)
print(f'R0: {r0}')
if (r0) > 1:
    print('Epidemic!')

# Initial conditions vector
y0 = S0, E0, I0, Q0, R0, D0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, phi, zeta, gamma, kappa, mu, alpha, epsilon))
S, E, I, Q, R, D = ret.T

Se = (N*(gamma+zeta))/beta
Ie = (phi*kappa*mu*N*(beta-gamma-zeta))/(beta*((kappa*(gamma+zeta)*(mu+phi))+(phi*mu*(kappa+zeta))))
Ee = (beta*Se*Ie)/(N*phi)
Qe = (zeta*Ie) / kappa
Re = (Ie*(gamma+zeta))/mu

print(f'Se: {Se}, Ee: {Ee}, Ie: {Ie}, Qe: {Qe}, Re: {Re}')

print(f'Imax = {max(I)}')
print(f'Cases(max) = {max(I+Q)}')

# R Plot
r_vals = r_plot(S, N, beta, gamma, zeta, alpha)
plt.plot(t, r_vals, label = "Reproductive Number")
plt.axhline(y = 1, linestyle = '--')
plt.title(f'Reproductive Number over Time ({test_case})')
plt.xlabel('Time (Days)')
plt.ylabel('Reproductive Number')
plt.autoscale()
if save:
    plt.savefig(f'pics\{test_case} Case_R_plot.png')
plt.show(block=False)

# Plot New Cases
plt.figure()
plt.plot(t, D)
plt.title(f'Cumulative Deaths over Time ({test_case})')
plt.xlabel('Time (Days)')
plt.ylabel('Deaths')
plt.autoscale()
if save:
    plt.savefig(f'pics\{test_case} Case_New_Cases.png')
plt.show(block=False)

#Plot Stacked Area Graph
plt.figure()
plt.stackplot(t, E, I, Q, S, R, D, labels=['Exposed', 'Infected', 'Quarantined','Susceptible','Recovered'])
plt.legend(loc='upper right')
plt.title(f'Stacked SEIQRS States over Time ({test_case} Case)')
plt.xlabel('Time (Days)')
plt.ylabel('Amount of Population (People)')
plt.autoscale()
if save:
    plt.savefig(f'pics\{test_case} Case_Stack.png')
plt.show(block=False)

#Plot Line Graph
plt.figure()
plt.plot(t, S, label = "Susceptible")
plt.plot(t, E, label = "Exposed")
plt.plot(t, I, label = "Infected")
plt.plot(t, Q, label = "Quarantined")
plt.plot(t, R, label = "Recovered")
plt.plot(t, D, label = "Dead")
# plt.axhline(y = Se, linestyle = '--')
# plt.axhline(y = Ee, linestyle = '--')
# plt.axhline(y = Ie, linestyle = '--')
# plt.axhline(y = Qe, linestyle = '--')
# plt.axhline(y = Re, linestyle = '--')
plt.legend(loc='upper right')
plt.title(f'SEIQRS State over Time ({test_case} Case)')
plt.xlabel('Time (Days)')
plt.ylabel('Amount of Population (People)')
plt.autoscale()
if save:
    plt.savefig(f'pics\{test_case} Case.png')
plt.show()