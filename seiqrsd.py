import numpy as np
import pandas as pd

from scipy.integrate import odeint
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
from r_plot import r_plot
from model import deriv

# Test Case
save = True
test_case = 'Vaccination'
print(f'Test Case: {test_case}')

# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
E0, I0, Q0, R0, D0 = 0, 1, 0, 0, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - E0 - I0 - Q0 - R0 - D0
# Contact rate(beta), incubation period(phi, in 1/days), quarantine rate(zeta, in 1/days), mean recovery rate in non-quarantine(gamma, in 1/days), mean recovery rate in quarantine(kappa, in 1/days), immunity wearoff rate(mu, in 1/days), death rate in non-quarantine(alpha, in 1/days), and death rate in quarantine(epsilon, in 1/days).
beta, phi, zeta, gamma, kappa, mu, alpha, epsilon = 0.85*0.9, 1./10, (1./10), 1./10, 1./10, 0.85*(1./60), 0.002, 0.002
# A grid of time points (in days)
t = np.linspace(0, 150, 150)

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

# N Array
Se = (N*(gamma+zeta))/beta
Ie = (phi*kappa*mu*N*(beta-gamma-zeta))/(beta*((kappa*(gamma+zeta)*(mu+phi))+(phi*mu*(kappa+zeta))))
Ee = (beta*Se*Ie)/(N*phi)
Qe = (zeta*Ie) / kappa
Re = (Ie*(gamma+zeta))/mu
De = N - (Se+Ee+Ie+Qe+Re)

print(f'Se: {Se}, Ee: {Ee}, Ie: {Ie}, Qe: {Qe}, Re: {Re}, De: {D[-1]}')

print(f'Imax = {max(I)}')
print(f'Cases(max) = {max(I+Q)}')


# Save Data to xlsx
# df = pd.DataFrame({'Time': t, 'Susceptible': S, 'Exposed': E, 'Infected': I, 'Quarantined': Q, 'Recovered': R, 'Dead': D})
# df.to_excel(f'seiqrs.xlsx', index=False)

# R Plot
r_vals = r_plot(S, N, beta, gamma, zeta, alpha)
plt.plot(t, r_vals, label = "Reproductive Number")
# plt.axhline(y = r_plot(S[-1], N, beta, gamma, zeta, alpha), linestyle = '--')
plt.axhline(y = 1, linestyle = '--')
plt.autoscale()
# Y Axis Max 4.5
plt.ylim(plt.ylim()[0], 4.65)
plt.title(f'Reproductive Number over Time ({test_case})')
plt.xlabel('Time (Days)')
plt.ylabel('Reproductive Number')
if save:
    plt.savefig(f'pics\{test_case} Case_R_plot.png')
plt.show(block=False)

# Plot Cumulative Deaths
plt.figure()
plt.plot(t, D)
# plt.plot(t, De, 'tab:blue', linestyle = '--')
plt.title(f'Cumulative Deaths over Time ({test_case})')
plt.xlabel('Time (Days)')
plt.ylabel('Deaths')
plt.autoscale()
if save:
    plt.savefig(f'pics\{test_case} Case_New_Cases.png')
plt.show(block=False)

#Plot Stacked Area Graph
plt.figure()
plt.stackplot(t, E, I, Q, S, R, D, labels=['Exposed', 'Infected', 'Quarantined','Susceptible','Recovered', 'Dead'])
plt.legend(loc='upper right')
plt.title(f'Stacked SEIQRSD States over Time ({test_case} Case)')
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
plt.axhline(Se, linestyle = '--')
plt.axhline(Ee, linestyle = '--')
plt.axhline(Ie, linestyle = '--')
plt.axhline(Qe, linestyle = '--')
plt.axhline(Re, linestyle = '--')
plt.axhline(De, linestyle = '--')
plt.legend(loc='upper right')
plt.title(f'SEIQRSD State over Time ({test_case} Case)')
plt.xlabel('Time (Days)')
plt.ylabel('Amount of Population (People)')
plt.autoscale()
if save:
    plt.savefig(f'pics\{test_case} Case.png')
plt.show()