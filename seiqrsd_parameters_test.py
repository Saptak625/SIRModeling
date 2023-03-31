import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
from r_plot import r_plot
from model import deriv
from tqdm import trange

# Make a linespace of 100 points between range
parameter = np.linspace(1/100, 1/2, 100)
parameter_name = 'zeta'
s_eq = []
e_eq = []
i_eq = []
q_eq = []
r_eq = []
d_eq = []
cases = []

for i in trange(len(parameter)):
    # Total population, N.
    N = 1000
    # Initial number of infected and recovered individuals, I0 and R0.
    E0, I0, Q0, R0, D0 = 0, 1, 0, 0, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - E0 - I0 - Q0 - R0 - D0
    # Contact rate(beta), incubation period(phi, in 1/days), quarantine rate(zeta, in 1/days), mean recovery rate in non-quarantine(gamma, in 1/days), mean recovery rate in quarantine(kappa, in 1/days), immunity wearoff rate(mu, in 1/days), death rate in non-quarantine(alpha, in 1/days), and death rate in quarantine(epsilon, in 1/days).
    beta, phi, zeta, gamma, kappa, mu, alpha, epsilon = 0.9, 1./10, 1.3 * (1./10), 1./10, 1./10, (1./60), 0.002, 0.002

    if parameter_name == 'beta':
        beta = parameter[i]
    elif parameter_name == 'mu':
        mu = parameter[i]
    elif parameter_name == 'zeta':
        zeta = parameter[i]

    # A grid of time points (in days)
    t = np.linspace(0, 500, 500)

    # print(f'Beta: {beta}, Gamma: {gamma}')
    r0 = r_plot(S0, N, beta, gamma, kappa, alpha)
    # print(f'R0: {r0}')
    # if (r0) > 1:
    #     print('Epidemic!')

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
    De = 0

    # print(f'Se: {Se}, Ee: {Ee}, Ie: {Ie}, Qe: {Qe}, Re: {Re}, De: {D[-1]}')

    # print(f'Imax = {max(I)}')
    # print(f'Cases(max) = {max(I+Q)}')

    s_eq.append(S[-1])
    e_eq.append(E[-1])
    i_eq.append(I[-1])
    q_eq.append(Q[-1])
    r_eq.append(R[-1])
    d_eq.append(D[-1])
    cases.append(max(I+Q))

    if i % 10 == 0:
        plt.figure()
        plt.plot(t, S, label = "Susceptible")
        plt.plot(t, E, label = "Exposed")
        plt.plot(t, I, label = "Infected")
        plt.plot(t, Q, label = "Quarantined")
        plt.plot(t, R, label = "Recovered")
        plt.plot(t, D, label = "Dead")
        plt.axhline(y = Se, linestyle = '--')
        plt.axhline(y = Ee, linestyle = '--')
        plt.axhline(y = Ie, linestyle = '--')
        plt.axhline(y = Qe, linestyle = '--')
        plt.axhline(y = Re, linestyle = '--')
        plt.axhline(y = De, linestyle = '--')
        plt.legend(loc='upper right')
        plt.title(f'SEIQRSD State over Time with {parameter_name} = {parameter[i]:.4f}')
        plt.xlabel('Time (Days)')
        plt.ylabel('Amount of Population (People)')
        plt.autoscale()
        plt.savefig(rf'model_parameters/{parameter_name}_graphs/graph_{parameter[i]:.4f}.png')
        plt.close()

for name, data in {'S Equilibrium': s_eq, 'E Equilibrium': e_eq, 'I Equilibrium': i_eq, 'Q Equilibrium': q_eq, 'R Equilibrium': r_eq, 'D Equilibrium': d_eq, 'Max Cases': cases}.items():
    plt.figure()
    plt.plot(parameter, data)
    plt.xlabel(parameter_name)
    plt.ylabel(name + ' (People)')
    plt.title(f'{name} vs {parameter_name}')

    # Save figure as png and close figure
    plt.savefig(rf'model_parameters/{parameter_name}/{name}.png')
    plt.close()
