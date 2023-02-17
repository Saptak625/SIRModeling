from scipy.integrate import odeint
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import deriv
from r_plot import r_plot

def Model(days, agegroups, beta, phi, zeta, gamma, kappa, mu):
    # agegroups is list with number of people per age group -> sum to get population
    N = sum(agegroups)  

    y0 = N-1.0, 1.0, 0.0, 0.0, 0.0  # one exposed, everyone else susceptible
    t = np.linspace(0, days, days)
    ret = odeint(deriv, y0, t, args=(N, beta, phi, zeta, gamma, kappa, mu))
    S, E, I, Q, R = ret.T

    R_0_over_time = r_plot(S, N, beta, gamma, zeta)  # get R0 over time for plotting

    return t, S, E, I, Q, R, R_0_over_time