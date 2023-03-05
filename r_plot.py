def r_plot(susceptible, N, beta, gamma, zeta, alpha):
    return (beta * (susceptible/N)) / (gamma + zeta + alpha)
