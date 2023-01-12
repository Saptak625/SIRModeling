def r_plot(susceptible, N, beta, gamma, kappa):
    return (beta * (susceptible/N)) / (gamma + kappa)
