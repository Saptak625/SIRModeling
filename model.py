# The SEIQRS model differential equations.
def deriv(y, t, N, beta, phi, zeta, gamma, kappa, mu, alpha, epsilon):
    S, E, I, Q, R, D = y
    dSdt = ((-beta * S * I)/N) + (mu * R)
    dEdt = ((beta * S * I)/N) - (phi * E)
    dIdt = (phi * E) - (gamma * I) - (zeta * I) - (alpha * I)
    dQdt = (zeta * I) - (kappa * Q) - (epsilon * Q)
    dRdt = (gamma * I) + (kappa * Q) - (mu * R)
    dDdt = (alpha * I) + (epsilon * Q)
    return dSdt, dEdt, dIdt, dQdt, dRdt, dDdt
