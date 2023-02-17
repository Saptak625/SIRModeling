# The SEIQRS model differential equations.
def deriv(y, t, N, beta, phi, zeta, gamma, kappa, mu):
    S, E, I, Q, R = y
    dSdt = ((-beta * S * I)/N) + (mu * R)
    dEdt = ((beta * S * I)/N) - (phi * E)
    dIdt = (phi * E) - (gamma * I) - (zeta * I)
    dQdt = (zeta * I) - (kappa * Q)
    dRdt = (gamma * I) + (kappa * Q) - (mu * R)
    return dSdt, dEdt, dIdt, dQdt, dRdt