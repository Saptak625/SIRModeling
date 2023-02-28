import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import datetime
from matplotlib import pyplot as plt
import lmfit

from fit_model import Model

# read the data
print("Loading data...")
time.sleep(0.5)
with tqdm(total=2) as pbar:
    # beds = pd.read_csv("https://raw.githubusercontent.com/hf2000510/infectious_disease_modelling/master/data/beds.csv", header=0)
    agegroups = pd.read_csv("https://raw.githubusercontent.com/hf2000510/infectious_disease_modelling/master/data/agegroups.csv")
    pbar.update(1)
    time.sleep(0.1)
    # probabilities = pd.read_csv("https://raw.githubusercontent.com/hf2000510/infectious_disease_modelling/master/data/probabilities.csv")
    covid_data = pd.read_csv("https://covid19.who.int/WHO-COVID-19-global-data.csv", parse_dates=["Date_reported"])
    covid_data["Location"] = covid_data["Country"]
    pbar.update(1)
    time.sleep(0.1)

agegroup_lookup = dict(zip(agegroups['Location'], agegroups[['0_9', '10_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80_89', '90_100']].values))

# parameters
country = "Philippines"
country_data = covid_data[covid_data["Location"] == "Philippines"]
data = country_data["New_cases"].values
times = country_data["Date_reported"].values
plt.title(f'New Cases COVID-19 Data ({country})')
plt.xlabel(f"Time in Days ({datetime.datetime.utcfromtimestamp(times[0].tolist()/1e9).date()} - {datetime.datetime.utcfromtimestamp(times[-1].tolist()/1e9).date()})")
plt.ylabel('New Cases')
plt.plot(data)
plt.autoscale()
plt.savefig(f'fitting\\new pics\{country}_new_cases.png')
plt.show(block=False)
# data = data[:len(data)//3]
# start at nth day
saved_data = data
data = data[:200]
# Moving Average
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
data = moving_average(data, 20)
times = times[:200]
print(len(data))
data -= data[0] # start cumulative sum at 0
agegroups = agegroup_lookup["Philippines"]
outbreak_shift = 250  # shift the outbreak by this many days (negative values are allowed)
params_init_min_max = {"beta": (0.9, 0.1, 2), "zeta": (1./10, 1./50, 1), "mu": (1./60, 1./300, 1./5)}  # form: {parameter: (initial guess, minimum value, max value)}
     

days = outbreak_shift + len(data)
if outbreak_shift >= 0:
    y_data = np.concatenate((np.zeros(outbreak_shift), data))
else:
    y_data = data[-outbreak_shift:]

x_data = np.linspace(0, days - 1, days, dtype=int)  # x_data is just [0, 1, ..., max_days] array

# Given Model Parameters (Based on COVID-19 Research Data)
phi = 1/9
gamma = 1/3
kappa = 1/3

# Must Fit Beta, Zeta, Mu
def fitter(x, beta, zeta, mu):
    ret = Model(days, agegroups, beta, phi, zeta, gamma, kappa, mu)
    exposed = ret[3]
    # print(deaths_predicted)
    # print(x)

    # print()
    return phi*exposed # Inflow of Infections aka New Cases


# Fit the model
mod = lmfit.Model(fitter)

for kwarg, (init, mini, maxi) in params_init_min_max.items():
    # mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)
    mod.set_param_hint(str(kwarg), value=init, min=0, vary=True)

params = mod.make_params()
fit_method = "leastsq"
     

result = mod.fit(y_data, params, method=fit_method, x=x_data)
     
plt.figure()
result.plot_fit(datafmt="-", xlabel=f"Time in Days ({datetime.datetime.utcfromtimestamp(times[0].tolist()/1e9).date()} - {datetime.datetime.utcfromtimestamp(times[-1].tolist()/1e9).date()}) (Outbreak Shift = {outbreak_shift})", ylabel="New Cases", title=f"Fitting SEIQRS Model to {country} COVID-19 Data")
plt.savefig(f"fitting\\new pics\\fit_{country}.png")
print(result.best_values)
plt.show(block=True)

# Using the fitted parameters to model the future outbreak
beta = result.best_values["beta"]
zeta = result.best_values["zeta"]
mu = result.best_values["mu"]
t, N, S, E, I, Q, R, r_vals = Model(days+700, agegroups, beta, phi, zeta, gamma, kappa, mu)

# Undo Modelling Outbreak Shift
t = t[outbreak_shift:]
t -= t[0] + 1 # shift the x axis to start at 1
S = S[outbreak_shift:]
E = E[outbreak_shift:]
I = I[outbreak_shift:]
Q = Q[outbreak_shift:]
R = R[outbreak_shift:]
r_vals = r_vals[outbreak_shift:]

Se = (N*(gamma+zeta))/beta
Ie = (phi*kappa*mu*N*(beta-gamma-zeta))/(beta*((kappa*(gamma+zeta)*(mu+phi))+(phi*mu*(kappa+zeta))))
Ee = (beta*Se*Ie)/(N*phi)
Qe = (zeta*Ie)/(kappa)
Re = (Ie*(gamma+zeta))/mu

print(f'Se: {Se}, Ee: {Ee}, Ie: {Ie}, Qe: {Qe}, Re: {Re}')

print(f'Imax = {max(I)}')
print(f'Cases(max) = {max(I+Q)}')

# R Plot
plt.figure()
plt.plot(t, r_vals, label = "Reproductive Number")
plt.axhline(y = 1, linestyle = '--')
plt.title(f'Reproductive Number over Time ({country})')
plt.xlabel('Time (Days)')
plt.ylabel('Reproductive Number')
plt.autoscale()
plt.savefig(f'fitting\\new pics\{country}_R_plot.png')
plt.show(block=False)

#Plot Stacked Area Graph
plt.figure()
plt.stackplot(t, E, I, Q, S, R, labels=['Exposed', 'Infected', 'Quarantined','Susceptible','Recovered'])
plt.legend(loc='upper right')
plt.title(f'Stacked SEIQRS States over Time ({country})')
plt.xlabel('Time (Days)')
plt.ylabel('Amount of Population (People)')
plt.autoscale()
plt.savefig(f'fitting\\new pics\{country}_Stack.png')
plt.show(block=False)

#Plot Line Graph
plt.figure()
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
plt.title(f'SEIQRS State over Time ({country})')
plt.xlabel('Time (Days)')
plt.ylabel('Amount of Population (People)')
plt.autoscale()
plt.savefig(f'fitting\\new pics\{country}.png')
plt.show(block=False)

#Plot Cumulative Infections Predicted
t, N, S, E, I, Q, R, r_vals = Model(len(saved_data)+outbreak_shift, agegroups, beta, phi, zeta, gamma, kappa, mu)

# Undo Modelling Outbreak Shift
t = t[outbreak_shift:]
t -= t[0] + 1 # shift the x axis to start at 1
S = S[outbreak_shift:]
E = E[outbreak_shift:]
I = I[outbreak_shift:]
Q = Q[outbreak_shift:]
R = R[outbreak_shift:]
r_vals = r_vals[outbreak_shift:]

plt.figure()
plt.plot(t, saved_data, label = "Actual New Cases")
plt.plot(t, phi*E, label = "Predicted New Cases")
plt.legend(loc='upper left')
plt.title(f'Predicted New Cases over Time ({country})')
plt.xlabel('Time (Days)')
plt.ylabel('New Cases (People)')
plt.autoscale()
plt.savefig(f'fitting\\new pics\{country}_New_cases.png')
plt.show()