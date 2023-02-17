import numpy as np
import pandas as pd
from tqdm import tqdm
import time
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
    covid_data = pd.read_csv("https://tinyurl.com/t59cgxn", parse_dates=["Date"], skiprows=[1])
    covid_data["Location"] = covid_data["Country/Region"]
    pbar.update(1)
    time.sleep(0.1)


agegroup_lookup = dict(zip(agegroups['Location'], agegroups[['0_9', '10_19', '20_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80_89', '90_100']].values))

# parameters
data = covid_data[covid_data["Location"] == "Italy"]["Value"].values[::-1]
agegroups = agegroup_lookup["Italy"]
outbreak_shift = 30
params_init_min_max = {"beta": (0.9, 0.1, 2), "zeta": (1./10, 1./50, 1), "mu": (1./60, 1./300, 1./5)}  # form: {parameter: (initial guess, minimum value, max value)}
     

days = outbreak_shift + len(data)
if outbreak_shift >= 0:
    y_data = np.concatenate((np.zeros(outbreak_shift), data))
else:
    y_data = y_data[-outbreak_shift:]

x_data = np.linspace(0, days - 1, days, dtype=int)  # x_data is just [0, 1, ..., max_days] array

# Given Model Parameters (Based on COVID-19 Research Data)
phi = 1/9
gamma = 1/3
kappa = 1/3

# Must Fit Beta, Zeta, Mu
def fitter(x, beta, zeta, mu):
    ret = Model(days, agegroups, beta, phi, zeta, gamma, kappa, mu)
    # Model returns bit tuple. 7-th value (index=6) is list with deaths per day.
    deaths_predicted = ret[5]
    # print(deaths_predicted)
    # print(x)

    # print()
    return deaths_predicted


# Fit the model
mod = lmfit.Model(fitter)

for kwarg, (init, mini, maxi) in params_init_min_max.items():
    mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)

params = mod.make_params()
fit_method = "leastsq"
     

result = mod.fit(y_data, params, method=fit_method, x=x_data)
     

result.plot_fit(datafmt="-")
plt.show()