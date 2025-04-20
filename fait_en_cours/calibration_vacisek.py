
import numpy as np
from sklearn.linear_model import LinearRegression

def vacisek(r,a,b,sigma,T,num_steps,num_paths):
    dt = T/num_steps
    rates = np.zeros((num_steps+1,num_paths))
    rates[0] = r

    for t in range(1,num_steps+1):
        dw = np.random.normal(0,1,num_paths)
        rates[t] = rates[t-1] + a*(k-rates[t-1])*dt + sigma*np.sqrt(dt)*dw
    return rates

r = 0.03
T = 10
num_steps = 1000
num_paths = 1
a = 0.1
k = 0.05
sigma = 0.01

simulated_rates = vacisek(r,a,k,sigma,T,num_steps,num_paths)
print(simulated_rates)

def calib_vacisek(r,dt):

    # Prepare the regression lineaire
    r_t = r[:-1]

    r_t1 = r[1:]

    reg = LinearRegression().fit(r_t, r_t1)
    print(reg)

calib_vacisek(simulated_rates[0],T/num_steps)