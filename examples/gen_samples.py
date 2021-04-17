import numpy as np
import pandas as pd
import scipy.stats as stat
import xarray as xr

import sys
from runBatch import *

beta_m = np.array([0.02584, 0.152, 0.13908, 0.30704, 0.1102, 0.02584])/100
beta_u = {}
beta_u["tuttle"] = np.array([7.72, 3.42, 7.46, 2.53, 5.24, 12.45])/100
beta_u["charlton"] = np.array([5.27, 2.03, 5.27, 1.89, 4.92, 13.5])/100
beta_u["mills"]   = np.array([2.78,2.15,5.89, 2.28, 6.45, 5.70])/100
beta_u["saleh"]   = np.array([9.02, 7.73, 8.25, 8.58, 7.30, 14.75])/100
beta_u["waldo"]   = np.array([8.01, 6.68, 7.99, 3.11, 5.50, 8.42])/100
beta_u["syentos"] = np.array([5.83, 3.76, 10.45, 3.17, 6.41, 12.76])/100
beta_u["radaideh"] = np.array([8.5, 6.3, 9.6, 7.5, 12.8, 16.9])/100
beta_u["radaideh_eol"] = np.array([15.9, 15.6, 16.5, 15.0, 20.3, 23.6])/100

#converting from relative uncertainty to variance
for key, value in beta_u.items():
    beta_u[key] = beta_m*value

lambda_m = np.array([0.0128, 0.0318, 0.119, 0.3181, 1.4027, 3.9286])
lambda_u = {}
lambda_u["tuttle"] = np.array([1.95, 2.66, 2.74, 2.95, 8.36, 7.19])/100
lambda_u["charlton"] = np.array([0.80, 1.89, 1.49, 2.31, 4.58, 6.48])/100
lambda_u["mills"] = np.array([0.79, 0.87, 2.84, 2.95, 5.23, 3.25])/100
lambda_u["saleh"] = np.array([5.89, 0.78, 4.36, 1.73, 3.22, 6.50])/100
lambda_u["waldo"] = np.array([1.91, 3.08, 2.61, 3.17, 5.84, 9.72])/100
lambda_u["syentos"] = np.array([0.81, 1.19, 7.42, 5.3, 8.18, 6.40])/100
lambda_u["radaideh"] = np.array([2.0, 2.7, 2.9, 3.2, 8.7, 7.9])/100
lambda_u["radaideh_eol"] = np.array([1.8, 2.4, 3.1, 3.2, 6.6, 7.6])/100

for key, value in lambda_u.items():
    lambda_u[key] = lambda_m*value

sets = list(lambda_u.keys())

corr = pd.read_csv("UQSA/corr.dat").values

def runsample(lambdas, betas):
    data, _, _ = loadData("cr_ejection_parcs.txt", 1000)
    pset = [Precursor(a, b) for a, b in zip(lambdas, betas)]
    data.setBeff(pset)
    data.write(pset, "epke_input.xml")
    runSolver("./epke-run", "examples/epke_input.xml")
    t, rho, p = extract_output("epke_output.xml")
    return t, rho, p

def gen_samples(sample_set, correlation = True, n_samples = 1):
    sd_vec = np.zeros(12)
    sd_vec[:6] = beta_u[sample_set]
    sd_vec[6:] = lambda_u[sample_set]
    if correlation:
        c = corr.copy()
    else:
        c = np.eye(12)

    cov = np.diag(sd_vec)@c@np.diag(sd_vec)

    m = np.zeros(12)
    m[:6] = beta_m
    m[6:] = lambda_m

    complete = np.random.multivariate_normal(m, cov, n_samples)
    betas = complete[:, :6]
    lambdas = complete[:, 6:]
    return betas, lambdas

samples = 1000
ctitles = ["sample" + str(a).zfill(4) for a in range(samples)]
btitles = ["beta_" + str(a + 1) for a in range(6)]
ltitles = ["lambda_" + str(a+1) for a in range(6)]
for s in sets:
    #with correlation
    betas, lambdas = gen_samples(s, True, samples)

    bdf = pd.DataFrame(betas.T, index = btitles, columns = ctitles)
    bdf.to_csv("UQSA/samples/betas_" + s + ".csv")
    ldf = pd.DataFrame(lambdas.T, index = ltitles, columns = ctitles)
    ldf.to_csv("UQSA/samples/lambdas_" + s + ".csv")

    rho = np.zeros((6000, samples))
    p = np.zeros((6000, samples))
    for i in range(betas.shape[0]):
        print(s + ": " + str(i) + "/ " + str(samples))
        t, rho[:, i], p[:, i] = runsample(lambdas[i], betas[i])
    rhout = pd.DataFrame(rho, columns = ctitles, index = t)
    pout = pd.DataFrame(p, columns = ctitles, index = t)
    rhout.to_csv("UQSA/samples/rho_" + s + ".csv")
    pout.to_csv("UQSA/samples/p_" + s + ".csv")

    #no correlation
    betas, lambdas = gen_samples(s, False, samples)

    bdf = pd.DataFrame(betas.T, index = btitles, columns = ctitles)
    bdf.to_csv("UQSA/samples/betas_" + s + "_nocorr.csv")
    ldf = pd.DataFrame(lambdas.T, index = ltitles, columns = ctitles)
    ldf.to_csv("UQSA/samples/lambdas_" + s + "_nocorr.csv")

    rho = np.zeros((6000, samples))
    p = np.zeros((6000, samples))
    for i in range(betas.shape[0]):
        print(s + "_nocorr: " + str(i) + "/ " + str(samples))
        t, rho[:, i], p[:, i] = runsample(lambdas[i], betas[i])
    rhout = pd.DataFrame(rho, columns = ctitles, index = t)
    pout = pd.DataFrame(p, columns = ctitles, index = t)
    rhout.to_csv("UQSA/samples/rho_" + s + "_nocorr.csv")
    pout.to_csv("UQSA/samples/p_" + s + "_nocorr.csv")
