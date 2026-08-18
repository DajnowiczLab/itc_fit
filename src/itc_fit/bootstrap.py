import numpy as np
from typing import NamedTuple
from functools import partial
from scipy.optimize import curve_fit
from tqdm import tqdm

# 01/31/19 Rejection of bad fits based on sum of squared errors
# Reference: http://www.isbg.fr/IMG/pdf/microcal-itc200-system-user-manual-malvern.pdf
# ITC settings
# XM is the variable combining the concentration of the guest (X) and the host (M)
# according to the manual referenced above
# 05/08/19 Changed how the code deals with skipping of the first n data points
# Renamed variables

class InitalGuess(NamedTuple):
    dH: float = -1000.0
    K: float = 1000.0
    N: float = 1.0

class FitResult(NamedTuple):
    dH: float
    K: float
    N: float
    dG: float
    dG_sem: float
    bootstrap_dQ: np.ndarray
    XM: np.ndarray

class InputData(NamedTuple):
    dQ: np.ndarray # cal
    dV: np.ndarray # L
    temperature: float # K
    skip: int
    cycles: int
    X0: float # Injectant concentration in Molar 
    M0: float # Cell concentration in Molar
    V0: float # Volume of the ITC cell in liters
    syringe_error: float = 0.02
    cell_error: float = 0.02
    heat_error: float = 0.01
    base_error: float = 0.000_000_15

def fit(XMa: np.ndarray, dH: float, K: float, N: float, input_data: InputData) -> np.ndarray:
    """
    Fit dQ given dH, K, N.
    """
    injections = len(XMa[0])
    Q = np.zeros(injections)  # Total Heat Array Initialized
    Q[0] = 0.0  # Total Heat before injections is zero
    dQ = np.zeros(injections)
    X = XMa[0, :]
    M = XMa[1, :]
    for i in range(1, injections):  # Loop over injections
        # Total Heat Equation 9
        Q[i] = (N * M[i] * dH * input_data.V0 / 2) * (
            1
            + (X[i] / (N * M[i]))
            + 1 / (N * K * M[i])
            - np.sqrt(
                ((1 + X[i] / (N * M[i]) + 1 / (N * K * M[i])) ** 2)
                - 4 * X[i] / (N * M[i])
            )
        )
        # Change in heat normalized by amount of injectant
        # Similar to equation 10 except for the normalization factor dV*X0.
        # Now the unit is cal/mol; same unit as for exp_dQ_normalized
        dQ[i] = (Q[i] + (input_data.dV[i - 1] / input_data.V0) * ((Q[i] + Q[i - 1]) / 2) - Q[i - 1]) / (
            input_data.dV[i - 1] * input_data.X0
        )
    # We are not fitting the first skip points and and heat release before injection 1 is 0.0
    return dQ[input_data.skip + 1 :]


def run_fitting(input_data: InputData, guess: InitalGuess) -> FitResult:
    """
    Bootstrap the fitting of dQ given re-sampled uncertainties.
    """
    # Increase the number of bootstrapping runs by 10 percent. We delete the bad fits afterwards and want to fill up
    # the array so that we always use the same number of bootstrapping cycles
    realcycles = input_data.cycles
    cycles = int(input_data.cycles * 1.3)
    # All variables which are different between different cycles of the bootstrapping process and stored  start with a
    # leading 'bootstrap_'
    # bootstrap_heat = np.zeros([len(dQ)-skip, cycles]) I think this is unnecessary
    bootstrap_dH = np.zeros([cycles])
    bootstrap_K = np.zeros([cycles])
    bootstrap_N = np.zeros([cycles])
    bootstrap_SS = np.zeros([cycles])
    # We are not fitting the first 'skip' elements. Therefore the array of fitted dQ is smaller.
    # As we are only fitting the differences we can ignore the first skip elements in the fitting easily
    # For every cycle we are storing the full set of calculate dQ values
    bootstrap_dQ = np.zeros([len(input_data.dQ) - input_data.skip, cycles])
    for cycle in tqdm(range(cycles)):
        # Picking concentrations from gaussian distribution
        sampled_syringe_concentration = np.random.normal(input_data.X0, 
                                                         abs(input_data.syringe_error * input_data.X0))
        sampled_cell_concentration = np.random.normal(input_data.M0, 
                                                      abs(input_data.cell_error * input_data.M0))
        # Find concentrations after each injection
        XM = np.zeros([2, len(input_data.dQ) + 1])
        # Concentration beore 1st injection
        XM[0, 0] = 0  # Guest concentration in cell
        XM[1, 0] = input_data.M0  # Starting host concentration in cell
        cumulative_volume = np.cumsum(input_data.dV)
        # New Injectant Concentration Equation 4
        XM[0, 1:] = (cumulative_volume * sampled_syringe_concentration / input_data.V0) * (
            1 / (1 + (cumulative_volume / (2 * input_data.V0)))
        )
        # New Cell Molecule Concentration Equation 2
        XM[1, 1:] = sampled_cell_concentration * (
            (1 - cumulative_volume / (2 * input_data.V0)) / (1 + cumulative_volume / (2 * input_data.V0))
        )
        # Add heat error
        exp_dQ = [
            np.random.normal(
                injection,
                abs(np.sqrt(((injection * input_data.heat_error) ** 2) + ((input_data.base_error) ** 2))),
            )
            for injection in input_data.dQ
        ]
        # Scale nominal Wiseman plot by new bootstrapped syringe concentration
        exp_dQ_normalized = [
            injection / (volume * sampled_syringe_concentration)
            for injection, volume in zip(exp_dQ, input_data.dV)
        ]
        # Fit the data
        # We are only fitting the experimental heat realizes after skip. Therefore, ignore the first 'skip'
        # heat releases
        # XM still has all datapoints
        initial_guess = np.array([guess.dH, guess.K, guess.N])
        fit_func = partial(fit, input_data=input_data)
        try:
            fitting_variables, _ = curve_fit(
                fit_func, XM, exp_dQ_normalized[input_data.skip:], initial_guess, maxfev=100
            )
        except RuntimeError:
            # print("Curve fit failure. Possibly weak binder.")
            fitting_variables, _ = curve_fit(
                fit_func, XM, exp_dQ_normalized[input_data.skip:], initial_guess, maxfev=10000
            )
            pass
        dH = fitting_variables[0]
        K = fitting_variables[1]
        N = fitting_variables[2]

        # (Print) Original Data, Fit, and find SumSqr
        fitdQ = fit(XM, dH, K, N, input_data)

        SumSqr = 0.0
        for i in range(input_data.skip, len(input_data.dQ)):
            SumSqr += (exp_dQ_normalized[i] - fitdQ[i - input_data.skip]) ** 2

        bootstrap_dH[cycle] = dH
        bootstrap_K[cycle] = K
        bootstrap_N[cycle] = N
        bootstrap_SS[cycle] = SumSqr
        bootstrap_dQ[:, cycle] = fitdQ

    # Reject all values which are over a threshold; definition is arbitrary.
    threshold = 1 * bootstrap_SS.mean() + 7.0 * np.sqrt(bootstrap_SS.std())
    for cycle in range(cycles - 1, -1, -1):
        if bootstrap_SS[cycle] > threshold:
            bootstrap_dH = np.delete(bootstrap_dH, cycle)
            bootstrap_K = np.delete(bootstrap_K, cycle)
            bootstrap_N = np.delete(bootstrap_N, cycle)
            bootstrap_SS = np.delete(bootstrap_SS, cycle)
            bootstrap_dQ = np.delete(bootstrap_dQ, cycle, 1)
    # Only use number of bootstrapping cycles defined in the main program.
    bootstrap_dH = bootstrap_dH[:realcycles]
    bootstrap_K = bootstrap_K[:realcycles]
    bootstrap_N = bootstrap_N[:realcycles]
    bootstrap_SS = bootstrap_SS[:realcycles]
    bootstrap_dQ = bootstrap_dQ[:realcycles]

    # Shortcut to get the exact uncertainty in delta G assuming both delta G and K are well-behaved Gaussians.
    # This could be resampled.
    R = 1.987_203_6 * 10 ** -3  # kcal K^-1 mol^-1
    dG_sem = R * input_data.temperature * np.std(bootstrap_K) / np.mean(bootstrap_K)
    dG = -R * input_data.temperature * np.log(np.mean(bootstrap_K))

    return bootstrap_dQ, XM, bootstrap_dH, bootstrap_K, bootstrap_N, dG, dG_sem
