import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from itc_fit.parser import parse_itc_data
from itc_fit.utils import plot, report
from itc_fit.bootstrap import run_fitting, InputData, InitialGuess

def parse_arguments(argv) -> argparse.Namespace:

    ap = argparse.ArgumentParser(description="Fit ITC data using a bootstrapping approach.")
    ap.add_argument("-f", "--file", type=str, help="Raw ITC data", required=True)
    ap.add_argument(
        "-s",
        "--skip",
        help="Number of injections to skip (default: 0)",
        type=int,
        default=0,
        required=False,
    )
    ap.add_argument(
        "-t",
        "--temperature",
        help="Temperature of the experiment (default: 300.15 K)",
        type=float,
        default=300.15,
        required=False,
    )
    ap.add_argument(
        "-M",
        "--M0",
        help="Cell concentration in Molar (default: 0.005 M)",
        type=float,
        default=0.005,
        required=False,
    )
    ap.add_argument(
        "-X",
        "--X0",
        help="Injectant concentration in Molar (default: 0.075 M)",
        type=float,
        default=0.075,
        required=False,
    )
    ap.add_argument(
        "-V",
        "--V0",
        help="Volume of the ITC cell in liters (default: 0.000202 L)",
        type=float,
        default=0.202 / 1000,
        required=False,
    )
    return ap.parse_args(argv)


def main(argv):
    args = parse_arguments(argv)
    dQ, dV = parse_itc_data(args.file)

    input_data = InputData(
        dQ=dQ,
        dV=dV,
        temperature=args.temperature,
        skip=args.skip,
        cycles=1000,
        X0=args.X0,
        M0=args.M0,
        V0=args.V0,
    )

    # dQ and XM include the complete data from the ITC output file.
    # We refer to dQ as the heat release measured by the ITC and dH as the reaction enthalpy obtained from the fitting
    # process
    vardQ, XM, dH, K, N, dG, dG_sem = run_fitting(input_data, InitialGuess(dH=1000.0))

    report(
        input_data.V0,
        input_data.M0,
        input_data.X0,
        input_data.syringe_error,
        input_data.cell_error,
        input_data.heat_error,
        input_data.base_error,
        vardQ,
        dH,
        K,
        N,
        dG,
        dG_sem,
        input_data.temperature,
    )

    ITC = dQ / (dV * input_data.X0)

    plot(XM, ITC, vardQ, args.file, np.mean(dH), np.mean(K), np.mean(N), dG, input_data.skip)

if __name__ == "__main__":
    main(sys.argv[1:])
