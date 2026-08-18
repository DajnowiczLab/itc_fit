from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot(XM, ITC, vardQ, name, dH, K, N, dG, skip):
    """
    Plot the heat as a function of molar ratio, as measured, and the final (?) curve fit.
    I think this takes the best fit from the last bootstrapped cycle, but I'm not sure.
    """
    fig, ax = plt.subplots(1, figsize=(6 * 1.2, 6))
    ax.scatter(XM[0, 1:] / XM[1, 1:], ITC, c="k", label="ITC data")
    for index, dQ in enumerate(vardQ.T):
        ax.plot(
            XM[0, 1 + skip :] / XM[1, 1 + skip :],
            dQ,
            c="r",
            label="Equation fit" if index == 0 else "",
            alpha=0.2,
            zorder=-1,
        )

    ax.set_xlabel("Molar ratio")
    ax.set_ylabel("Heat (cal/mol)")
    ax.set_title(
        r"dH = {:9.2f} cal/mol, K = {:9.2f} M$^{{-1}}$, N = {:5.3f}, dG = {:9.2f} kcal/mol".format(
            dH, K, N, dG
        )
    )
    ax.legend(loc=4)
    fig.savefig("{}.png".format(Path(name).stem), bbox_inches="tight")

def report(
    V0,
    M0,
    X0,
    syringe_error,
    cell_error,
    heat_error,
    base_error,
    vardQ,
    dH,
    K,
    N,
    dG,
    dG_sem,
    temperature,
):
    print(f"{'Cell volume = ':<20} {V0:5.7f} L")
    print(f"{'Cell conc. = ':<20} {M0:5.7f} M")
    print(f"{'Injectant conc. = ':<20} {X0:5.7f} M")
    print(f"{'Syringe error = ':<20} {syringe_error * 100:4.2f} percent")
    print(f"{'Cell error = ':<20} {cell_error * 100 :4.2f} percent")
    print(f"{'Heat error = ':<20} {heat_error * 100:4.2f} percent")
    print(f"{'Base error = ':<20} {base_error * 1e6:4.2f} ucal")

    print(
        f"{'# dH = ':<20} {np.mean(dH) / 1000.:>10.2f} +/- {np.std(dH) / 1000.:>10.5f} kcal/mol"
    )

    print(f"{'# K = ':<20} {np.mean(K):>10.2f} +/- {np.std(K):>10.5f} M^{{-1}}")
    print(f"{'# N = ':<20} {np.mean(N):>10.2f} +/- {np.std(N):>10.5f}")

    print(f"{'# dG = ':<20} {dG:>10.2f} +/- {dG_sem:>10.5f} kcal/mol")
    TdS = -1 * (dG - np.mean(dH) / 1000.0)
    dS_sem = np.sqrt(dG_sem ** 2 + (np.std(dH) / 1000.0) ** 2)
    print(f"{'# TdS = ':<20} {TdS:>10.5f} +/- {dS_sem:>10.5f} kcal/mol")
