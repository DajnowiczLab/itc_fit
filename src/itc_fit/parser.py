import numpy as np


def parse_itc_data(file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read in the raw ITC data and return the heat and volume injected.
    Always skip the first row which is the header from the instrument.
    Always skip the final row which is a summary of total heat and total injected
    volume from the instrument.
    Optionally skip additional injections from the beginning of the experiment.

    :param file: Path to the raw ITC data file.
    :return: Tuple of heat and volume injected.
    """

    data = np.genfromtxt(file, skip_header=1, skip_footer=1)

    heat = data[:, 0] / 1e6  # microcalories to calories
    volume = data[:, 1] / 1e6  # microliters to liters

    return heat, volume