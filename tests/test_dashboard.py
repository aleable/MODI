import sys
import pytest

# Adjust the path to make sure you can import modules from notebooks directory
sys.path.insert(0, "notebooks")

import utils
from modi_flows.main import *


def test_dashboard_multicom():
    """
    Test the multicommodity algorithm as used in the dashboard.

    Returns:
       None
    """
    ms = 40
    flowers_dataframe = utils.preprocessing(ms)

    otp_alg = "multicom"
    sigma = 0.5
    pooling_size = 1
    theta = 0.5
    alpha = 0.5

    time_step = 0.5
    tol = 1e-1
    time_tol = 1e3
    beta = 1.0

    seed = 0
    VERBOSE = False

    img1 = flowers_dataframe["image"][0]
    img2 = flowers_dataframe["image"][1]

    C, g, h = preprocessing(img2, img1, otp_alg, pooling_size, sigma, theta)
    thresh = 0.05

    modi = MODI(
        g,
        h,
        C,
        beta=beta,
        dt=time_step,
        tol=tol,
        time_tol=time_tol,
        alpha=alpha,
        t=thresh,
        verbose=VERBOSE,
    )

    info_dyn = modi.exec()

    expected_cost = 4925.39472375424
    assert pytest.approx(info_dyn[0], rel=1e-6) == expected_cost


def test_dashboard_unicom():
    """
    Test the unicommodity algorithm as used in the dashboard.

    Returns:
       None
    """

    ms = 40
    flowers_dataframe = utils.preprocessing(ms)

    otp_alg = "unicom"  # "multicom"/"unicom" for M=3 and M=1, respectively
    sigma = 0.5  # in case additional gaussian smoothing of the images is desired
    pooling_size = 1  # in case further pooling of the images is desired
    theta = 0.5  # convex weight for the construction of C
    alpha = 0.5  # penalty for Kirchhoff's law (https://ieeexplore.ieee.org/document/5459199)

    time_step = 0.5  # forward Euler discretization time step
    tol = 1e-1  # convergence tolerance
    time_tol = 1e3  # stopping time step limit (for safety)
    beta = 1.0  # regularization parameter

    seed = 0
    VERBOSE = False

    img1 = flowers_dataframe["image"][0]
    img2 = flowers_dataframe["image"][1]

    C, g, h = preprocessing(img2, img1, otp_alg, pooling_size, sigma, theta)
    thresh = 0.05  # trimming threshold (https://ieeexplore.ieee.org/document/5459199)

    modi = MODI(
        g,
        h,
        C,
        beta=beta,
        dt=time_step,
        tol=tol,
        time_tol=time_tol,
        alpha=alpha,
        t=thresh,
        verbose=VERBOSE,
    )

    info_dyn = modi.exec()

    expected_cost = 2990.5899531108944
    assert np.isclose(
        info_dyn[0], expected_cost, rtol=1e-5
    ), f"Expected cost: {expected_cost}, Actual cost: {info_dyn[0]}"
