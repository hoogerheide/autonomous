import numpy as np
from scipy.stats import poisson, norm

def sim_data_N(R, incident_neutrons: np.ndarray, resid_bkg=0.0, meas_bkg=0.0):
    R = np.array(R, ndmin=1)
    _bR = np.ones_like(R)*(meas_bkg - resid_bkg)*incident_neutrons
    _R = (R + meas_bkg - resid_bkg)*incident_neutrons

    # np.float64 type avoids ValueError: lam is too large
    _R.astype(np.float64)
    N = poisson.rvs(_R, size=_R.shape)

    _bR.astype(np.float64)
    bNp = poisson.rvs(_bR, size=_bR.shape)
    bNm = poisson.rvs(_bR, size=_bR.shape)

    _Ninc = np.array(incident_neutrons, ndmin=1)
    _Ninc.astype(np.float64)
    try:
        Ninc = poisson.rvs(_Ninc, size=_Ninc.shape)
    except ValueError:
        print(_Ninc)
        print('warning: switching to Gaussian point simulator')
        Ninc = norm.rvs(_Ninc, size=_Ninc.shape)

    return N, (bNp, bNm), Ninc

def calc_expected_R(fitness, T, dT, L, dL, oversampling=None, resolution='normal'):
    # currently requires sorted values (by Q) because it returns sorted values.
    # this will need to be modified for CANDOR.
    fitness.probe._set_TLR(T, dT, L, dL, R=None, dR=None, dQ=None)
    fitness.probe.resolution = resolution
    if oversampling is not None:
        fitness.probe.oversample(oversampling)
    fitness.update()
    return fitness.reflectivity()[1]
