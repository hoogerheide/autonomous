import numpy as np
from scipy.stats import poisson

def sim_data_N(R, incident_neutrons, resid_bkg=0, meas_bkg=0):
    R = np.array(R, ndmin=1)
    _bR = np.ones_like(R)*(meas_bkg - resid_bkg)*incident_neutrons
    _R = (R + meas_bkg - resid_bkg)*incident_neutrons
    N = poisson.rvs(_R, size=_R.shape)
    bNp = poisson.rvs(_bR, size=_bR.shape)
    bNm = poisson.rvs(_bR, size=_bR.shape)
    _Ninc = np.array(incident_neutrons, ndmin=1)
    Ninc = poisson.rvs(_Ninc, size=_Ninc.shape)

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
