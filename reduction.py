from typing import Union, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from bumps.fitters import DreamFit, _fill_defaults
from reflred.candor import edges, _rebin_bank, QData
from reflred.background import apply_background_subtraction, _ordinate
from reflred.scale import apply_intensity_norm
from reflred.refldata import ReflData, Sample, Detector, Monochromator, Monitor
from dataflow.lib.uncertainty import Uncertainty as U, interp
from datastruct import DataPoint, data_attributes

def interpolate_background(Qbasis, backp: Union[ReflData, None] = None, backm: Union[ReflData, None] = None):
    
    """
    Interpolates background and returns background and variance. Backgrounds should be normalized first

    Inputs:
    Qbasis -- points to interpolate
    backp -- collection of DataPoints containing background+ data
    backm -- collection of DataPoints containing background- data
    """
    backp_v = U(backp.v, backp.dv**2) if backp is not None else None
    backm_v = U(backm.v, backm.dv**2) if backm is not None else None

    backp_v = interp(Qbasis, backp.Qz, backp_v) if backp is not None else None
    backm_v = interp(Qbasis, backm.Qz, backm_v) if backm is not None else None

    if backp and backm:
        bkg = (backp_v + backm_v)/2
    elif backp:
        bkg = backp_v
    elif backm:
        bkg = backm_v
    else:
        return None, None  # no background to subtract

    return bkg.x, bkg.variance

def DataPoint2ReflData(Qbasis, data: List[DataPoint], normbase='none') -> Tuple[Union[ReflData, None], Union[ReflData, None]]:
    """
    Converts a list of DataPoint objects to tuple of numpy arrays (T, dT, L, dL, N, Ninc). Filters
    out any data points with zero incident intensity (protects against division by zero)

    Inputs:
    data -- list of DataPoint objects
    normbase -- normalization basis. Currently only 'time' (default) is supported.

    Returns:
    idata -- dict of (T, dT, L, dL, N, Ninc). Each element is a numpy array
    """

    idata = {attr: [val for pt in data for val in getattr(pt, attr)] for attr in data_attributes}
    idata['t'] = [pt.t for pt in data for _ in getattr(pt, 'T')] 
    idata['Ninc'] = np.round(idata['Ninc'])
    crit = idata['Ninc'] > 0

    for k in idata.keys():
        idata[k] = np.array(idata[k])[crit]

    if len(idata['T']):

        q_edges = edges(Qbasis, extended=True)
        #argsort = np.argsort(idata['T'])
        t = idata['t'][:, None, None]
        v = idata['N'][:, None, None]
#        v += (v == 0)
        dv = np.sqrt(v)
        dv += (dv == 0)

        vinc = idata['Ninc'][:, None, None]
#        vinc += (vinc == 0)
        dvinc = np.sqrt(vinc)
        T, dT, L, dL = (idata[key][:, None, None] for key in ['T', 'dT', 'L', 'dL'])
        wavelength_resolution = dL / L

        data = ReflData(monochromator=Monochromator(wavelength=L, wavelength_resolution=wavelength_resolution),
                    sample=Sample(angle_x=T),
                    angular_resolution=dT,
                    detector=Detector(angle_x=2*T, wavelength=L, wavelength_resolution=wavelength_resolution),
                    _v=v, _dv=dv, Qz_basis='actual', normbase=normbase)

        qz, dq, v, dv, _Ti, _dT, _L, _dL = _rebin_bank(data, 0, q_edges, 'poisson')
        data = QData(data, qz, dq, v, dv, _Ti, _dT, _L, _dL)

        inc = ReflData(monochromator=Monochromator(wavelength=L, wavelength_resolution=wavelength_resolution),
                        sample=Sample(angle_x=T),
                        angular_resolution=dT,
                        detector=Detector(angle_x=2*T, wavelength=L, wavelength_resolution=wavelength_resolution),
                        _v=vinc, _dv=dvinc, Qz_basis='actual', normbase=normbase)

        qz, dq, v, dv, _Ti, _dT, _L, _dL = _rebin_bank(inc, 0, q_edges, 'poisson')
        inc = QData(data, qz, dq, v, dv, _Ti, _dT, _L, _dL)

        apply_intensity_norm(data, inc, align_by='Qz')

        return data
    
    else:

        return None

def reduce(Qbasis, specdata: List[DataPoint], backp: List[DataPoint], backm: List[DataPoint], normbase='none'):
    """Reduces data
    
        1. Bin all data
        2. Subtract background
        3. Normalize by intensity
        
        Inputs:
        Qbasis -- measurement Q values on which to bin
        specdata -- list of DataPoint objects containing specular data
        backp -- list of DataPoint objects containing back+ data
        backm -- list of DataPoint objects containing back- data
        normbase -- how to normalize data (default 'none')
        """

    spec = DataPoint2ReflData(Qbasis, specdata, normbase=normbase)
    bkgp = DataPoint2ReflData(Qbasis, backp, normbase=normbase) if backp is not None else None
    bkgm = DataPoint2ReflData(Qbasis, backm, normbase=normbase) if backm is not None else None

    if spec is not None:

        # subtract background
        apply_background_subtraction(spec, bkgp, bkgm)

        # TODO: divide intensity here instead of in DataPoint2ReflData
        # apply_intensity_norm(specdata, incdata)

        return spec #_Ti, _dT, _L, _dL, specdata.v, specdata.dv, qz, dq

    else:

        return None #tuple([np.array([]) for _ in range(8)])
