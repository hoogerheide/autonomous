import numpy as np
from typing import Tuple, Union, List

from reflred.intent import Intent
from refl1d.resolution import TL2Q
from bumps.dream.state import MCMCDraw

data_tuple = Tuple[Union[np.ndarray, list], Union[np.ndarray, list],
                                   Union[np.ndarray, list], Union[np.ndarray, list],
                                   Union[np.ndarray, list], Union[np.ndarray, list]]

data_attributes = ['T', 'dT', 'L', 'dL', 'N', 'Ninc']

class DataPoint(object):
    """ Container object for a single data point.

    A "single data point" normally corresponds to a single instrument configuration.
    Note that for polychromatic and time-of-flight instruments, this may involve multiple
    Q values. As a result, all of the "data" fields (described below) are stored 
    as lists or numpy.ndarrays.

    Required attributes:
    x -- a description of the instrument configuration, usually as a single number
        whose interpretation is determined by the instrument class (e.g. Q for MAGIK,
        Theta for CANDOR)
    meastime -- the total measurement time 
    modelnum -- the index of the bumps.FitProblem model with which the data point
             is associated
    data -- a tuple of data attributes (see below)
    intent -- a description of the intent. One of Intent.spec, Intent.backp, Intent.backm from
            the reflred.intent module

    Optional attributes:
    movet -- the total movement time. Note that this varies depending on what the 
            previous point was. Zero for background points.
    merit -- if calculated, the figure of merit of this data point. Mainly used for plotting.

    Data attributes. When initializing, these are required as the argument "data" 
    in a tuple of lists or arrays.
    T -- theta array
    dT -- angular resolution array
    L -- wavelength array
    dL -- wavelength uncertainty array
    N -- neutron counts at this instrument configuration
    Ninc -- incident neutron counts at this instrument configuration

    Methods:
    Q -- returns an array of Q points corresponding to T and L.
    """

    def __init__(self, x: float, meastime: float, modelnum: int,
                       data: data_tuple,
                       intent: str,
                       merit: Union[bool, None] = None,
                       movet: float = 0.0):
        self.model = modelnum
        self.t = meastime
        self.movet = movet
        self.merit = merit
        self.x = x
        self._data: data_tuple = None
        self.data = data
        self.intent = intent

    def __repr__(self):

        def format_obj(obj, fmt: str) -> str:

            if isinstance(obj, np.ndarray) | isinstance(obj, list):
                return ', '.join(format_obj(itm, fmt) for itm in obj)
            else:
                return f'{obj:{fmt}}'

#        try:
        qrep = format_obj(self.Q(), '0.4f')
        reprq = f'Q: {qrep} Ang^-1'
#        except TypeError:
#            reprq = 'Q: ' + ', '.join('{:0.4f}'.format(q) for q in self.Q()) + ' Ang^-1'
        
        nrep = format_obj(self.N, '0.0f')
        return ('Model: %i\t' % self.model) + reprq + ('\tIntent: ' + self.intent) + ('\tTime: %0.1f s' %  self.t) + f'\tCounts: {nrep}'

    @property
    def data(self):
        """ gets the internal data variable"""
        return self._data

    @data.setter
    def data(self, newdata) -> None:
        """populates T, dT, L, dL, N, Ninc.
            newdata is a length-6 tuple of lists"""
        self._data = newdata
        self.T, self.dT, self.L, self.dL, self.N, self.Ninc = newdata

    def Q(self):
        return TL2Q(self.T, self.L)

class ExperimentStep(object):
    """ Container object for a single experiment step.

        Attributes:
        points -- a list of DataPoint objects
        H -- MVN entropy in all parameters
        dH -- change in H from the initial step (with no data and calculated
                only from the bounds of the model parameters)
        H_marg -- MVN entropy from selected parameters (marginalized entropy)
        dH_marg -- change in dH from the initial step
        foms -- list of the figures of merit for each model
        scaled_foms -- figures of merit after various penalties are applied. Possibly
                        not useful
        meastimes -- list of the measurement time proposed for each Q value of each model
        bkgmeastimes -- list of the background measurement time proposed for each Q value
        qprofs -- list of Q profile arrays calculated from each sample from the MCMC posterior
        qbkgs -- not used
        best_logp -- best nllf after fitting
        final_chisq -- final chi-squared string (including uncertainty) after fitting
        draw -- an MCMCDraw object containing the best fit results
        chain_pop -- MCMC chain heads for use in DreamFitPlus for initializing the MCMC
                     fit. Useful for restarting fits from an arbitrary step.
        use -- a flag for whether the step contains real data and should be used in furthur
                analysis.
        
        TODO: do not write draw.state, which is inflating file sizes!

        Methods:
        getdata -- returns all data of type "attr" for data points from a specific model
        meastime -- returns the total measurement time or the time from a specific model
        movetime -- returns the total movement time or the time from a specific model
    """

    def __init__(self, points: List[DataPoint], use=True) -> None:
        self.points = points
        self.H: Union[float, None] = None
        self.dH: Union[float, None] = None
        self.H_marg: Union[float, None] = None
        self.dH_marg: Union[float, None] = None
        self.foms: Union[List[np.ndarray], None] = None
        self.scaled_foms: Union[List[np.ndarray], None] = None
        self.meastimes: Union[List[np.ndarray], None] = None
        self.bkgmeastimes: Union[List[np.ndarray], None] = None
        self.qprofs: Union[List[np.ndarray], None] = None
        self.qbkgs: Union[List[np.ndarray], None] = None
        self.best_logp: Union[float, None] = None
        self.final_chisq: Union[str, None] = None
        self.draw: Union[MCMCDraw, None] = None
        self.chain_pop: Union[np.ndarray, None] = None
        self.use: bool = use

    def getdata(self, attr: str, modelnum: int) -> list:
        # returns all data of type "attr" for a specific model
        if self.use:
            return [getattr(pt, attr) for pt in self.points if pt.model == modelnum]
        else:
            return []

    def meastime(self, modelnum: Union[int, None] = None) -> float:
        if modelnum is None:
            return sum([pt.t for pt in self.points])
        else:
            return sum([pt.t for pt in self.points if pt.model == modelnum])

    def movetime(self, modelnum: Union[int, None] = None) -> float:
        if modelnum is None:
            return sum([pt.movet for pt in self.points])
        else:
            return sum([pt.movet for pt in self.points if pt.model == modelnum])
