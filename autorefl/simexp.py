import copy
import numpy as np
from typing import Union, List, Tuple

from .autorefl import AutoReflBase
from . import instrument

from refl1d.names import FitProblem, Experiment

from .datastruct import DataPoint, Intent
from entropy import default_entropy_options
from .inference import default_fit_options
from .simulation import calc_expected_R, sim_data_N

class SimReflExperiment(AutoReflBase):
    """
    A simulated reflectometry experiment.

    Contains methods for defining the experiment (via a bumps.FitProblem) object,
    simulating data from a specific instrument (via a ReflectometerBase-d object from
    the instrument module), fitting simulated data (via Refl1D), and determining the
    next optimal measurement point. Also allows saving and loading.

    Typical workflow:
        exp = SimReflExperiment(...)
        exp.add_initial_step()
        while (condition):
            exp.fit_step()
            exp.take_step()

    Additional Inputs:
    
    bestpars -- numpy array or list or None: best fit (ground truth) parameters (length P
                parameters). Used for simulating new data
                
    """

    def __init__(self, problem: FitProblem,
                       Q: Union[np.ndarray, List[np.ndarray]],
                       instrument: instrument.ReflectometerBase = instrument.MAGIK(),
                       eta: float = 0.68,
                       npoints: int = 1,
                       switch_penalty: float = 1.0,
                       switch_time_penalty: float = 0.0,
                       fit_options: dict = default_fit_options,
                       entropy_options: dict = default_entropy_options,
                       oversampling: int = 11,
                       meas_bkg: Union[float, List[float]] = 1e-6,
                       startmodel: int = 0,
                       min_meas_time: float = 10.0,
                       select_pars: Union[list, None] = None,
                       bestpars: Union[np.ndarray, list, None] = None) -> None:
        
        super().__init__(problem, Q, instrument, eta, npoints, switch_penalty,
                       switch_time_penalty, fit_options, entropy_options, oversampling,
                       meas_bkg, startmodel, min_meas_time, select_pars)

        calcmodel = copy.deepcopy(problem)
        self.calcmodels: List[Union[Experiment, FitProblem]] = [calcmodel] if hasattr(calcmodel, 'fitness') else list(calcmodel.models)
        if bestpars is not None:
            calcmodel.setp(bestpars)

        # add residual background
        self.resid_bkg: np.ndarray = np.array([c.fitness.probe.background.value for c in self.calcmodels])

    def request_data(self, newpoints, foms) -> List[DataPoint]:
        """
        Requests new data. In simulation mode, generates new data points. In experiment mode, should
        add new data points to a point queue to be measured
        """
        
        # Determine next measurement point(s).
        # Number of points to be used is determined from n_forecast (self.npoints)

        points = []
        for pt, fom in zip(newpoints, foms):
            mnum, idx, newx, new_meastime, new_bkgmeastime = pt
            newpoints = self._generate_new_point(mnum, newx, new_meastime, new_bkgmeastime, fom[mnum][idx])
            newpoints[0].movet = self.instrument.movetime(newpoints[0].x)[0]
            for newpoint in newpoints:
                points.append(newpoint)
                print('New data point:\t' + repr(newpoint))

            # Once a new point is added, update the current model so model switching
            # penalties can be reapplied correctly
            self.curmodel = newpoint.model

            # "move" instrument to new location for calculating the next movement penalty
            self.instrument.x = newpoint.x
        
        return points

    def _generate_new_point(self, mnum: int,
                                  newx: float,
                                  new_meastime: float,
                                  new_bkgmeastime: float,
                                  maxfom: Union[float, None] = None) -> List[DataPoint]:
        """ Generates a new data point with simulated data from the specified x
            position, model number, and measurement time
            
            Inputs:
            mnum -- the model number of the new point
            newx -- the x position of the new point
            new_meastime -- the measurement time
            maxfom -- the maximum of the figure of merit. Only used for record-keeping

            Returns a single DataPoint object
        """
        
        T = self.instrument.T(newx)[0]
        dT = self.instrument.dT(newx)[0]
        L = self.instrument.L(newx)[0]
        dL = self.instrument.dL(newx)[0]

        # for simulating data, need to subtract theta_offset from calculation models
        # not all probes have theta_offset, however
        # for now this is turned off. 
        if False:
            try:
                to_calc = self.calcmodels[mnum].fitness.probe.theta_offset.value
            except AttributeError:
                to_calc = 0.0

        calcR = calc_expected_R(self.calcmodels[mnum].fitness, T, dT, L, dL, oversampling=self.oversampling, resolution='normal')
        #print('expected R:', calcR)
        incident_neutrons = self.instrument.intensity(newx) * new_meastime
        N, Nbkg, Ninc = sim_data_N(calcR, incident_neutrons, resid_bkg=self.resid_bkg[mnum], meas_bkg=self.meas_bkg[mnum])
        Nbkgp, Nbkgm = Nbkg
        pts = [DataPoint(newx, new_meastime, mnum, (T, dT, L, dL, N[0], Ninc[0]), merit=maxfom, intent=Intent.spec)]

        if new_bkgmeastime > 0:
            pts.append(DataPoint(newx, new_bkgmeastime / 2.0, mnum, (T, dT, L, dL, Nbkgp[0], Ninc[0]), intent=Intent.backp))
            pts.append(DataPoint(newx, new_bkgmeastime / 2.0, mnum, (T, dT, L, dL, Nbkgm[0], Ninc[0]), intent=Intent.backm))

        return pts

class SimReflExperimentControl(SimReflExperiment):
    r"""Control experiment with even or scaled distribution of count times
    
    Subclasses SimReflExperiment.

    Additional input:
    model_weights -- a vector of weights, with length equal to number of problems
                    in self.problem. Scaled by the sum.
    NOTE: an instrument-defined default weighting is additionally applied to each Q point
    """

    def __init__(self, problem: FitProblem,
                       Q: Union[np.ndarray, List[np.ndarray]],
                       model_weights: Union[List[float], None] = None,
                       frac_background: float = 0.33,
                       instrument: instrument.ReflectometerBase = instrument.MAGIK(),
                       eta: float = 0.68,
                       npoints: int = 1,
                       switch_penalty: float = 1.0,
                       switch_time_penalty: float = 0.0,
                       bestpars: Union[np.ndarray, list, None] = None,
                       fit_options: dict = default_fit_options,
                       entropy_options: dict = default_entropy_options,
                       oversampling: int = 11,
                       meas_bkg: Union[float, List[float]] = 1e-6,
                       startmodel: int = 0,
                       min_meas_time: float = 10.0,
                       select_pars: Union[list, None] = None) -> None:
        super().__init__(problem, Q, instrument=instrument, eta=eta, npoints=npoints,
            switch_penalty=switch_penalty, switch_time_penalty=switch_time_penalty,
            bestpars=bestpars, fit_options=fit_options, entropy_options=entropy_options,
            oversampling=oversampling, meas_bkg=meas_bkg, startmodel=startmodel,
            min_meas_time=min_meas_time, select_pars=select_pars)

        if model_weights is None:
            model_weights = np.ones(self.nmodels)
        else:
            assert (len(model_weights) == self.nmodels), "weights must have same length as number of models"
        
        model_weights = np.array(model_weights) / np.sum(model_weights)

        self.meastimeweights = list()
        for x, weight in zip(self.x, model_weights):
            f = self.instrument.meastime(x, weight)
            self.meastimeweights.append(f)

        self.frac_background = frac_background

    def take_step(self, total_time: float) -> None:
        r"""Overrides SimReflExperiment.take_step
        
        Generates a simulated reflectivity curve based on weighted / scaled
        measurement times.
        """

        points = list()

        for mnum, (newx, mtimeweight) in enumerate(zip(self.x, self.meastimeweights)):
            for x, t in zip(newx, total_time * mtimeweight):
                pts = self._generate_new_point(mnum, x, t * (1 - self.frac_background), t * self.frac_background, None)
                pts[0].movet = self.instrument.movetime(x)[0]
                for pt in pts:
                    points.append(pt)
                self.instrument.x = x

        self.add_step(points)
