import copy
import numpy as np
from typing import Union, List, Tuple

from . import instrument
from .inference import default_fit_options
from .reduction import ReflData

from .autorefl import AutoReflBase, AutoReflExperiment

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
                       meas_bkg, startmodel, min_meas_time, select_pars, bestpars)

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
            newpoints = [self._simulate_datapoint(mnum, newx, new_meastime, Intent.spec, fom[mnum][idx])]
            if new_bkgmeastime > 0:
                newpoints.append(self._simulate_datapoint(mnum, newx, new_bkgmeastime / 2.0, Intent.backp, fom[mnum][idx]))
                newpoints.append(self._simulate_datapoint(mnum, newx, new_bkgmeastime / 2.0, Intent.backm, fom[mnum][idx]))

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
                pts = [self._simulate_datapoint(mnum, x, t * (1 - self.frac_background), Intent.spec, None),
                       self._simulate_datapoint(mnum, x, 0.5 * t * self.frac_background, Intent.backp, None),
                       self._simulate_datapoint(mnum, x, 0.5 * t * self.frac_background, Intent.backm, None)]

                pts[0].movet = self.instrument.movetime(x)[0]
                for pt in pts:
                    points.append(pt)
                self.instrument.x = x

        self.add_step(points)
