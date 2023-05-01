import numpy as np
import copy
import time
import dill
from typing import Tuple, Union, List
from threading import Event, Semaphore
from queue import Queue

from bumps.fitters import ConsoleMonitor, _fill_defaults, StepMonitor
from bumps.initpop import generate

from refl1d.names import FitProblem, Experiment
from scipy.interpolate import interp1d

# local imports
from entropy import calc_entropy, calc_init_entropy, default_entropy_options
from .datastruct import DataPoint, ExperimentStep, Intent, MeasurementPoint
from .reduction import DataPoint2ReflData, interpolate_background, reduce, ReflData
from .inference import MPMapper, _MP_calc_qprofile, DreamFitPlus, default_fit_options
from .simulation import sim_data_N, calc_expected_R
from . import instrument

class AutoReflBase(object):
    """
    Autonomous reflectometry experiment base class

    Inputs:

    problem -- a Refl1d FitProblem describing the reflectometry experiment. Multiple models (M)
                are supported. Required.
    Q -- numpy array or nested list of Q bins for reducing data. Can be either a single Q vector
        (applied to each model), or an M-element list of Q bins, one for each model. Required.
    instrument -- instrument definition based on the instrument.ReflectometerBase class; default
                   MAGIK
    eta -- confidence interval for measurement time determination; default 0.68
    npoints -- integer number of points to measure in each step; if > 1, forecasting is used to 
                determine subsequent points; default 1
    switch_penalty -- scaling factor on the figure of merit to switch models, i.e. applied only
                      to models that are not the current one; default 1.0 (no penalty)
    switch_time_penalty -- time required to switch models, i.e. applied only
                      to models that are not the current one; default 0.0 (no penalty)
    fit_options -- dictionary of Bumps fitter fit options; default {'pop': 10, 'burn': 1000,
                    'steps': 500, 'init': 'lhs', 'alpha': 0.001}
    entropy_options -- dictionary of entropy options; default entropy.default_entropy_options
    oversampling -- integer oversampling value for calculating Refl1D models. Default 11; should be
                    > 1 for accurate simulations.
    meas_bkg -- single float of list of M floats representing the measurement background level
                (unsubtracted) for each model.
    startmodel -- integer index of starting model; default 0.
    min_meas_time -- minimum measurement time (float); default 10.0 seconds
    select_pars -- selected parameters for entropy determination. None uses all parameters, otherwise
                    list of parameter indices
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
                       select_pars: Union[list, None] = None) -> None:
        
        # Load instrument
        self.instrument = instrument

        # Analysis options
        self.eta = eta
        self.npoints = int(npoints)
        self.switch_penalty = switch_penalty
        self.switch_time_penalty = switch_time_penalty
        self.min_meas_time = min_meas_time

        # Initialize the fit problem
        self.problem = problem
        models: List[Union[Experiment, FitProblem]] = [problem] if hasattr(problem, 'fitness') else list(problem.models)
        self.models = models
        self.nmodels = len(models)
        self.curmodel = startmodel
        self.oversampling = oversampling
        for m in self.models:
            m.fitness.probe.oversample(oversampling)
            m.fitness.probe.resolution = self.instrument.resolution
            m.fitness.update()

        # Condition Q vector to a list of arrays, one for each model
        if isinstance(Q, np.ndarray):
            if len(Q.shape) == 1:
                self.measQ = np.broadcast_to(Q, (self.nmodels, len(Q)))
            elif len(Q.shape) == 2:
                assert (Q.shape[0]==self.nmodels), "Q array must be a single vector or have first dimension equal to the number of models in problem"
                self.measQ = Q
            else:
                raise Exception('Bad Q shape')
        else:
            if any(isinstance(i, (list, np.ndarray)) for i in Q): # is a nested list
                assert (len(Q) == self.nmodels), "Q array must be a single vector or a list of vectors with length equal to the number of models in problem"
                self.measQ = Q
            else:
                self.measQ = [Q for _ in range(self.nmodels)]

        # define measurement space. Contains same number of points per model as self.measQ
        # measurement space is instrument specific (e.g. for MAGIK x=Q but for polychromatic
        # or TOF instruments x = Theta). In principle x can be anything that can be mapped
        # to a specific instrument configuration; this is defined in the instrument module.
        # TODO: Make separate measurement list. Because Q is used for rebinning, it should
        # have a different length from "x"
        self.x: List[np.ndarray] = list()
        for Q in self.measQ:
            minx, maxx = self.instrument.qrange2xrange([min(Q), max(Q)])
            self.x.append(np.linspace(minx, maxx, len(Q), endpoint=True))

        # Create a copy of the problem for calculating the "true" reflectivity profiles
        self.npars = len(problem.getp())
        self.orgQ = [list(m.fitness.probe.Q) for m in models]

        # deal with inherent measurement background
        if not isinstance(meas_bkg, (list, np.ndarray)):
            self.meas_bkg: np.ndarray = np.full(self.nmodels, meas_bkg)
        else:
            self.meas_bkg: np.ndarray = np.array(meas_bkg)

        self.resid_bkg: np.ndarray = np.full_like(self.meas_bkg, 1e-12)

        # these are not used
        self.newmodels = [m.fitness for m in models]
        self.par_scale: np.ndarray = np.diff(problem.bounds(), axis=0)

        # set and condition selected parameters for marginalization; use all parameters
        # if none are specified
        if select_pars is None:
            self.sel: np.ndarray = np.arange(self.npars)
        else:
            self.sel: np.ndarray = np.array(select_pars, ndmin=1)

        # initialize objects required for fitting
        self.fit_options = {**default_fit_options, **fit_options}
        self.steps: List[ExperimentStep] = []
        self.restart_pop: Union[np.ndarray, None] = None

# calculate initial MVN entropy in the problem
        self.entropy_options = {**default_entropy_options, **entropy_options}
        self.thinning = int(self.fit_options['steps']*0.8)
        self.init_entropy, _, _ = calc_init_entropy(problem, pop=self.fit_options['pop'] * self.fit_options['steps'] / self.thinning, options=self.entropy_options)
        self.init_entropy_marg, _, _ = calc_init_entropy(problem, select_pars=select_pars, pop=self.fit_options['pop'] * self.fit_options['steps'] / self.thinning, options=self.entropy_options)

    def get_all_points(self, modelnum: Union[int, None]) -> List[DataPoint]:
        # returns all data points associated with model with index modelnum
        return [pt for step in self.steps for pt in step.points if pt.model == modelnum]

    def get_data(self) -> List[Tuple[ReflData, Union[ReflData, None],
                                     Union[ReflData, None]]]:
        """
        Gets all data and returns specular, background+, background- as ReflData objects
        """

        modeldata = list()
        for i, mQ in enumerate(self.measQ):
            specdata = [pt for step in self.steps for pt in step.points if (pt.model == i) & (pt.intent == Intent.spec)]
            bkgpdata = [pt for step in self.steps for pt in step.points if (pt.model == i) & (pt.intent == Intent.backp)]
            bkgmdata = [pt for step in self.steps for pt in step.points if (pt.model == i) & (pt.intent == Intent.backm)]
            modeldata.append((DataPoint2ReflData(mQ, specdata),
                              DataPoint2ReflData(mQ, bkgpdata),
                              DataPoint2ReflData(mQ, bkgmdata)))

        return modeldata

    def add_initial_step(self, dRoR=10.0) -> None:
        """ Generate initial data set. This is only necessary because of the requirement that
            dof > 0 in Refl1D (not strictly required for DREAM fit)
            
            Inputs:
            dRoR -- target uncertainty relative to the average of the reflectivity, default 10.0;
                    determines the "measurement time" for the initial data set. This should be
                    > 3 so as not to constrain the parameters before collecting any real data.
        """

        # evenly spread the Q points over the models in the problem
        nQs = [((self.npars + 1) // self.nmodels) + 1 if i < ((self.npars + 1) % self.nmodels) else ((self.npars + 1) // self.nmodels) for i in range(self.nmodels)]
        newQs = [np.linspace(min(Qvec), max(Qvec), nQ) for nQ, Qvec in zip(nQs, self.measQ)]

        # generate an initial population and calculate the associated q-profiles
        initpts = generate(self.problem, init='lhs', pop=self.fit_options['pop'], use_point=False)

        # Set attributes of "problem" for passing into multiprocessing routines
        newvars = [self.instrument.Q2TdTLdL(nQ, mx, mQ) for nQ, mQ, mx in zip(newQs, self.measQ, self.x)]
        setattr(self.problem, 'calcTdTLdL', newvars)
        setattr(self.problem, 'oversampling', self.oversampling)
        setattr(self.problem, 'resolution', self.instrument.resolution)

        # Start multiprocessing mapper
        mapper = MPMapper.start_mapper(self.problem, None, cpus=0)

        init_qprof = self.calc_qprofiles(initpts)

        # Terminate the multiprocessing pool (required to avoid memory issues
        # if run is stopped after current fit step)
        MPMapper.stop_mapper(mapper)
        MPMapper.pool = None

        points = []

        # simulate data based on the q profiles. The uncertainty in the parameters is estimated
        # from the dRoR paramter
        for mnum, (newQ, qprof, meas_bkg, resid_bkg) in enumerate(zip(newQs, init_qprof, self.meas_bkg, self.resid_bkg)):
            # calculate target mean and uncertainty from unconstrained profiles
            newR, newdR = np.mean(qprof, axis=0), dRoR * np.std(qprof, axis=0)

            # calculate target number of measured neutrons to give the correct uncertainty with
            # Poisson statistics
            targetN = (newR / newdR) ** 2

            # calculate the target number of incident neutrons to give the target reflectivity
            target_incident_neutrons = targetN / newR

            # simulate the data
            Ns, Nbkgs, Nincs = sim_data_N(newR, target_incident_neutrons, resid_bkg=resid_bkg, meas_bkg=meas_bkg)

            # Calculate T, dT, L, dL. Note that because these data don't constrain the model at all,
            # these values are brought in from MAGIK (not instrument-specific) because they don't need
            # to be.
            Ts = instrument.q2a(newQ, 5.0)
            # Resolution function doesn't matter here at all because these points don't have any effect
            dTs = np.polyval(np.array([ 2.30358547e-01, -1.18046955e-05]), newQ)
            Ls = np.ones_like(newQ)*5.0
            dLs = np.ones_like(newQ)*0.01648374 * 5.0

            # Append the data points with zero measurement time
            points.append(DataPoint(0.0, 0.0, mnum, (Ts, dTs, Ls, dLs, Ns, Nincs), Intent.spec))

        # Add the step with the new points
        self.add_step(points, use=False)

    def update_models(self) -> None:
        # Update the models in the fit problem with new data points. Should be run every time
        # new data are to be incorporated into the model
        modeldata = self.get_data()

        for m, measQ, (specdata, bkgpdata, bkgmdata) in zip(self.models, self.measQ, modeldata):
            # run reduction
            # TODO: instrument-specific reduction
            spec = reduce(specdata, bkgpdata, bkgmdata)

            mT, mdT, mL, mdL, mR, mdR, mQ, mdQ = spec.sample.angle_x, spec.angular_resolution, \
                            spec.detector.wavelength, spec.detector.wavelength_resolution,  \
                            spec.v, spec.dv, spec.Qz, spec.dQ

            m.fitness.probe._set_TLR(mT, mdT, mL, mdL, mR, mdR, dQ=mdQ)
            m.fitness.probe.oversample(self.oversampling)
            m.fitness.probe.resolution = self.instrument.resolution
            m.fitness.update()
        
        # protect against too few data points
        self.problem.partial = True

        # Triggers recalculation of all models
        self.problem.model_reset()
        self.problem.chisq_str()

    def calc_qprofiles(self, drawpoints: np.ndarray) -> np.ndarray:
        # NOTE: must have MPMapper started before calling this!!
        mappercalc = lambda points: MPMapper.pool.map(_MP_calc_qprofile, ((MPMapper.problem_id, p) for p in points))

        # q-profile calculator using multiprocessing for speed
        res = mappercalc(drawpoints)

        # condition output of mappercalc to a list of q-profiles for each model
        qprofs = list()
        for i in range(self.nmodels):
            qprofs.append(np.array([r[i] for r in res]))

        return qprofs

    def fit_step(self, abort_test=None, outfid=None) -> None:
        """Analyzes most recent step"""
        
        # Update models
        self.update_models()

        # Set attributes of "problem" for passing into multiprocessing routines
        newvars = [self.instrument.Q2TdTLdL(mQ, mx, mQ) for mQ, mx in zip(self.measQ, self.x)]
        setattr(self.problem, 'calcTdTLdL', newvars)
        setattr(self.problem, 'oversampling', self.oversampling)
        setattr(self.problem, 'resolution', self.instrument.resolution)

        # initialize mappers for Dream fit and for Q profile calculations
        mapper = MPMapper.start_mapper(self.problem, None, cpus=0)

        # set output stream
        if outfid is not None:
            monitor = StepMonitor(self.problem, outfid)
        else:
            monitor = ConsoleMonitor(self.problem)
        
        # Condition and run fit
        fitter = DreamFitPlus(self.problem)
        options=_fill_defaults(self.fit_options, fitter.settings)
        result = fitter.solve(mapper=mapper, monitors=[monitor], abort_test=abort_test, initial_population=self.restart_pop, **options)

        if not abort_test():
            # Save head state for initializing the next fit step
            _, chains, _ = fitter.state.chains()
            self.restart_pop = chains[-1, : ,:]

            # Analyze the fit state and save values
            fitter.state.keep_best()
            fitter.state.mark_outliers()

            step = self.steps[-1]
            step.chain_pop = chains[-1, :, :]
            step.draw = fitter.state.draw(thin=self.thinning)
            step.best_logp = fitter.state.best()[1]
            self.problem.setp(fitter.state.best()[0])
            step.final_chisq = self.problem.chisq_str()
            step.H, _, _ = calc_entropy(step.draw.points, select_pars=None, options=self.entropy_options)
            step.dH = self.init_entropy - step.H
            step.H_marg, _, _ = calc_entropy(step.draw.points, select_pars=self.sel, options=self.entropy_options)
            step.dH_marg = self.init_entropy_marg - step.H_marg

            # Calculate the Q profiles associated with posterior distribution
            print('Calculating %i Q profiles:' % (step.draw.points.shape[0]))
            init_time = time.time()
            step.qprofs = self.calc_qprofiles(step.draw.points)
            print('Calculation time: %f' % (time.time() - init_time))

        # Terminate the multiprocessing pool (required to avoid memory issues
        # if run is stopped after current fit step)
        MPMapper.stop_mapper(mapper)
        MPMapper.pool = None

    def take_step(self, allow_repeat=True) -> None:
        """Analyze the last fitted step and add the next one
        
        Procedure:
            1. Calculate the figures of merit
            2. Identify the next self.npoints data points
                to simulate/measure
            (1 and 2 are currently done in _fom_from_draw)
            3. Simulate the new data points
            4. Add a new step for fitting.

        Inputs:
            allow_repeat -- toggles whether to allow measurement at the same point over and over;
                            default True. (Can cause issues if MCMC fit isn't converged in
                            high-gradient areas, e.g. around the critical edge)
        """

        # Focus on the last step
        step = self.steps[-1]
        
        # Calculate figures of merit and proposed measurement times with forecasting
        print('Calculating figures of merit:')
        init_time = time.time()
        pts = step.draw.points[:, self.sel]

        # can scale parameters for entropy calculations. Helps with GMM.
        # TODO: check that this works. Does it need to be implemented in entropy.calc_entropy?
        if self.entropy_options['scale']:
            pts = copy.copy(pts) / self.par_scale[:, self.sel]

        foms, meastimes, bkgmeastimes, _, newpoints = self._fom_from_draw(pts, step.qprofs, select_ci_level=0.68, meas_ci_level=self.eta, n_forecast=self.npoints, allow_repeat=allow_repeat)
        print('Total figure of merit calculation time: %f' % (time.time() - init_time))

        # populate step foms
        # TODO: current analysis code can't handle multiple foms, could pass all of them in here
        step.foms, step.meastimes, step.bkgmeastimes = foms[0], meastimes[0], bkgmeastimes[0]

        points = self.request_data(newpoints, foms)

        return points

    def request_data(self, newpoints, foms=None) -> List[DataPoint]:
        """
        Requests new data. In simulation mode, generates new data points. In experiment mode, should
        add new data points to a point queue to be measured

        newpoints -- Required. New points to measure. List of items:
                    modelnum: int. Model number of new point
                    idx: int. Index of x value (self.x[modelnum])
                    newx: float. x value (self.x[modelnum])
                    new_meastime: float. Measurement time at new position
                    new_bkgmeastime: float. Background measurement time at new position.
                                            Typically 0 indicates to skip background measurement.
        foms -- Optional (default: None). Figures of merit. Used only for populating new data points
                    and plotting.
        """
        
        # Determine next measurement point(s).
        # Number of points to be used is determined from n_forecast (self.npoints)

        return []

    def add_step(self, points, use=True) -> None:
        """Adds a set of DataPoint objects as a new ExperimentStep

        Inputs:
            points: list of DataPoints for the step\
            use: boolean toggle for whether to use this step for plotting. Default true.
        """
        self.steps.append(ExperimentStep(points, use=use))

    def _apply_fom_penalties(self, foms, curmodel=None) -> List[np.ndarray]:
        """
        Applies any penalties that scale the figures of merit directly

        Inputs:
        foms -- list of of figure of merits, one for each model
        curmodel -- integer index of current model

        Returns:
        scaled_foms -- scaled list of figures of merit (list of numpy arrays)
        """

        if curmodel is None:
            curmodel = self.curmodel

        # Calculate switching penalty
        spenalty = [1.0 if j == curmodel else self.switch_penalty for j in range(self.nmodels)]

        # Perform scaling
        scaled_foms = [fom  / pen for fom, pen in zip(foms, spenalty)]

        return scaled_foms

    def _apply_time_penalties(self, foms, meastimes, bkgmeastimes, curmodel=None) -> List[np.ndarray]:
        """
        Applies any penalties that act to increase the measurement time, e.g. movement penalties or model switch time penalities
        NOTE: uses current state of the instrument (self.instrument.x).

        Inputs:
        foms -- list of of figure of merits, one for each model
        meastimes -- list of proposed measurement time vectors, one for each model
        bkgmeastimes -- list of proposed background measurement times, one for each model
        curmodel -- integer index of current model

        Returns:
        scaled_foms -- scaled list of figures of merit (list of numpy arrays)
        """

        if curmodel is None:
            curmodel = self.curmodel

        # Apply minimum to proposed measurement times
        min_meas_times = [np.maximum(np.full_like(meastime, self.min_meas_time), meastime) for meastime in meastimes]

        # Apply minimum to non-zero background measurement times
        min_bkgmeastimes = copy.deepcopy(bkgmeastimes)
        for bkgmeastime in min_bkgmeastimes:
            bkgmeastime[bkgmeastime > 0] = np.clip(bkgmeastime[bkgmeastime > 0], a_min = self.min_meas_time, a_max = None)

        # Calculate time penalty to switch models
        switch_time_penalty = [0.0 if j == curmodel else self.switch_time_penalty for j in range(self.nmodels)]

        # Add all movement time penalties together.
        movepenalty = [meastime / (meastime + bkgmeastime + self.instrument.movetime(x) + pen) for x, meastime, bkgmeastime, pen in zip(self.x, min_meas_times, min_bkgmeastimes, switch_time_penalty)]

        # Perform scaling
        scaled_foms = [fom * movepen for fom,movepen in zip(foms, movepenalty)]

        return scaled_foms

    def _fom_from_draw(self, pts: np.ndarray,
                        qprofs: List[np.ndarray],
                        select_ci_level: float = 0.68,
                        meas_ci_level: float = 0.68,
                        n_forecast: int = 1,
                        allow_repeat: bool = True) -> Tuple[List[List[np.ndarray]],
                                                            List[List[np.ndarray]],
                                                            List[float],
                                                            List[Tuple[int, int, float, float]]]:
        """ Calculate figure of merit from a set of draw points and associated q profiles
        
            Inputs:
            pts -- draw points. Should be already selected for marginalized paramters
            qprofs -- list of q profiles, one for each model of size <number of samples in pts> x <number of measQ values>
            select_ci_level -- confidence interval level to use for selection (default 0.68)
            meas_ci_level -- confidence interval level to target for measurement (default 0.68, typically use self.eta)
            n_forecast -- number of forecast steps to take (default 1)
            allow_repeat -- whether or not the same point can be measured twice in a row. Turn off to improve stability.

            Returns:
            all_foms -- list (one for each forecast step) of lists of figures of merit (one for each model)
            all_meastimes -- list (one for each forecast step) of lists of proposed measurement times (one for each model)
            all_H0 -- list (one for each forecast step) of maximum entropy (not entropy change) before that step
            all_new -- list of forecasted optimal data points (one for each forecast step). Each element in the list is a list
                        of properties of the new point with format: [<model number>, <x index>, <x value>, <measurement time>])
        """

        """shape definitions:
            X -- number of x values in xs
            D -- number of detectors
            N -- number of samples
            M -- number of samples after selecting those inside the confidence interval
            P -- number of marginalized parameters"""

        import matplotlib.pyplot as plt

        # Cycle through models, with model-specific x, Q, calculated q profiles, and measurement background level
        # Populate q vectors, interpolated q profiles (slow), and intensities
        intensities = list()
        intens_shapes = list()
        qs = list()
        xqbkgs = list()
        xdbkgs = list()
        xqprofs = list()
        init_time = time.time()
        modeldata = self.get_data()
        for mnum, (xs, Qth, qprof, qbkg_default, mdata) in enumerate(zip(self.x, self.measQ, qprofs, self.meas_bkg, modeldata)):

            # get the incident intensity and q values for all x values (should have same shape X x D).
            # flattened dimension is XD
            incident_neutrons = self.instrument.intensity(xs)
            init_shape = incident_neutrons.shape
            incident_neutrons = incident_neutrons.flatten()
            q = self.instrument.x2q(xs).flatten()

            # calculate the existing background uncertainty. This is used to determine if additional
            # background measurements must be performed
            specdata, bkgpdata, bkgmdata = mdata
            #spec = DataPoint2ReflData(Qth, specdata)
            #bkgp = DataPoint2ReflData(Qth, bkgpdata)
            #bkgm = DataPoint2ReflData(Qth, bkgmdata)
            
            bkg, bkgvar = interpolate_background(Qth, bkgpdata, bkgmdata)

            # if real measurement backgrounds are available, use them
            qbkg = bkg if bkg is not None else np.full_like(Qth, qbkg_default)
            
            # also calculate the background uncertainty. If real backgrounds are not available, use
            # 1.0 so they will always be measured.
            dbkg = np.sqrt(bkgvar) if bkgvar is not None else np.full_like(Qth, 1.0)

            if False:
                print(qprof.shape, qbkg.shape)
                if spec is not None:
                    plt.errorbar(spec.Qz, spec.v, spec.dv, fmt='.', capsize=4)
                    spec = reduce(Qth, specdata, bkgpdata, bkgmdata)
                    plt.errorbar(spec.Qz, spec.v, spec.dv, fmt='o', capsize=4)
                plt.plot(Qth, qbkg)
                plt.plot(Qth, qbkg + dbkg, Qth, qbkg - dbkg)
                plt.plot(Qth, np.median(qprof, axis=0), Qth, np.median(qprof, axis=0) + np.std(qprof, axis=0),
                        Qth, np.median(qprof, axis=0) - np.std(qprof, axis=0))
                plt.yscale('log')
                plt.show()

            if False:
                # define signal to background. For now, this is just a scaling factor on the effective rate
                # reference: Hoogerheide et al. J Appl. Cryst. 2022
                sbr = qprof / qbkg
                refl = qprof/(1+2/sbr)
                refl = np.clip(refl, a_min=0, a_max=None)
            else:
                refl = np.clip(qprof, a_min=0, a_max=None)
            
            # perform interpolation. xqprof should have shape N x XD. This is a slow step (and should only be done once)
            interp_refl = interp1d(Qth, refl, axis=1, fill_value=(refl[:,0], refl[:,-1]), bounds_error=False)
            xqprof = np.array(interp_refl(q))

            # interpolate the background. Should also have shape XD
            #interp_refl_bkg = interp1d(Qth, qbkg, fill_value=(qbkg[0], qbkg[-1]), bounds_error=False)
            #xqbkg = np.array(interp_refl_bkg(q))
            xqbkg, xdbkg2 = interpolate_background(q, bkgpdata, bkgmdata)
            xqbkg = xqbkg if xqbkg is not None else np.full_like(q, qbkg_default)
            xdbkg2 = xdbkg2 if xdbkg2 is not None else np.full_like(q, 1.0)

            intensities.append(incident_neutrons)
            intens_shapes.append(init_shape)
            qs.append(q)
            xqprofs.append(xqprof)
            xqbkgs.append(xqbkg)
            xdbkgs.append(np.sqrt(xdbkg2))

        print(f'Forecast setup time: {time.time() - init_time}')

        all_foms = list()
        all_meas_times = list()
        all_bkg_meas_times = list()
        all_H0 = list()
        all_new = list()
        org_curmodel = self.curmodel
        org_x = self.instrument.x

        """For each stage of the forecast, go through:
            1. Calculate the foms
            2. Select the new points
            3. Repeat
        """
        for i in range(n_forecast):
            init_time = time.time()
            Hlist = list()
            foms = list()
            meas_times = list()
            bkg_meas_times = list()
            #newidxs_select = list()
            newidxs_meas = list()
            newxqprofs = list()
            N, P = pts.shape
            minci_sel, maxci_sel =  int(np.floor(N * (1 - select_ci_level) / 2)), int(np.ceil(N * (1 + select_ci_level) / 2))
            minci_meas, maxci_meas =  int(np.floor(N * (1 - meas_ci_level) / 2)), int(np.ceil(N * (1 + meas_ci_level) / 2))
            H0, _, predictor = calc_entropy(pts, select_pars=None, options=self.entropy_options, predictor=None)   # already marginalized!!
            if predictor is not None:
                predictor.warm_start = True

            all_H0.append(H0)
            # cycle though models
            for incident_neutrons, init_shape, q, xqprof, xqbkg, xdbkg in zip(intensities, intens_shapes, qs, xqprofs, xqbkgs, xdbkgs):

                #init_time2a = time.time()
                # TODO: Shouldn't these already be sorted by the second step?
                idxs = np.argsort(xqprof, axis=0)
                #print(f'Sort time: {time.time() - init_time2a}')
                #print(idxs.shape)

                # Select new points and indices in CI. Now has dimension M x XD X P
                A = np.take_along_axis(pts[:, None, :], idxs[:, :, None], axis=0)[minci_sel:maxci_sel]
                
                #init_time2a = time.time()
                # calculate new index arrays and xqprof values
                # this also works: meas_sigma = 0.5*np.diff(np.take_along_axis(xqprof, idxs[[minci, maxci],:], axis=0), axis=0)
                newidx = idxs[minci_meas:maxci_meas]
                meas_xqprof = np.take_along_axis(xqprof, newidx, axis=0)#[minci:maxci]
                meas_sigma = 0.5 * (np.max(meas_xqprof, axis=0) - np.min(meas_xqprof, axis=0))
                sel_xqprof = np.take_along_axis(xqprof, idxs[minci_sel:maxci_sel], axis=0)#[minci:maxci]
                sel_sigma = 0.5 * (np.max(sel_xqprof, axis=0) - np.min(sel_xqprof, axis=0))

                #print(f'Sel calc time: {time.time() - init_time2a}')
                
                #sel_sigma = 0.5 * np.diff(np.take_along_axis(xqprof, idxs[[minci_sel, maxci_sel],:], axis=0), axis=0)
                #meas_sigma = 0.5 * np.diff(np.take_along_axis(xqprof, idxs[[minci_meas, maxci_meas],:], axis=0), axis=0)

                init_time2 = time.time()

                # Condition shape (now has dimension M X P X XD)
                A = np.moveaxis(A, -1, 1)
                Hs, _, predictor = calc_entropy(A, None, options=self.entropy_options, predictor=predictor)

                # Calculate measurement times (shape XD)
                med = np.median(xqprof, axis=0)
                xrefl_sel = (incident_neutrons * med * (sel_sigma / med) ** 2)
                xrefl_meas = (incident_neutrons * med * (meas_sigma / med) ** 2)
                meastime_sel = 1.0 / xrefl_sel
                meastime_meas = 1.0 / xrefl_meas

                # Compare uncertainty in background to expected uncertainty in reflectivity
                t_0 = 1.0 / (incident_neutrons * xqbkg * (xdbkg / xqbkg) ** 2)
                t_bkg = 1.0 / (incident_neutrons * xqbkg * (meas_sigma / xqbkg) ** 2)
                if False:
                    plt.plot(t_0)
                    plt.plot(t_bkg)
                    plt.yscale('log')
                    plt.show()

                    #plt.plot(xdbkg)
                    #plt.plot(meas_sigma)
                    #plt.yscale('log')
                    #plt.show()

                # Calculate time required to achieve target background; negative times indicate
                # that background measurement is not necessary
                # TODO: Should a_min be self.min_meas_time?
                bkg_meastime = np.clip(t_bkg - t_0, a_min=0.0, a_max=None)

                # apply min measurement time (turn this off initially to test operation)
                #meastime = np.maximum(np.full_like(meastime, self.min_meas_time), meastime)

                # figure of merit is dHdt (reshaped to X x D)
                dHdt = (H0 - Hs) / meastime_sel
                dHdt = np.reshape(dHdt, init_shape)

                # calculate fom and average time (shape X)
                fom = np.sum(dHdt, axis=1)
                meas_time = 1./ np.sum(1./np.reshape(meastime_meas, init_shape), axis=1)
                bkg_meas_time = 1./ np.sum(1./np.reshape(bkg_meastime, init_shape), axis=1)

                Hlist.append(Hs)
                foms.append(fom)
                meas_times.append(meas_time)
                bkg_meas_times.append(bkg_meas_time)
                newxqprofs.append(meas_xqprof)
                newidxs_meas.append(newidx)
                
            # populate higher-level lists
            all_foms.append(foms)
            all_meas_times.append(meas_times)
            all_bkg_meas_times.append(bkg_meas_times)

            # apply penalties
            scaled_foms = self._apply_fom_penalties(foms, curmodel=self.curmodel)
            scaled_foms = self._apply_time_penalties(scaled_foms, meas_times, bkg_meas_times, curmodel=self.curmodel)

            # remove current point from contention if allow_repeat is False
            if (not allow_repeat) & (self.instrument.x is not None):
                curidx = np.where(self.x[self.curmodel]==self.instrument.x)[0][0]
                scaled_foms[self.curmodel][curidx] = 0.0

            # perform point selection
            top_n = self._find_fom_maxima(scaled_foms, start=0)
            #print(top_n)
            if top_n is not None:
                _, mnum, idx = top_n
                newx = self.x[mnum][idx]
                new_meastime = max(meas_times[mnum][idx], self.min_meas_time)
                new_bkgmeastime = max(bkg_meas_times[mnum][idx], self.min_meas_time) if bkg_meas_times[mnum][idx] > 0 else 0.0
                
                all_new.append([mnum, idx, newx, new_meastime, new_bkgmeastime])
            else:
                break

            # apply point selection
            self.instrument.x = newx
            self.curmodel = mnum

            # choose new points. This is not straightforward if there is more than one detector, because
            # each point in XD may choose a different detector. We will choose without replacement by frequency.
            # idx_array has shape M x D
            idx_array = newidxs_meas[mnum].reshape(-1, *intens_shapes[mnum])[:, idx, :]
            #print(idx_array.shape)
            if idx_array.shape[1] == 1:
                # straightforward case, with 1 detector
                chosen = np.squeeze(idx_array)
            else:
                # select those that appear most frequently
                #print(idx_array.shape)
                freq = np.bincount(idx_array.flatten(), minlength=len(pts))
                freqsort = np.argsort(freq)
                chosen = freqsort[-idx_array.shape[0]:]
                
            newpts = pts[chosen]
            newxqprofs = [xqprof[chosen] for xqprof in xqprofs]

            # set up next iteration
            xqprofs = newxqprofs
            pts = newpts

            print(f'Forecast step {i}:\tNumber of samples: {N}\tCalculation time: {time.time() - init_time}')

        # reset instrument state
        self.instrument.x = org_x
        self.curmodel = org_curmodel

        return all_foms, all_meas_times, all_bkg_meas_times, all_H0, all_new

    def _find_fom_maxima(self, scaled_foms: List[np.ndarray],
                         start: int = 0) -> List[Tuple[float, int, int]]:
        """Finds all maxima in the figure of merit, including the end points
        
            Inputs:
            scaled_foms -- figures of merit. They don't have to be scaled, but it should be the "final"
                            FOM with any penalties already applied
            start -- index of the first peak to select. Defaults to zero (start with the highest).

            Returns:
            top_n -- sorted list 

        """

        # TODO: Implement a more random algorithm (probably best appplied in a different function 
        #       to the maxima themselves). One idea is to define a partition function
        #       Z = np.exp(fom / np.mean(fom)) - 1. The fom is then related to ln(Z(x)). Points are chosen
        #       using np.random.choice(x, size=self.npoints, p=Z/np.sum(Z)).
        #       I think that penalties will have to be applied differently, potentially directly to Z.

        # finds a single point to measure
        maxQs = []
        maxidxs = []
        maxfoms = []

        # find maximum figures of merit in each model

        for fom, Qth in zip(scaled_foms, self.measQ):
            
            # a. calculate whether gradient is > 0
            dfom = np.sign(np.diff(np.append(np.insert(fom, 0, 0),0))) < 0
            # b. find zero crossings
            xings = np.diff(dfom.astype(float))
            maxidx = np.where(xings>0)[0]
            maxfoms.append(fom[maxidx])
            maxQs.append(Qth[maxidx])
            maxidxs.append(maxidx)

        # condition the maximum indices
        maxidxs_m = [[fom, m, idx] for m, (idxs, mfoms) in enumerate(zip(maxidxs, maxfoms)) for idx, fom in zip(idxs, mfoms)]
        #print(maxidxs_m)
        # select top point
        top_n = sorted(maxidxs_m, reverse=True)[start:min(start+1, len(maxidxs_m))][0]

        # returns sorted list of lists, each with entries [max fom value, model number, measQ index]
        return top_n


    def save(self, fn) -> None:
        """Save a pickled version of the experiment"""

        for step in self.steps[:-2]:
            step.draw.state = None

        with open(fn, 'wb') as f:
            dill.dump(self, f, recurse=True)

    @classmethod
    def load(cls, fn) -> None:
        """ Load a pickled version of the experiment
        
        Usage: <variable> = SimReflExperiment.load(<filename>)
        """

        with open(fn, 'rb') as f:
            exp = dill.load(f)
        
        # for back compatibility
        if not hasattr(exp, 'entropy_options'):
            exp.entropy_options = default_entropy_options
        
        return exp


class AutoReflExperiment(AutoReflBase):
    """
    Autonomous reflectometry experiment with instrument control.

    Only works with single-model problems at the moment.

    Additional Inputs:

    filename -- string representing the filename. Recommended to have a scheme to use a unique name.

    """

    def __init__(self, filename: str,
                       problem: FitProblem,
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
                       select_pars: Union[list, None] = None) -> None:
        
        super().__init__(problem, Q, instrument, eta, npoints, switch_penalty,
                       switch_time_penalty, fit_options, entropy_options, oversampling,
                       meas_bkg, startmodel, min_meas_time, select_pars)

        self.filename = filename

    def get_data(self) -> List[Tuple[ReflData, Union[ReflData, None],
                                     Union[ReflData, None]]]:

        #data = self.instrument.load_data(self.filename)

        #for dataset in data:
        #    pass

        # TODO: make sure each entry is intensity normalized!
        return super().get_data()

    def request_data(self, newpoints, foms) -> List[List[MeasurementPoint]]:
        """
        Requests new data. Creates data points to be added, e.g. to a measurement point queue.

        Returns:
        List of measurement point lists, each with 1 or 3 dictionaries containing specular or
            specular + 2 background measurement points
        """
        
        step_id = len(self.steps)

        def create_msg(point_id: int, model_num: int, x: float, t: float, intent: Intent, merit:float):

            return MeasurementPoint(step_id,
                                    point_id=point_id,
                                    base=DataPoint(x, t, model_num,
                                                    (self.instrument.T(x)[0],
                                                        self.instrument.dT(x)[0],
                                                        self.instrument.L(x)[0],
                                                        self.instrument.dL(x)[0],
                                                        None,
                                                        self.instrument.intensity(x)[0]),
                                                    intent=intent),
                                    movements=self.instrument.trajectoryData(x, intent=intent))

            # NOTE: DataPoint.T may be different from actual T for backgorunds;
            # this is the reference T for specular data

        points = []

        for i, (pt, fom) in enumerate(zip(newpoints, foms)):
            ptlist = []
            mnum, idx, newx, new_meastime, new_bkgmeastime = pt

            ptlist.append(create_msg(i, mnum, newx, new_meastime, Intent.spec, fom[mnum][idx]))

            if new_bkgmeastime > 0:
                ptlist.append(create_msg(i, mnum, newx, new_bkgmeastime / 2.0, Intent.backp, fom[mnum][idx]))
                ptlist.append(create_msg(i, mnum, newx, new_bkgmeastime / 2.0, Intent.backm, fom[mnum][idx]))

            points.append(ptlist)

        return points