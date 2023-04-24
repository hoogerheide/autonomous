from typing import Union, List, Tuple

import numpy as np
from bumps.fitters import DreamFit, _fill_defaults
from reflred.candor import edges, _rebin_bank, QData
from reflred.background import apply_background_subtraction
from reflred.scale import apply_intensity_norm
from reflred.refldata import ReflData, Sample, Detector, Monochromator
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
        bkg = None  # no background to subtract

    return bkg.x, bkg.variance

def DataPoint2ReflData(data: List[DataPoint], normbase='time') -> Tuple[Union[ReflData, None], Union[ReflData, None]]:
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
    idata['Ninc'] = np.round(idata['Ninc'])
    crit = idata['Ninc'] > 0

    for k in idata.keys():
        idata[k] = np.array(idata[k])[crit]

    if len(idata['T']):

        v = idata['N']
        dv = np.sqrt(v)
        vinc = idata['Ninc']
        dvinc = np.sqrt(vinc)
        T, dT, L, dL = (idata[key] for key in ['T', 'dT', 'L', 'dL'])
        wavelength_resolution = dL[:,None,None]  / L[:,None,None]

        data = ReflData(monochromator=Monochromator(wavelength=L[:,None,None], wavelength_resolution=wavelength_resolution),
                    sample=Sample(angle_x=T[:,None,None]),
                    angular_resolution=dT[:,None,None],
                    detector=Detector(angle_x=2*T[:,None,None], wavelength=L[:,None,None], wavelength_resolution=wavelength_resolution),
                    _v=v[:,None,None], _dv=dv[:,None,None], Qz_basis='actual', normbase=normbase)

        inc = ReflData(monochromator=Monochromator(wavelength=L[:,None,None], wavelength_resolution=wavelength_resolution),
                        sample=Sample(angle_x=T[:,None,None]),
                        angular_resolution=dT[:,None,None],
                        detector=Detector(angle_x=2*T[:,None,None], wavelength=L[:,None,None], wavelength_resolution=wavelength_resolution),
                        _v=vinc[:, None, None], _dv=dvinc[:,None, None], Qz_basis='actual', normbase=normbase)

        apply_intensity_norm(data, inc)

        return data
    
    else:

        return None

def reduce(Qbasis, specdata: List[DataPoint], backp: List[DataPoint], backm: List[DataPoint], normbase='time'):
    """Reduces data
    
        1. Bin all data
        2. Subtract background
        3. Normalize by intensity
        
        Inputs:
        Qbasis -- measurement Q values on which to bin
        specdata -- list of DataPoint objects containing specular data
        backp -- list of DataPoint objects containing back+ data
        backm -- list of DataPoint objects containing back- data
        normbase -- how to normalize data (default 'time')
        """

    spec = DataPoint2ReflData(specdata, normbase=normbase)
    bkgp = DataPoint2ReflData(backp, normbase=normbase) if backp is not None else None, None
    bkgm = DataPoint2ReflData(backm, normbase=normbase) if backm is not None else None, None

    if spec is not None:
        q_edges = edges(Qbasis, extended=True)

        if bkgp is not None:
            qz, dq, v, dv, _Ti, _dT, _L, _dL = _rebin_bank(bkgp, 0, q_edges, 'poisson')
            bkgpdata = QData(bkgp, qz, dq, v, dv, _Ti, _dT, _L, _dL)
        
        if bkgm is not None:
            qz, dq, v, dv, _Ti, _dT, _L, _dL = _rebin_bank(bkgm, 0, q_edges, 'poisson')
            bkgmdata = QData(bkgm, qz, dq, v, dv, _Ti, _dT, _L, _dL)

        # Bin values
        qz, dq, vspec, dvspec, _Ti, _dT, _L, _dL = _rebin_bank(spec, 0, q_edges, 'poisson')
        specdata = QData(spec, qz, dq, vspec, dvspec, _Ti, _dT, _L, _dL)

        # subtract background
        apply_background_subtraction(specdata, bkgpdata, bkgmdata)

        # TODO: divide intensity here instead of in DataPoint2ReflData
        # apply_intensity_norm(specdata, incdata)

        return _Ti, _dT, _L, _dL, specdata.v, specdata.dv, qz, dq

    else:

        return tuple([np.array([]) for _ in range(8)])

class DreamFitPlus(DreamFit):
    def __init__(self, problem):
        super().__init__(problem)

    def solve(self, monitors=None, abort_test=None, mapper=None, initial_population=None, **options):
        from bumps.dream import Dream
        from bumps.fitters import MonitorRunner, initpop
        if abort_test is None:
            abort_test = lambda: False
        options = _fill_defaults(options, self.settings)
        #print(options, flush=True)

        if mapper:
            self.dream_model.mapper = mapper
        self._update = MonitorRunner(problem=self.dream_model.problem,
                                     monitors=monitors)

        population = initpop.generate(self.dream_model.problem, **options) if initial_population is None else initial_population
        pop_size = population.shape[0]
        draws, steps = int(options['samples']), options['steps']
        if steps == 0:
            steps = (draws + pop_size-1) // pop_size
        # TODO: need a better way to announce number of steps
        # maybe somehow print iteration # of # iters in the monitor?
        print("# steps: %d, # draws: %d"%(steps, pop_size*steps))
        population = population[None, :, :]
        sampler = Dream(model=self.dream_model, population=population,
                        draws=pop_size * steps,
                        burn=pop_size * options['burn'],
                        thinning=options['thin'],
                        monitor=self._monitor, alpha=options['alpha'],
                        outlier_test=options['outliers'],
                        DE_noise=1e-6)

        self.state = sampler.sample(state=self.state, abort_test=abort_test)

        self._trimmed = self.state.trim_portion() if options['trim'] else 1.0
        #print("trimming", options['trim'], self._trimmed)
        self.state.mark_outliers(portion=self._trimmed)
        self.state.keep_best()
        self.state.title = self.dream_model.problem.name

        # TODO: Temporary hack to apply a post-mcmc action to the state vector
        # The problem is that if we manipulate the state vector before saving
        # it then we will not be able to use the --resume feature.  We can
        # get around this by just not writing state for the derived variables,
        # at which point we can remove this notice.
        # TODO: Add derived/visible variable support to other optimizers
        fn, labels = getattr(self.problem, 'derive_vars', (None, None))
        if fn is not None:
            self.state.derive_vars(fn, labels=labels)
        visible_vars = getattr(self.problem, 'visible_vars', None)
        if visible_vars is not None:
            self.state.set_visible_vars(visible_vars)
        integer_vars = getattr(self.problem, 'integer_vars', None)
        if integer_vars is not None:
            self.state.set_integer_vars(integer_vars)

        x, fx = self.state.best()

        # Check that the last point is the best point
        #points, logp = self.state.sample()
        #assert logp[-1] == fx
        #print(points[-1], x)
        #assert all(points[-1, i] == xi for i, xi in enumerate(x))
        return x, -fx


