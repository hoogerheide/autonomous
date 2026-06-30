from .simulation import calc_expected_R

default_fit_options = {'pop': 10, 'burn': 1000, 'steps': 500, 'init': 'lhs', 'alpha': 0.001}


def _MP_calc_qprofile(problem_point_pair):
    """ Calculate q profiles based on a sample draw, for use with
        multiprocessing

        Adapted from refl1d.mapper
    """

    # given a problem and a sample draw and a Q-vector, calculate the profiles associated with each sample
    problem_id, point = problem_point_pair
    if problem_id != MPMapper.problem_id:
        #print(f"Fetching problem {problem_id} from namespace")
        # Problem is pickled using dill when it is available
        try:
            import dill
            MPMapper.problem = dill.loads(MPMapper.namespace.pickled_problem)
        except ImportError:
            MPMapper.problem = MPMapper.namespace.problem
        MPMapper.problem_id = problem_id
    return _calc_qprofile(MPMapper.problem, point)

def _calc_qprofile(calcproblem, point):
    """Calculation function of q profiles using _MP_calc_qprofiles
    
    Inputs:
    calcproblem -- a bumps.BaseFitProblem or bumps.MultiFitProblem, prepopulated
                    with attributes:
                        calcTdTLdL (derived from SimReflExperiment.measQ via Q2TdTLdL);
                        oversampling
                        resolution (either 'normal' or 'uniform', instrument-dependent)
    point -- parameter vector
    """
    
    mlist = list(calcproblem.models)
    qprof = list()
    for m, newvar in zip(mlist, calcproblem.calcTdTLdL):
        calcproblem.setp(point)
        calcproblem.chisq_str()
        Rth = calc_expected_R(m, *newvar, oversampling=calcproblem.oversampling, resolution=calcproblem.resolution)
        qprof.append(Rth)

    return qprof
