import numpy as np
import json
import warnings
from typing import Tuple, List, Union
from reductus.reflred.candor import edges, Candor
from reductus.reflred.resolution import divergence
from reductus.reflred.intent import Intent
from reductus.reflred.nexusref import NCNRNeXusRefl, load_nexus_entries
from reductus.reflred.refldata import ReflData, Monitor, Detector, Sample, Slit, Monochromator

def q2a(q, L):
    return np.degrees(np.arcsin(np.array(q)*L/(4*np.pi)))

def a2q(T, L):
    return 4*np.pi/L * np.sin(np.radians(T))

class ReflectometerBase(object):
    def __init__(self, bank=None) -> None:
        self._L = None
        self._dL = None
        self.xlabel = ''
        self.name = None
        self.resolution = 'normal'

        # assumes that the detector arm motion is the slowest component
        self.topspeed = 1.0 # units of degrees / second for detector arm
        self.basespeed = 0.2 # units of degrees / second for detector arm
        self.acceleration = 0.5 # units of degrees / second^2 for detector arm
        self.x = None   # current position

        # instrument geometry
        self._L12 = None
        self._L2S = None
        self._LS3 = None
        self._L34 = None
        self.footprint = None
        self.fixed_slits = None
        self.sample_width = None
        self._S3Offset = 0.0
        self._R12 = 1.0

        # default t(Q) scaling parameters. Here t(Q) \propto _mon0 + _mon1 * Q^Qpow
        self._mon0 = 0.0
        self._mon1 = 1.0
        self._Qpow = 2.0
        
        # Loader-specific parameters
        self._data_loader = None
        self._data_type = None

        # Support for multidimensional detectors
        # None if a point detector
        self.bank = bank

        # Intensity calibration table: set by calibrate_intensity() or populated as a
        # simulation fallback in subclass __init__. Shape: s1_intens_calib (N,),
        # intens_calib (N,) for single-bank or (N, D) for multi-bank instruments.
        self.s1_intens_calib = None
        self.intens_calib = None

        # Polarization: None for unpolarized; set to a dict mapping xs name →
        # {motor_name: value} for polarized campaigns (set at campaign start by
        # AutoReflDevice, not at instrument construction time).
        self.polarization_states: dict | None = None

    def x2q(self, x):
        raise NotImplementedError

    def x2a(self, x):
        raise NotImplementedError

    def qrange2xrange(self, qmin, qmax):
        raise NotImplementedError

    def intensity(self, x):
        if self.s1_intens_calib is not None:
            s1 = self.get_slits(np.array(x, ndmin=1))[0]
            if self.intens_calib.ndim == 1:
                return np.array(np.interp(s1, self.s1_intens_calib, self.intens_calib), ndmin=2).T
            else:
                incident_neutrons = [np.interp(s1, self.s1_intens_calib, self.intens_calib[:, d])
                                     for d in range(self.intens_calib.shape[1])]
                return np.array(incident_neutrons, ndmin=2).T
        raise NotImplementedError

    def meastime(self, x, totaltime):

        q = self.x2q(np.array(x))

        f = self._mon0 + self._mon1 * q ** self._Qpow

        return totaltime * f / sum(f)

    def T(self, x):
        
        return self.x2a(x)

    def dT(self, x):
        usesample = True if self.footprint > self.sample_width else False 
        return divergence(self.get_slits(x), self.get_slit_distances(), T=np.array(self.T(x)), sample_width=self.sample_width, use_sample=usesample)

    def L(self, x):
        
        return np.array(np.ones_like(x) * self._L, ndmin=1)

    def dL(self, x):
        
        return np.array(np.ones_like(x) * self._dL, ndmin=1)

    def Q2TdTLdL(self, qs, measx, measQ):
        """
        Converts a Q value into T, dT, L, dL variables.
        Replaces gen_new_variables. Used for calculating R(Q) profiles
        Logic derived from reflred.candor._rebin_bank

        Inputs:
        qs -- Q values to convert to variables
        measx -- possible x values
        measQ -- possible Q bins

        Returns:
        T -- average angle over all measx, one for each value of qs
        dT -- average angular divergence
        L -- average wavelength
        dL -- average wavelength spread
        """

        # calculate all variables
        _Q = self.x2q(measx)
        _T = self.T(measx)
        _dT = self.dT(measx)
        _L = self.L(measx)
        _dL = self.dL(measx)

        # calculate q bin edges
        q_edges = edges(measQ, extended=True)
        nbins = len(q_edges) - 1

        # calculate flattened bin index
        bin_index = np.searchsorted(q_edges, _Q).ravel() - 1

        # calculate normalization factor
        sum_w = np.bincount(bin_index, minlength=nbins)
        sum_w += (sum_w == 0)  # protect against divide by zero

        # Combine wavelengths
        sum_L = np.bincount(bin_index, weights=_L.ravel(), minlength=nbins)
        sum_dLsq = np.bincount(bin_index, weights=(_dL.ravel()**2+_L.ravel()**2), minlength=nbins)
        bar_L = sum_L/sum_w
        bar_dL = np.sqrt(sum_dLsq/sum_w - (sum_L/sum_w)**2)

        # Combine angles
        sum_T = np.bincount(bin_index, weights=_T.ravel(), minlength=nbins)
        sum_dT = np.bincount(bin_index, weights=_dT.ravel()**2, minlength=nbins)
        bar_T = sum_T/sum_w
        bar_dT = np.sqrt(sum_dT/sum_w)

        # find indices corresponding to requested values and return results
        idxs = np.searchsorted(measQ, qs) + 1
        #res = [(bar_T[idx], bar_dT[idx], bar_L[idx], bar_dL[idx]) for idx in idxs]

        return (bar_T[idxs], bar_dT[idxs], bar_L[idxs], bar_dL[idxs])


    def get_slits(self, x, force_angle=None):
        """
        Returns slit values. If "fixed_slits" is set to a tuple of slit values (s1, s2, s3, s4),
        this is returned. Otherwise, calculates slits based on "footprint".

        Inputs:
        x -- the x value at which to get the slits
        force_angle (optional) -- force theta to be this particular value (ignores x). Useful for
                                    backgrounds where a constant footprint is desired but theta
                                    deviates from the specular condition
        """
        if self.fixed_slits is None:
            
            x = np.array(x, ndmin=1)

            if force_angle is None:
                sintheta = np.sin(np.radians(self.x2a(x)))
            else:
                sintheta = np.sin(np.radians(np.array(force_angle, ndmin=1)))

            s2 = self.footprint * sintheta / ((self._R12 + 1) * self._L2S / self._L12 + 1)
            s1 = self._R12 * s2
            s3 = (s1 + s2) * (self._L2S + self._LS3) / self._L12 + s2 + self._S3Offset
            s4 = (s1 + s2) * (self._L2S + self._LS3 + self._L34) / self._L12 + s2 + self._S3Offset

        else:
            s1, s2, s3, s4 = self.fixed_slits

        return s1, s2, s3, s4

    def get_slit_distances(self):

        return -(self._L12 + self._L2S), -self._L2S, self._LS3, self._LS3 + self._L34

    def movetime(self, x):

        if self.x is None:
            return np.array([0])
        else:
            x = np.array(x, ndmin=1)

            # convert x to angle units
            newT = self.x2a(x)
            curT = self.x2a(self.x)

            # detector arm motion is 2 * dTheta
            dx = 2 * np.abs(newT - curT)

            t = np.empty_like(dx)

            # total time that arm is accelerating
            accel_t = (self.topspeed - self.basespeed) / self.acceleration

            # total distance that can be traversed in one acceleration / deceleration cycle without achieving top speed
            max_accel_dx = 2 * (0.5 * self.acceleration * accel_t ** 2 + self.basespeed * accel_t)

            # select points in the acceleration only regime
            accel_crit = (dx < max_accel_dx)

            # top velocity reached
            t[~accel_crit] = 2 * accel_t + (dx[~accel_crit] - max_accel_dx) / self.topspeed

            # top velocity not reached
            t[accel_crit] = 2 * self.basespeed / self.acceleration * (-1 + np.sqrt(1 + 2 * (dx[accel_crit] / 2) * self.acceleration / self.basespeed ** 2))

            return t

    def _flipper_motor_names(self) -> List[str]:
        """Returns flipper motor names from polarization_states, or [] if unpolarized."""
        if self.polarization_states is None:
            return []
        return list(next(iter(self.polarization_states.values())).keys())

    def trajectoryMotors(self) -> List[str]:
        """
        Generates list of motors moved for trajectory
        """

        raise NotImplementedError

    def trajectoryData(self, x, intent, dtheta_bkg=0.5) -> List[str]:
        """
        Generates string for NICE instrument control movement.

        Backgrounds are implemented as sample angle movements. Assumes detectorTableMotor has been
        zeroed for the bank of interest

        Inputs:
        x -- the (single) x value at which to generate trajectory movement string
        intent -- sets the intent of the measurement (specular, backp, or backm). Compared against
                    reflred.intent.Intent.(spec, backp, or backm)
        dtheta_bkg (default=0.5) -- angle offset in degrees for backp and backm measurements
        """

        raise NotImplementedError

    def load_data(self, filename: str) -> Tuple[ReflData, ReflData, ReflData]:
        """
        Loads data of the appropriate type
        """

        spec, bkgp, bkgm = (self._data_type(
                                load_nexus_entries(filename, None, entry, False, self._data_loader))
                                for entry in ['spec', 'bkgp', 'bkgm'])

        return spec, bkgp, bkgm
    
    def x2ReflData(self, x: float,
                   v: Union[np.ndarray, list, float], dv: Union[np.ndarray, list, float],
                   intent: Intent) -> List[ReflData]:
        """
        Converts single x value into ReflData object.

        Inputs:
        x -- instrument configuration (single value, array, list). Required. One ReflData object
             will be created for each x.
        v -- number of counts if intent is specular or background; otherwise target incident intensity
                counts for intensity. Required. Should have the same shape as self.T(x)[0]
        dv -- error in v (standard deviation). Required. Same shape as v
        intent -- reflred.intent.Intent object. Required.

        Returns:
        rd -- ReflData object


        """

        _T = self.T(x)[0]
        _dT = self.dT(x)[0]
        _L = self.L(x)[0]
        _dL = self.dL(x)[0]

        s1, s2, s3, s4 = self.get_slits(x)
        d1, d2, d3, d4 = self.get_slit_distances()

        rd = ReflData(slit1=Slit(distance=d1, x=s1),
                        slit2=Slit(distance=d2, x=s2),
                        slit3=Slit(distance=d3, x=s3),
                        slit4=Slit(distance=d4, x=s4),
                        monochromator=Monochromator(wavelength=_L, wavelength_resolution=_dL),
                        sample=Sample(angle_x=_T, angle_x_target=_T),
                        angular_resolution=_dT,
                        detector=Detector(angle_x=2*_T,
                                        angle_x_target=2*_T,
                                        wavelength=_L,
                                        wavelength_resolution=_dL),
                        Qz_basis='target', normbase='none', _v=v, _dv=dv)
    
        rd.intent = intent

        return rd

class MAGIK(ReflectometerBase):
    """ MAGIK Reflectometer
    x = Q """
    def __init__(self, bank=None) -> None:
        super().__init__(bank=bank)
        self._L = np.array([5.0])
        self._dL = 0.01648374 * self._L / 2.355
        self.xlabel = r'$Q_z$ (' + u'\u212b' + r'$^{-1}$)'
        self.name = 'MAGIK'
        self.resolution = 'normal'
        self.topspeed = 1.0
        self.basespeed = 0.2
        self.acceleration = 0.5
        # As of 1/24/2022:
        # Base: 0.2 deg / sec
        # Acceleration: 0.5 deg / sec^2
        # Top velocity: 1.0 deg / sec

        # instrument geometry
        self._L12 = 1403.
        self._L2S = 330.
        self._LS3 = 229.
        self._L34 = 939.
        self.footprint = 45.
        self._S3Offset = 1.22
        self._R12 = 1.0
        self.sample_width = np.inf

        # best practice Q scaling
        self._mon0 = 30.0
        self._mon1 = 1250.
        self._Qpow = 2.0

        # data loader
        self._data_loader = NCNRNeXusRefl
        self._data_type = ReflData

        # load calibration files
        try:
            d_intens = np.loadtxt('calibration/magik_intensity_hw106.refl')
            self.d_intens = d_intens
            self.p_intens, self.p_intens_cov = np.polyfit(d_intens[:,0], d_intens[:,1], 3, w=1/d_intens[:,2], cov='unscaled')
        except OSError:
            warnings.warn('MAGIK calibration files not found, using defaults')
            self.p_intens = np.array([ 5.56637543e+02,  7.27944632e+04,  2.13479802e+02, -4.37052050e+01])

        # populate simulation fallback table from polynomial
        _s1_grid = np.linspace(0.01, 5.0, 200)
        self.s1_intens_calib = _s1_grid
        self.intens_calib = np.polyval(self.p_intens, _s1_grid)

    def x2q(self, x):
        return x

    def x2a(self, x):
        return q2a(x, self._L)

    def qrange2xrange(self, bounds):
        return min(bounds), max(bounds)

    def intensity(self, x):
        return super().intensity(x)

    def intensity_with_error(self, x):
        # NOTE: Code validated using a bumps polynomial fit
        x = np.array(x, ndmin=1)
        s1 = []
        incident_neutron_rate = []
        incident_neutron_rate_error = []

        for ix in x:
            news1 = self.get_slits(ix)[0]
            s1.append(news1)
            iincident_neutron_rate = np.polyval(self.p_intens, news1)
            j = np.flipud(np.array([news1 ** i for i in range(len(self.p_intens))]))
            iincident_neutron_rate_error = np.sqrt(j.T.dot(self.p_intens_cov.dot(j)))
            incident_neutron_rate.append(iincident_neutron_rate)
            incident_neutron_rate_error.append(iincident_neutron_rate_error)
    
        #print(np.array(incident_neutron_rate).shape, np.array(incident_neutron_rate_error).shape)
        return np.array(s1, ndmin=2).reshape(x.shape), np.array(incident_neutron_rate, ndmin=2).reshape(x.shape), np.array(incident_neutron_rate_error, ndmin=2).reshape(x.shape)

    def T(self, x):

        x = np.array(x, ndmin=1)
        return np.broadcast_to(self.x2a(x), (len(self._L), len(x))).T

    def dT(self, x):
        x = np.array(x, ndmin=1)
        dTs = super().dT(x).T
        return np.broadcast_to(dTs, (len(self._L), len(x))).T

    def L(self, x):
        x = np.array(x, ndmin=1)
        return np.broadcast_to(self._L, (len(x), len(self._L)))

    def dL(self, x):
        x = np.array(x, ndmin=1)
        return np.broadcast_to(self._dL, (len(x), len(self._L)))

    def trajectoryMotors(self) -> List[str]:
        """
        Generates list of motors moved for trajectory
        """

        return ['detectorAngle', 'sampleAngle', 'slitAperture1', 'slitAperture2', 'slitAperture3',
                'slitAperture4', 'counter', 'pointDetector'] + self._flipper_motor_names()

    def trajectoryData(self, x, intent, dtheta_bkg=0.5) -> List[str]:
        """
        Generates string for NICE instrument control movement.

        Backgrounds are implemented as sample angle movements.

        Inputs:
        x -- the (single) x value at which to generate trajectory movement string
        intent -- sets the intent of the measurement (specular, backp, or backm). Compared against
                    reflred.intent.Intent.(spec, backp, or backm)
        dtheta_bkg (default=0.5) -- angle offset in degrees for backp and backm measurements
        """

        movements = []

        # detector angle
        detectorAngle = 2 * np.squeeze(self.x2a(x))
        movements.append(['detectorAngle', f'{detectorAngle:0.3f}'])

        # sample angle depends on intent
        if intent == Intent.spec:
            sampleAngle = detectorAngle / 2.0
        elif intent == Intent.backp:
            sampleAngle = detectorAngle / 2.0 + dtheta_bkg
        elif intent == Intent.backm:
            sampleAngle = detectorAngle / 2.0 - dtheta_bkg

        movements.append(['sampleAngle', f'{sampleAngle:0.3f}'])

        # slit values
        s1, s2, s3, s4 = self.get_slits(x, force_angle=sampleAngle)
        movements.append(['slitAperture1', f'{s1[0]:0.4f}'])
        movements.append(['slitAperture2', f'{s2[0]:0.4f}'])
        movements.append(['slitAperture3', f'{s3[0]:0.4f}'])
        movements.append(['slitAperture4', f'{s4[0]:0.4f}'])

        # return flattened list
        return [item for movelist in movements for item in movelist]


class CANDOR(ReflectometerBase):
    """ CANDOR Reflectometer with a single bank
    x = T """

    # Flipper motor positions per cross-section.
    # TODO: confirm exact NICE motor names and values with instrument scientists.
    polarization_states_half: dict = {
        'mm': {'frontPolarization': 'DOWN', 'backPolarization': 'DOWN'},
        'pp': {'frontPolarization': 'UP',   'backPolarization': 'UP'},
    }
    polarization_states_full: dict = {
        'mm': {'frontPolarization': 'DOWN', 'backPolarization': 'DOWN'},
        'mp': {'frontPolarization': 'DOWN', 'backPolarization': 'UP'},
        'pm': {'frontPolarization': 'UP',   'backPolarization': 'DOWN'},
        'pp': {'frontPolarization': 'UP',   'backPolarization': 'UP'},
    }

    def __init__(self, bank=0) -> None:
        super().__init__(bank=bank)

        self.name = 'CANDOR'
        self.xlabel = r'$\Theta$ $(\degree)$'
        self.resolution = 'uniform'
        self.topspeed = 2.0
        self.basespeed = 0.1
        self.acceleration = 0.1
        # As of 1/24/2022:
        # Base: 0.1 deg / sec
        # Acceleration: 0.1 deg / sec^2
        # Top velocity: 2.0 deg / sec        
        # NOTE: dominated by acceleration and base for most moves!!

        # instrument geometry
        self._L12 = 4000.
        self._L2S = 356.
        self._LS3 = 356.
        self._L34 = 3000.
        self.footprint = 45.
        self._S3Offset = 5.0
        self._R12 = 2.5
        self.detector_mask = 8.0
        self.sample_width = np.inf

        # best practice Q scaling
        self._mon0 = 20.0
        self._mon1 = 20000.
        self._Qpow = 3.0

        # data loading
        self._data_type = Candor
        self._data_loader = Candor

        # load wavelength calibration
        wvcal = np.flipud(np.loadtxt(f'calibration/DetectorWavelengths_PG_integrate_sumeff_bank{bank}.csv', delimiter=',', usecols=[1, 2]))
        self._L = wvcal[:,0]
        self._dL = wvcal[:,1]

        # load intensity calibration
        with open('calibration/flowcell_d2o_r12_2_5_maxbeam_60_qoverlap0_751388_unpolarized_intensity.json', 'r') as f:
            d = json.load(f)
        
        self.intens_calib = np.squeeze(np.array(d['outputs'][0]['v']))
        self.dintens_calib = np.squeeze(np.array(d['outputs'][0]['dv']))
        self.s1_intens_calib = np.squeeze(d['outputs'][0]['x'])
        #ps1 = np.polynomial.polynomial.polyfit(s1, intens, 1)

        print(self.intens_calib.shape)

    def x2q(self, x):
        return a2q(self.T(x), self.L(x))

    def x2a(self, x):
        return x

    def qrange2xrange(self, qbounds):
        qbounds = np.array(qbounds)
        minx = q2a(min(qbounds), max(self._L))
        maxx = q2a(max(qbounds), min(self._L))
        return minx, maxx

    def intensity(self, x):
        return super().intensity(x)

    def meastime(self, x, totaltime):

        q = a2q(np.array(x), 5.0)
        f = self._mon0 + self._mon1 * q ** self._Qpow

        return totaltime * f / sum(f)

    def get_slits(self, x, force_angle=None):
        s1, s2, s3, _ = super().get_slits(x, force_angle=force_angle)

        return s1, s2, s3, self.detector_mask

    def T(self, x):
        x = np.array(x, ndmin=1)
        return np.broadcast_to(x, (len(self._L), len(x))).T

    def dT(self, x):
        x = np.array(x, ndmin=1)
        dTs = super().dT(x).T
        return np.broadcast_to(dTs, (len(self._L), len(x))).T

    def L(self, x):
        x = np.array(x, ndmin=1)
        return np.broadcast_to(self._L, (len(x), len(self._L)))

    def dL(self, x):
        x = np.array(x, ndmin=1)
        return np.broadcast_to(self._dL, (len(x), len(self._L)))

    def trajectoryMotors(self) -> List[str]:
        """
        Generates list of motors moved for trajectory
        """

        return ['detectorArmAngle', 'sampleAngle', 'slitAperture1', 'slitAperture2',
                'slitAperture3', 'counter', 'multiDetector'] + self._flipper_motor_names()

    def trajectoryData(self, x, intent, dtheta_bkg=0.5) -> List[str]:
        """
        Generates string for NICE instrument control movement.

        Backgrounds are implemented as sample angle movements. Assumes detectorTableMotor has been
        zeroed for the bank of interest

        Inputs:
        x -- the (single) x value at which to generate trajectory movement string
        intent -- sets the intent of the measurement (specular, backp, or backm). Compared against
                    reflred.intent.Intent.(spec, backp, or backm)
        dtheta_bkg (default=0.5) -- angle offset in degrees for backp and backm measurements
        """

        movements = []

        # detector angle
        detectorAngle = 2.0 * np.squeeze(self.x2a(x))
        movements.append(['detectorArmAngle', f'{detectorAngle:0.4f}'])

        # sample angle depends on intent
        if intent == Intent.spec:
            sampleAngle = detectorAngle / 2.0
        elif intent == Intent.backp:
            sampleAngle = detectorAngle / 2.0 + dtheta_bkg
        elif intent == Intent.backm:
            sampleAngle = detectorAngle / 2.0 - dtheta_bkg

        movements.append(['sampleAngle', f'{sampleAngle:0.4f}'])

        # slit values
        s1, s2, s3, s4 = self.get_slits(x, force_angle=sampleAngle)
        movements.append(['slitAperture1', f'{s1[0]:0.4f}'])
        movements.append(['slitAperture2', f'{s2[0]:0.4f}'])
        movements.append(['slitAperture3', f'{s3[0]:0.4f}'])

        # return flattened list
        return [item for movelist in movements for item in movelist]


class LIQREF(ReflectometerBase):
    """ LIQREF TOF Reflectometer
    x = integer index of predefined angle/chopper configurations.
    Each configuration covers a contiguous Q range at a fixed angle.
    Calibration data is loaded from files in calibration/liqref/.
    """
    def __init__(self) -> None:
        super().__init__()

        self.name = 'LIQREF'
        self.xlabel = r'Buffer index'
        self.resolution = 'normal'
        self.topspeed = 1. / 40
        # Top velocity: 1.0 deg / 40 sec

        # instrument geometry
        self._L12 = 1350.
        self._L2S = 135.
        self.footprint = 25.
        self._R12 = 1.5
        self.sample_width = np.inf

        self.load_calibration_files()

    def load_calibration_files(self):
        import glob

        beam_current = 1. / 0.656  # s / mC

        caldata = list()
        for f in glob.glob('calibration/liqref/*.txt'):
            Q, L, N, Ne = np.loadtxt(f, unpack=True)

            # convert counts / mC to counts / s
            N *= beam_current
            Ne *= beam_current

            with open(f, 'r') as fn:
                headerdata = fn.readlines()[:3]
                T = float(headerdata[0].split(':')[-1])
                s1 = float(headerdata[1].split(':')[-1].split('x')[0])
                s2 = float(headerdata[2].split(':')[-1].split('x')[0])

            caldata.append(dict(Q=Q, L=L, N=N, Ne=Ne, T=T, s1=s1, s2=s2))

        caldata.sort(key=lambda c: c['Q'][0])
        self.calibration_data = caldata

    def get_slits(self, x):
        x = np.array(x, ndmin=1)
        s1 = np.array([self.calibration_data[ix]['s1'] for ix in x])
        s2 = np.array([self.calibration_data[ix]['s2'] for ix in x])
        return s1, s2

    def get_slit_distances(self):
        return -(self._L12 + self._L2S), -self._L2S

    def x2q(self, x):
        x = np.array(x, ndmin=1)
        return [self.calibration_data[ix]['Q'] for ix in x]

    def x2a(self, x):
        x = np.array(x, ndmin=1)
        return [self.calibration_data[ix]['T'] for ix in x]

    def qrange2xrange(self, qbounds):
        qbounds = np.array(qbounds)
        minx = next(ix for ix, cd in enumerate(self.calibration_data) if cd['Q'][-1] > min(qbounds))
        maxx = [ix for ix, cd in enumerate(self.calibration_data) if cd['Q'][0] < max(qbounds)][-1]
        return minx, maxx

    def intensity(self, x):
        x = np.array(x, ndmin=1)
        return [self.calibration_data[ix]['N'] for ix in x]

    def meastime(self, x, totaltime):
        q = a2q(np.array(x), 5.0)
        f = self._mon0 + self._mon1 * q ** self._Qpow
        return totaltime * f / sum(f)

    def movetime(self, x):
        x = np.array(x, ndmin=1)
        if self.x is None:
            movetimes = np.zeros_like(x).tolist()
        else:
            movetimes = []
            for ix in x:
                if ix == self.x:
                    movetimes.append(0)
                else:
                    curT = self.calibration_data[self.x]['T']
                    newT = self.calibration_data[ix]['T']
                    two_theta_movetime = 2 * abs(curT - newT) / self.topspeed

                    cur_lowL = self.calibration_data[self.x]['L'][0]
                    new_lowL = self.calibration_data[ix]['L'][0]

                    chopper_movetime = 0.0 if np.isclose(cur_lowL, new_lowL, atol=0.1) else 45.0

                    movetimes.append(max(two_theta_movetime, chopper_movetime))
        return movetimes

    def T(self, x):
        x = np.array(x, ndmin=1)
        return [self.calibration_data[ix]['T'] * np.ones_like(self.calibration_data[ix]['L'])
                for ix in x]

    def dT(self, x):
        x = np.array(x, ndmin=1)
        return [ReflectometerBase.dT(self, ix)[0] * np.ones_like(self.calibration_data[ix]['L'])
                for ix in x]

    def L(self, x):
        x = np.array(x, ndmin=1)
        return [self.calibration_data[ix]['L'] for ix in x]

    def dL(self, x):
        x = np.array(x, ndmin=1)
        dLs = []
        for ix in x:
            Ls = self.calibration_data[ix]['L']
            center_points = 0.5 * (Ls[1:] + Ls[:-1])
            first_center_point = Ls[0] - (center_points[0] - Ls[0])
            last_center_point = Ls[-1] + (Ls[-1] - center_points[-1])
            center_points = np.insert(center_points, 0, first_center_point)
            center_points = np.append(center_points, last_center_point)
            dLs.append(-0.5 * ((Ls - center_points[:-1]) + (center_points[1:] - Ls)))
        return dLs

    def Q2TdTLdL(self, qs, measx, measQ):
        """
        Converts Q values into (T, dT, L, dL) by averaging over all measx configurations.
        Overrides the base class version to handle ragged per-configuration arrays.
        """
        def flatten(a):
            return np.array([iia for ia in a for iia in ia])

        _Q = flatten(self.x2q(measx))
        _T = flatten(self.T(measx))
        _dT = flatten(self.dT(measx))
        _L = flatten(self.L(measx))
        _dL = flatten(self.dL(measx))

        q_edges = edges(measQ, extended=True)
        nbins = len(q_edges) - 1

        bin_index = np.searchsorted(q_edges, _Q) - 1

        sum_w = np.bincount(bin_index, minlength=nbins)
        sum_w += (sum_w == 0)

        sum_L = np.bincount(bin_index, weights=_L, minlength=nbins)
        sum_dL = np.bincount(bin_index, weights=_dL ** 2, minlength=nbins)
        bar_L = sum_L / sum_w
        bar_dL = np.sqrt(sum_dL / sum_w)

        sum_T = np.bincount(bin_index, weights=_T, minlength=nbins)
        sum_dT = np.bincount(bin_index, weights=_dT ** 2, minlength=nbins)
        bar_T = sum_T / sum_w
        bar_dT = np.sqrt(sum_dT / sum_w)

        idxs = np.searchsorted(measQ, qs) + 1
        return (bar_T[idxs], bar_dT[idxs], bar_L[idxs], bar_dL[idxs])
