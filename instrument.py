import numpy as np
import json
import warnings
from autorefl import q2a, a2q
from reflred.resolution import divergence
from reflred.candor import edges

class ReflectometerBase(object):
    def __init__(self) -> None:
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
        self.sample_width = None
        self._S3Offset = 0.0
        self._R12 = 1.0

        # default t(Q) scaling parameters. Here t(Q) \propto _mon0 + _mon1 * Q^Qpow
        self._mon0 = 0.0
        self._mon1 = 1.0
        self._Qpow = 2.0

    def x2q(self, x):
        pass

    def x2a(self, x):
        pass

    def qrange2xrange(self, qmin, qmax):
        pass

    def intensity(self, x):
        pass

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

    def get_slits(self, x):
        x = np.array(x, ndmin=1)
        sintheta = np.sin(np.radians(self.x2a(x)))
        s2 = self.footprint * sintheta / ((self._R12 + 1) * self._L2S / self._L12 + 1)
        s1 = self._R12 * s2
        s3 = (s1 + s2) * (self._L2S + self._LS3) / self._L12 + s2 + self._S3Offset
        s4 = (s1 + s2) * (self._L2S + self._LS3 + self._L34) / self._L12 + s2 + self._S3Offset

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

class MAGIK(ReflectometerBase):
    """ MAGIK Reflectometer
    x = Q """
    def __init__(self) -> None:
        super().__init__()
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

        # load calibration files
        try:
            d_intens = np.loadtxt('calibration/magik_intensity_hw106.refl')

            self.p_intens = np.polyfit(d_intens[:,0], d_intens[:,1], 3, w=1/d_intens[:,2])
        except OSError:
            warnings.warn('MAGIK calibration files not found, using defaults')
            self.p_intens = np.array([ 5.56637543e+02,  7.27944632e+04,  2.13479802e+02, -4.37052050e+01])

    def x2q(self, x):
        return x

    def x2a(self, x):
        return q2a(x, self._L)

    def qrange2xrange(self, bounds):
        return min(bounds), max(bounds)

    def intensity(self, x):
        news1 = self.get_slits(x)[0]
        incident_neutrons = np.polyval(self.p_intens, news1)
    
        return np.array(incident_neutrons, ndmin=2).T

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

class CANDOR(ReflectometerBase):
    """ CANDOR Reflectometer with a single bank
    x = T """
    def __init__(self, bank=0) -> None:
        super().__init__()
        
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

        # load wavelength calibration
        wvcal = np.flipud(np.loadtxt(f'calibration/DetectorWavelengths_PG_integrate_sumeff_bank{bank}.csv', delimiter=',', usecols=[1, 2]))
        self._L = wvcal[:,0]
        self._dL = wvcal[:,1]

        # load intensity calibration
        with open('calibration/flowcell_d2o_r12_2_5_maxbeam_60_qoverlap0_751388_unpolarized_intensity.json', 'r') as f:
            d = json.load(f)
        
        self.intens_calib = np.squeeze(np.array(d['outputs'][0]['v']))
        self.s1_intens_calib = np.squeeze(d['outputs'][0]['x'])
        crit = self.s1_intens_calib < 1
        self.p_intens = np.polynomial.polynomial.polyfit(self.s1_intens_calib[crit], self.intens_calib[crit], 2)

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

        news1 = self.get_slits(x)[0]
        incident_neutrons = [np.interp(news1, self.s1_intens_calib, intens) for intens in self.intens_calib.T]
    
        return np.array(incident_neutrons, ndmin=2).T
    
    def intensity_interp(self, x):
        news1 = self.get_slits(x)[0]
        incident_neutrons = np.polynomial.polynomial.polyval(news1, self.p_intens)
    
        return np.array(incident_neutrons, ndmin=2).T    

    def meastime(self, x, totaltime):

        q = a2q(np.array(x), 5.0)
        f = self._mon0 + self._mon1 * q ** self._Qpow

        return totaltime * f / sum(f)

    def get_slits(self, x):
        s1, s2, s3, _ = super().get_slits(x)

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
    

class LIQREF(ReflectometerBase):
    """ LIQREF TOF Reflectometer
    x = integer index of predefined buffers """
    def __init__(self, bank=0) -> None:
        super().__init__()
        
        self.name = 'LIQREF'
        self.xlabel = r'Buffer index'
        self.resolution = 'normal'
        self.topspeed = 1./ 40
        # As of 6/24/2024:
        # Top velocity: 1.0 deg /  40 sec        

        # instrument geometry
        self._L12 = 1350.
        self._L2S = 135.
        self.footprint = 25.
        self._R12 = 1.5
        self.sample_width = np.inf

        # load calibration files
        self.load_calibration_files()

    def load_calibration_files(self):
        import glob

        beam_current = 1.4 # mA

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
            # if not currently at a position, assume no movement time
            movetimes = np.zeros_like(x).tolist()
        
        else:

            movetimes = []
            for ix in x:
                if ix == self.x:
                    # if the instrument doesn't have to move, move time is zero.
                    movetimes.append(0)
                else:
                    # calculate two-theta movement time
                    curT = self.calibration_data[self.x]['T']
                    newT = self.calibration_data[ix]['T']
                    two_theta_movetime = 2 * abs(curT - newT) / self.topspeed

                    # calculate chopper rephasing time
                    cur_lowL = self.calibration_data[self.x]['L'][0]
                    new_lowL = self.calibration_data[ix]['L'][0]

                    if np.isclose(cur_lowL, new_lowL, atol=0.1):
                        chopper_movetime = 0.0
                    else:
                        chopper_movetime = 45.0

                    # choose maximum of chopper rephasing time and two theta movement time
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

            # note that this is not exactly the same as np.diff(Ls) / 2
            dLs.append(-0.5 * ((Ls - center_points[:-1]) + (center_points[1:] - Ls)))
        return [self.calibration_data[ix]['L']*0.02 for ix in x]

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

        def flatten(a: list):
            return np.array([iia for ia in a for iia in ia])

        # calculate all variables
        _Q = flatten(self.x2q(measx))
        _T = flatten(self.T(measx))
        _dT = flatten(self.dT(measx))
        _L = flatten(self.L(measx))
        _dL = flatten(self.dL(measx))

        # calculate q bin edges
        q_edges = edges(measQ, extended=True)
        nbins = len(q_edges) - 1

        # calculate flattened bin index
        bin_index = np.searchsorted(q_edges, _Q) - 1

        # calculate normalization factor
        sum_w = np.bincount(bin_index, minlength=nbins)
        sum_w += (sum_w == 0)  # protect against divide by zero

        # Combine wavelengths
        sum_L = np.bincount(bin_index, weights=_L, minlength=nbins)
        sum_dLsq = np.bincount(bin_index, weights=(_dL**2+_L**2), minlength=nbins)
        sum_dL = np.bincount(bin_index, weights=_dL**2, minlength=nbins)
        bar_L = sum_L/sum_w
        bar_dL = np.sqrt(sum_dLsq/sum_w - (sum_L/sum_w)**2)
        #bar_dL = np.sqrt(sum_dL/sum_w)

        # Combine angles
        sum_T = np.bincount(bin_index, weights=_T, minlength=nbins)
        sum_dT = np.bincount(bin_index, weights=_dT**2, minlength=nbins)
        bar_T = sum_T/sum_w
        bar_dT = np.sqrt(sum_dT/sum_w)

        # find indices corresponding to requested values and return results
        idxs = np.searchsorted(measQ, qs) + 1
        #res = [(bar_T[idx], bar_dT[idx], bar_L[idx], bar_dL[idx]) for idx in idxs]

        return (bar_T[idxs], bar_dT[idxs], bar_L[idxs], bar_dL[idxs])
    
