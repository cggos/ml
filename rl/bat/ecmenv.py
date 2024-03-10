import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import sys
# sys.path.append(os.path.join('.', 'esctoolbox'))
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
import random
import esctoolbox
# from esctoolbox.dyn_model.funcs import simCell_ecm,processDynamic, processDynamic_short
# from esctoolbox.dyn_model.models import DataModel, DataModelshort, ModelOcv, ModelDyn
from pathlib import Path
import pybamm
import pandas as pd
from scipy.optimize import fminbound, nnls, minimize_scalar
from scipy.signal import dlsim, dlti
import matplotlib.pyplot as plt

def update_input(current):
    noise = np.random(-1,1)[0]
    update_input = {
        "Input currrent [A]": current + noise,
    }

    return update_input

class DataModel:
    """
    Data from battery script tests. Requires the Script class which reads the
    csv file and assigns the data to class attributes.
    """

    def __init__(self, temp, csvfiles):
        """
        Initialize from script data.
        """
        self.temp = temp
        self.s1 = Script(csvfiles[0])
        self.s2 = Script(csvfiles[1])
        self.s3 = Script(csvfiles[2])

class DataModelshort:
    """
    Data from battery script tests. Requires the Script class which reads the
    csv file and assigns the data to class attributes.
    """

    def __init__(self, temp, csvfiles):
        """
        Initialize from script data.
        """
        self.temp = temp
        self.s1 = Scriptshort(csvfiles[0])

class Script:
    """
    Object to represent script data.
    """

    def __init__(self, csvfile):
        """
        Initialize with data from csv file.
        """
        df = pd.read_csv(csvfile)
        time = df['time'].values
        step = df[' step'].values
        current = df[' current'].values
        voltage = df[' voltage'].values
        chgAh = df[' chgAh'].values
        disAh = df[' disAh'].values

        self.time = time
        self.step = step
        self.current = current
        self.voltage = voltage
        self.chgAh = chgAh
        self.disAh = disAh

class Scriptshort:
    """
    Object to represent script data.
    """

    def __init__(self, csvfile):
        """
        Initialize with data from csv file.
        """
        df = pd.read_csv(csvfile)
        time    = df['time'].values
        current = df[' current'].values
        voltage = df[' voltage'].values
        chgAh   = df[' chgAh'].values
        disAh   = df[' disAh'].values

        self.time = time
        self.current = current
        self.voltage = voltage
        self.chgAh = chgAh
        self.disAh = disAh

class ModelOcv:
    """
    Model representing OCV results.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, OCV0, OCVrel, SOC, OCV, SOC0, SOCrel, OCVeta, OCVQ):
        self.OCV0 = np.array(OCV0)
        self.OCVrel = np.array(OCVrel)
        self.SOC = np.array(SOC)
        self.OCV = np.array(OCV)
        self.SOC0 = np.array(SOC0)
        self.SOCrel = np.array(SOCrel)
        self.OCVeta = np.array(OCVeta)
        self.OCVQ = np.array(OCVQ)

    @classmethod
    def load(cls, pfile):
        """
        Load attributes from json file where pfile is string representing
        path to the json file.
        """
        ocv = json.load(open(pfile, 'r'))
        return cls(ocv['OCV0'], ocv['OCVrel'], ocv['SOC'], ocv['OCV'], ocv['SOC0'], ocv['SOCrel'], ocv['OCVeta'], ocv['OCVQ'])

class ModelDyn:
    """
    Model representing results from the dynamic calculations.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.temps = None
        self.etaParam = None
        self.QParam = None
        self.GParam = None
        self.M0Param = None
        self.MParam = None
        self.R0Param = None
        self.RCParam = None
        self.RParam = None
        self.SOC = None
        self.OCV0 = None
        self.OCVrel = None
        self.OCV = None
        self.SOC0 = None
        self.SOCrel = None

def OCVfromSOCtemp(soc, temp, model):
    """ OCV function """
    SOC = model.SOC          # force to be column vector
    OCV0 = model.OCV0        # force to be column vector
    OCVrel = model.OCVrel    # force to be column vector

    # if soc is scalar then make it a vector
    soccol = np.asarray(soc)
    if soccol.ndim == 0:
        soccol = soccol[None]

    tempcol = temp * np.ones(np.size(soccol))

    diffSOC = SOC[1] - SOC[0]           # spacing between SOC points - assume uniform
    ocv = np.zeros(np.size(soccol))     # initialize output to zero
    I1, = np.where(soccol <= SOC[0])    # indices of socs below model-stored data
    I2, = np.where(soccol >= SOC[-1])   # and of socs above model-stored data
    I3, = np.where((soccol > SOC[0]) & (soccol < SOC[-1]))   # the rest of them
    I6 = np.isnan(soccol)               # if input is "not a number" for any locations

    # for voltages less than lowest stored soc datapoint, extrapolate off
    # low end of table
    if I1.any():
        dv = (OCV0[1] + tempcol*OCVrel[1]) - (OCV0[0] + tempcol*OCVrel[0])
        ocv[I1] = (soccol[I1] - SOC[0])*dv[I1]/diffSOC + OCV0[0] + tempcol[I1]*OCVrel[0]

    # for voltages greater than highest stored soc datapoint, extrapolate off
    # high end of table
    if I2.any():
        dv = (OCV0[-1] + tempcol*OCVrel[-1]) - (OCV0[-2] + tempcol*OCVrel[-2])
        ocv[I2] = (soccol[I2] - SOC[-1])*dv[I2]/diffSOC + OCV0[-1] + tempcol[I2]*OCVrel[-1]

    # for normal soc range, manually interpolate (10x faster than "interp1")
    I4 = (soccol[I3] - SOC[0])/diffSOC  # using linear interpolation
    I5 = np.floor(I4)
    I5 = I5.astype(int)
    I45 = I4 - I5
    omI45 = 1 - I45
    ocv[I3] = OCV0[I5]*omI45 + OCV0[I5+1]*I45
    ocv[I3] = ocv[I3] + tempcol[I3]*(OCVrel[I5]*omI45 + OCVrel[I5+1]*I45)
    ocv[I6] = 0     # replace NaN SOCs with zero voltage
    return ocv

def SOCfromOCVtemp(ocv,temp,model):
    OCV = model.OCV          # force to be column vector
    SOC0 = model.SOC0        # force to be column vector
    SOCrel = model.SOCrel    # force to be column vector

    # if soc is scalar then make it a vector
    ocvcol = np.asarray(ocv)
    if ocvcol.ndim == 0:
        ocvcol = ocvcol[None]

    tempcol = temp * np.ones(np.size(ocvcol))

    diffOCV = OCV[1] - OCV[0]           # spacing between OCV points - assume uniform
    soc = np.zeros(np.size(ocvcol))     # initialize output to zero
    I1, = np.where(ocvcol <= OCV[0])    # indices of ocvs below model-stored data
    I2, = np.where(ocvcol >= OCV[-1])   # and of ocvs above model-stored data
    I3, = np.where((ocvcol > OCV[0]) & (ocvcol < OCV[-1]))   # the rest of them
    I6 = np.isnan(ocvcol)               # if input is "not a number" for any locations

    # for ocvs less than lowest voltage, extrapolate off
    # low end of table
    if I1.any():
        dz = (SOC0[1] + tempcol*SOCrel[1]) - (SOC0[0] + tempcol*SOCrel[0])
        soc[I1] = (ocvcol[I1] - OCV[0])*dz[I1]/diffOCV + SOC0[0] + tempcol[I1]*SOCrel[0] 
    
    # for ocvs greater than highest voltage, extrapolate off
    # high end of table
    if I2.any():
        dz = (SOC0[-1] + tempcol*SOCrel[-1]) - (SOC0[-2] + tempcol*SOCrel[-2])
        soc[I2] = (ocvcol[I2] - OCV[-1])*dz[I2]/diffOCV + SOC0[-1] + tempcol[I2]*SOCrel[-1]

    # for normal ocv range, manually interpolate (10x faster than "interp1")
    I4 = (ocvcol[I3] - OCV[0])/diffOCV  # using linear interpolation
    I5 = np.floor(I4)
    I5 = I5.astype(int)
    I45 = I4 - I5
    omI45 = 1 - I45
    soc[I3] = SOC0[I5]*omI45 + SOC0[I5+1]*I45
    soc[I3] = soc[I3] + tempcol[I3]*(SOCrel[I5]*omI45 + SOCrel[I5+1]*I45)
    soc[I6] = 0     # replace NaN OCVs with zero SOC

    # time = np.arange(len(soc))
    # plt.figure("SOC")
    # plt.plot(time[::10]/60, soc[::10], label='soc')
    # plt.xlabel('Time (min)')
    # plt.ylabel('soc')
    # plt.title(f'Current at T = {temp} C')
    # plt.show()
    return soc

def SISOsubid(y, u, n):
    """
    Identify state-space "A" matrix from input-output data.
    y: vector of measured outputs
    u: vector of measured inputs
    n: number of poles in solution
    A: discrete-time state-space state-transition matrix.

    Theory from "Subspace Identification for Linear Systems Theory - Implementation
    - Applications" Peter Van Overschee / Bart De Moor (VODM) Kluwer Academic
      Publishers, 1996. Combined algorithm: Figure 4.8 page 131 (robust). Robust
      implementation: Figure 6.1 page 169.

    Code adapted from "subid.m" in "Subspace Identification for Linear Systems"
    toolbox on MATLAB CENTRAL file exchange, originally by Peter Van Overschee,
    Dec. 1995
    """
    
    ny = len(y)
    i = 2*n
    twoi = 4*n

    # Determine the number of columns in the Hankel matrices
    j = ny - twoi + 1

    # Make Hankel matrices Y and U
    Y = np.zeros((twoi, j))
    U = np.zeros((twoi, j))

    for k in range(2*i):
        Y[k] = y[k:k+j]
        U[k] = u[k:k+j]

    # Compute the R factor
    UY = np.concatenate((U, Y))     # combine U and Y into one array
    _, r = np.linalg.qr(UY.T)       # QR decomposition
    R = r.T                         # transpose of upper triangle

    # STEP 1: Calculate oblique and orthogonal projections
    # ------------------------------------------------------------------

    Rf = R[-i:]                                 # future outputs
    Rp = np.concatenate((R[:i], R[2*i:3*i]))    # past inputs and outputs
    Ru = R[i:twoi, :twoi]                       # future inputs

    RfRu = np.linalg.lstsq(Ru.T, Rf[:, :twoi].T, rcond=None)[0].T
    RfRuRu = RfRu.dot(Ru)
    tm1 = Rf[:, :twoi] - RfRuRu
    tm2 = Rf[:, twoi:4*i]
    Rfp = np.concatenate((tm1, tm2), axis=1)    # perpendicular future outputs

    RpRu = np.linalg.lstsq(Ru.T, Rp[:, :twoi].T, rcond=None)[0].T
    RpRuRu = RpRu.dot(Ru)
    tm3 = Rp[:, :twoi] - RpRuRu
    tm4 = Rp[:, twoi:4*i]
    Rpp = np.concatenate((tm3, tm4), axis=1)    # perpendicular past inputs and outputs

    # The oblique projection is computed as (6.1) in VODM, page 166.
    # obl/Ufp = Yf/Ufp * pinv(Wp/Ufp) * (Wp/Ufp)
    # The extra projection on Ufp (Uf perpendicular) tends to give
    # better numerical conditioning (see algo on VODM page 131)

    # Funny rank check (SVD takes too long)
    # This check is needed to avoid rank deficiency warnings

    nmRpp = np.linalg.norm(Rpp[:, 3*i-3:-i], ord='fro')
    if nmRpp < 1e-10:
        # oblique projection as (Rfp*pinv(Rpp')') * Rp
        Ob = Rfp.dot(np.linalg.pinv(Rpp.T).T).dot(Rp)
    else:
        # oblique projection as (Rfp/Rpp) * Rp
        Ob = (np.linalg.lstsq(Rpp.T, Rfp.T, rcond=None)[0].T).dot(Rp)

    # STEP 2: Compute weighted oblique projection and its SVD
    #         Extra projection of Ob on Uf perpendicular
    # ------------------------------------------------------------------

    ObRu = np.linalg.lstsq(Ru.T, Ob[:, :twoi].T, rcond=None)[0].T
    ObRuRu = ObRu.dot(Ru)
    tm5 = Ob[:, :twoi] - ObRuRu
    tm6 = Ob[:, twoi:4*i]
    WOW = np.concatenate((tm5, tm6), axis=1)

    U, S, _ = np.linalg.svd(WOW, full_matrices=False)
    ss = S       # In np.linalg.svd S is already the diagonal, generally ss = diag(S)

    # STEP 3: Partitioning U into U1 and U2 (the latter is not used)
    # ------------------------------------------------------------------

    U1 = U[:, :n]       # determine U1

    # STEP 4: Determine gam = Gamma(i) and gamm = Gamma(i-1)
    # ------------------------------------------------------------------

    gam = U1 @ np.diag(np.sqrt(ss[:n]))
    gamm = gam[0:(i-1),:]
    gam_inv = np.linalg.pinv(gam)               # pseudo inverse of gam
    gamm_inv = np.linalg.pinv(gamm)             # pseudo inverse of gamm

    # STEP 5: Determine A matrix (also C, which is not used)
    # ------------------------------------------------------------------

    tm7 = np.concatenate((gam_inv @ R[3*i:4*i, 0:3*i], np.zeros((n,1))), axis=1)
    tm8 = R[i:twoi, 0:3*i+1]
    Rhs = np.vstack((tm7, tm8))
    tm9 = gamm_inv @ R[3*i+1:4*i, 0:3*i+1]
    tm10 = R[3*i:3*i+1, 0:3*i+1]
    Lhs = np.vstack((tm9, tm10))
    sol = np.linalg.lstsq(Rhs.T, Lhs.T, rcond=None)[0].T    # solve least squares for [A; C]
    A = sol[0:n, 0:n]                           # extract A

    return A

def minfn(data, model, theTemp, doHyst):
    """
    Using an assumed value for gamma (already stored in the model), find optimum
    values for remaining cell parameters, and compute the RMS error between true
    and predicted cell voltage
    """

    alltemps = [d.temp for d in data]
    ind, = np.where(np.array(alltemps) == theTemp)[0]

    G = abs(model.GParam[ind])

    Q = abs(model.QParam[ind])
    eta = abs(model.etaParam[ind])
    RC = abs(model.RCParam[ind])
    numpoles = len(RC)

    ik = data[ind].s1.current.copy()
    vk = data[ind].s1.voltage.copy()
    tk = np.arange(len(vk))
    etaik = ik.copy()
    etaik[ik < 0] = etaik[ik < 0] * eta

    hh = 0*ik
    sik = 0*ik
    fac = np.exp(-abs(G * etaik/(3600*Q)))

    for k in range(1, len(ik)):
        hh[k] = (fac[k-1]*hh[k-1]) - ((1-fac[k-1])*np.sign(ik[k-1]))
        sik[k] = np.sign(ik[k])
        if abs(ik[k]) < Q/100:
            sik[k] = sik[k-1]

    # First modeling step: Compute error with model = OCV only
    vest1 = data[ind].OCV
    verr = vk - vest1

    # Second modeling step: Compute time constants in "A" matrix
    y = -np.diff(verr)
    u = np.diff(etaik)
    A = SISOsubid(y, u, numpoles)

    # Modify results to ensure real, preferably distinct, between 0 and 1

    eigA = np.linalg.eigvals(A)
    eigAr = eigA + 0.001 * np.random.normal(loc=0.0, scale=1.0, size=eigA.shape)
    eigA[eigA != np.conj(eigA)] = abs(eigAr[eigA != np.conj(eigA)]) # Make sure real
    eigA = np.real(eigA)                                            # Make sure real
    eigA[eigA<0] = abs(eigA[eigA<0])    # Make sure in range 
    eigA[eigA>1] = 1 / eigA[eigA>1]
    RCfact = np.sort(eigA)
    RCfact = RCfact[-numpoles:]
    # print(f"RCfact is {RCfact}")
    RC = -1 / np.log(RCfact)

    # Compute RC time constants as Plett's Matlab ESCtoolbox 
    # nup = numpoles
    # while 1:
    #     A = SISOsubid(y, u, nup)

    #     # Modify results to ensure real, preferably distinct, between 0 and 1
    #     eigA = np.linalg.eigvals(A)
    #     eigA = np.real(eigA[eigA == np.conj(eigA)])   # Make sure real
    #     eigA = eigA[(eigA>0) & (eigA<1)]    # Make sure in range 
    #     okpoles = len(eigA)
    #     nup = nup + 1
    #     if okpoles >= numpoles:
    #         break
    #     # print(nup)

    # RCfact = np.sort(eigA)
    # RCfact = RCfact[-numpoles:]
    # RC = -1 / np.log(RCfact)

    # Simulate the R-C filters to find R-C currents
    stsp = dlti(np.diag(RCfact), np.vstack(1-RCfact), np.eye(numpoles), np.zeros((numpoles, 1))) 
    [tout, vrcRaw, xout] = dlsim(stsp, etaik)

    # Third modeling step: Hysteresis parameters
    if doHyst:
        H = np.column_stack((hh, sik, -etaik, -vrcRaw))
        W = nnls(H, verr)
        M = W[0][0]
        M0 = W[0][1]
        R0 = W[0][2]
        Rfact = W[0][3:].T
    else:
        H = np.column_stack((-etaik, -vrcRaw))
        W = np.linalg.lstsq(H,verr, rcond=None)[0]
        M = 0
        M0 = 0
        R0 = W[0]
        Rfact = W[1:].T

    idx, = np.where(np.array(model.temps) == data[ind].temp)[0]
    model.R0Param[idx] = R0
    model.M0Param[idx] = M0
    model.MParam[idx] = M
    model.RCParam[idx] = RC.T
    model.RParam[idx] = Rfact.T
    # print(R0,
    #       RC,
    #       Rfact)

    soc = SOCfromOCVtemp(vest1, data[ind].temp, model)

    # plt.figure("SOC")
    # plt.plot(tk[::10]/60, soc[::10], label='soc')
    # plt.xlabel('Time (min)')
    # plt.ylabel('soc')
    # plt.title(f'Current at T = {data[ind].temp} C')
    # plt.legend(loc='best', numpoints=1)
    # # plt.show()

    vest2 = vest1 + M*hh + M0*sik - R0*etaik - vrcRaw @ Rfact.T
    verr = vk - vest2

    # plot voltages
    # plt.figure(1)
    # plt.plot(tk[::10]/60, vk[::10], label='voltage')
    # plt.plot(tk[::10]/60, vest1[::10], label='vest1 (OCV)')
    # plt.plot(tk[::10]/60, vest2[::10], label='vest2 (DYN)')
    # plt.xlabel('Time (min)')
    # plt.ylabel('Voltage (V)')
    # plt.title(f'Voltage and estimates at T = {data[ind].temp} C')
    # plt.legend(loc='best', numpoints=1)
    # #plt.show()

    # plot modeling errors
    # plt.figure(1)
    # plt.plot(tk[::10]/60, verr[::10], label='verr')
    # plt.xlabel('Time (min)')
    # plt.ylabel('Error (V)')
    # plt.title(f'Modeling error at T = {data[ind].temp} C')
    # plt.legend(loc='best', numpoints=1)
    # # plt.show()

    # Compute RMS error only on data roughly in 5% to 95% SOC
    v1 = OCVfromSOCtemp(0.95, data[ind].temp, model)[0]
    
    v2 = OCVfromSOCtemp(0.05, data[ind].temp, model)[0]
    N1 = np.where(vk < v1)[0][0]
    N2 = np.where(vk < v2)[0][0]
    # print(N1,N2)

    rmserr = np.sqrt(np.mean(verr[N1:N2]**2))
    cost = np.sum(rmserr)
    # print(f'RMS error = {cost*1000:.2f} mV')

    return cost, model, vest2

def simCell_ecm(i, model, z0, theTemp, doHyst):
    """
    Using an assumed value for gamma (already stored in the model), find optimum
    values for remaining cell parameters, and compute the RMS error between true
    and predicted cell voltage
    """

    alltemps = theTemp
    ind, = np.where(np.array(alltemps) == theTemp)[0]
    G = 0
    # G = abs(model.GParam[ind])

    Q = abs(model.QParam[ind])
    eta = abs(model.etaParam[ind])
    RC = abs(model.RCParam[ind])

    RC = np.array(RC).flatten()
    print(f'RC in simCell ecm = {RC}')
    numpoles = len(RC)
    # RC = RC[0]
    ik = i[-1]
    # tk = np.arange(len(ik))
    etaik = ik
    print(f'etaik in simcell_ecm is {etaik}')
    if etaik < 0:
        etaik = etaik * eta

    hh = 0*ik
    sik = 0*ik
    fac = np.exp(-abs(G * etaik/(3600*Q)))

    for k in range(1, len(i)):
        hh[k] = (fac[k-1]*hh[k-1]) - ((1-fac[k-1])*np.sign(ik[k-1]))
        sik[k] = np.sign(ik[k])
        if abs(ik[k]) < Q/100:
            sik[k] = sik[k-1]

    # First modeling step: Compute error with model = OCV only
    pfile = os.path.join(os.path.dirname(__file__), 'modelocv.json')
    modelocv = ModelOcv.load(Path(pfile))
    print(f'z0 in simCell = {z0}')
    vest1 = OCVfromSOCtemp(z0, alltemps[ind], modelocv)
    
    # vest1 = 3.5
    print(f'vest1 is {vest1}')
    
    RCfact = np.exp(-1/abs(RC[0]))
    RCfact = np.array([RCfact])
    RCfact = RCfact[-numpoles:]
    print(f'RCfact is {RCfact}')
    # print(f"RCfact_inv is {RCfact}")

    # Simulate the R-C filters to find R-C currents
    stsp = dlti(np.diag(RCfact), np.vstack(1-RCfact), np.eye(numpoles), np.zeros((numpoles, 1))) 
    [tout, vrcRaw, xout] = dlsim(stsp, etaik)
    print(f'vrcRaw is {vrcRaw}')

    # Third modeling step: Hysteresis parameters
    if doHyst:
        H = np.column_stack((hh, sik, -etaik, -vrcRaw))
        # W = nnls(H, verr)
        # M = W[0][0]
        # M0 = W[0][1]
        # R0 = W[0][2]
        # Rfact = W[0][3:].T
    else:
        # H = np.column_stack((-etaik, -vrcRaw))
        # W = np.linalg.lstsq(H,verr, rcond=None)[0]
        M = 0
        M0 = 0
        R0 = abs(model.R0Param[ind])
        print(f'R0 is {R0}')
        Rfact = abs(model.RParam[ind])
        Rfact = np.array([Rfact])
        # print(f'Rfact is {Rfact}')

    model.R0Param[ind] = R0
    model.M0Param = M0
    model.MParam = M
    model.RCParam = RC.T
    model.RParam = Rfact.T
    print(f'RCparam in step = {model.RCParam}')
    print(f'Rfact is {model.RParam[ind]}')
    # print(R0,
    #       RC,
    #       Rfact)

    # soc = SOCfromOCVtemp(vest1, data[ind].temp, model)

    # plt.figure("SOC")
    # plt.plot(tk[::10]/60, soc[::10], label='soc')
    # plt.xlabel('Time (min)')
    # plt.ylabel('soc')
    # plt.title(f'Current at T = {data[ind].temp} C')
    # plt.legend(loc='best', numpoints=1)
    # plt.show()

    vest2 = vest1 + M*hh + M0*sik - R0*etaik - vrcRaw @ Rfact.T
    vest2 = vest2[0]
    print(f'vest2 is {vest2}')
    return model, vest2

def simCell(data, model, theTemp, doHyst):
    """
    Using an assumed value for gamma (already stored in the model), find optimum
    values for remaining cell parameters, and compute the RMS error between true
    and predicted cell voltage
    """

    alltemps = [d.temp for d in data]
    ind, = np.where(np.array(alltemps) == theTemp)[0]
    G = 0
    # G = abs(model.GParam[ind])

    Q = abs(model.QParam[ind])
    eta = abs(model.etaParam[ind])
    RC = abs(model.RCParam[ind])
    RC = np.array([RC])
    numpoles = len(RC)

    ik = data[ind].s1.current.copy()
    tk = np.arange(len(ik))
    etaik = ik.copy()
    etaik[ik < 0] = etaik[ik < 0] * eta

    hh = 0*ik
    sik = 0*ik
    fac = np.exp(-abs(G * etaik/(3600*Q)))

    for k in range(1, len(ik)):
        hh[k] = (fac[k-1]*hh[k-1]) - ((1-fac[k-1])*np.sign(ik[k-1]))
        sik[k] = np.sign(ik[k])
        if abs(ik[k]) < Q/100:
            sik[k] = sik[k-1]

    # First modeling step: Compute error with model = OCV only
    vest1 = data[ind].OCV
    
    RCfact = np.exp(-1/abs(RC))
    RCfact = np.array([RCfact])
    RCfact = RCfact[-numpoles:]
    # print(f"RCfact_inv is {RCfact}")

    # Simulate the R-C filters to find R-C currents
    stsp = dlti(np.diag(RCfact), np.vstack(1-RCfact), np.eye(numpoles), np.zeros((numpoles, 1))) 
    [tout, vrcRaw, xout] = dlsim(stsp, etaik)

    # Third modeling step: Hysteresis parameters
    if doHyst:
        H = np.column_stack((hh, sik, -etaik, -vrcRaw))
        # W = nnls(H, verr)
        # M = W[0][0]
        # M0 = W[0][1]
        # R0 = W[0][2]
        # Rfact = W[0][3:].T
    else:
        # H = np.column_stack((-etaik, -vrcRaw))
        # W = np.linalg.lstsq(H,verr, rcond=None)[0]
        M = 0
        M0 = 0
        R0 = abs(model.R0Param[ind])
        Rfact = abs(model.RParam[ind])
        Rfact = np.array([Rfact])

    idx, = np.where(np.array(model.temps) == data[ind].temp)[0]
    model.R0Param[idx] = R0
    model.M0Param[idx] = M0
    model.MParam[idx] = M
    model.RCParam[idx] = RC.T
    model.RParam[idx] = Rfact.T
    # print(R0,
    #       RC,
    #       Rfact)

    # soc = SOCfromOCVtemp(vest1, data[ind].temp, model)

    # plt.figure("SOC")
    # plt.plot(tk[::10]/60, soc[::10], label='soc')
    # plt.xlabel('Time (min)')
    # plt.ylabel('soc')
    # plt.title(f'Current at T = {data[ind].temp} C')
    # plt.legend(loc='best', numpoints=1)
    # plt.show()

    vest2 = vest1 + M*hh + M0*sik - R0*etaik - vrcRaw @ Rfact.T

    return model, vest2
    
def optfn(x, data, model, theTemp, doHyst):
    """
    This minfn works for the enhanced self-correcting cell model
    """

    idx, = np.where(np.array(model.temps) == theTemp)
    model.GParam[idx] = abs(x)

    cost, _, vest2= minfn(data, model, theTemp, doHyst)
    return cost, vest2

def processDynamic(data, modelocv, numpoles, doHyst):
    """
    Technical note: PROCESSDYNAMIC assumes that specific Arbin test scripts have
    been executed to generate the input files.  "makeMATfiles.m" converts the raw
    Excel data files into "MAT" format where the MAT files have fields for time,
    step, current, voltage, chgAh, and disAh for each script run.

    The results from three scripts are required at every temperature.
    The steps in each script file are assumed to be:
    Script 1 (thermal chamber set to test temperature):
        Step 1: Rest @ 100% SOC to acclimatize to test temperature
        Step 2: Discharge @ 1C to reach ca. 90% SOC
        Step 3: Repeatedly execute dynamic profiles (and possibly intermediate
        rests) until SOC is around 10%
    Script 2 (thermal chamber set to 25 degC):
        Step 1: Rest ca. 10% SOC to acclimatize to 25 degC
        Step 2: Discharge to min voltage (ca. C/3)
        Step 3: Rest
        Step 4: Constant voltage at vmin until current small (ca. C/30)
        Steps 5-7: Dither around vmin
        Step 8: Rest
    Script 3 (thermal chamber set to 25 degC):
        Step 2: Charge @ 1C to max voltage
        Step 3: Rest
        Step 4: Constant voltage at vmax until current small (ca. C/30)
        Steps 5-7: Dither around vmax
        Step 8: Rest

    All other steps (if present) are ignored by PROCESSDYNAMIC. The time step
    between data samples must be uniform -- we assume a 1s sample period in this
    code.

    The inputs:
    - data: An array, with one entry per temperature to be processed.
        One of the array entries must be at 25 degC. The fields of "data" are:
        temp (the test temperature), script1, script 2, and script 3, where the
        latter comprise data collected from each script.  The sub-fields of
        these script structures that are used by PROCESSDYNAMIC are the
        vectors: current, voltage, chgAh, and disAh
    - model: The output from processOCV, comprising the OCV model
    - numpoles: The number of R-C pairs in the model
    - doHyst: 0 if no hysteresis model desired; 1 if hysteresis desired

    The output:
    - model: A modified model, which now contains the dynamic fields filled in.
    """

    # used by minimize_scalar later on
    options = {
        'xatol': 1e-08, 
        'maxiter': 1e5, 
        'disp': 0
    }

    # Step 1: Compute capacity and coulombic efficiency for every test
    # ------------------------------------------------------------------

    alltemps = [d.temp for d in data]
    alletas = np.zeros(len(alltemps))
    allQs = np.zeros(len(alltemps))

    ind25, = np.where(np.array(alltemps) == 25)[0]
    not25, = np.where(np.array(alltemps) != 25)

    k = ind25

    totDisAh = data[k].s1.disAh[-1] + data[k].s2.disAh[-1] + data[k].s3.disAh[-1]
    totChgAh = data[k].s1.chgAh[-1] + data[k].s2.chgAh[-1] + data[k].s3.chgAh[-1]
    eta25 = totDisAh/totChgAh
    data[k].eta = eta25
    alletas[k] = eta25
    data[k].s1.chgAh = data[k].s1.chgAh * eta25
    data[k].s2.chgAh = data[k].s2.chgAh * eta25
    data[k].s3.chgAh = data[k].s3.chgAh * eta25

    Q25 = data[k].s1.disAh[-1] + data[k].s2.disAh[-1] - data[k].s1.chgAh[-1] - data[k].s2.chgAh[-1]
    data[k].Q = Q25
    allQs[k] = Q25

    eta25 = np.mean(alletas[ind25])

    for k in not25:
        data[k].s2.chgAh = data[k].s2.chgAh*eta25
        data[k].s3.chgAh = data[k].s3.chgAh*eta25
        eta = (data[k].s1.disAh[-1] + data[k].s2.disAh[-1] + data[k].s3.disAh[-1] - data[k].s2.chgAh[-1] - data[k].s3.chgAh[-1])/data[k].s1.chgAh[-1]

        data[k].s1.chgAh = eta*data[k].s1.chgAh
        data[k].eta = eta
        alletas[k] = eta

        Q = data[k].s1.disAh[-1] + data[k].s2.disAh[-1] - data[k].s1.chgAh[-1] - data[k].s2.chgAh[-1]
        data[k].Q = Q
        allQs[k] = Q

    modeldyn = ModelDyn()
    modeldyn.temps = alltemps
    modeldyn.etaParam = alletas
    modeldyn.QParam = allQs

    # Step 2: Compute OCV for "discharge portion" of test
    # ------------------------------------------------------------------

    for k, _ in enumerate(data):
        etaParam = modeldyn.etaParam[k]
        etaik = data[k].s1.current.copy()
        etaik[etaik < 0] = etaParam*etaik[etaik < 0]
        data[k].Z = 1 - np.cumsum(etaik) * 1/(data[k].Q * 3600)
        data[k].OCV = OCVfromSOCtemp(data[k].Z, alltemps[k], modelocv)
        # print(f"ocv is {data[k].OCV}")
        
    # Step 3: Now, optimize!
    # ------------------------------------------------------------------

    modeldyn.GParam = np.zeros(len(modeldyn.temps))   # gamma hysteresis parameter
    modeldyn.M0Param = np.zeros(len(modeldyn.temps))  # M0 hysteresis parameter
    modeldyn.MParam = np.zeros(len(modeldyn.temps))   # M hysteresis parameter
    modeldyn.R0Param = np.zeros(len(modeldyn.temps))  # R0 ohmic resistance parameter
    modeldyn.RCParam = np.zeros((len(modeldyn.temps), numpoles))  # time constant
    modeldyn.RParam = np.zeros((len(modeldyn.temps), numpoles))   # Rk

    modeldyn.SOC = modelocv.SOC        # copy SOC values from OCV model
    modeldyn.OCV0 = modelocv.OCV0      # copy OCV0 values from OCV model
    modeldyn.OCVrel = modelocv.OCVrel  # copy OCVrel values from OCV model
    modeldyn.OCV = modelocv.OCV        # copy SOC values from OCV model
    modeldyn.SOC0 = modelocv.SOC0      # copy OCV0 values from OCV model
    modeldyn.SOCrel = modelocv.SOCrel  # copy OCVrel values from OCV model

    for theTemp in range(len(modeldyn.temps)):
        temp = modeldyn.temps[theTemp]
        print('Processing temperature', temp, 'C')

        if doHyst:
            g = abs(minimize_scalar(optfn, bounds=(1, 250), args=(data, modeldyn, temp, doHyst), method='bounded', options=options).x)
            print('g =', g)

        else:
            modeldyn.GParam[theTemp] = 0
            theGParam = 0 
            _, vest2 = optfn(theGParam, data, modeldyn, temp, doHyst)
    return modeldyn, vest2

def processDynamic_short(data, modelocv, numpoles, doHyst):
    """
    """

    # Step 1: Compute capacity and coulombic efficiency for every test
    # ------------------------------------------------------------------
    
    alltemps = [d.temp for d in data]
    alletas = np.zeros(len(alltemps))
    allQs = np.zeros(len(alltemps))

    ind25, = np.where(np.array(alltemps) == 25)[0]

    k = ind25

    eta25 = 0.9944503313637096
    data[k].eta = eta25
    alletas[k] = eta25
    data[k].s1.chgAh = data[k].s1.chgAh * eta25

    Q25 = 2.0495322455503873
    data[k].Q = Q25
    allQs[k] = Q25

    modeldyn = ModelDyn()
    modeldyn.temps = alltemps
    modeldyn.etaParam = alletas
    modeldyn.QParam = allQs

    # Step 2: Compute OCV for "discharge portion" of test
    # ------------------------------------------------------------------

    for k, _ in enumerate(data):
        etaParam = modeldyn.etaParam[k]
        etaik = data[k].s1.current.copy()
        etaik[etaik < 0] = etaParam*etaik[etaik < 0]
        data[k].Z = 1 - np.cumsum(etaik) * 1/(data[k].Q * 3600)
        data[k].OCV = OCVfromSOCtemp(data[k].Z, alltemps[k], modelocv)
        # print(f"ocv is {data[k].OCV}")
        
    # Step 3: Now, optimize!
    # ------------------------------------------------------------------

    modeldyn.GParam = np.zeros(len(modeldyn.temps))   # gamma hysteresis parameter
    modeldyn.M0Param = np.zeros(len(modeldyn.temps))  # M0 hysteresis parameter
    modeldyn.MParam = np.zeros(len(modeldyn.temps))   # M hysteresis parameter
    modeldyn.R0Param = np.zeros(len(modeldyn.temps))  # R0 ohmic resistance parameter
    modeldyn.RCParam = np.zeros((len(modeldyn.temps), numpoles))  # time constant
    modeldyn.RParam = np.zeros((len(modeldyn.temps), numpoles))   # Rk

    modeldyn.SOC = modelocv.SOC        # copy SOC values from OCV model
    modeldyn.OCV0 = modelocv.OCV0      # copy OCV0 values from OCV model
    modeldyn.OCVrel = modelocv.OCVrel  # copy OCVrel values from OCV model
    modeldyn.OCV = modelocv.OCV        # copy SOC values from OCV model
    modeldyn.SOC0 = modelocv.SOC0      # copy OCV0 values from OCV model
    modeldyn.SOCrel = modelocv.SOCrel  # copy OCVrel values from OCV model

    for theTemp in range(len(modeldyn.temps)):
        temp = modeldyn.temps[theTemp]
        print('Processing temperature', temp, 'C')

        modeldyn.GParam[theTemp] = 0
        theGParam = 0 
        _, vest2 = optfn(theGParam, data, modeldyn, temp, doHyst)

    return modeldyn, vest2

import time

class EcmEnv_v0(gym.Env):

    def __init__(self, Vmax = 4.2, Tmax = 350, delta_t=10, render_mode = 'rgb_array'):

        # Observations are dictionaries
        
        self.Vmax = Vmax
        self.Tmax = Tmax

        self.delta_t = delta_t
        
        self.observation_space = spaces.Dict(
            {
                "Current function [A]": spaces.Box(-1, 1, shape=(1,), dtype=np.float16),
                "Voltage known [V]": spaces.Box(0, 1, shape=(1,), dtype=np.float16),
                "Voltage unknown [V]": spaces.Box(0, 1, shape=(1,), dtype=np.float16),
            }
        )
        # we have 1 actions, current
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, seed: int|None = None, options: int|None = None):
        print('RESET ------------------------------')
        super().reset(seed=seed)
        
        self.reward = 0
        # self.rewards = []
        self.reward_opt = []
        self.terminated = False
        self.truncated = False
        self.step_count = 0

        # self.etaik = []

        model_known = pybamm.equivalent_circuit.Thevenin(options)

        self.params_known = pybamm.ParameterValues("ECM_Example").copy()
        self.params_known.update({"Current function [A]": [0.0],
                                  })
        self.params_known["Current function [A]"] = pybamm.InputParameter("Input current [A]")

        model_known1 = self.params_known.process_model(model_known, inplace=False)

        #setting geometry
        geometry_known = model_known.default_geometry
        submesh_types_known = model_known.default_submesh_types
        var_pts_known = model_known.default_var_pts
        self.params_known.process_geometry(geometry_known)
        mesh_known = pybamm.Mesh(geometry_known, submesh_types_known, var_pts_known)
        spatial_methods_known = model_known.default_spatial_methods

        self.disc_known = pybamm.Discretisation(mesh_known, spatial_methods_known)
        self.solver_known = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)

        #model unknown
        pfile = os.path.join(os.path.dirname(__file__), 'modelocv.json')
        modelocv = ModelOcv.load(Path(pfile))
        self.modelocv = modelocv
        cellID = 'A123'
        numpoles = 1
        temps = [25]
        self.temps = temps
        mags = [50]
        doHyst = 0
        data = np.zeros(len(mags), dtype=object)
        print('Load files')
        for idx, temp in enumerate(temps):
            mag = mags[idx]
            if temp < 0:
                tempfmt = f'{abs(temp):02}'
                files = [Path(f'../dyn_data/{cellID}_DYN_{mag}_N{tempfmt}_s1.csv'),
                        Path(f'../dyn_data/{cellID}_DYN_{mag}_N{tempfmt}_s2.csv'),
                        Path(f'../dyn_data/{cellID}_DYN_{mag}_N{tempfmt}_s3.csv')]
                data[idx] = DataModel(temp, files)
                print(*files, sep='\n')
            else:
                tempfmt = f'{abs(temp):02}'
                files = [Path(f'./dyn_data/{cellID}_DYN_{mag}_P{tempfmt}_s1.csv'),
                        Path(f'./dyn_data/{cellID}_DYN_{mag}_P{tempfmt}_s2.csv'),
                        Path(f'./dyn_data/{cellID}_DYN_{mag}_P{tempfmt}_s3.csv')]
                data[idx] = DataModel(temp, files)
                print(*files, sep='\n')
        self.data = data
        modeldyn, _ = processDynamic(data, modelocv, numpoles, doHyst)
        # print(modeldyn)
        modeldyn = {k:v.tolist() if isinstance(v, np.ndarray) else v 
                    for k,v in modeldyn.__dict__.items()}
        # print(type(modeldyn))
        if True:
            if doHyst:
                with open('modeldyn_hys.json', 'w') as json_file:
                    json.dump(modeldyn, json_file, indent=4)
            else:
                modeldyn_nohys = 'modeldyn_nohys.json'
                with open(modeldyn_nohys, 'w') as json_file:
                    json.dump(modeldyn,json_file, indent=4)

        # open json with modeldyn
        with open(modeldyn_nohys) as json_file:
            params_inv = json.load(json_file)

        noise = abs(np.random.normal(0, 0.1, size = 1))

        print(f'noise is {noise}')
        model_unknown = ModelDyn()
        if True:
            model_unknown.temps = params_inv["temps"]
            model_unknown.GParam = params_inv['GParam']
            model_unknown.QParam = params_inv['QParam']
            # print(model_unknown.QParam)
            model_unknown.etaParam = params_inv['etaParam'] + noise
            model_unknown.RCParam = params_inv['RCParam'][0] + noise
            print(f'RCParam is {model_unknown.RCParam}')
            model_unknown.RParam = params_inv["RParam"][0] + noise
            model_unknown.R0Param = params_inv["R0Param"] + noise
            model_unknown.M0Param = params_inv["M0Param"]
            model_unknown.MParam = params_inv["MParam"]
            model_unknown.SOC = params_inv["SOC"]
            model_unknown.OCV0 = params_inv["OCV0"]
            model_unknown.OCVrel = params_inv["OCVrel"]
            model_unknown.OCV = params_inv["OCV"]
            model_unknown.SOC0 = params_inv["SOC0"]
            model_unknown.SOCrel = params_inv["SOCrel"]

        self.etaik = [0.0]
        z0 = [0.5] #soc
        model_unknown1, voltage_unknown = simCell_ecm(self.etaik, model_unknown, z0, temps, 0)
        self.voltage_unknown = voltage_unknown
        # print(f'model_unknown is {model_unknown1.RCParam}')
        print(f'voltage_unknown is {self.voltage_unknown}')

        #set initial solution
        self.solutions_known = []
        self.solutions_unknown = []

        self.model_known = self.disc_known.process_model(model=model_known1, inplace=False)
        self.model_unknown = model_unknown1
        initial_solution_known = self.solver_known.step(model=self.model_known, dt=1, old_solution=None, npts=3, save=False, inputs={"Input current [A]": 0.0})
        self.solutions_known.append(initial_solution_known)
        self.solutions_unknown.append(voltage_unknown)

        #showing initial conditions
        self.observation = {
            "Current function [A]": np.array([self.solutions_known[-1]["Current [A]"].data[-1]],dtype = np.float16)/7.5,
            "Voltage known [V]":    np.array([self.solutions_known[-1]["Voltage [V]"].data[-1]], dtype = np.float16)/4.2,
            "Voltage unknown [V]":  np.array([self.solutions_unknown[-1]], dtype = np.float16)/4.2,
        }

        print("Observation space in reset:", self.observation)
        info = {}
        
        return self.observation, info
        
    def step(self, action):
        
        
        self.step_count += 1
        print(f'STEP {self.step_count}_______________________________________')

        noise1 = 0
        ratio = 5
        c_rate = 2.05

        current = c_rate*ratio*action[-1] # 5C max, 1C = 2.05
        print(f'current of step is {current}')

        self.solutions_known += [self.solver_known.step(self.solutions_known[-1].last_state,
                                                        self.model_known,
                                                        1,
                                                        npts=3,
                                                        save=False,
                                                        inputs={"Input current [A]": current + noise1}
                                                        )]
        self.voltage_known = self.solutions_known[-1]["Voltage [V]"].data[-1]

        etaik = current + noise1
        if current <= 0:
            etaik = etaik * self.model_unknown.etaParam[0]
        # print(f'etaik is {etaik}')
        # print(f'Qparam is {self.model_unknown.QParam}')
        z0 = [0.5 - (etaik/(3600*self.model_unknown.QParam[0]))]
        # z0 = z0[-1]
        print(f'z0 is {z0}')
        etaik = [etaik]
        print(f'etaik is {etaik}')
        self.model_unknown, voltage_unknown = simCell_ecm(etaik, self.model_unknown, z0, self.temps, 0)

        print(f'voltage unknown of simCell is {voltage_unknown}')

        #dense reward
        #terminted can also be done by checking pybamm.solution.termination(for example)

        rmse = np.sqrt(np.mean((self.voltage_known-voltage_unknown)**2))
        reward = -rmse
        self.reward += reward
        # print(f"reward of two battery is {self.reward}")

        #optimize

        with open('modeldyn_nohys.json') as json_file:
            params_inv = json.load(json_file)
        
        costs = []
        step_count = 1
        for step in range(step_count):
            print(f'step {step+1}')
            # print(f'len costs is {len(costs)}')
            # if len(costs) >= 2:
            
            if True:
                self.model_unknown.temps = params_inv["temps"]
                self.model_unknown.GParam = params_inv['GParam']
                self.model_unknown.QParam = params_inv['QParam']
                self.model_unknown.etaParam = params_inv['etaParam']
                self.model_unknown.RCParam = params_inv['RCParam'][0]
                self.model_unknown.RParam = params_inv["RParam"][0]
                self.model_unknown.R0Param = params_inv["R0Param"]
                self.model_unknown.M0Param = params_inv["M0Param"]
                self.model_unknown.MParam = params_inv["MParam"]
                self.model_unknown.SOC = params_inv["SOC"]
                self.model_unknown.OCV0 = params_inv["OCV0"]
                self.model_unknown.OCVrel = params_inv["OCVrel"]
                self.model_unknown.OCV = params_inv["OCV"]
                self.model_unknown.SOC0 = params_inv["SOC0"]
                self.model_unknown.SOCrel = params_inv["SOCrel"]

            self.model_unknown, vest2_inv = simCell(self.data, self.model_unknown, 25, 0)

            voltage = np.round(vest2_inv, 8)
            for k, _ in enumerate(self.data):
                etaParam = self.model_unknown.etaParam[k]
                etaik = self.data[k].s1.current.copy()
                etaik[etaik < 0] = etaParam*etaik[etaik < 0]

            tk = self.data[0].s1.time
            # tk_chg = tk[self.data[0].s1.current<0]
            # tk_dis = tk[self.data[0].s1.current>0]
            # print(tk_chg)

            chgAh = [0]*len(tk)
            disAh = [0]*len(tk)

            for i in range(len(etaik)):
                if etaik[i] <= 0:
                    chgAh[i] = abs(etaik[i])/3600
                elif etaik[i] > 0:
                    disAh[i] = etaik[i]/3600

            chgAh = np.cumsum(chgAh)
            # print(f'charge_capacity is {chgAh}')

            disAh = np.cumsum(disAh)
            # print(f'dischg_capacity is {disAh}')

            data_inv = {
                        'time'     : np.round(tk, 4),
                        ' current'  : np.round(self.data[k].s1.current.copy(), 4),
                        ' voltage'  : np.round(voltage, 4),
                        ' chgAh'    : np.round(chgAh, 4),
                        ' disAh'    : np.round(disAh, 4)
                        }
            df = pd.DataFrame(data_inv)
            df.to_csv('data_inverse.csv', index=False, sep = ',')
            data1 = []
            files = [Path('data_inverse.csv')]
            data1 = [DataModelshort(25, files)]
            
            modeldyn, voltage_neu = processDynamic_short(data1, self.modelocv, 1, 0)
            print(f'R0Param is {modeldyn.R0Param} RCParam is {modeldyn.RCParam} RParam is {modeldyn.RParam}')

            verr = voltage_unknown - voltage_neu
            verr = verr[np.isfinite(verr)] 
            # print(modeldyn)
            v1 = OCVfromSOCtemp(0.95, 25, modeldyn)[0]
            v2 = OCVfromSOCtemp(0.05, 25, modeldyn)[0]

            N1 = np.where(voltage_neu < v1)[0][0]
            N2 = np.where(voltage_neu < v2)[0][0]
            
            rmserr = np.sqrt(np.mean(verr[N1:N2]**2))
            cost = np.sum(rmserr)
            costs.append(cost)

            # if len(costs) > 1:
                
            #     print(f'converged in step {i+1}')
            #     self.reward += 150_000

            #     break
            # else:
            #     pass
            self.costs = costs
            
            print(f'RMS error = {cost*1000:.8f} mV')

            modeldyn = {k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in modeldyn.__dict__.items()}
            with open('model_unknown_inverse.json', 'w') as json_file:
                json.dump(modeldyn,json_file, indent=4)
            with open('model_unknown_inverse.json') as json_file:
                params_inv = json.load(json_file)
            print('__________________________________')
        
        # # plt.figure('rmse')
        # fig, ax = plt.subplots()
        # plt.plot(np.arange(len(costs))+1, costs[::], label='rmse')
        # from matplotlib import ticker

        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6e'))
        # plt.xlabel('Step')
        # plt.ylabel('RMSE')
        # plt.title('origin at T = 25 C')
        # plt.legend(loc='best', numpoints=1)
        # # plt.show()
        # plt.close(fig)
        self.solutions_unknown.append(voltage_unknown)
        
        self.voltage_unknown = self.solutions_unknown[-1]
        print(f'voltage unknown is {self.voltage_unknown}')
        
        if self.step_count > 720:
        
            self.truncated = True
            self.terminated = True

        

        self.observation = {
                "Current function [A]"  : np.array([current/c_rate/ratio]).astype(np.float16),
                "Voltage known [V]"     : np.array([self.voltage_known/4.2]).astype(np.float16),
                "Voltage unknown [V]"   : np.array([self.voltage_unknown/4.2]).flatten().astype(np.float16),
                }
        print("Observation space in step:", self.observation)

        info = {'terminated': self.terminated, 'truncated': self.truncated,
                }

        # time reward
        if self.truncated or self.terminated:
            self.reward += -0.2*(self.step_count*self.delta_t)**1.5
            solutions_known = self.solutions_known
            current = [solution_known["Current [A]"].data for solution_known in solutions_known]
            voltage_known = [solution_known["Voltage [V]"].data for solution_known in solutions_known]

            solutions_unknown = self.solutions_unknown
            voltage_unknown = [solution_unknown for solution_unknown in solutions_unknown]

            self.rmse = cost
            reward_opt = -self.rmse
            self.reward_opt.append(reward_opt)

            print(f"optimized reward of two battery is {self.reward_opt}")

            print(f'Final RMS error = {cost*1000:.8f} mV')
            
            
            info = {"model_known": self.model_known, 
                    "model_unknown": self.model_unknown,
                    "param_known": self.params_known, 
                    "param_unknown": params_inv, 
                    "disc_known": self.disc_known, 

                    "reward": self.reward_opt, 
                    'current': current, 'voltage_known': voltage_known, 'voltage_unknown': voltage_unknown,
                    }
                
        return self.observation, self.reward, self.terminated, self.truncated, info

    def render(self):    
        output_variables = [
        "Voltage [V]",
        "Current [A]",
        ]
        plot1 = pybamm.QuickPlot(self.solutions_known, output_variables)
        plot1.dynamic_plot()

        # plot2 = pybamm.QuickPlot(self.solutions_unknown, output_variables)
        # plot2.dynamic_plot()
        fig, ax = plt.subplots()
        plt.plot(np.arange(len(self.costs))+1, self.costs[::], label='RMSE')
        from matplotlib import ticker

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.6e'))
        plt.xlabel('Step')
        plt.ylabel('RMSE')
        plt.title('origin at T = 25 C')
        plt.legend(loc='best', numpoints=1)
        plt.show(fig)
        # plt.close(fig)

    def close(self):
        pass