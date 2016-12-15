'''
Module for char conversion calculation.
'''

import numpy as np
from scipy.integrate import ode
from scipy.optimize import brentq


# parameters
rhoAshT = 2100.0
Rgas = 8314.33
atm = 101325.

sigma = 5.670373e-8

M1 = 28.01
TC1 = 126.2
Vc1 = 89.49
# q1=3.798
M2 = 44.01
TC2 = 304.1
Vc2 = 94.04

DT1 = (2.616e-4 * np.sqrt(1. / M1 + 1. / M2) /
       pow(TC1 * TC2 / 10000., 0.1405) /
       pow(pow(Vc1 / 100., 0.4) + pow(Vc2 / 100., 0.4), 2))


def diffusivity(pressure, temperature):
    '''
    Calculate diffusivity of CO2 in a N2/CO2 mixture
    calculation is based on
    http://www.sciencedirect.com/science/article/pii/S1352231097003919

    Parameters
    ----------
    pressure: float
        pressure in Pa
    temperature: float
        temperature in K

    Returns
    -------
    diffusivity: float
        diffusivity in m2/s
    '''

    return DT1 * pow(temperature / 273.15, 1.81) * (101325. / pressure)


def trueDensity(ua, yash):
    '''
    calculate the true density of the coal using the correlation from
    Merrick

    Parameters
    ----------
    ua: array
        ultimate analysis C,H,O,N,S
    yash: float
        daf mass fraction

    Returns
    -------
    float
        true density of dry coal, kg/m3
    '''
    coeff = np.array([0.0053, 0.00577, 0.00346, 0.0669, 0.0384])
    MW = np.array([12., 1., 16., 14., 32.])
    rhoDafT = 1. / np.sum(coeff * ua / MW)
    return 1. / ((1. - yash) / rhoDafT + yash / rhoAshT)


def intrinsicRate(pressure, temperature, kinetics=[1e8, 160e6, 0.4]):
    '''
    Calculate the intrinsic rate in kg/s-m2

    Parameters
    ----------
    pressure: float
        pressure in atm
    temperature: float
        temperature in K
    kinetics: array
        kinetics array
        kinetics[0]: pre-exp factor
        kinetics[1]: activation energy J/kmol
        kinetics[2]: reaction order

    Returns
    -------
    float
        intrinsic kinetic rate in kg/s-m2
    '''
    return (kinetics[0] * np.exp(-kinetics[1] / Rgas / temperature) *
            pow(pressure, kinetics[2]))


def thiele(dp, pressure, temperature, rhoc, Aint, Deff,
           kinetics=[1e8, 160e6, 0.4], nu=1):
    '''
    Calculate the Thiele modulus

    Parameters
    ----------
    dp: float
        particle diameter, m
    pressure: float
        partial pressure of reactants
    temperature: float
        temperature, K
    rhoc: float
        char density kg/m3
    Aint: float
        specific intrinsic surface m2/kg
    Deff: float
        effective diffusivity m2/s
    kinetics: array
        kinetics[0]: pre-exp factor
        kinetics[1]: activation energy J/kmol
        kinetics[2]: reaction order
    nu: stoichimetric coefficient n_reactant/n_char

    Returns
    -------
    float
        Thiele modulus
    '''
    rate = intrinsicRate(pressure, temperature, kinetics=kinetics)
    if pressure > 0:
        return 0.5 * dp * np.sqrt(
            nu * (kinetics[2] + 1) * rate * Aint * rhoc * Rgas *
            temperature / 2. / 12.0 / Deff / pressure / atm)
    else:
        return 1.0


def effectivenessFactor(dp, pressure, temperature, rhoc, Aint, Deff,
                        kinetics=[1e8, 160e6, 0.4], nu=1):
    '''
    Calculate effectiveness factor

    Parameters
    ----------
    dp: float
        particle diameter, m
    pressure: float
        partial pressure of reactants
    temperature: float
        temperature, K
    rhoc: float
        char density kg/m3
    Aint: float
        specific intrinsic surface m2/kg
    Deff: float
        effective diffusivity m2/s
    kinetics: array
        kinetics[0]: pre-exp factor
        kinetics[1]: activation energy J/kmol
        kinetics[2]: reaction order
    nu: float
        stoichimetric coefficient n_reactant/n_char

    Returns
    -------
    tuple
        phi, Th
    '''
    Th = thiele(dp, pressure, temperature, rhoc, Aint, Deff, kinetics,
                nu)
    if pressure > 0:
        phi = 3. / Th * (1. / np.tanh(Th) - 1. / Th)
    else:
        phi = 1.0

    return phi, Th


def diffusionFlux(Ptot, Pinf, Tinf, Tp, dp, Ps=0, Sh=2):
    '''
    Calculate diffusion flux for the given pressures

    Parameters
    ----------
    Ptot: flaot
        total pressure
    Pinf: float
        bulk partial pressure
    Ps: float
        surface partial pressure
    Tinf: float
        bulk temperature
    Tp: float
        particle temperature
    dp: float
        particle diameter
    Sh: float
        Sherwood number (default 2)

    Returns
    -------
    tuple
        (Diffusion flux in kg/s-m2, kdiff kg/s-m2..)
    '''
    Tm = 0.5 * (Tp + Tinf)
    diff = diffusivity(pressure=Ptot * atm, temperature=Tm)
    kdiff = Sh * diff * 12. / dp / Rgas / Tm
    rate = kdiff * Ptot * atm * np.log((
        1 + Pinf / Ptot) / (1 + Ps / Ptot))
    return rate, kdiff


def chi_parameter(dp, Ptot, Pinf, Tinf, Tp, rhoc, Aint, Deff,
                  kinetics=[1e8, 160e6, 0.4], nu=1.0, Sh=2.0):
    '''
    Calculate chi parameter and rates including film and pore diffusion.
    It uses the Brent method

    Parameters
    ----------
    dp: particle diameter
    Ptot: total pressure, atm
    Pinf: bulk partial pressure, atm
    Tinf: bulk temperature, K
    Tp: particle temperature, K
    rhoc: char density, kg/m3
    Aint: intrinsic surface m2/kg
    Deff: effective diffusivity
    kinetics:
    nu: reactant/char molar fraction
    Sh: Sherwood number

    Returns
    -------
    tuple
        (Ps, chi, rate, eff, thiele)
    '''
    def f(x):
        '''
        Balance between the oxidant consumption due to the surface
        reaction and the transport through the particle Boundary layer.

        Parameters
        ----------
        x: float
            Mole fraction of the reactant on the particle surface

        Returns
        -------
        float
            kinetic rate - diffusion rate
        '''
        Ps = x * Ptot
        ratei = intrinsicRate(pressure=Ps, temperature=Tp,
                              kinetics=kinetics) * Aint  # 1/s
        eff, thiele = effectivenessFactor(
            dp=dp, pressure=Ps, temperature=Tp, rhoc=rhoc, Aint=Aint,
            Deff=Deff, kinetics=kinetics, nu=nu)
        ratek = eff * ratei  # kinetic rate 1/s
        rated, kdiff = diffusionFlux(
            Ptot=Ptot, Pinf=Pinf, Tinf=Tinf, Tp=Tp, dp=dp, Ps=Ps, Sh=Sh)

        rated *= 6. / rhoc / dp  # diffusion rate 1/s
        return ratek - rated

    tol = 1e-6
    if np.abs(f(0)) < tol:
        xs = 0
    elif np.abs(f(Pinf / Ptot)) < tol:
        xs = Pinf / Ptot
    else:
        try:
            xs = brentq(f, a=0.0, b=Pinf / Ptot)
        except:
            raise RuntimeError('Brentq does not work\nf(a)={}'
                               '\nf(b)={}'.format(f(0), f(Pinf / Ptot)))
    Ps = xs * Ptot
    chi = 1 - np.log(1 + Ps / Ptot) / np.log(1 + Pinf / Ptot)
    # print f(xs)
    eff, thiele = effectivenessFactor(
        dp=dp, pressure=Ps, temperature=Tp, rhoc=rhoc, Aint=Aint,
        Deff=Deff, kinetics=kinetics)
    rateo = intrinsicRate(pressure=Ps, temperature=Tp,
                          kinetics=kinetics) * Aint * eff

    return Ps, chi, rateo, eff, thiele


def RPM(X, As0, psi):
    '''
    Calculate the specific intrinsic surface per unit of mass using the
    Random Pore Model.

    Parameters
    ----------
    X: float
        char conversion
    psi: float
        structural parameter

    Returns
    -------
    float
        specific surface per unit of char mass a=A/mc
    '''
    return As0 * np.sqrt(1. - psi * np.log(1. - X))


def drop_tube(Tp0, time, dp0, Ptot, Pinf, Tg, Twall, rhop0, A0, yash0,
              psi, tauf, DH=172e6, kinetics=[1e8, 160e6, 0.4], nu=1.0,
              Sh=2.0, ug=0):
    '''
    Solve the Drop Tube Reactor using an ODE solver

    Parameters
    ----------
    Tp0: float
        Initial particle temperature
    time: float
        Residence time
    dp0: float
        initial diameter
    '''
    alpha = 0.95
    rhoAsh = 800.
    cp = 1600.  # J/kgK
    cpg = 1100.0  # J/kgK
    # calculate initial properties
    X = 0
    Vp0 = np.pi / 6. * pow(dp0, 3)
    mp0 = Vp0 * rhop0
    mc0 = mp0 * (1 - yash0)
    mash = mp0 - mc0
    rhoc0 = (1. - yash0) / (1. / rhop0 - yash0 / rhoAsh)
    ua = np.array([1, 0, 0, 0, 0])
    rhoct = trueDensity(ua, yash0)  # true density of char

    # mc = None
    Mc = []
    rhoc = None
    Rhoc = []
    Rhocs = []
    rhop = None
    Rhop = []
    chi = None
    Chi = []
    robs = None
    Robs = []
    blow = None
    Blow = []
    eff = None
    Eff = []
    Dp = []

    def inside(time, y):
        X = y[0]
        if X > 1:
            X = 1
        Tp = y[1]
        rhoc = y[2]
        rhocp = y[3]
        up = y[4]  # particle velocity
        z = y[5]  # particle position

        # update properties using X
        Aint = RPM(X, As0=A0, psi=psi)  # specific intrinsic surface
        # rhoc = rhoc0 * pow(1 - X, alpha)  #updated density

        mc = mc0 * (1 - X)
        mp = mc + mash
        yash = mash / mp  # new ash fraction

        rhop = 1. / (yash / rhoAsh + (1 - yash) / rhoc)
        Vp = mp / rhop
        dp = pow(6. * mp / np.pi / rhop, 1. / 3.)  # diameter
        Ap = 6 * Vp / dp

        rhoT = trueDensity(ua, yash)  # true density
        epsilon = 1 - rhop / rhoT  # porosity

        Tm = 2. / 3. * Tp + 1. / 3. * Tg
        # reactant diffusivity at particle temperature
        diff = diffusivity(Ptot * 101325, Tm)
        # effective diffusivity in the particle pores
        Deff = diff * epsilon / tauf

        Mmix = 44. * Pinf / Ptot + 28. * (Ptot - Pinf) / Ptot
        rhog = Ptot * atm * Mmix / Rgas / Tg  # density of gas on the bulk phase

        # calculate reaction rate / rate is 1/s
        Ps, chi, robs, eff, thiele = chi_parameter(
            dp=dp, Ptot=Ptot, Pinf=Pinf, Tinf=Tg, Tp=Tp, rhoc=rhoc,
            Aint=Aint, Deff=Deff, kinetics=kinetics)

        # dXdt = robs*(1.-X)

        # calculate heat
        Le = 1
        k = rhog * cpg * diff * Le
        h = Sh * k / dp
        B = robs * mc * cpg / 2. / np.pi / k / dp
        if B > 1e-7:
            blow = B / (np.exp(B) - 1)
        else:
            blow = 1.0
        Qc = h * Ap * blow * (Tg - Tp)  # convection W
        # radiation
        Qr = sigma * 0.9 * Ap * (pow(Twall, 4) - pow(Tp, 4))
        Qch = -DH * robs * mc

        Qtot = (Qc + Qr + Qch)

        Fd = 18. * 1.7e-5 / pow(dp, 2) / rhop
        dudt = Fd * (ug - up) + (rhop - rhog) / rhop * 9.81

        return [dp, mc, mp, thiele, rhop, chi, robs, blow, eff, Qtot,
                dudt]

    def dydt(time, y):

        X = y[0]
        if X > 1:
            X = 1
        Tp = y[1]
        rhoc = y[2]
        rhocs = y[3]
        up = y[4]  # particle velocity
        z = y[5]  # particle position
        soli = inside(time, y)
        dp = soli[0]
        mc = soli[1]
        mp = soli[2]
        thiele = soli[3]
        rhop = soli[4]
        chi = soli[5]
        robs = soli[6]
        blow = soli[7]
        eff = soli[8]
        Qtot = soli[9]
        dudt = soli[10]

        #
        dXdt = robs * (1. - X)
        dTdt = Qtot / mp / cp

        # density is calculated by the methods proposed by
        # Haugen C&F 2014

        threshold = 1.0
        if X > 0.99:
            drhocsdt = 0.0
            drhocdt = 0.0
        else:
            if rhocs / rhoc0 > threshold:
                # constant volume
                drhocdt = -robs * rhoc
                drhocsdt = -robs / eff * rhocs
            else:
                if thiele > 1e-5:
                    drhocdt = -robs * rhoc * (
                        1. - 9. * (1. - eff) / pow(eff * thiele, 2))
                else:
                    drhocdt = -robs * rhoc
                drhocsdt = 0.0

        return [dXdt, dTdt, drhocdt, drhocsdt, dudt, up]

    import warnings

    y0 = [0, Tp0, rhoc0, rhoc0, 0, 0]  # intial values of ODE
    backend = 'dopri5'

    t0 = 0
    solver = ode(dydt).set_integrator(backend, nsteps=1)
    solver.set_initial_value(y0, t0)
    T = []
    Xc = []
    TP = []
    Up = []
    Zp = []

    warnings.filterwarnings("ignore", category=UserWarning)
    while solver.t < time:
        solver.integrate(time, step=True)
        soli = inside(solver.t, solver.y)
        dp = soli[0]
        mc = soli[1]
        mp = soli[2]
        rhoc = soli[3]
        rhop = soli[4]
        chi = soli[5]
        robs = soli[6]
        blow = soli[7]
        eff = soli[8]
        Qtot = soli[9]

        Dp.append(dp)
        Mc.append(mc)

        Rhop.append(rhop)
        Chi.append(chi)
        Robs.append(robs)
        Blow.append(blow)
        Eff.append(eff)

        # print(solver.t,solver.y)
        # print('Time={:5.4f} s - X={:5.4f}'.format(solver.t,solver.y[0]))
        T.append(solver.t)
        Xc.append(solver.y[0])
        TP.append(solver.y[1])
        Rhoc.append(solver.y[2])
        Rhocs.append(solver.y[3])
        Up.append(solver.y[4])
        Zp.append(solver.y[5])

    warnings.resetwarnings()
    return [T, Xc, TP, Mc, Rhoc, Rhop, Chi, Robs, Blow, Eff, Dp, Rhocs,
            Up, Zp]
