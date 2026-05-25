import math
import numpy as np
from collections import deque


class ThreeNodePlant:
    """
    3-node thermal plant for closed-loop simulation with FOPID.

    Structure (physically correct — more PWM → lower Tc):
    -------------------------------------------------------
    dTc = [(Tamb-Tc)/Rca  -  (Tc-Tm)/Rcm  -  P] / Cc
    dTm = [(Tc-Tm)/Rcm    -  (Tm-Th)/Rmh       ] / Cp
    dTh = [(Tm-Th)/Rmh    +  P  -  (Th-Tamb)/Rh] / Ch

    Peltier delay modelled as first-order lag (tau).

    Parameters fan ON: identified from dynamic fan-ON experiment.
    R_hot_fan_off:     identified from dynamic fan-OFF experiment.
    All other params:  invariant (same physical hardware).
    """

    # ------------------------------------------------------------------
    # THERMAL RESISTANCES  [K/W]  — identified from real plant data
    # ------------------------------------------------------------------
    R_cold_amb = 4.897706   # Rca  — cold face ↔ ambient (convective)
    R_cold_mid = 4.022803   # Rcm  — cold face ↔ mid node
    R_mid_hot  = 6.003481   # Rmh  — mid node  ↔ heatsink

    # Hot side → ambient: fan modulates this resistance
    R_hot_fan_on  =  1.04      # K/W  fan=255  (measured directly)
    R_hot_fan_off = 17.07925   # K/W  fan=0    (identified from dynamic fan-OFF experiment)

    # ------------------------------------------------------------------
    # THERMAL CAPACITANCES  [J/K]  — identified from real plant data
    # ------------------------------------------------------------------
    Cc = 10.305737    # cold face
    Cp = 824.187344   # mid mass
    Ch = 56.890349    # heatsink

    # ------------------------------------------------------------------
    # PELTIER LAG  [s]
    # First-order smoothing of electrical→thermal power transfer.
    # Identified from real step-response data.
    # ------------------------------------------------------------------
    tau = 31.17

    # ------------------------------------------------------------------
    def __init__(self, dt: float = 0.2):
        self._dt = dt
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, T_init: float = 20.0):
        self.Tc    = T_init
        self.Tm    = T_init
        self.Th    = T_init
        self.P_eff = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def peltier_power(pwm: float) -> float:
        """
        Electrical→thermal power curve fitted from real V/I data.
            Measured: PWM=[0,50,150,200,255], I=[0,0.29,0.80,0.93,1.09] A
            V = 12 V constant  →  quadratic fit on P = V·I
        """
        p = -1.0867e-4 * pwm**2 + 7.8994e-2 * pwm - 5.815e-2
        return max(0.0, p)

    # ------------------------------------------------------------------
    def r_hot(self, fan_on: bool) -> float:
        return self.R_hot_fan_on if fan_on else self.R_hot_fan_off

    # ------------------------------------------------------------------
    def step(self, pwm_peltier: float, Tamb: float,
             fan_on: bool = True):
        """
        Advance the simulation by one time step dt.

        Parameters
        ----------
        pwm_peltier : float   PWM command [0–255]
        Tamb        : float   Ambient temperature [°C]
        fan_on      : bool    Fan state

        Returns
        -------
        Tc, Tm, Th  : float   Node temperatures [°C]
        """
        # Peltier power with first-order lag
        P_target = self.peltier_power(pwm_peltier)
        self.P_eff += (P_target - self.P_eff) * (self._dt / self.tau)
        P = self.P_eff

        R_hot = self.r_hot(fan_on)

        # Heat flows [W]
        q_amb_cold = (Tamb - self.Tc) / self.R_cold_amb   # ambient → cold
        q_cold_mid = (self.Tc - self.Tm) / self.R_cold_mid
        q_mid_hot  = (self.Tm - self.Th) / self.R_mid_hot
        q_hot_amb  = (self.Th - Tamb)    / R_hot           # heatsink → ambient

        # Energy balance
        dTc = (q_amb_cold - q_cold_mid - P) / self.Cc
        dTm = (q_cold_mid - q_mid_hot)       / self.Cp
        dTh = (q_mid_hot  + P - q_hot_amb)   / self.Ch

        # Euler integration
        self.Tc += dTc * self._dt
        self.Tm += dTm * self._dt
        self.Th += dTh * self._dt

        # Physical limits
        self.Tc = float(np.clip(self.Tc, -20.0, 85.0))
        self.Tm = float(np.clip(self.Tm, -20.0, 85.0))
        self.Th = float(np.clip(self.Th, -20.0, 85.0))

        return self.Tc, self.Tm, self.Th

    # ------------------------------------------------------------------
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value


# -----------------------------------------------------------------------
def make_plant(dt: float = 0.2) -> ThreeNodePlant:
    return ThreeNodePlant(dt=dt)
