import math
import numpy as np
from collections import deque


class ThreeNodePlant:
    """
    3-node thermal plant — equations identical to the identification
    model (three_node.py) used in the NSGA-II characterisation.

    Two independent parameter sets are stored: one for fan ON and one
    for fan OFF.  The active set is selected automatically by the fan
    state passed to step().

    Node layout
    -----------
    Tc  — cold face (Peltier cold side)
    Tm  — intermediate thermal mass
    Th  — heatsink (hot side)

    Equations (consistent with three_node.py)
    ------------------------------------------
    dTc = (Tm - Tc) / (R1 * Cc)  -  alpha * P_eff / Cc
    dTm = (Tc - Tm) / (R1 * Cp)  +  (Th - Tm) / (R2 * Cp)
    dTh = (Tm - Th) / (R2 * Ch)  +  beta * P_eff / Ch
          - (Th - Tamb) / (Rconv * Ch)

    where  beta = 1 - alpha
    and    P_eff = P(t - td)   [discrete delay buffer]

    Power curve (fitted from real V/I data)
    ----------------------------------------
    P = -1.0867e-4 * pwm^2  +  7.8994e-2 * pwm  -  5.815e-2   [W]
    """

    # ------------------------------------------------------------------
    # PARAMETERS — fan ON  (identified from dynamic fan-ON experiment)
    # ------------------------------------------------------------------
    PARAMS_FAN_ON = dict(
        R1    =  4.022803,   # K/W  cold <-> mid
        R2    =  6.003481,   # K/W  mid  <-> hot
        Rconv =  4.897706,   # K/W  hot  -> ambient
        Cc    = 10.305737,   # J/K  cold face
        Cp    = 824.187344,  # J/K  mid mass
        Ch    = 56.890349,   # J/K  heatsink
        alpha =  0.302829,   # —    power fraction to cold node
        td    =  0.894863,   # s    transport delay
    )

    # ------------------------------------------------------------------
    # PARAMETERS — fan OFF  (identified from dynamic fan-OFF experiment)
    # ------------------------------------------------------------------
    PARAMS_FAN_OFF = dict(
        R1    =  0.597337,   # K/W
        R2    =  8.208479,   # K/W
        Rconv = 17.079250,   # K/W
        Cc    = 17.171414,   # J/K
        Cp    = 14.771706,   # J/K
        Ch    = 30.015057,   # J/K
        alpha =  0.319792,   # —
        td    =  2.446485,   # s
    )

    # ------------------------------------------------------------------
    def __init__(self, dt: float = 0.2):
        self._dt = dt
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, T_init: float = 20.0):
        """Reset all state to T_init and flush the delay buffers."""
        self.Tc = T_init
        self.Tm = T_init
        self.Th = T_init

        # Separate delay buffers for each fan mode so a mode switch does
        # not corrupt the history of the other.
        max_delay_steps = int(max(
            self.PARAMS_FAN_ON['td'],
            self.PARAMS_FAN_OFF['td']
        ) / self._dt) + 2

        self._buf_on  = deque([0.0] * max_delay_steps,
                               maxlen=max_delay_steps)
        self._buf_off = deque([0.0] * max_delay_steps,
                               maxlen=max_delay_steps)

    # ------------------------------------------------------------------
    @staticmethod
    def peltier_power(pwm: float) -> float:
        """
        Electrical-to-thermal power curve fitted from real V/I data.
            Measured: PWM=[0,50,150,200,255], I=[0,0.29,0.80,0.93,1.09] A
            V = 12 V constant  ->  quadratic fit on P = V*I
        """
        p = -1.0867e-4 * pwm**2 + 7.8994e-2 * pwm - 5.815e-2
        return max(0.0, p)

    # ------------------------------------------------------------------
    def step(self, pwm_peltier: float, Tamb: float,
             fan_on: bool = True):
        """
        Advance the simulation by one time step dt.

        Parameters
        ----------
        pwm_peltier : float   PWM command [0-255]
        Tamb        : float   Ambient temperature [deg C]
        fan_on      : bool    Fan state

        Returns
        -------
        Tc, Tm, Th  : float   Node temperatures [deg C]
        """
        # Select parameter set
        p = self.PARAMS_FAN_ON if fan_on else self.PARAMS_FAN_OFF
        R1    = p['R1']
        R2    = p['R2']
        Rconv = p['Rconv']
        Cc    = p['Cc']
        Cp    = p['Cp']
        Ch    = p['Ch']
        alpha = p['alpha']
        beta  = 1.0 - alpha
        td    = p['td']

        # Select the right delay buffer
        buf = self._buf_on if fan_on else self._buf_off

        # Current power (undelayed)
        P_now = self.peltier_power(pwm_peltier)

        # Append current power and read the delayed value
        buf.append(P_now)
        delay_steps = int(td / self._dt)
        delay_steps = max(0, min(delay_steps, len(buf) - 1))
        P_eff = buf[-(delay_steps + 1)]   # element delay_steps ago

        # ------------------------------------------------------------------
        # Differential equations  (mirror of three_node.py)
        # ------------------------------------------------------------------
        dTc = (self.Tm - self.Tc) / (R1 * Cc) - alpha * P_eff / Cc

        dTm = ((self.Tc - self.Tm) / (R1 * Cp)
               + (self.Th - self.Tm) / (R2 * Cp))

        dTh = ((self.Tm - self.Th) / (R2 * Ch)
               + beta * P_eff / Ch
               - (self.Th - Tamb) / (Rconv * Ch))

        # Euler integration
        self.Tc += dTc * self._dt
        self.Tm += dTm * self._dt
        self.Th += dTh * self._dt

        # Physical limits
        self.Tc = float(np.clip(self.Tc, -10.0, 120.0))
        self.Tm = float(np.clip(self.Tm, -10.0, 120.0))
        self.Th = float(np.clip(self.Th, -10.0, 120.0))

        return self.Tc, self.Tm, self.Th

    # ------------------------------------------------------------------
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value
        self.reset()   # buffers depend on dt, must rebuild


# ----------------------------------------------------------------------
def make_plant(dt: float = 0.2) -> ThreeNodePlant:
    return ThreeNodePlant(dt=dt)
