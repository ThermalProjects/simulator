import math

class ThreeNodePlant:

    def __init__(self, dt=0.2):
        # --------------------------------------------------
        # THERMAL RESISTANCES  [K/W]
        # --------------------------------------------------
        self.R_cold_amb = 5.976   # Tc ↔ ambient (cold side convection)
        self.R_cold_mid = 3.638   # Tc ↔ Tm
        self.R_mid_hot  = 4.670   # Tm ↔ Th

        # Hot side to ambient — fan modulates this
        self.R_hot_fan_on  = 1.04   # K/W  (measured, fan=255)
        self.R_hot_fan_off = 1.84   # K/W  (measured, fan=0)

        # --------------------------------------------------
        # THERMAL CAPACITANCES  [J/K]
        # --------------------------------------------------
        self.Cc = 6.69     # cold face
        self.Cp = 963.11   # mid mass
        self.Ch = 46.72    # heatsink

        # --------------------------------------------------
        # PELTIER SMOOTHING TIME CONSTANT  [s]
        # --------------------------------------------------
        self.tau = 8.0     # calibrated from real rise time

        # --------------------------------------------------
        # STATE
        # --------------------------------------------------
        self.Tc    = 20.0
        self.Tm    = 20.0
        self.Th    = 20.0
        self.P_eff = 0.0   # smoothed peltier power [W]

    # ------------------------------------------------------
    def reset(self, T_init=20.0):
        self.Tc    = T_init
        self.Tm    = T_init
        self.Th    = T_init
        self.P_eff = 0.0

    # ------------------------------------------------------
    # PELTIER POWER from PWM  (fitted from real V/I data)
    #   Measured points: PWM=[0,50,150,200,255]
    #                    I  =[0, 0.29, 0.80, 0.93, 1.09] A
    #                    V  = 12 V constant
    #   Quadratic fit:  P = a*pwm² + b*pwm + c
    #   Coeffs: a=-1.087e-4  b=7.899e-2  c=-5.815e-2
    # ------------------------------------------------------
    def peltier_power(self, pwm):
        p = -1.0867e-4 * pwm**2 + 7.8994e-2 * pwm - 5.815e-2
        return max(0.0, p)   # [W], physically bounded

    # ------------------------------------------------------
    # EFFECTIVE R_hot depending on fan state
    # ------------------------------------------------------
    def r_hot(self, fan_on):
        return self.R_hot_fan_on if fan_on else self.R_hot_fan_off

    # ------------------------------------------------------
    # STEP
    # ------------------------------------------------------
    def step(self, pwm_peltier, Tamb, fan_on=True):
        # Peltier power with first-order smoothing
        P_target = self.peltier_power(pwm_peltier)
        self.P_eff += (P_target - self.P_eff) * (self.dt / self.tau)
        P = self.P_eff

        # Effective R_hot (fan modulates hot-side dissipation)
        R_hot = self.r_hot(fan_on)

        # Heat flows  [W]
        q_amb_cold = (Tamb - self.Tc) / self.R_cold_amb   # ambient → cold face
        q_cold_mid = (self.Tc - self.Tm) / self.R_cold_mid
        q_mid_hot  = (self.Tm - self.Th) / self.R_mid_hot
        q_hot_amb  = (self.Th - Tamb)    / R_hot           # heatsink → ambient

        # Energy balance on each node
        # Cold face: gains from ambient, loses to mid, loses pumped heat P
        dTc = (q_amb_cold - q_cold_mid - P) / self.Cc

        # Mid node: gains from cold, loses to hot
        dTm = (q_cold_mid - q_mid_hot) / self.Cp

        # Hot node: gains from mid + P (peltier dumps heat here), loses to ambient
        dTh = (q_mid_hot + P - q_hot_amb) / self.Ch

        # Euler integration
        self.Tc += dTc * self.dt
        self.Tm += dTm * self.dt
        self.Th += dTh * self.dt

        # Physical limits
        self.Tc = max(-20.0, min(85.0, self.Tc))
        self.Tm = max(-20.0, min(85.0, self.Tm))
        self.Th = max(-20.0, min(85.0, self.Th))

        return self.Tc, self.Tm, self.Th

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value


def make_plant(dt=0.2):
    p = ThreeNodePlant(dt=dt)
    p._dt = dt
    return p
