# ======================================================
# plant.py (REALISTIC 3-NODE THERMAL MODEL - FIXED)
# ======================================================

import math

class ThreeNodePlant:
    def __init__(self, params, dt=0.2):

        # ==================================================
        # IDENTIFIED PARAMETERS
        # ==================================================
        self.R1 = 5.976550
        self.R2 = 3.638349
        self.R3 = 4.669510

        self.fc = 0.209390
        self.Cc = 6.689968
        self.Cp = 963.111279
        self.Ch = 46.719172

        self.tau = 10.0
        self.dt = dt

        # ==================================================
        # STATES
        # ==================================================
        self.Tc = 20.0
        self.Tm = 20.0
        self.Th = 20.0

        # ==================================================
        # THERMAL MEMORY
        # ==================================================
        self.P_eff = 0.0

    def reset(self):
        self.Tc = 20.0
        self.Tm = 20.0
        self.Th = 20.0
        self.P_eff = 0.0

    # ======================================================
    # PELTIER NONLINEARITY
    # ======================================================
    def peltier_efficiency(self, pwm):
        x = pwm / 255.0
        return x * (1.0 - 0.25 * x)

    # ======================================================
    # FAN EFFECT (NEW)
    # ======================================================
    def fan_cooling(self, pwm_fan):
        """
        Fan does NOT inject heat flow directly,
        but increases convective cooling (R1 reduction)
        """
        return 1.0 + 0.8 * (pwm_fan / 255.0)

    # ======================================================
    # STEP
    # ======================================================
    def step(self, pwm_peltier, Tamb, pwm_fan=0.0):

        # ==================================================
        # PELTIER POWER (SMOOTHED)
        # ==================================================
        P_raw = self.peltier_efficiency(pwm_peltier) * 50.0
        self.P_eff += (P_raw - self.P_eff) * (self.dt / self.tau)
        P = self.P_eff

        # ==================================================
        # FAN EFFECT ON CONVECTION
        # ==================================================
        fan_gain = self.fan_cooling(pwm_fan)
        R1_eff = self.R1 / fan_gain  # ↓ resistance → ↑ cooling

        # ==================================================
        # HEAT FLOWS
        # ==================================================
        q_env = (Tamb - self.Tc) / R1_eff
        q_ct  = (self.Tc - self.Tm) / self.R2
        q_tm  = (self.Tm - self.Th) / self.R3

        # ==================================================
        # ENERGY BALANCE
        # ==================================================
        dTc = (q_env - q_ct - P) / self.Cc
        dTm = (q_ct - q_tm) / self.Cp
        dTh = (q_tm) / self.Ch

        # ==================================================
        # INTEGRATION (STABLE EULER)
        # ==================================================
        self.Tc += dTc * self.dt
        self.Tm += dTm * self.dt
        self.Th += dTh * self.dt

        # ==================================================
        # PHYSICAL LIMITS
        # ==================================================
        self.Tc = max(-10, min(80, self.Tc))
        self.Tm = max(-10, min(80, self.Tm))
        self.Th = max(-10, min(80, self.Th))

        return self.Tc, self.Tm, self.Th