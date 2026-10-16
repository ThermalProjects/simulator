"""
simulator_server_ws.py
WebSocket server for FOPID digital twin — calibrated plant
"""

import asyncio
import math
import os
from plant import make_plant

# ── CONFIG ────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5005))
dt   = 0.1      # control period [s]; same as the Arduino firmware (Ts = 0.1)

# ── PLANT ────────────────────────────────────────────────
plant = make_plant(dt=dt)

# ── ARDUINO CONSTANTS ─────────────────────────────────────
M            = 30    # GL memory length (same as Arduino)
MAX_PWM      = 255
THERMAL_BIAS = 0.0   # firmware v3+: controller tracks exactly Tref
TDEW_OFFSET  = 0.0   # firmware v6: controller tracks exactly Tdew


class FOPID:
    """
    Fractional-order PID, Grünwald-Letnikov discretisation (same algorithm as the firmware).

        u = Kp*e + Ki*I + Kd * D_reg / h^mu

    * Integral of order lambda:  s^-lambda = (1/s) * s^(1-lambda)
          v[n] = h^-(1-lambda) * sum_k g_k^(1-lambda) e[n-k]     (GL, memory M)
          I[n] = I[n-1] + h * v[n]                               (true accumulator)
    * Derivative of order mu:    D[n] = sum_k g_k^(mu) e[n-k]    (GL, memory M),
          regularised as D / (1 + |D|) and scaled by h^-mu
    * GL weights with alternating sign:  g_0 = 1,  g_k = g_{k-1} * (k-1-a) / k
      (lambda = mu = 1 recovers the classical discrete PID)
    * Conditional anti-windup: the accumulator is frozen while the command is
      saturated in the direction of v, and clamped to +-MAX_PWM/Ki.
    """

    def __init__(self, Kp, Ki, Kd, lambda_, mu, h):
        self.Kp, self.Ki, self.Kd, self.h = Kp, Ki, Kd, h
        a_v = 1.0 - lambda_
        self.wV = [1.0] * (M + 1)      # GL weights of order (1 - lambda)
        self.wD = [1.0] * (M + 1)      # GL weights of order mu
        for k in range(1, M + 1):
            self.wV[k] = self.wV[k-1] * ((k - 1) - a_v) / k
            self.wD[k] = self.wD[k-1] * ((k - 1) - mu)  / k
        self.hV = h ** a_v
        self.hD = h ** mu
        self.e_hist = [0.0] * (M + 1)
        self.I = 0.0

    def step(self, error):
        self.e_hist.insert(0, error)
        self.e_hist.pop()

        sumV = sum(w * e for w, e in zip(self.wV, self.e_hist))
        sumD = sum(w * e for w, e in zip(self.wD, self.e_hist))

        v    = sumV / self.hV
        uP   = self.Kp * error
        uD   = self.Kd * (sumD / (1.0 + abs(sumD))) / self.hD

        u_try = uP + self.Ki * (self.I + self.h * v) + uD
        sat_high = (u_try > MAX_PWM) and (v > 0)
        sat_low  = (u_try < 0)       and (v < 0)

        if not sat_high and not sat_low:
            self.I += self.h * v
            if self.Ki > 1e-6:
                lim = MAX_PWM / self.Ki
                self.I = max(-lim, min(lim, self.I))

        return uP + self.Ki * self.I + uD


def calc_dew(T, RH):
    RH    = max(1.0, min(float(RH), 100.0))
    a     = 17.27
    b     = 237.7
    alpha = (a * T / (b + T)) + math.log(RH / 100.0)
    return (b * alpha) / (a - alpha)

# ── STATE ─────────────────────────────────────────────────
state = dict(
    running   = False,
    Tref      = 12.0,
    Kp        = 49.9933,
    Ki        = 2.9958,
    Kd        = 1.7547,
    lambda_   = 0.7809,
    mu        = 1.2402,
    Tamb      = 20.0,
    RH        = 70.0,
    Tsup      = 20.0,
    Tdew_const= 0.0,
    fan_on    = False,
    show_tdew = False,
)

def new_controller():
    return FOPID(state['Kp'], state['Ki'], state['Kd'],
                 state['lambda_'], state['mu'], dt)

ctrl = new_controller()

# ── HANDLER ───────────────────────────────────────────────
async def handler(websocket):
    global ctrl

    print(f"[WS] Client connected: {websocket.remote_address}")

    plant.reset(T_init=state['Tamb'])
    state['running']   = False
    state['fan_on']    = False
    state['show_tdew'] = False
    ctrl = new_controller()

    queue = asyncio.Queue()

    async def rx():
        try:
            async for msg in websocket:
                for line in msg.split("\n"):
                    line = line.strip()
                    if line:
                        await queue.put(line)
        except Exception:
            pass

    async def loop():
        global ctrl
        clock  = asyncio.get_running_loop().time
        next_t = clock()

        while True:

            # ── RX ─────────────────────────────────────
            while not queue.empty():
                parts = queue.get_nowait().split(',')
                cmd   = parts[0]

                if cmd == "START":
                    state['running'] = True

                    if len(parts) > 1:  state['Tref']    = float(parts[1])
                    if len(parts) > 2:  state['Kp']      = float(parts[2])
                    if len(parts) > 3:  state['Ki']      = float(parts[3])
                    if len(parts) > 4:  state['Kd']      = float(parts[4])
                    if len(parts) > 5:  state['lambda_'] = float(parts[5])
                    if len(parts) > 6:  state['mu']      = float(parts[6])
                    if len(parts) > 8:  state['Tamb']    = float(parts[8])
                    if len(parts) > 9:  state['RH']      = float(parts[9])
                    if len(parts) > 10:
                        state['fan_on'] = (parts[10].strip() == '1')

                    state['Tdew_const'] = calc_dew(state['Tamb'], state['RH'])
                    plant.reset(T_init=state['Tamb'])
                    ctrl = new_controller()

                elif cmd == "STOP":
                    state['running'] = False
                    plant.reset(T_init=state['Tamb'])

                elif cmd == "FAN_ON":
                    state['fan_on'] = True

                elif cmd == "FAN_OFF":
                    state['fan_on'] = False

                elif cmd == "MODE_TDEW_ON":
                    state['show_tdew'] = True

                elif cmd == "MODE_TDEW_OFF":
                    state['show_tdew'] = False

            # ── CONTROL LOOP ───────────────────────────
            if state['running']:

                if state['show_tdew']:
                    target = state['Tdew_const'] - TDEW_OFFSET
                else:
                    target = state['Tref'] - THERMAL_BIAS

                error = state['Tsup'] - target
                u     = ctrl.step(error)
                pwm   = int(max(0, min(MAX_PWM, u)))

                Tc, _, Th = plant.step(pwm, state['Tamb'], fan_on=state['fan_on'])
                state['Tsup'] = Tc

                pwm_fan_out = 255 if state['fan_on'] else 0

                msg = (
                    f"{state['Tamb']:.2f},"
                    f"{Tc:.4f},"
                    f"{state['RH']:.2f},"
                    f"{state['Tdew_const']:.4f},"
                    f"{pwm},"
                    f"{pwm_fan_out},"
                    f"{Th:.4f}\n"
                )

                try:
                    await websocket.send(msg)
                except Exception:
                    return

            # Fixed-rate schedule: exactly one step per dt
            next_t += dt
            delay = next_t - clock()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_t = clock()

    t1 = asyncio.create_task(rx())
    t2 = asyncio.create_task(loop())

    await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)

    for t in [t1, t2]:
        t.cancel()

    print("[WS] Client disconnected")


# ── MAIN ──────────────────────────────────────────────────
async def main():
    import websockets
    print(f"FOPID WebSocket server running on {HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
