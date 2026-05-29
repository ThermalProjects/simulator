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
dt   = 0.2

# ── PLANT ────────────────────────────────────────────────
plant = make_plant(dt=dt)

# ── FOPID MEMORY ──────────────────────────────────────────
M = 30          # matches Arduino M=30
I_MAX = 80.0    # matches Arduino I_MAX=80
e_hist = [0.0] * (M + 1)
wI     = [0.0] * (M + 1)
wD     = [0.0] * (M + 1)

# ── ARDUINO CONSTANTS ─────────────────────────────────────
THERMAL_BIAS = 1.0   # Arduino subtracts this from Tref before controlling
TDEW_OFFSET  = 1.5   # Arduino subtracts this from Tdew in dew point mode

def init_gl(lambda_, mu):
    wI[0] = wD[0] = 1.0
    for k in range(1, M + 1):
        wI[k] = wI[k-1] * ((lambda_ - (k-1)) / k)
        wD[k] = wD[k-1] * ((mu     - (k-1)) / k)

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
    Kp        = 28.361,
    Ki        = 0.9195,
    Kd        = 2.5452,
    lambda_   = 1.1946,
    mu        = 1.3013,
    Tamb      = 20.0,
    RH        = 70.0,
    Tsup      = 20.0,
    Tdew_const= 0.0,
    fan_on    = False,
    show_tdew = False,
)

MAX_PWM = 255

# ── HANDLER ───────────────────────────────────────────────
async def handler(websocket):
    global e_hist

    print(f"[WS] Client connected: {websocket.remote_address}")

    plant.reset(T_init=state['Tamb'])
    state['running']   = False
    state['fan_on']    = False
    state['show_tdew'] = False
    e_hist = [0.0] * (M + 1)
    init_gl(state['lambda_'], state['mu'])

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
        global e_hist
        last = asyncio.get_event_loop().time()

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
                    e_hist = [0.0] * (M + 1)
                    init_gl(state['lambda_'], state['mu'])

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

                # Target matches Arduino: Tref - THERMAL_BIAS (or Tdew - TDEW_OFFSET)
                if state['show_tdew']:
                    target = state['Tdew_const'] - TDEW_OFFSET
                else:
                    target = state['Tref'] - THERMAL_BIAS

                error = state['Tsup'] - target

                for i in range(M, 0, -1):
                    e_hist[i] = e_hist[i-1]
                e_hist[0] = error

                sumI = sum(wI[k] * e_hist[k] for k in range(M + 1))
                sumD = sum(wD[k] * e_hist[k] for k in range(M + 1))

                # Match Arduino: clamp integral, regularize derivative
                sumI_clamped = max(-I_MAX, min(I_MAX, sumI))
                sumD_reg     = sumD / (1.0 + abs(sumD))

                u = (
                    state['Kp'] * error
                    + state['Ki'] * (dt ** state['lambda_']) * sumI_clamped
                    + state['Kd'] * (sumD_reg / (dt ** state['mu']))
                )

                pwm = int(max(0, min(MAX_PWM, u)))

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

            now = asyncio.get_event_loop().time()
            await asyncio.sleep(max(0, dt - (now - last)))
            last = now

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