"""
simulator_server_ws.py
WebSocket server for FOPID digital twin (Render/VPS ready)
"""

import asyncio
import math
import os
from plant import ThreeNodePlant

# ── CONFIG ────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5005))  # <<< CRÍTICO PARA RENDER
dt = 0.2

# ── PLANT ────────────────────────────────────────────────
params = [25, 40, 60, 200, 220, 250, 0.55, 1.0]
plant = ThreeNodePlant(params, dt=dt)

# ── FOPID MEMORY ──────────────────────────────────────────
M = 40
e_hist = [0.0] * (M + 1)
wI = [0.0] * (M + 1)
wD = [0.0] * (M + 1)

def init_gl(lambda_, mu):
    wI[0] = wD[0] = 1.0
    for k in range(1, M + 1):
        wI[k] = wI[k-1] * ((lambda_ - (k-1)) / k)
        wD[k] = wD[k-1] * ((mu - (k-1)) / k)

def calc_dew(T, RH):
    RH = max(1.0, min(float(RH), 100.0))
    a = 17.27
    b = 237.7
    alpha = (a * T / (b + T)) + math.log(RH / 100.0)
    return (b * alpha) / (a - alpha)

# ── STATE ────────────────────────────────────────────────
state = dict(
    running=False,
    Tref=25.0,
    Kp=35.0, Ki=2.0, Kd=1.2,
    lambda_=0.8, mu=1.2,
    Tamb=20.0, RH=70.0,
    Tsup=20.0,
    Tdew_const=0.0,
    pwmFan=0.0,
)

MAX_PWM = 255

# ── HANDLER ───────────────────────────────────────────────
async def handler(websocket):
    global e_hist

    print(f"[WS] Client connected: {websocket.remote_address}")

    plant.reset()
    state['running'] = False
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
        except:
            pass

    async def loop():
        global e_hist
        last = asyncio.get_event_loop().time()

        while True:

            # ── RX ─────────────────────────────
            while not queue.empty():
                parts = queue.get_nowait().split(',')
                cmd = parts[0]

                if cmd == "START":
                    state['running'] = True

                    if len(parts) > 1: state['Tref'] = float(parts[1])
                    if len(parts) > 2: state['Kp'] = float(parts[2])
                    if len(parts) > 3: state['Ki'] = float(parts[3])
                    if len(parts) > 4: state['Kd'] = float(parts[4])
                    if len(parts) > 5: state['lambda_'] = float(parts[5])
                    if len(parts) > 6: state['mu'] = float(parts[6])
                    if len(parts) > 8: state['Tamb'] = float(parts[8])
                    if len(parts) > 9: state['RH'] = float(parts[9])

                    state['Tdew_const'] = calc_dew(state['Tamb'], state['RH'])
                    plant.reset()
                    e_hist = [0.0] * (M + 1)
                    init_gl(state['lambda_'], state['mu'])

                elif cmd == "STOP":
                    state['running'] = False
                    plant.reset()

                elif cmd == "DATA":
                    try:
                        state['pwmFan'] = float(parts[2])
                    except:
                        pass

            # ── CONTROL ───────────────────────
            if state['running']:

                error = state['Tsup'] - state['Tref']

                for i in range(M, 0, -1):
                    e_hist[i] = e_hist[i-1]
                e_hist[0] = error

                sumI = sum(wI[k] * e_hist[k] for k in range(M+1))
                sumD = sum(wD[k] * e_hist[k] for k in range(M+1))

                u = (
                    state['Kp'] * error +
                    state['Ki'] * (dt ** state['lambda_']) * sumI +
                    state['Kd'] * (sumD / (dt ** state['mu']))
                )

                pwm = int(max(0, min(MAX_PWM, u)))

                Tc, _, _ = plant.step(pwm, state['Tamb'], state['pwmFan'])
                state['Tsup'] = Tc

                msg = f"{state['Tamb']:.2f},{Tc:.2f},{state['RH']:.2f},{state['Tdew_const']:.2f},{pwm},{state['pwmFan']}\n"

                try:
                    await websocket.send(msg)
                except:
                    return

            now = asyncio.get_event_loop().time()
            await asyncio.sleep(max(0, dt - (now - last)))
            last = now

    t1 = asyncio.create_task(rx())
    t2 = asyncio.create_task(loop())

    done, pending = await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)

    for t in pending:
        t.cancel()

    print("[WS] Client disconnected")


# ── MAIN ────────────────────────────────────────────────
async def main():
    import websockets

    print(f"FOPID WebSocket running on {HOST}:{PORT}")

    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())