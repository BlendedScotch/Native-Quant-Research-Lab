"""
Implied Volatility Surface  --  live SSVI  (Gatheral-Jacquier)
================================================================
A rotatable, breathing implied-vol surface IV(k, T) over

    k = log-moneyness  ln(K / F)        (downside = left, negative k)
    T = tenor in years                  (front month -> 2y)

The surface is NOT painted by hand. Each maturity slice is a genuine
Surface-SVI (SSVI) total-variance curve

    w(k, theta) = (theta/2) * [ 1 + rho*phi*k + sqrt( (phi*k + rho)^2 + (1 - rho^2) ) ]
    IV(k, T)    = sqrt( w / T ),   theta = ATM total variance = sigma_atm(T)^2 * T,
    phi(theta)  = eta / theta^gamma   (skew term-structure),  rho = spot/vol correlation.

Two recognizable facts fall straight out of the parameterization:
  * rho < 0           -> the equity skew: downside puts trade at higher IV.
  * phi ~ theta^-gamma -> ATM skew flattens with maturity (~ T^-gamma).

A live STRESS variable (0 = calm, 1 = crisis) drives the regime:
  * level up           : whole surface lifts.
  * skew steepens       : rho -> -0.85, eta rises  (left wing tears upward).
  * term structure flips: front-end vol spikes ABOVE the back end (inversion).

SSVI stays free of static (butterfly + calendar) arbitrage as long as
theta*phi^2*(1+|rho|) <= 4 and theta(T) is non-decreasing; eta is auto-capped
each frame so the stressed surface is still arbitrage-free (status shown).

Side panels slice the surface: the SMILE (IV vs k at one tenor) and the
ATM TERM STRUCTURE (IV vs T) -- watch both reshape as stress moves.

Controls
--------
  ARROWS : rotate  (left/right yaw, up/down tilt)     SPACE : pause
  S : stress shock (crash)   C : force calm   X : toggle auto-stress breathing
  [ / ] : move the smile-slice tenor                  R : reset view+state
  + / - : animation speed   G : record GIF/MP4 (1m)   ESC : quit
"""

import pygame
import math
import random
import numpy as np
import imageio
import os
import threading

# ---------------- SSVI model parameters ----------------
GAMMA          = 0.35                  # skew term-structure decay exponent
RHO_CALM       = -0.55                 # spot/vol correlation (skew sign), calm
RHO_STRESS     = -0.85                 # ... crisis
ETA_CALM       = 0.95                  # skew magnitude, calm
ETA_STRESS     = 1.90                  # ... crisis (auto-capped for no-arb)

# ATM variance term structure: sigma_atm^2(T) = v_long + (v_short - v_long)*exp(-T/TAU)
TAU            = 0.55
VSHORT_CALM    = 0.17 ** 2             # calm: short < long  -> upward sloping (contango)
VLONG_CALM     = 0.21 ** 2
VSHORT_STRESS  = 0.55 ** 2             # crisis: short >> long -> inverted (backwardation)
VLONG_STRESS   = 0.30 ** 2

# stress dynamics (OU toward a low base, with shocks that decay)
STRESS_BASE    = 0.05
STRESS_KAPPA   = 0.9
STRESS_VOL     = 0.10
SHOCK_PROB     = 0.004

# ---------------- grid ----------------
# Increased resolution for a smoother surface
NK, NT     = 55, 41
K_LO, K_HI = -0.55, 0.55
T_LO, T_HI = 1.0 / 12.0, 2.0

# color + height scales (fixed so animation is comparable frame to frame)
IV_COL_LO, IV_COL_HI = 0.10, 0.70
IV_Z_LO,  IV_Z_HI    = 0.05, 0.85

# ---------------- display ----------------
WIDTH, HEIGHT = 1600, 1000
SX0, SX1 = 30, 1060
SY0, SY1 = 120, 930
CX0 = (SX0 + SX1) // 2
CY0 = (SY0 + SY1) // 2 + 60
SCALE = 290

PANEL_L, PANEL_R = 1090, 1570
SM_T, SM_B = 150, 470          # smile panel
TS_T, TS_B = 545, 865          # term-structure panel

BG    = (11, 15, 25)           # #0b0f19
INK   = (228, 230, 238)
DIM   = (120, 124, 140)
GRID  = (34, 39, 52)
FLOOR = (26, 31, 44)
RED   = (255, 96, 110)
AMBER = (245, 190, 90)
GREEN = (90, 210, 140)
BLUE  = (88, 166, 255)
VIOLET = (180, 150, 255)

VIRIDIS = [(68, 1, 84), (72, 34, 115), (64, 67, 135), (52, 94, 141), (41, 120, 142),
           (32, 144, 140), (34, 167, 132), (68, 190, 112), (121, 209, 81),
           (189, 222, 38), (253, 231, 37)]


def cmap(t):
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    x = t * (len(VIRIDIS) - 1)
    i = int(x); f = x - i
    if i >= len(VIRIDIS) - 1:
        return VIRIDIS[-1]
    a, b = VIRIDIS[i], VIRIDIS[i + 1]
    return (int(a[0] + (b[0] - a[0]) * f),
            int(a[1] + (b[1] - a[1]) * f),
            int(a[2] + (b[2] - a[2]) * f))


def lerp(a, b, s):
    return a + (b - a) * s


# ---------------- the surface model ----------------
class Surface:
    def __init__(self):
        self.ks = [K_LO + (K_HI - K_LO) * i / (NK - 1) for i in range(NK)]
        self.ts = [T_LO + (T_HI - T_LO) * j / (NT - 1) for j in range(NT)]
        self.reset()

    def reset(self):
        self.stress = STRESS_BASE
        self.auto = True
        self.sel = NT // 4            # smile slice tenor index
        self.noarb = True
        self.eta_used = ETA_CALM
        self.iv = [[SIGMA(0.2)] for _ in range(NT)]
        self.recompute()

    def shock(self, amt=0.8):
        self.stress = min(1.0, self.stress + amt)

    def calm(self):
        self.stress = STRESS_BASE

    def evolve(self, dt):
        if self.auto:
            self.stress += STRESS_KAPPA * (STRESS_BASE - self.stress) * dt
            self.stress += STRESS_VOL * math.sqrt(dt) * random.gauss(0, 1)
            if random.random() < SHOCK_PROB:
                self.stress += random.uniform(0.4, 0.9)
            self.stress = min(max(self.stress, 0.0), 1.0)

    def _params(self):
        s = self.stress
        rho = lerp(RHO_CALM, RHO_STRESS, s)
        eta = lerp(ETA_CALM, ETA_STRESS, s)
        vshort = lerp(VSHORT_CALM, VSHORT_STRESS, s)
        vlong = lerp(VLONG_CALM, VLONG_STRESS, s)
        # no-arb cap on eta: max over grid of theta*phi^2*(1+|rho|) <= 4
        # phi = eta/theta^gamma  ->  theta*phi^2 = eta^2 * theta^(1-2gamma); worst at theta_max
        theta_max = (vlong + (vshort - vlong) * math.exp(-T_HI / TAU)) * T_HI
        worst_unit = (theta_max ** (1.0 - 2.0 * GAMMA)) * (1.0 + abs(rho))
        eta_cap = math.sqrt(4.0 / worst_unit) * 0.98 if worst_unit > 0 else eta
        self.noarb = eta <= eta_cap
        eta = min(eta, eta_cap)
        self.eta_used = eta
        return rho, eta, vshort, vlong

    def sigma_atm(self, T, vshort, vlong):
        return math.sqrt(vlong + (vshort - vlong) * math.exp(-T / TAU))

    def iv_at(self, k, T, rho, eta, vshort, vlong):
        theta = self.sigma_atm(T, vshort, vlong) ** 2 * T
        phi = eta / (theta ** GAMMA)
        w = 0.5 * theta * (1.0 + rho * phi * k +
                           math.sqrt((phi * k + rho) ** 2 + (1.0 - rho ** 2)))
        return math.sqrt(max(w, 1e-9) / T)

    def recompute(self):
        rho, eta, vshort, vlong = self._params()
        self.iv = [[self.iv_at(self.ks[i], self.ts[j], rho, eta, vshort, vlong)
                    for i in range(NK)] for j in range(NT)]
        self.atm = [self.sigma_atm(self.ts[j], vshort, vlong) for j in range(NT)]
        self.rho, self.eta = rho, eta

    def metrics(self):
        # ATM 1m / 1y, 10%-money risk-reversal proxy at selected tenor, term slope
        atm_1m = self.atm[0]
        # nearest index to 1y
        j1y = min(range(NT), key=lambda j: abs(self.ts[j] - 1.0))
        atm_1y = self.atm[j1y]
        j = self.sel
        # interpolate iv at k = -0.10 and +0.10
        def iv_k(target):
            return float(np.interp(target, self.ks, self.iv[j]))
        rr = iv_k(-0.10) - iv_k(0.10)
        return atm_1m, atm_1y, rr, atm_1y - atm_1m


def SIGMA(x):
    return x


# ---------------- 3D projection ----------------
def project(cx, cy, cz, yaw, pitch):
    # world: x=k, y=T (floor), z=iv (up). yaw about z, then pitch about x.
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    x1 = cx * cyaw - cy * syaw
    y1 = cx * syaw + cy * cyaw
    z1 = cz
    cp, sp = math.cos(pitch), math.sin(pitch)
    y2 = y1 * cp - z1 * sp
    z2 = y1 * sp + z1 * cp
    sx = CX0 + SCALE * x1
    sy = CY0 - SCALE * z2
    return sx, sy, y2           # y2 = depth (larger = farther)


def to_cube(i, j, iv):
    cx = (i / (NK - 1)) * 2.0 - 1.0
    cy = (j / (NT - 1)) * 2.0 - 1.0
    cz = ((iv - IV_Z_LO) / (IV_Z_HI - IV_Z_LO)) * 1.6 - 0.8
    return cx, cy, cz


# ---------------- drawing ----------------
def draw_surface(screen, fonts, surf, yaw, pitch):
    small, font, big, huge = fonts
    # project all vertices
    P = [[project(*to_cube(i, j, surf.iv[j][i]), yaw, pitch)
          for i in range(NK)] for j in range(NT)]

    # floor frame (base plane at cz=-0.8)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    fp = [project(cx, cy, -0.8, yaw, pitch) for cx, cy in corners]
    pygame.draw.polygon(screen, FLOOR, [(p[0], p[1]) for p in fp])
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        pygame.draw.line(screen, GRID, fp[a][:2], fp[b][:2], 1)

    # build quads with average depth for painter's sort
    quads = []
    for j in range(NT - 1):
        for i in range(NK - 1):
            v = (P[j][i], P[j][i + 1], P[j + 1][i + 1], P[j + 1][i])
            ivm = 0.25 * (surf.iv[j][i] + surf.iv[j][i + 1] +
                          surf.iv[j + 1][i + 1] + surf.iv[j + 1][i])
            depth = 0.25 * (v[0][2] + v[1][2] + v[2][2] + v[3][2])
            quads.append((depth, v, ivm))
    quads.sort(key=lambda q: q[0], reverse=True)   # far first

    # depth range for shading
    dmin = min(q[0] for q in quads); dmax = max(q[0] for q in quads)
    drange = (dmax - dmin) or 1.0
    for depth, v, ivm in quads:
        t = (ivm - IV_COL_LO) / (IV_COL_HI - IV_COL_LO)
        base = cmap(t)
        shade = 0.62 + 0.38 * (1.0 - (depth - dmin) / drange)   # nearer = brighter
        col = (int(base[0] * shade), int(base[1] * shade), int(base[2] * shade))
        poly = [(p[0], p[1]) for p in v]
        
        # Draw only the filled polygon to create the seamless smooth surface
        pygame.draw.polygon(screen, col, poly)
        # We explicitly skip drawing the gridline borders here

    # ATM ridge (k = 0 column) and selected smile slice, drawn on top
    i0 = min(range(NK), key=lambda i: abs(surf.ks[i]))
    ridge = [(P[j][i0][0], P[j][i0][1]) for j in range(NT)]
    pygame.draw.lines(screen, INK, False, ridge, 2)
    smile = [(P[surf.sel][i][0], P[surf.sel][i][1]) for i in range(NK)]
    pygame.draw.lines(screen, AMBER, False, smile, 3)

    # axis labels at back corners
    lx = project(0.0, 1.05, -0.8, yaw, pitch)
    screen.blit(small.render("log-moneyness  k", True, DIM), (lx[0] - 70, lx[1]))
    ly = project(1.08, 0.0, -0.8, yaw, pitch)
    screen.blit(small.render("tenor  T", True, DIM), (ly[0] - 20, ly[1]))
    screen.blit(small.render("implied vol", True, DIM), (SX0 + 4, SY0 + 4))
    screen.blit(small.render("downside puts", True, RED),
                (project(-1.0, 1.05, -0.8, yaw, pitch)[0] - 40,
                 project(-1.0, 1.05, -0.8, yaw, pitch)[1] + 18))


def draw_smile(screen, fonts, surf):
    small, font, big, huge = fonts
    j = surf.sel
    title = big.render(f"Smile slice   T = {surf.ts[j]*12:.1f}m", True, INK)
    screen.blit(title, (PANEL_L, SM_T - 38))
    pygame.draw.rect(screen, GRID, (PANEL_L, SM_T, PANEL_R - PANEL_L, SM_B - SM_T), 1)
    ivs = surf.iv[j]
    vmax = max(0.7, max(ivs) * 1.08); vmin = max(0.0, min(ivs) * 0.85)

    def xx(k): return PANEL_L + (k - K_LO) / (K_HI - K_LO) * (PANEL_R - PANEL_L)
    def yy(v): return SM_B - (v - vmin) / (vmax - vmin) * (SM_B - SM_T)

    for gv in (0.2, 0.4, 0.6, 0.8):
        if vmin < gv < vmax:
            y = yy(gv)
            pygame.draw.line(screen, GRID, (PANEL_L, y), (PANEL_R, y), 1)
            screen.blit(small.render(f"{int(gv*100)}%", True, DIM), (PANEL_R + 6, y - 9))
    x0 = xx(0.0)
    pygame.draw.line(screen, (60, 64, 80), (x0, SM_T), (x0, SM_B), 1)
    screen.blit(small.render("ATM", True, DIM), (x0 - 14, SM_B + 6))
    pygame.draw.lines(screen, AMBER, False, [(xx(surf.ks[i]), yy(ivs[i])) for i in range(NK)], 3)
    # ATM dot
    iatm = float(np.interp(0.0, surf.ks, ivs))
    pygame.draw.circle(screen, INK, (int(x0), int(yy(iatm))), 4)
    screen.blit(small.render("left wing = downside-put skew", True, RED), (PANEL_L + 8, SM_T + 8))


def draw_term(screen, fonts, surf):
    small, font, big, huge = fonts
    title = big.render("ATM term structure", True, INK)
    screen.blit(title, (PANEL_L, TS_T - 38))
    pygame.draw.rect(screen, GRID, (PANEL_L, TS_T, PANEL_R - PANEL_L, TS_B - TS_T), 1)
    atm = surf.atm
    vmax = max(0.6, max(atm) * 1.1); vmin = max(0.0, min(atm) * 0.82)

    def xx(T): return PANEL_L + (T - T_LO) / (T_HI - T_LO) * (PANEL_R - PANEL_L)
    def yy(v): return TS_B - (v - vmin) / (vmax - vmin) * (TS_B - TS_T)

    for gv in (0.2, 0.3, 0.4, 0.5, 0.6):
        if vmin < gv < vmax:
            y = yy(gv)
            pygame.draw.line(screen, GRID, (PANEL_L, y), (PANEL_R, y), 1)
            screen.blit(small.render(f"{int(gv*100)}%", True, DIM), (PANEL_R + 6, y - 9))
    for Tm, lab in ((1.0/12, "1m"), (0.5, "6m"), (1.0, "1y"), (2.0, "2y")):
        if T_LO <= Tm <= T_HI:
            x = xx(Tm)
            pygame.draw.line(screen, GRID, (x, TS_T), (x, TS_B), 1)
            screen.blit(small.render(lab, True, DIM), (x - 8, TS_B + 6))
    pts = [(xx(surf.ts[j]), yy(atm[j])) for j in range(NT)]
    slope = atm[-1] - atm[0]
    col = GREEN if slope >= 0 else RED
    pygame.draw.lines(screen, col, False, pts, 3)
    tag = "upward (contango)" if slope >= 0 else "INVERTED (backwardation)"
    screen.blit(small.render(tag, True, col), (PANEL_L + 8, TS_T + 8))
    # mark selected tenor
    xs = xx(surf.ts[surf.sel])
    pygame.draw.line(screen, AMBER, (xs, TS_T), (xs, TS_B), 1)


def draw_header(screen, fonts, surf, paused, speed):
    small, font, big, huge = fonts
    title = huge.render("Implied Volatility Surface  -  live SSVI", True, INK)
    screen.blit(title, ((WIDTH - title.get_width()) // 2, 22))

    s = surf.stress
    if s < 0.25:
        reg, rc = "CALM", BLUE
    elif s < 0.6:
        reg, rc = "STRESSED", AMBER
    else:
        reg, rc = "CRISIS", RED
    screen.blit(big.render(f"regime: {reg}", True, rc), (30, 70))

    # stress bar
    bx, by, bw, bh = 250, 78, 280, 16
    pygame.draw.rect(screen, GRID, (bx, by, bw, bh), 1)
    pygame.draw.rect(screen, rc, (bx + 1, by + 1, int((bw - 2) * s), bh - 2))
    screen.blit(small.render(f"stress {s*100:.0f}%", True, DIM), (bx + bw + 10, by - 1))

    atm_1m, atm_1y, rr, slope = surf.metrics()
    stats = [
        (f"ATM 1m: {atm_1m*100:.0f}%", INK),
        (f"ATM 1y: {atm_1y*100:.0f}%", INK),
        (f"25d-RR proxy: {rr*100:+.1f} vol pts", VIOLET),
        (f"term slope (1y-1m): {slope*100:+.0f}", GREEN if slope >= 0 else RED),
        (f"rho {surf.rho:+.2f}   eta {surf.eta:.2f}", DIM),
        (f"no-arb (SSVI): {'free' if surf.noarb else 'free (eta capped)'}",
         GREEN if surf.noarb else AMBER),
    ]
    for i, (t, c) in enumerate(stats):
        screen.blit(font.render(t, True, c), (1180, 30 + i * 30))

    auto = "auto" if surf.auto else "manual"
    hint = (f"ARROWS rotate   SPACE pause   S shock   C calm   X {auto}   "
            f"[ ] slice   +/- speed   G GIF" + ("   [PAUSED]" if paused else f"   x{speed}"))
    screen.blit(font.render(hint, True, DIM), (30, HEIGHT - 30))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Implied Volatility Surface - live SSVI")
    clock = pygame.time.Clock()
    fonts = (pygame.font.SysFont(None, 24),
             pygame.font.SysFont(None, 28),
             pygame.font.SysFont(None, 36),
             pygame.font.SysFont(None, 48))

    surf = Surface()
    yaw, pitch = -0.7, 1.02
    running, paused = True, False
    speed = 1
    recording, frames = False, []

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_s:
                    surf.shock()
                elif e.key == pygame.K_c:
                    surf.calm()
                elif e.key == pygame.K_x:
                    surf.auto = not surf.auto
                elif e.key == pygame.K_r:
                    surf.reset(); yaw, pitch = -0.7, 1.02
                elif e.key == pygame.K_LEFTBRACKET:
                    surf.sel = max(0, surf.sel - 1)
                elif e.key == pygame.K_RIGHTBRACKET:
                    surf.sel = min(NT - 1, surf.sel + 1)
                elif e.key == pygame.K_g:
                    recording = not recording
                    if recording:
                        frames = []
                        print("Recording started (MP4)...")
                    else:
                        print("Recording stopped. Saving...")
                        save_video(frames)
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(speed + 1, 8)
                elif e.key == pygame.K_MINUS:
                    speed = max(speed - 1, 1)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  yaw -= 0.03
        if keys[pygame.K_RIGHT]: yaw += 0.03
        if keys[pygame.K_UP]:    pitch = min(pitch + 0.02, 1.45)
        if keys[pygame.K_DOWN]:  pitch = max(pitch - 0.02, 0.30)

        if not paused:
            yaw += 0.0035                       # gentle auto-rotation
            for _ in range(speed):
                surf.evolve(1.0 / 60.0)
            surf.recompute()

        screen.fill(BG)
        draw_surface(screen, fonts, surf, yaw, pitch)
        draw_smile(screen, fonts, surf)
        draw_term(screen, fonts, surf)
        draw_header(screen, fonts, surf, paused, speed)

        if recording:
            view = pygame.surfarray.array3d(screen).transpose([1, 0, 2])
            frames.append(view)
            pygame.draw.circle(screen, RED, (WIDTH - 40, 40), 14)
            if len(frames) >= 60 * 60:
                save_video(frames); recording = False

        pygame.display.flip()
        clock.tick(60)


def save_video(frames):
    if not frames:
        return

    def _save():
        print("Saving MP4 in background...")
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        path = os.path.join(downloads, f"vol_surface_{int(pygame.time.get_ticks())}.mp4")
        imageio.mimsave(path, frames, fps=60, macro_block_size=1)
        print(f"Saved {path}")

    threading.Thread(target=_save, daemon=True).start()


if __name__ == "__main__":
    main()