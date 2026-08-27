"""
Beyond Geometric Brownian Motion
================================================================
Two live price paths share the SAME Gaussian shocks every step:

  * GBM  (blue)  -- pure geometric Brownian motion, the Black-Scholes world.
                    Constant volatility, continuous path. It can NEVER gap.

  * MARKET (red) -- the same diffusion PLUS two ingredients GBM lacks:
                      (1) Poisson jumps        -> discontinuities (crashes/gaps)
                      (2) stochastic volatility -> vol clustering (calm vs storm)

Both consume the same base shock z_t, so they track while calm and tear apart
where the market does something GBM cannot. Jumps use the Merton martingale
compensator, so they add RISK without dragging the price to zero.

The volatility risk premium (VRP) is built from exactly the two non-GBM pieces:
implied vol = sqrt( risk-premium x (diffusive variance + jump variance) ), which
sits above realized vol on the large majority of days. A short-vol book harvests
that gap and gets run over on jumps.

Controls
--------
  SPACE : pause / resume      R : reset      J : force a jump now
  + / - : speed up / slow down                 ESC : quit
"""

import pygame
import math
import random
import imageio
import os
import threading
from collections import deque

# ---------------- Model parameters ----------------
DT          = 1.0 / 252.0     # one step = one trading day
MU          = 0.08            # annual drift
SIGMA_BASE  = 0.20            # annual baseline volatility (GBM uses this, constant)
S0          = 100.0

KAPPA       = 5.0             # mean-reversion speed of log-vol
VOL_OF_VOL  = 1.5             # vol-of-vol -> clustering strength
LOGV_TARGET = math.log(SIGMA_BASE)

JUMP_PROB   = 0.018           # per-day probability of a jump
JUMP_MEAN   = -0.030          # mean jump in log-return
JUMP_STD    = 0.055           # jump size dispersion
JUMP_COMP   = JUMP_PROB * (math.exp(JUMP_MEAN + 0.5 * JUMP_STD ** 2) - 1.0)  # Merton compensator

VRP_MARKUP  = 1.35            # risk-neutral variance inflation (tuned: IV>RV ~75% of days)
NOTIONAL    = 100.0           # scales the short-vol P&L to readable units

STEPS_PER_FRAME_DEFAULT = 1

# ---------------- Display ----------------
WIDTH, HEIGHT = 1620, 1240
# left column
CHART_L, CHART_R = 50, 790       # Market vs GBM (the reference size)
CHART_T, CHART_B = 210, 650
WINDOW = CHART_R - CHART_L
LEG_Y1, LEG_Y2 = 660, 696        # GBM / MARKET definitions under the chart (wrapped)
PNL_L, PNL_R = 50, 790           # short-vol P&L (under the chart)
PNL_T, PNL_B = 800, 1182
# right column (both panels same size as the Market vs GBM chart)
HIST_L, HIST_R = 850, 1590       # Daily return distribution
HIST_T, HIST_B = 210, 650
VOL_L, VOL_R = 850, 1590         # Implied vs realized vol (the VRP)
VOL_T, VOL_B = 742, 1182

BG    = (16, 18, 24)
INK   = (228, 230, 238)
DIM   = (120, 124, 140)
GRID  = (38, 41, 50)
BLUE  = (88, 166, 255)
RED   = (255, 96, 110)
CURVE = (110, 114, 130)
AMBER = (245, 190, 90)
GREEN = (90, 210, 140)


class Sim:
    def __init__(self):
        self.reset()

    def reset(self):
        self.day = 0
        self.log_gbm = math.log(S0)
        self.log_mkt = math.log(S0)
        self.logv = LOGV_TARGET
        self.gbm = deque(maxlen=WINDOW)
        self.mkt = deque(maxlen=WINDOW)
        self.mkt_rets = []
        self.gbm_rets = []
        self.rvol_gbm = deque(maxlen=WINDOW)
        self.rvol_mkt = deque(maxlen=WINDOW)
        self.jump_days = deque(maxlen=WINDOW)
        self.jump_count = 0
        self.big5_mkt = 0
        self.big5_gbm = 0
        self.worst_sigma = 0.0
        self.last_jump_flash = -999
        self.gbm.append(math.exp(self.log_gbm))
        self.mkt.append(math.exp(self.log_mkt))
        self._force_jump = False
        self.iv = SIGMA_BASE
        self.iv_hist = deque(maxlen=WINDOW)
        self.vrp = 0.0
        self.pnl = 0.0
        self.pnl_hist = deque(maxlen=WINDOW)
        self.pnl_peak = 0.0
        self.max_dd = 0.0
        self.iv_gt_rv = 0          # count of days IV > RV
        self.iv_obs = 0            # total IV observations

    def force_jump(self):
        self._force_jump = True

    def step(self):
        self.day += 1
        sdt = math.sqrt(DT)
        z = random.gauss(0.0, 1.0)

        r_gbm = (MU - 0.5 * SIGMA_BASE ** 2) * DT + SIGMA_BASE * sdt * z
        self.log_gbm += r_gbm

        w = random.gauss(0.0, 1.0)
        self.logv += KAPPA * (LOGV_TARGET - self.logv) * DT + VOL_OF_VOL * sdt * w
        sig_t = math.exp(self.logv)

        r_mkt = (MU - 0.5 * sig_t ** 2) * DT - JUMP_COMP + sig_t * sdt * z
        if self._force_jump or random.random() < JUMP_PROB:
            r_mkt += random.gauss(JUMP_MEAN, JUMP_STD)
            self.jump_days.append(self.day)
            self.jump_count += 1
            self.last_jump_flash = self.day
            self._force_jump = False
        self.log_mkt += r_mkt

        self.gbm.append(math.exp(self.log_gbm))
        self.mkt.append(math.exp(self.log_mkt))
        self.gbm_rets.append(r_gbm)
        self.mkt_rets.append(r_mkt)

        ref = SIGMA_BASE * sdt
        if abs(r_mkt) > 5 * ref:
            self.big5_mkt += 1
        if abs(r_gbm) > 5 * ref:
            self.big5_gbm += 1
        self.worst_sigma = max(self.worst_sigma, abs(r_mkt) / ref)

        self.rvol_gbm.append(_realized_vol(self.gbm_rets))
        rv_now = _realized_vol(self.mkt_rets)
        self.rvol_mkt.append(rv_now)

        exp_diff_var = 0.5 * sig_t ** 2 + 0.5 * SIGMA_BASE ** 2
        jump_var = (JUMP_PROB / DT) * (JUMP_MEAN ** 2 + JUMP_STD ** 2)
        iv = math.sqrt(VRP_MARKUP * (exp_diff_var + jump_var))
        self.iv = iv
        self.iv_hist.append(iv)
        self.vrp = iv - rv_now
        self.iv_obs += 1
        if iv > rv_now:
            self.iv_gt_rv += 1

        pnl_day = NOTIONAL * (iv ** 2 * DT - r_mkt ** 2)
        self.pnl += pnl_day
        self.pnl_hist.append(self.pnl)
        self.pnl_peak = max(self.pnl_peak, self.pnl)
        self.max_dd = max(self.max_dd, self.pnl_peak - self.pnl)

    def iv_gt_rv_pct(self):
        return 100.0 * self.iv_gt_rv / self.iv_obs if self.iv_obs else 0.0

    def regime(self):
        sig_t = math.exp(self.logv)
        if sig_t < SIGMA_BASE * 1.4:
            return "CALM", BLUE
        if sig_t < SIGMA_BASE * 2.5:
            return "STRESSED", AMBER
        return "CRISIS", RED


def _realized_vol(rets, win=20):
    if len(rets) < 2:
        return SIGMA_BASE
    w = rets[-win:]
    m = sum(w) / len(w)
    var = sum((x - m) ** 2 for x in w) / max(len(w) - 1, 1)
    return math.sqrt(var / DT)


def _wrap(text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for wd in words:
        trial = wd if not cur else cur + " " + wd
        if font.size(trial)[0] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


# ---------------- drawing ----------------
def draw_chart(screen, fonts, sim):
    small, font, big, huge = fonts
    title = big.render("Market vs GBM   (log price)", True, INK)
    screen.blit(title, ((CHART_L + CHART_R - title.get_width()) // 2, CHART_T - 38))
    if len(sim.mkt) < 2:
        return
    logs = [math.log(p) for p in sim.gbm] + [math.log(p) for p in sim.mkt]
    lmin, lmax = min(logs), max(logs)
    if lmax - lmin < 1e-6:
        lmax += 0.1
    pad = (lmax - lmin) * 0.06
    lmin -= pad; lmax += pad

    def to_px(i, price, slen):
        x = CHART_L + (i / max(slen - 1, 1)) * WINDOW
        y = CHART_B - (math.log(price) - lmin) / (lmax - lmin) * (CHART_B - CHART_T)
        return x, y

    pygame.draw.rect(screen, GRID, (CHART_L, CHART_T, WINDOW, CHART_B - CHART_T), 1)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = CHART_B - f * (CHART_B - CHART_T)
        pygame.draw.line(screen, GRID, (CHART_L, y), (CHART_R, y), 1)
        screen.blit(small.render(f"{math.exp(lmin + f*(lmax-lmin)):7.1f}", True, DIM), (CHART_R + 8, y - 9))

    base_day = sim.day - len(sim.mkt) + 1
    for jd in sim.jump_days:
        idx = jd - base_day
        if 0 <= idx < len(sim.mkt):
            x = CHART_L + (idx / max(len(sim.mkt) - 1, 1)) * WINDOW
            pygame.draw.line(screen, (70, 58, 30), (x, CHART_T), (x, CHART_B), 1)

    gbm_pts = [to_px(i, p, len(sim.gbm)) for i, p in enumerate(sim.gbm)]
    mkt_pts = [to_px(i, p, len(sim.mkt)) for i, p in enumerate(sim.mkt)]
    pygame.draw.lines(screen, BLUE, False, gbm_pts, 2)
    pygame.draw.lines(screen, RED, False, mkt_pts, 2)
    for pts, col, lbl, series in ((gbm_pts, BLUE, "GBM", sim.gbm), (mkt_pts, RED, "MARKET", sim.mkt)):
        ex, ey = pts[-1]
        pygame.draw.circle(screen, col, (int(ex), int(ey)), 5)
        screen.blit(font.render(f"{lbl}  {series[-1]:.1f}", True, col), (int(ex) - 150, int(ey) - 24))

    if 0 <= sim.day - sim.last_jump_flash <= 18:
        screen.blit(big.render("JUMP  -  GBM cannot do this", True, AMBER), (CHART_L + 12, CHART_T + 12))


def draw_histogram(screen, fonts, sim):
    small, font, big, huge = fonts
    title = big.render("Daily Return Distribution", True, INK)
    screen.blit(title, ((HIST_L + HIST_R - title.get_width()) // 2, HIST_T - 38))
    if len(sim.mkt_rets) < 5:
        return
    ref = SIGMA_BASE * math.sqrt(DT)
    lo, hi = -8 * ref, 8 * ref
    nb = 41
    bw = (hi - lo) / nb
    mh = [0] * nb
    gh = [0] * nb
    for r in sim.mkt_rets:
        b = int((r - lo) / bw)
        if 0 <= b < nb:
            mh[b] += 1
    for r in sim.gbm_rets:
        b = int((r - lo) / bw)
        if 0 <= b < nb:
            gh[b] += 1
    peak = max(max(mh), max(gh), 1)
    w = (HIST_R - HIST_L) / nb

    pygame.draw.line(screen, INK, (HIST_L, HIST_B), (HIST_R, HIST_B), 1)
    for b in range(nb):
        x = HIST_L + (b / nb) * (HIST_R - HIST_L)
        if mh[b]:
            h = (mh[b] / peak) * (HIST_B - HIST_T)
            pygame.draw.rect(screen, RED, (int(x), int(HIST_B - h), max(int(w) - 1, 1), int(h)))
        if gh[b]:
            h = (gh[b] / peak) * (HIST_B - HIST_T)
            pygame.draw.rect(screen, BLUE, (int(x), int(HIST_B - h), max(int(w) - 1, 1), int(h)), 1)

    n = len(sim.mkt_rets)
    pts = []
    for k in range(nb + 1):
        r = lo + k * bw
        dens = math.exp(-r * r / (2 * ref * ref)) / (ref * math.sqrt(2 * math.pi))
        h = (dens * bw * n / peak) * (HIST_B - HIST_T)
        pts.append((HIST_L + (k / nb) * (HIST_R - HIST_L), HIST_B - h))
    pygame.draw.lines(screen, CURVE, False, pts, 2)

    for s in (-5, 5):
        x = HIST_L + ((s * ref - lo) / (hi - lo)) * (HIST_R - HIST_L)
        pygame.draw.line(screen, GRID, (x, HIST_T), (x, HIST_B), 1)
        screen.blit(small.render(f"{s} sd", True, DIM), (x - 16, HIST_B + 8))
    screen.blit(small.render("red = market     blue = GBM     grey = normal", True, DIM), (HIST_L, HIST_B + 32))


def draw_vrp(screen, fonts, sim):
    small, font, big, huge = fonts
    title = big.render("Implied vs Realized Volatility (VRP)", True, INK)
    screen.blit(title, ((VOL_L + VOL_R - title.get_width()) // 2, VOL_T - 38))
    if len(sim.rvol_mkt) < 2 or len(sim.iv_hist) < 2:
        return
    n = min(len(sim.rvol_mkt), len(sim.iv_hist))
    rv = list(sim.rvol_mkt)[-n:]
    iv = list(sim.iv_hist)[-n:]
    vmax = max(max(rv), max(iv), SIGMA_BASE) * 1.1
    vmin = 0.0

    def yv(v):
        return VOL_B - (v - vmin) / (vmax - vmin) * (VOL_B - VOL_T)

    def xv(i):
        return VOL_L + (i / max(n - 1, 1)) * (VOL_R - VOL_L)

    pygame.draw.rect(screen, GRID, (VOL_L, VOL_T, VOL_R - VOL_L, VOL_B - VOL_T), 1)
    # horizontal gridlines with vol labels
    for vv in (0.20, 0.40, 0.60):
        if vv < vmax:
            y = yv(vv)
            pygame.draw.line(screen, GRID, (VOL_L, y), (VOL_R, y), 1)
            screen.blit(small.render(f"{int(vv*100)}%", True, DIM), (VOL_R + 8, y - 9))

    for i in range(n):
        x = xv(i); a, b = yv(iv[i]), yv(rv[i])
        col = (62, 50, 24) if iv[i] >= rv[i] else (64, 26, 30)
        pygame.draw.line(screen, col, (x, min(a, b)), (x, max(a, b)), 1)
    pygame.draw.lines(screen, AMBER, False, [(xv(i), yv(iv[i])) for i in range(n)], 2)
    pygame.draw.lines(screen, RED, False, [(xv(i), yv(rv[i])) for i in range(n)], 2)

    screen.blit(font.render(f"implied {iv[-1]*100:.0f}%", True, AMBER), (VOL_L + 12, VOL_T + 12))
    screen.blit(font.render(f"realized {rv[-1]*100:.0f}%", True, RED), (VOL_R - 170, VOL_T + 12))
    vc = AMBER if sim.vrp >= 0 else RED
    screen.blit(font.render(f"VRP = {sim.vrp*100:+.0f} vol pts", True, vc), (VOL_L + 12, VOL_B - 56))
    screen.blit(font.render(f"implied > realized on {sim.iv_gt_rv_pct():.0f}% of days", True, AMBER),
                (VOL_L + 12, VOL_B - 30))


def draw_legend(screen, fonts, sim):
    small, font, big, huge = fonts
    g = ("GBM (blue): Geometric Brownian Motion, The Black-Scholes World - "
         "Constant Vol, Continuous, Can NEVER gap.")
    m = ("MARKET (red): Same diffusion + Poisson jumps (Gaps) + Stochastic vol "
         "(Clustering); Fat Tails Emerge, Not Assumed.")
    y = LEG_Y1
    lh = 22
    for text, col in ((g, BLUE), (m, RED)):
        for line in _wrap(text, small, WINDOW):
            screen.blit(small.render(line, True, col), (CHART_L, y))
            y += lh
        y += 4


def draw_pnl(screen, fonts, sim):
    small, font, big, huge = fonts
    title = big.render("Short-vol P&L  -  harvesting the VRP", True, INK)
    screen.blit(title, ((PNL_L + PNL_R - title.get_width()) // 2, PNL_T - 38))
    pygame.draw.rect(screen, GRID, (PNL_L, PNL_T, PNL_R - PNL_L, PNL_B - PNL_T), 1)
    if len(sim.pnl_hist) < 2:
        return
    pnl = list(sim.pnl_hist)
    n = len(pnl)
    lo, hi = min(min(pnl), 0.0), max(max(pnl), 0.0)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo -= pad; hi += pad

    def yv(v):
        return PNL_B - (v - lo) / (hi - lo) * (PNL_B - PNL_T)

    def xv(i):
        return PNL_L + (i / max(n - 1, 1)) * (PNL_R - PNL_L)

    pygame.draw.line(screen, DIM, (PNL_L, yv(0.0)), (PNL_R, yv(0.0)), 1)
    base_day = sim.day - n + 1
    for jd in sim.jump_days:
        idx = jd - base_day
        if 0 <= idx < n:
            x = xv(idx)
            pygame.draw.line(screen, (70, 58, 30), (x, PNL_T), (x, PNL_B), 1)
    col = GREEN if pnl[-1] >= 0 else RED
    pygame.draw.lines(screen, col, False, [(xv(i), yv(pnl[i])) for i in range(n)], 2)

    pc = GREEN if sim.pnl >= 0 else RED
    screen.blit(font.render(f"Premium Harvested: {sim.pnl:+.1f}", True, pc), (PNL_L + 12, PNL_T + 10))
    screen.blit(font.render(f"Worst Drawdown: -{sim.max_dd:.1f}", True, RED), (PNL_R - 250, PNL_T + 10))
    screen.blit(small.render("Grinds up collecting premium   .   Vertical Drops = jumps run it over",
                             True, DIM), (PNL_L + 12, PNL_B - 26))


def draw_header(screen, fonts, sim, paused, speed):
    small, font, big, huge = fonts
    title = huge.render("Geometric Brownian Motion (Live)", True, INK)
    screen.blit(title, ((WIDTH - title.get_width()) // 2, 28))
    reg, rc = sim.regime()
    screen.blit(big.render(f"regime: {reg}", True, rc), (50, 104))
    screen.blit(font.render(f"day {sim.day}", True, DIM), (50, 148))
    
    stats = [
        (f"market > 5 sigma days: {sim.big5_mkt}", RED),
        (f"GBM > 5 sigma days: {sim.big5_gbm}", BLUE),
        (f"jumps fired: {sim.jump_count}", AMBER),
        (f"worst move: {sim.worst_sigma:.1f} sigma", RED),
    ]
    for i, (t, c) in enumerate(stats):
        screen.blit(font.render(t, True, c), (1190, 34 + i * 34))

    hint = "SPACE pause   R reset   J force jump   G record GIF (1m)   +/- speed" + ("   [PAUSED]" if paused else f"   x{speed}")
    screen.blit(font.render(hint, True, DIM), (50, HEIGHT - 32))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("GBM Vs Market Stochastic Volatility")
    clock = pygame.time.Clock()
    fonts = (pygame.font.SysFont(None, 24),    # small
             pygame.font.SysFont(None, 28),    # font
             pygame.font.SysFont(None, 36),    # big
             pygame.font.SysFont(None, 48))    # huge

    sim = Sim()
    running, paused = True, False
    speed = STEPS_PER_FRAME_DEFAULT
    
    recording = False
    frames = []

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    sim.reset()
                elif e.key == pygame.K_j:
                    sim.force_jump()
                elif e.key == pygame.K_g:
                    recording = not recording
                    if recording:
                        frames = []
                        print("Recording started (MP4)...")
                    else:
                        print("Recording stopped. Saving...")
                        save_video(frames)
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(speed + 1, 10)
                elif e.key == pygame.K_MINUS:
                    speed = max(speed - 1, 1)

        if not paused:
            for _ in range(speed):
                sim.step()

        screen.fill(BG)
        draw_header(screen, fonts, sim, paused, speed)
        draw_chart(screen, fonts, sim)
        draw_histogram(screen, fonts, sim)
        draw_vrp(screen, fonts, sim)
        draw_legend(screen, fonts, sim)
        draw_pnl(screen, fonts, sim)

        if recording:
            # Capture frame
            view = pygame.surfarray.array3d(screen)
            view = view.transpose([1, 0, 2])  # Pygame uses (w,h), imageio needs (h,w)
            frames.append(view)
            # Visual indicator for recording
            pygame.draw.circle(screen, RED, (WIDTH - 50, 50), 15)
            
            if len(frames) >= 60 * 60:  # 1 minute at 60 FPS
                save_video(frames)
                recording = False 

        pygame.display.flip()
        clock.tick(60)

def save_video(frames):
    if not frames: return
    def _save():
        print("Saving MP4 in background... please wait.")
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        vid_path = os.path.join(downloads_path, f"simulation_{int(pygame.time.get_ticks())}.mp4")
        imageio.mimsave(vid_path, frames, fps=60, macro_block_size=1)
        print(f"Saved as {vid_path}")
    
    threading.Thread(target=_save, daemon=True).start()

if __name__ == "__main__":
    main()