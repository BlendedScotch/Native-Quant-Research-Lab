import pygame
import math
import random

# ---------- Config ----------
WIDTH, HEIGHT = 1000, 760
ROWS = 16                 # number of random steps per ball
PPU = 9.5                 # pixels per unit of horizontal displacement
TOTAL_BALLS = 800         # per panel
SPAWN_EVERY = 3           # frames between spawns
FALL_SPEED = 0.14         # rows per frame
DROP_SPEED = 7
SIGMA = math.sqrt(ROWS)   # theoretical std of the NORMAL walk, in units
VISIBLE_UNITS = 22        # half-range; beyond this a ball has "left the board"

FIELD_TOP, FIELD_BOTTOM = 70, 430
BASE = 700                # histogram baseline
PX_PER_BALL = 2.5

# ---------- Dark palette ----------
BG     = (16, 18, 24)     # near-black, faintly blue
INK    = (228, 230, 238)  # off-white text/lines
GREY   = (70, 74, 86)     # pegs: dim, sits behind the action
DIVIDER= (40, 43, 52)     # center split + sigma ticks
CURVE  = (120, 124, 140)  # Gaussian overlay: muted so bars dominate
BLUE   = (88, 166, 255)   # luminous cyan-blue
RED    = (255, 96, 110)   # hot coral-red
SUBTLE = (140, 144, 158)  # subtitle text

# ---------- Step distributions ----------
def normal_step():
    # bounded, finite variance -> classic CLT -> bell curve
    return random.choice([-1.0, 1.0])

def market_step():
    # Student-t, nu=1.5 -> INFINITE variance, power-law tails -> no bell
    nu = 1.5
    z = random.gauss(0.0, 1.0)
    chi2 = random.gammavariate(nu/2.0, 2.0)
    return 0.7 * (z / math.sqrt(chi2/nu))   # 0.7 scales the central bulk to ~match

def pdf(u):
    return math.exp(-u*u/(2*SIGMA*SIGMA)) / (SIGMA*math.sqrt(2*math.pi))

# ---------- Entities ----------
class Ball:
    def __init__(self, step_fn):
        self.p = 0.0
        self.cum = [0.0]
        for _ in range(ROWS):
            self.cum.append(self.cum[-1] + step_fn())
        self.final = self.cum[-1]
        self.y = FIELD_TOP

    def x_unit(self):
        if self.p >= ROWS:
            return self.final
        k = int(self.p); f = self.p - k
        return self.cum[k] + (self.cum[k+1]-self.cum[k])*f

class Panel:
    def __init__(self, cx, title, subtitle, step_fn, color):
        self.cx, self.title, self.subtitle = cx, title, subtitle
        self.step_fn, self.color = step_fn, color
        self.counts = {}; self.overflow = 0; self.landed = 0; self.beyond3 = 0
        self.balls = []; self.spawned = 0

    def maybe_spawn(self):
        if self.spawned < TOTAL_BALLS:
            self.balls.append(Ball(self.step_fn)); self.spawned += 1

    def update(self):
        kept = []
        half_px = VISIBLE_UNITS * PPU
        for b in self.balls:
            b.p += FALL_SPEED
            u = b.x_unit()
            x_px = self.cx + u*PPU
            if b.p < ROWS:
                b.y = FIELD_TOP + (b.p/ROWS)*(FIELD_BOTTOM-FIELD_TOP)
            else:
                if b.y < FIELD_BOTTOM: b.y = FIELD_BOTTOM
                b.y += DROP_SPEED
            b.x_px = x_px
            if abs(x_px - self.cx) > half_px:          # flew off the side
                self.overflow += 1; self.landed += 1; continue
            if b.y >= BASE:                             # landed in a bin
                bn = int(round(u))
                self.counts[bn] = self.counts.get(bn, 0) + 1
                self.landed += 1
                if abs(u) > 3*SIGMA: self.beyond3 += 1
                continue
            kept.append(b)
        self.balls = kept

# ---------- Drawing ----------
def draw_panel(screen, font, big, p):
    # peg lattice (decorative, matches the +-1 walk)
    for r in range(ROWS):
        for c in range(r+1):
            u = -r + 2*c
            x = p.cx + u*PPU
            y = FIELD_TOP + (r/ROWS)*(FIELD_BOTTOM-FIELD_TOP)
            pygame.draw.circle(screen, GREY, (int(x), int(y)), 2)
    # +-3 sigma "impossible zone" markers
    for s in (-3*SIGMA, 3*SIGMA):
        x = int(p.cx + s*PPU)
        for yy in range(FIELD_BOTTOM, BASE, 10):
            pygame.draw.line(screen, DIVIDER, (x, yy), (x, yy+5))
    # histogram bars
    for bn, c in p.counts.items():
        x = p.cx + bn*PPU
        h = c*PX_PER_BALL
        top = max(BASE - h, FIELD_BOTTOM)              # clip tall spikes
        pygame.draw.rect(screen, p.color, (int(x-PPU/2), int(top), int(PPU)-1, int(BASE-top)))
    # Gaussian "if it were normal" overlay
    pts = []
    for i in range(-VISIBLE_UNITS, VISIBLE_UNITS+1):
        h = p.landed * pdf(i) * PX_PER_BALL
        pts.append((p.cx + i*PPU, BASE - h))
    if len(pts) > 1:
        pygame.draw.lines(screen, CURVE, False, pts, 2)
    pygame.draw.line(screen, INK, (p.cx-VISIBLE_UNITS*PPU, BASE), (p.cx+VISIBLE_UNITS*PPU, BASE), 1)
    # balls in flight
    for b in p.balls:
        pygame.draw.circle(screen, p.color, (int(b.x_px), int(b.y)), 3)
    # labels
    screen.blit(big.render(p.title, True, INK), (p.cx-160, 20))
    screen.blit(font.render(p.subtitle, True, SUBTLE), (p.cx-160, 48))
    pct = 100*p.beyond3/max(p.landed,1)
    stat = f"{p.landed} balls   beyond 3sigma: {pct:.1f}%   off the board: {p.overflow}"
    screen.blit(font.render(stat, True, p.color), (p.cx-160, BASE+14))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Two kinds of randomness")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    big = pygame.font.SysFont(None, 30)

    left = Panel(WIDTH*0.27, "NORMAL RANDOMNESS",
                 "coin-flip steps . finite variance . a Bell Curve", normal_step, BLUE)
    right = Panel(WIDTH*0.73, "MARKET RANDOMNESS",
                  "power-law steps . infinite variance . No Bell Curve", market_step, RED)
    panels = [left, right]

    running, frame = True, 0
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
        if frame % SPAWN_EVERY == 0:
            for p in panels: p.maybe_spawn()
        for p in panels: p.update()

        screen.fill(BG)
        pygame.draw.line(screen, DIVIDER, (WIDTH//2, 60), (WIDTH//2, HEIGHT-30), 1)
        for p in panels: draw_panel(screen, font, big, p)
        pygame.display.flip()
        clock.tick(60)
        frame += 1
    pygame.quit()

main()