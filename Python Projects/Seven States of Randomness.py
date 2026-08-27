import pygame
import math
import random

# ---------- Config ----------
WIDTH, HEIGHT = 1100, 820
ROWS = 20                 # peg rows AND random steps per ball
PPU = 15.0                # pixels per unit of horizontal displacement
TOTAL_BALLS = 700
SPAWN_EVERY = 2
FALL_SPEED = 0.16
DROP_SPEED = 7
SIGMA = math.sqrt(ROWS)   # std of the NORMAL walk (unit-variance steps)
VISIBLE_UNITS = 22        # half-range; beyond this a ball has left the board
CX = WIDTH // 2
FIELD_TOP, FIELD_BOTTOM = 160, 480
BASE = 700
PX_PER_BALL = 2.2

# ---------- Dark palette ----------
BG     = (16, 18, 24)
INK    = (228, 230, 238)
DIM    = (120, 124, 140)
GRID   = (40, 43, 52)
PEG    = (54, 58, 72)     # the Galton lattice: dim, sits behind the action
CURVE  = (110, 114, 130)  # Gaussian reference overlay
SPECTRUM = {
    1: (88, 200, 255), 2: (90, 210, 180), 3: (150, 210, 120), 4: (230, 210, 100),
    5: (245, 165, 80), 6: (255, 96, 110), 7: (220, 80, 230),
}

# ---------- Step distributions (one per state) ----------
def _student_t(nu):
    z = random.gauss(0.0, 1.0)
    chi2 = random.gammavariate(nu / 2.0, 2.0)
    return z / math.sqrt(chi2 / nu)

def _centered_lognormal(sigma):
    v = math.exp(random.gauss(0.0, sigma))
    mean = math.exp(sigma * sigma / 2.0)
    std = math.sqrt(math.exp(sigma * sigma) - 1.0) * mean
    return (v - mean) / std            # mean 0, std 1

def step_1(): return random.gauss(0.0, 1.0)                       # proper mild
def step_2(): return random.expovariate(1.0) - 1.0               # borderline mild
def step_3(): return _centered_lognormal(0.5)                    # slow, delocalized
def step_4(): return _centered_lognormal(1.0)                    # slow, localized
def step_5():                                                    # pre-wild
    nu = 3.0
    return _student_t(nu) / math.sqrt(nu / (nu - 2.0))
def step_6(): return 0.5 * _student_t(1.3)                       # wild
def step_7():                                                    # extreme
    c = math.tan(math.pi * (random.random() - 0.5))
    sign = 1.0 if random.random() < 0.5 else -1.0
    return sign * 0.15 * math.exp(min(abs(c) * 0.5, 9.0))

#         name                 distribution        moment condition                       real-world example                          step
STATES = {
    1: ("PROPER MILD",        "Normal",            "all moments finite . CLT holds",       "adult human height",                       step_1),
    2: ("BORDERLINE MILD",    "Exponential",       "concentrated at N=2, evens out",       "time between radioactive decays",          step_2),
    3: ("SLOW (delocalized)", "mild lognormal",    "moments finite, scale grows fast",     "particle / organism sizes",                step_3),
    4: ("SLOW (localized)",   "lognormal sigma=1", "moments finite, convergence glacial",  "personal incomes (the bulk)",              step_4),
    5: ("PRE-WILD",           "Student-t nu=3",    "variance finite, kurtosis infinite",   "daily stock-market returns",               step_5),
    6: ("WILD",               "Student-t nu=1.3",  "infinite variance . one event rules",  "earthquake energy . top-end wealth",       step_6),
    7: ("EXTREME",            "log-Cauchy",        "all moments infinite . never in practice", "war / pandemic death tolls (debated)", step_7),
}

def gauss_pdf(u):
    return math.exp(-u * u / (2 * SIGMA * SIGMA)) / (SIGMA * math.sqrt(2 * math.pi))

# ---------- Entities ----------
class Ball:
    def __init__(self, step_fn):
        self.cum = [0.0]
        biggest = total = 0.0
        for _ in range(ROWS):
            s = step_fn()
            self.cum.append(self.cum[-1] + s)
            biggest = max(biggest, abs(s)); total += abs(s)
        self.final = self.cum[-1]
        self.concentration = biggest / total if total > 0 else 0.0   # portioning
        self.p = 0.0
        self.y = FIELD_TOP

    def x_unit(self):
        if self.p >= ROWS:
            return self.final
        k = int(self.p); f = self.p - k
        return self.cum[k] + (self.cum[k + 1] - self.cum[k]) * f

class Board:
    def __init__(self, state): self.set_state(state)

    def set_state(self, state):
        self.state = state
        self.name, self.dist, self.cond, self.example, self.step_fn = STATES[state]
        self.color = SPECTRUM[state]
        self.counts = {}; self.balls = []; self.spawned = 0
        self.landed = self.overflow = self.beyond3 = 0; self.conc_sum = 0.0

    def maybe_spawn(self):
        if self.spawned < TOTAL_BALLS:
            self.balls.append(Ball(self.step_fn)); self.spawned += 1

    def update(self):
        kept = []; half_px = VISIBLE_UNITS * PPU
        for b in self.balls:
            b.p += FALL_SPEED
            u = b.x_unit()
            x_px = CX + u * PPU
            if b.p < ROWS:
                b.y = FIELD_TOP + (b.p / ROWS) * (FIELD_BOTTOM - FIELD_TOP)
            else:
                if b.y < FIELD_BOTTOM: b.y = FIELD_BOTTOM
                b.y += DROP_SPEED
            b.x_px = x_px
            if abs(x_px - CX) > half_px:
                self.overflow += 1; self.landed += 1; self.conc_sum += b.concentration; continue
            if b.y >= BASE:
                bn = int(round(u))
                self.counts[bn] = self.counts.get(bn, 0) + 1
                self.landed += 1; self.conc_sum += b.concentration
                if abs(u) > 3 * SIGMA: self.beyond3 += 1
                continue
            kept.append(b)
        self.balls = kept

    def mean_conc(self): return self.conc_sum / self.landed if self.landed else 0.0

# ---------- Drawing ----------
def draw(screen, fonts, board):
    font, big, huge = fonts

    # --- Galton peg pyramid (decorative scaffold of the IDEAL mild board) ---
    for r in range(ROWS):
        yy = FIELD_TOP + (r / ROWS) * (FIELD_BOTTOM - FIELD_TOP)
        for c in range(r + 1):
            xx = CX + (-r + 2 * c) * PPU
            pygame.draw.circle(screen, PEG, (int(xx), int(yy)), 2)

    # reference grid: +-1,2,3 sigma
    for k in (1, 2, 3):
        for s in (-k * SIGMA, k * SIGMA):
            x = int(CX + s * PPU)
            col = GRID if k < 3 else (70, 74, 90)
            for yy in range(FIELD_TOP, BASE, 9):
                pygame.draw.line(screen, col, (x, yy), (x, yy + 4))

    # histogram
    for bn, c in board.counts.items():
        x = CX + bn * PPU; h = c * PX_PER_BALL
        top = max(BASE - h, FIELD_TOP)
        pygame.draw.rect(screen, board.color,
                         (int(x - PPU / 2), int(top), max(int(PPU) - 1, 2), int(BASE - top)))

    # Gaussian reference overlay (always the SAME bell)
    pts = [(CX + i * PPU, BASE - board.landed * gauss_pdf(i) * PX_PER_BALL)
           for i in range(-VISIBLE_UNITS, VISIBLE_UNITS + 1)]
    if len(pts) > 1: pygame.draw.lines(screen, CURVE, False, pts, 2)
    pygame.draw.line(screen, INK, (CX - VISIBLE_UNITS * PPU, BASE),
                     (CX + VISIBLE_UNITS * PPU, BASE), 1)

    # balls
    for b in board.balls:
        pygame.draw.circle(screen, board.color, (int(b.x_px), int(b.y)), 3)

    # ---- header ----
    screen.blit(huge.render(f"{board.state}.  {board.name}", True, board.color), (40, 22))
    screen.blit(big.render(board.dist, True, INK), (40, 60))
    screen.blit(font.render(board.cond, True, DIM), (40, 88))
    screen.blit(font.render("real-world:", True, DIM), (40, 116))
    screen.blit(font.render(board.example, True, board.color), (135, 116))

    # ---- live stats ----
    pct = 100 * board.beyond3 / max(board.landed, 1)
    stats = [(f"landed: {board.landed}", INK),
             (f"beyond 3 sigma: {pct:4.1f}%", board.color),
             (f"off the board: {board.overflow}", board.color),
             (f"concentration: {board.mean_conc():4.2f}", board.color)]
    for i, (txt, col) in enumerate(stats):
        screen.blit(font.render(txt, True, col), (WIDTH - 320, 28 + i * 26))
    screen.blit(font.render("largest step / total path: 0=mild, high=wild", True, DIM),
                (WIDTH - 320, 28 + 4 * 26 + 4))

    # ---- footer spectrum ----
    seg = 120; x0 = (WIDTH - seg * 7) // 2; y = 760
    for st in range(1, 8):
        x = x0 + (st - 1) * seg; active = (st == board.state); col = SPECTRUM[st]
        pygame.draw.rect(screen, col if active else tuple(c // 3 for c in col),
                         (x + 4, y, seg - 8, 26))
        if active: pygame.draw.rect(screen, INK, (x + 4, y, seg - 8, 26), 2)
        screen.blit(font.render(str(st), True, BG if active else DIM), (x + seg // 2 - 4, y + 5))
    screen.blit(font.render("press 1-7 to change state  .  R reset  .  mild <------> wild",
                            True, DIM), (x0, y + 34))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mandelbrot's seven states of randomness")
    clock = pygame.time.Clock()
    fonts = (pygame.font.SysFont(None, 24),
             pygame.font.SysFont(None, 28),
             pygame.font.SysFont(None, 38))

    board = Board(1)
    running, frame = True, 0
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if pygame.K_1 <= e.key <= pygame.K_7:
                    board.set_state(e.key - pygame.K_0)
                elif e.key == pygame.K_r:
                    board.set_state(board.state)
                elif e.key == pygame.K_ESCAPE:
                    running = False
        if frame % SPAWN_EVERY == 0:
            board.maybe_spawn()
        board.update()
        screen.fill(BG)
        draw(screen, fonts, board)
        pygame.display.flip()
        clock.tick(60)
        frame += 1
    pygame.quit()

main()