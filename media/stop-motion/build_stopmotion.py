import numpy as np, os, json, math, random
from PIL import Image

SCR = os.path.dirname(os.path.abspath(__file__))
random.seed(7)
N, S = 9, 160          # 9 cells de 160 px -> grille 1440 px

# ---------- outils geometrie ----------
def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u*x, -u*y, -u],
              [0, 0, 0, x, y, 1, -v*x, -v*y, -v]]
    _, _, V = np.linalg.svd(np.array(A))
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]

def warp(img, quad, w, h):
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], float)
    H = homography(dst, quad)
    yy, xx = np.mgrid[0:h, 0:w]
    P = np.stack([xx, yy, np.ones_like(xx)], -1).reshape(-1,3).T.astype(float)
    Q = H @ P; Q = (Q[:2]/Q[2]).T.reshape(h, w, 2)
    a = np.asarray(img).astype(np.float32); ih, iw = a.shape[:2]
    x = np.clip(Q[...,0], 0, iw-2); y = np.clip(Q[...,1], 0, ih-2)
    x0 = x.astype(int); y0 = y.astype(int)
    fx = (x-x0)[...,None]; fy = (y-y0)[...,None]
    out = (a[y0,x0]*(1-fx)*(1-fy) + a[y0,x0+1]*fx*(1-fy) +
           a[y0+1,x0]*(1-fx)*fy + a[y0+1,x0+1]*fx*fy)
    return Image.fromarray(out.astype(np.uint8))

def expand(quad, f):
    c = quad.mean(0)
    return c + (quad - c) * f

corners = {k: np.array(v) for k, v in json.load(open(os.path.join(SCR,'corners.json'))).items()}

# ---------- 1. plateau de base, nettoye ----------
BASE = '8def6165-image.jpg'
img = Image.open(os.path.join(SCR,'src',BASE)).convert('RGB')
grid = np.asarray(warp(img, corners[BASE], N*S, N*S)).astype(np.float32)
# la seule case occupee (r2,c3) + son halo sont remplaces par une case voisine propre
def graft_from(grid, donor, r, c, pad=1.5, feather=48):
    """Remplace la case (r,c) par la meme case prise sur une autre photo redressee.
    Les deux images partagent la geometrie canonique : les lignes de grille coincident."""
    p = int(S*pad); h = S + 2*p
    y, x = r*S - p, c*S - p
    dst = grid[y:y+h, x:x+h]; src = donor[y:y+h, x:x+h].copy()
    ring = np.ones((h, h), bool); ring[feather:-feather, feather:-feather] = False
    src *= (dst[ring].mean(0) / np.maximum(src[ring].mean(0), 1e-6))   # calage d'exposition
    rr = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    w = np.clip(rr/feather, 0, 1)
    a = (np.minimum(w[:, None], w[None, :])**2)[..., None]
    grid[y:y+h, x:x+h] = dst*(1-a) + np.clip(src, 0, 255)*a

DONOR = 'a5c13ec8-image.jpg'            # plateau vide sur toute la zone haute
_d = Image.open(os.path.join(SCR, 'src', DONOR)).convert('RGB')
graft_from(grid, np.asarray(warp(_d, corners[DONOR], N*S, N*S)).astype(np.float32), 2, 3)

board_clean = grid.copy()

# ---------- 2. sprites de pions reels ----------
# ---- detourage du pion : masque alpha, sinon le fond photo (lignes de grille)
#      est recopie par-dessus le plateau et laisse des bavures noires
def _shift(m, dy, dx):
    return np.roll(np.roll(m, dy, 0), dx, 1)

def _close(m, k=3):
    d = m.copy()
    for dy in range(-k, k+1):
        for dx in range(-k, k+1):
            d |= _shift(m, dy, dx)
    e = d.copy()
    for dy in range(-k, k+1):
        for dx in range(-k, k+1):
            e &= _shift(d, dy, dx)
    return e

def _dilate(m, k):
    d = m.copy()
    for dy in range(-k, k+1):
        for dx in range(-k, k+1):
            d |= _shift(m, dy, dx)
    return d

def _erode(m, k):
    e = m.copy()
    for dy in range(-k, k+1):
        for dx in range(-k, k+1):
            e &= _shift(m, dy, dx)
    return e

def _blur(a, r=2, n=2):
    out = a.astype(np.float32)
    for _ in range(n):
        acc = np.zeros_like(out); c = 0
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                acc += _shift(out, dy, dx); c += 1
        out = acc / c
    return out


def sprite_from(fname, r, c, pad=0.34):
    im = Image.open(os.path.join(SCR,'src',fname)).convert('RGB')
    g = np.asarray(warp(im, corners[fname], N*S, N*S)).astype(np.float32)
    p = int(S*pad)
    sp = g[r*S-p:(r+1)*S+p, c*S-p:(c+1)*S+p].copy()
    lum = sp.mean(2); dark = lum < 118
    ys, xs = np.nonzero(dark)
    dy = int(round(sp.shape[0]/2 - ys.mean())); dx = int(round(sp.shape[1]/2 - xs.mean()))
    sp = np.roll(np.roll(sp, dy, 0), dx, 1)
    m = _erode(_close(sp.mean(2) < 118, 2), 6)
    h, w = m.shape
    # on ne garde que la tache connexe qui contient le centre du pion
    from collections import deque
    seen = np.zeros_like(m); cy, cx = h//2, w//2
    if m[cy, cx]:
        dq = deque([(cy, cx)]); seen[cy, cx] = True
        while dq:
            y, x = dq.popleft()
            for ny, nx in ((y+1,x),(y-1,x),(y,x+1),(y,x-1)):
                if 0 <= ny < h and 0 <= nx < w and m[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx] = True; dq.append((ny,nx))
    # bouchage des trous : enveloppe par ligne INTERSECTEE avec l'enveloppe par colonne
    h_, w_ = seen.shape
    xi = np.arange(w_)[None, :]; yi = np.arange(h_)[:, None]
    rows = seen.any(1, keepdims=True)
    rlo = np.where(seen, xi, w_).min(1, keepdims=True)
    rhi = np.where(seen, xi, -1).max(1, keepdims=True)
    rowfill = (xi >= rlo) & (xi <= rhi) & rows
    cols = seen.any(0, keepdims=True)
    clo = np.where(seen, yi, h_).min(0, keepdims=True)
    chi = np.where(seen, yi, -1).max(0, keepdims=True)
    colfill = (yi >= clo) & (yi <= chi) & cols
    fill = _dilate(rowfill & colfill, 6)   # on restitue l'erosion
    lim = np.zeros_like(fill); k = int(S*1.06/2)
    lim[h//2-k:h//2+k, w//2-k:w//2+k] = True
    fill &= lim
    return sp, _blur(fill.astype(np.float32), 1, 2)

SPRITES = [sprite_from('8def6165-image.jpg', 2, 3),
           sprite_from('9f571f1c-image.jpg', 4, 2),
           sprite_from('9f571f1c-image.jpg', 4, 6),
           sprite_from('fcdc0548-image.jpg', 4, 2),
           sprite_from('acb99fb4-image.jpg', 6, 2),
           sprite_from('edfa49a3-image.jpg', 6, 2)]

def jitter(sprite, ang, dx, dy):
    rgb, al = sprite
    r = np.asarray(Image.fromarray(rgb.astype(np.uint8)).rotate(
        ang, resample=Image.BICUBIC, fillcolor=(250, 250, 250))).astype(np.float32)
    a = np.asarray(Image.fromarray((al*255).astype(np.uint8)).rotate(
        ang, resample=Image.BICUBIC, fillcolor=0)).astype(np.float32)/255.0
    return _shift(r, dy, dx), _shift(a, dy, dx)

# placement deterministe mais "fait main" pour chaque case
PLACE = {}
def tile_for(r, c):
    if (r, c) not in PLACE:
        PLACE[(r, c)] = jitter(SPRITES[random.randrange(len(SPRITES))],
                               random.uniform(-1.3, 1.3),
                               random.randint(-2, 2), random.randint(-2, 2))
    return PLACE[(r, c)]

def render(cells):
    """cells = set de (row, col) -> plateau 1440x1440 en float32"""
    out = board_clean.copy()
    for (r, c) in sorted(cells):
        rgb, al = tile_for(r, c); h, w = rgb.shape[:2]
        y0 = int((r+0.5)*S - h/2); x0 = int((c+0.5)*S - w/2)
        sh = _blur(_shift(al, 8, 7), 4, 2)[..., None] * 0.42
        reg = out[y0:y0+h, x0:x0+w]
        reg *= (1 - sh)
        a = al[..., None]
        out[y0:y0+h, x0:x0+w] = reg*(1-a) + rgb*a
    return out

# ---------- 3. police 3x5, calee sur les lettres reelles des photos ----------
FONT = {
 'E': ["#####","#....","#....","####.","#....","#....","#####"],
 'V': ["#...#","#...#","#...#","#...#","#...#",".#.#.","..#.."],
 'G': [".###.","#...#","#....","#..##","#...#","#...#",".###."],
 'F': ["#####","#....","#....","####.","#....","#....","#...."],
 'L': ["#....","#....","#....","#....","#....","#....","#####"],
 'O': [".###.","#...#","#...#","#...#","#...#","#...#",".###."],
}
R0, C0 = 1, 2     # lettre 5x7 centree sur la grille 9x9

def cells_of(letter, k=None):
    seq = [(R0+r, C0+c) for r, row in enumerate(FONT[letter])
                        for c, ch in enumerate(row) if ch == '#']
    return seq if k is None else set(seq[:k])

# ---------- 4. cadrage 9:16 ----------
W, H = 1080, 1920
frame_quad = expand(corners[BASE], 1.135)      # grille + liseré noir du plateau
plate_px = 980
bg = Image.new('RGB', (W, H), (14, 14, 16))
gy = (H - plate_px)//2
gx = (W - plate_px)//2

def to_frame(board):
    """board (grille seule) -> image 1080x1920 avec le cadre noir du plateau"""
    plate = np.asarray(warp(img, frame_quad, plate_px, plate_px)).astype(np.float32)
    inset = int(plate_px * (1/1.135) / 2)
    g0 = plate_px//2 - inset
    gimg = Image.fromarray(board.astype(np.uint8)).resize((inset*2, inset*2), Image.LANCZOS)
    p = Image.fromarray(plate.astype(np.uint8)); p.paste(gimg, (g0, g0))
    f = bg.copy(); f.paste(p, (gx, gy))
    return f

# ---------- 5. sequence ----------
FPS = 30
states = []                       # (set de cases, duree en secondes)
states.append((set(), 0.7))
for i, L in enumerate("EVGFLO"):
    full = cells_of(L)
    for k in range(1, len(full)+1):
        states.append((cells_of(L, k), 0.13))
    last = (L == 'O')
    states.append((set(full), 2.2 if last else 1.05))
    if not last:
        states.append((set(), 0.55 if L != 'G' else 0.85))

out = os.path.join(SCR, 'frames'); os.makedirs(out, exist_ok=True)
for f in os.listdir(out): os.remove(os.path.join(out, f))

cache, lines, idx = {}, [], 0
for cells, dur in states:
    key = tuple(sorted(cells))
    if key not in cache:
        p = os.path.join(out, f'f{idx:04d}.png'); idx += 1
        to_frame(render(cells)).save(p, optimize=True)
        cache[key] = p
    lines.append((cache[key], dur))

with open(os.path.join(SCR, 'concat.txt'), 'w') as fh:
    for p, d in lines:
        fh.write(f"file '{p}'\nduration {d:.3f}\n")
    fh.write(f"file '{lines[-1][0]}'\n")

print(f"{idx} images uniques, {len(lines)} plans, duree {sum(d for _,d in lines):.1f}s")
