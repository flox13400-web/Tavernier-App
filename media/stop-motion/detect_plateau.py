import numpy as np
from PIL import Image
import glob, os, json

SCR = os.path.dirname(os.path.abspath(__file__))

def whiteish(arr):
    a = arr.astype(np.int16)
    mx = a.max(2); mn = a.min(2)
    sat = mx - mn
    return (mn > 140) & (sat < 45)

def largest_cc(mask):
    # simple flood fill via label propagation using scipy-free BFS on downscaled mask
    from collections import deque
    h, w = mask.shape
    lab = np.zeros((h,w), np.int32)
    cur = 0; best = (0, 0)
    for y in range(h):
        for x in range(w):
            if mask[y,x] and lab[y,x]==0:
                cur += 1; n=0
                dq = deque([(y,x)]); lab[y,x]=cur
                while dq:
                    cy,cx = dq.popleft(); n+=1
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx = cy+dy, cx+dx
                        if 0<=ny<h and 0<=nx<w and mask[ny,nx] and lab[ny,nx]==0:
                            lab[ny,nx]=cur; dq.append((ny,nx))
                if n > best[1]: best = (cur, n)
    return lab == best[0]

def corners_from_mask(m):
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(float)
    s = pts[:,0]+pts[:,1]; d = pts[:,0]-pts[:,1]
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]; bl = pts[np.argmin(d)]
    return np.array([tl,tr,br,bl])

def homography(src, dst):
    A=[]
    for (x,y),(u,v) in zip(src,dst):
        A.append([x,y,1,0,0,0,-u*x,-u*y,-u])
        A.append([0,0,0,x,y,1,-v*x,-v*y,-v])
    A=np.array(A)
    _,_,V=np.linalg.svd(A)
    H=V[-1].reshape(3,3)
    return H/H[2,2]

def warp(img, src_quad, size):
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], float)
    H = homography(dst, src_quad)  # dst -> src (inverse map)
    yy, xx = np.mgrid[0:size, 0:size]
    ones = np.ones_like(xx)
    P = np.stack([xx, yy, ones], -1).reshape(-1,3).T.astype(float)
    Q = H @ P
    Q = (Q[:2]/Q[2]).T.reshape(size, size, 2)
    a = np.asarray(img).astype(np.float32)
    h,w = a.shape[:2]
    x = np.clip(Q[...,0], 0, w-2); y = np.clip(Q[...,1], 0, h-2)
    x0=x.astype(int); y0=y.astype(int); fx=(x-x0)[...,None]; fy=(y-y0)[...,None]
    out = (a[y0,x0]*(1-fx)*(1-fy) + a[y0,x0+1]*fx*(1-fy) +
           a[y0+1,x0]*(1-fx)*fy + a[y0+1,x0+1]*fx*fy)
    return Image.fromarray(out.astype(np.uint8))

res={}
for f in sorted(glob.glob(os.path.join(SCR,'src','*.jpg'))):
    im = Image.open(f).convert('RGB')
    small = im.resize((im.width//6, im.height//6))
    m = whiteish(np.asarray(small))
    m = largest_cc(m)
    c = corners_from_mask(m) * 6.0
    res[os.path.basename(f)] = c.tolist()
    # detect grid content
    N=9; S=90
    w = warp(im, c, N*S)
    g = np.asarray(w.convert('L')).astype(float)
    grid=[]
    for r in range(N):
        row=''
        for col in range(N):
            patch = g[r*S+25:r*S+65, col*S+25:col*S+65]
            row += '#' if patch.mean() < 128 else '.'
        grid.append(row)
    print(os.path.basename(f))
    print('\n'.join(grid)); print()
    w.save(os.path.join(SCR,'rect_'+os.path.basename(f)+'.png'))
json.dump(res, open(os.path.join(SCR,'corners.json'),'w'))
