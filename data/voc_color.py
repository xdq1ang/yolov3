import numpy as np

def voc_colormap(N=21):
    def bitget(val, idx): return ((val & (1 << idx)) != 0)
 
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3
        cmap[i, :] = [r, g, b]
    return cmap
