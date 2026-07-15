import numpy as np
from engine.io.dem import load_dem
from engine.core.terrain_analysis import compute_slope_aspect
from engine.core.four_beasts import _sector_mask

dem = load_dem(r'D:\Xunlong\engine\tests\fixtures\synth_dem.tif')
slope, _ = compute_slope_aspect(dem)
h, w = dem.data.shape
cy, cx = 100, 100
mpx, mpy = 30.0, 30.0
yy, xx = np.mgrid[0:h, 0:w]
dx_m = (xx - cx) * mpx
dy_m = (cy - yy) * mpy
dist_m = np.hypot(dx_m, dy_m)
bearing = (np.degrees(np.arctan2(dx_m, dy_m)) + 360) % 360
within = (dist_m <= 300) & np.isfinite(dem.data)
ql = _sector_mask(bearing, 90, 45) & within
bh = _sector_mask(bearing, 270, 45) & within
print('ql count', ql.sum(), 'max_elev_in_ql', float(np.nanmax(dem.data[ql])))
print('bh count', bh.sum(), 'max_elev_in_bh', float(np.nanmax(dem.data[bh])))
print('qinglong pos bearing:', bearing[100, 118], 'dist:', dist_m[100, 118])
print('baihu pos bearing:', bearing[100, 82], 'dist:', dist_m[100, 82])
print('baseline elev:', dem.data[100, 100])
