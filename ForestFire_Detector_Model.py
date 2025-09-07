import os
from datetime import datetime, timedelta
import ee, geemap
PROJECT_ID = 'earth-engine-project-470617'
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate(); ee.Initialize(project=PROJECT_ID)
print('EE initialized with project:', PROJECT_ID)

Uttarakhand = ee.Geometry.Rectangle([77.5, 28.7, 81.5, 31.3])


START_DATE = '2016-04-09'
END_DATE   = '2016-06-15'

OUTPUT_DIR = 'daily_stacks'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPORT_DAILY_STACKS = True
print(f'Config ready: {START_DATE} -> {END_DATE}  EXPORT_DAILY_STACKS={EXPORT_DAILY_STACKS}')

BUILD_STACK_IN_RAM = False

if BUILD_STACK_IN_RAM:
    stack_list = []
    for band_file in stack_files:
        
        pass
   
    stack = np.stack(stack_list, axis=0)
else:
    print('Skipping in-RAM stack build; use the Out-of-core stack builder (memmap) cell below.')

EXPORT_SEPARATE_BANDS = True   
SEPARATE_DIR = 'daily_bands'

if EXPORT_DAILY_STACKS:
    import pathlib
    scale = 500
    crs = 'EPSG:4326'
    pathlib.Path(SEPARATE_DIR).mkdir(exist_ok=True)
    
    lulc = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map').rename('LULC').clip(Uttarakhand)
    dem = ee.Image('USGS/SRTMGL1_003').clip(Uttarakhand).rename('DEM')
    terrain = ee.Terrain.products(dem)
    slope = terrain.select('slope').rename('Slope')
    aspect = terrain.select('aspect').rename('Aspect')
    hillshade = ee.Terrain.hillshade(dem).rename('Hillshade')
    
    era5 = ee.ImageCollection('ECMWF/ERA5/DAILY')
    ndvi_ic = ee.ImageCollection('MODIS/061/MOD13Q1').select('NDVI')
    burn_ic = ee.ImageCollection('MODIS/061/MCD64A1').select('BurnDate')
    start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_dt   = datetime.strptime(END_DATE, '%Y-%m-%d')
    cur = start_dt
    print('Beginning daily export loop .')
    while cur <= end_dt:
        dstr = cur.strftime('%Y-%m-%d')
        day = ee.Date(dstr)
        
        temp = era5.filterDate(day, day.advance(1,'day')).select('mean_2m_air_temperature').mean().add(-273.15).rename('TempC')
        u    = era5.filterDate(day, day.advance(1,'day')).select('u_component_of_wind_10m').mean().rename('U10')
        v    = era5.filterDate(day, day.advance(1,'day')).select('v_component_of_wind_10m').mean().rename('V10')
        wspd = u.pow(2).add(v.pow(2)).sqrt().rename('WindSpeed')
        ndvi = ndvi_ic.filterDate(day.advance(-16,'day'), day.advance(1,'day')).mean().multiply(0.0001).rename('NDVI')
        burn = burn_ic.filterDate(day.advance(-32,'day'), day.advance(1,'day')).mosaic().rename('BurnDate')
        
        daily_stack = temp.addBands([u, v, wspd, ndvi, burn, lulc, dem, slope, aspect, hillshade])
        out_stack_path = os.path.join(OUTPUT_DIR, f'Uttarakhand_stack_{dstr}.tif')

        if not os.path.exists(out_stack_path):
            print('Exporting stacked', out_stack_path)
            geemap.ee_export_image(daily_stack, filename=out_stack_path, region=Uttarakhand, scale=scale, crs=crs, file_per_band=False)
        else:
            print('Stack exists, skip', out_stack_path)
       
        if EXPORT_SEPARATE_BANDS:
            band_dir = os.path.join(SEPARATE_DIR, dstr)
            pathlib.Path(band_dir).mkdir(parents=True, exist_ok=True)
            band_names = ['TempC','U10','V10','WindSpeed','NDVI','BurnDate','LULC','DEM','Slope','Aspect','Hillshade']
            for idx, bname in enumerate(band_names):
                band_file = os.path.join(band_dir, f'{bname}.tif')
                if os.path.exists(band_file):
                    continue
                
                single = daily_stack.select(idx)
                try:
                    geemap.ee_export_image(single, filename=band_file, region=Uttarakhand, scale=scale, crs=crs, file_per_band=False)
                except Exception as e:
                    print('Band export failed', bname, e)
        cur += timedelta(days=1)
    print('Daily exports complete.')
else:
    print('Skipping export (EXPORT_DAILY_STACKS False).')
import glob, rasterio, numpy as np, os
STACK_PATTERN = os.path.join(OUTPUT_DIR, 'Uttarakhand_stack_*.tif')
stack_files = sorted(glob.glob(STACK_PATTERN))
stack_list = []
dates = []
if stack_files:
    print(f'Found {len(stack_files)} stacked daily files.')
    for f in stack_files:
        dates.append(f.split('_')[-1].replace('.tif',''))
        with rasterio.open(f) as src:
            arr = src.read()  
        if arr.shape[0] != 11:
            raise RuntimeError('Unexpected band count in ' + f)
        stack_list.append(arr.astype('float32'))
else:
   
    BAND_DIR_ROOT = 'daily_bands'
    band_names = ['TempC','U10','V10','WindSpeed','NDVI','BurnDate','LULC','DEM','Slope','Aspect','Hillshade']
    date_dirs = sorted([d for d in os.listdir(BAND_DIR_ROOT) if os.path.isdir(os.path.join(BAND_DIR_ROOT,d))])
    if not date_dirs:
        raise SystemExit('No stacked files or per-band directories found. Run Cell 2.')
    print('Reconstructing stack from per-band directories...')
    for d in date_dirs:
        band_arrays = []
        valid = True
        first_shape = None
        for b in band_names:
            path = os.path.join(BAND_DIR_ROOT, d, f'{b}.tif')
            if not os.path.exists(path):
                print('Missing band', b, 'for date', d, 'skipping date.')
                valid = False; break
            with rasterio.open(path) as src:
                arr = src.read(1) 
            if first_shape is None:
                first_shape = arr.shape
            else:
                if arr.shape != first_shape:
                    print('Shape mismatch for', d, b, 'skipping date.')
                    valid = False; break
            band_arrays.append(arr.astype('float32'))
        if not valid:
            continue
        stack_list.append(np.stack(band_arrays, axis=0))
        dates.append(d)
    if not stack_list:
        raise SystemExit('No valid reconstructed days.')

stack = np.stack(stack_list, axis=0)  
T,B,H,W = stack.shape
print('Loaded data shape:', stack.shape, '| Days:', len(dates))

B_TEMP,B_U,B_V,B_WS,B_NDVI,B_BURN = 0,1,2,3,4,5
B_LULC,B_DEM,B_SLOPE,B_ASPECT,B_HILL = 6,7,8,9,10
means_day0 = stack[0].reshape(11,-1).mean(axis=1)
print('Day0 band means:', means_day0)

WRITE_RECONSTRUCTED_STACKS = True  
STACK_DTYPE = 'float32'
if WRITE_RECONSTRUCTED_STACKS and not stack_files:
    import rasterio
    from rasterio.transform import from_bounds
   
    sample_day_dir = os.path.join('daily_bands', dates[0])
    sample_band_path = os.path.join(sample_day_dir, 'TempC.tif')
    with rasterio.open(sample_band_path) as ssrc:
        profile = ssrc.profile.copy()
        transform = ssrc.transform
        crs = ssrc.crs
    profile.update(count=11, dtype=STACK_DTYPE)
    print('Writing stacked daily GeoTIFFs to', OUTPUT_DIR)
    band_names_order = ['TempC','U10','V10','WindSpeed','NDVI','BurnDate','LULC','DEM','Slope','Aspect','Hillshade']
    for arr, dstr in zip(stack_list, dates):
        out_path = os.path.join(OUTPUT_DIR, f'pauri_stack_{dstr}.tif')
        if os.path.exists(out_path):
            continue
        with rasterio.open(out_path, 'w', **profile) as dst:
            for i in range(11):
                dst.write(arr[i].astype(STACK_DTYPE), i+1)
            dst.update_tags(**{f'B{i+1}': band_names_order[i] for i in range(11)})
        print('Wrote', out_path)
    print('Local stacking complete.')
elif WRITE_RECONSTRUCTED_STACKS and stack_files:
    print('Stacked files already exist; skipping local write.')
else:
    print('Skipped writing reconstructed stacks.')

SHOW_LAST = True     
SAVE_FIGS = False     
FIG_DIR = 'figures'
KEY_BANDS = [
    ('TempC', B_TEMP, '°C air temperature (higher = hotter fuel/air)', 'inferno'),
    ('WindSpeed', B_WS, 'm/s wind speed (higher = more spread potential)', 'plasma'),
    ('NDVI', B_NDVI, 'NDVI vegetation index (-1..1)', 'Greens'),
    ('Burn mask', B_BURN, 'Already burned (red = burned)', 'Reds'),
    ('DEM', B_DEM, 'Elevation (m)', 'terrain'),
    ('Slope', B_SLOPE, 'Slope (degrees)', 'viridis')
]
import numpy as np, os
import matplotlib.pyplot as plt
if SAVE_FIGS:
    os.makedirs(FIG_DIR, exist_ok=True)
sel_days = [0] + ([T-1] if SHOW_LAST and T>1 else [])
for di in sel_days:
    fig, axes = plt.subplots(2,3, figsize=(11,6))
    fig.suptitle(f'Day {di+1}/{T} ({dates[di] if di < len(dates) else di})')
    for ax, (name, idx, desc, cmap) in zip(axes.ravel(), KEY_BANDS):
        data = stack[di, idx]
        if name == 'Burn mask':
            data_plot = (data>0).astype(float)
            vmin, vmax = 0,1
        else:
            data_plot = data
            vmin, vmax = np.nanpercentile(data_plot,2), np.nanpercentile(data_plot,98)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin==vmax:
                vmin, vmax = np.nanmin(data_plot), np.nanmax(data_plot)
        im = ax.imshow(data_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'{name}', fontsize=10)
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(desc, fontsize=7)
    expl = (
        'Interpretation: Higher TempC & WindSpeed usually increase fire spread risk. NDVI: 0.2-0.5 moderate veg; >0.6 dense fuels; near 0 sparse/barren. '
        'Burn mask shows existing burned areas (cannot ignite again immediately). Terrain (DEM, Slope): fires move faster upslope; steep slopes raise intensity.'
    )
    fig.text(0.5, 0.005, expl, ha='center', va='bottom', fontsize=8, wrap=True)
    plt.tight_layout(rect=(0,0.03,1,0.97))
    if SAVE_FIGS:
        out = os.path.join(FIG_DIR, f'key_bands_day{di+1}.png')
        plt.savefig(out, dpi=150)
        print('Saved figure', out)
    plt.show()


import numpy as np
import os
from scipy.ndimage import binary_dilation

# Feature engineering with on-disk memmaps (prevents OOM)
LAG_DAYS = 3                 # number of lagged dynamic days
INCLUDE_DIFFS = True         # include per-lag diffs for dynamics
DILATE_LABELS = True         # optional label dilation
DILATION_RADIUS = 1          # not used directly (3x3 struct now)
MIN_TRAIN_POS_IGNITION = 50  # ignition model off if too few positives

DYN_BASE_IDXS = [B_TEMP, B_U, B_V, B_WS, B_NDVI]
STATIC_IDXS = [B_LULC, B_DEM, B_SLOPE, B_ASPECT, B_HILL]

if LAG_DAYS < 1:
    raise ValueError('LAG_DAYS must be >=1')

structure = np.ones((3,3), dtype=bool) if DILATE_LABELS else None

# Determine number of usable transitions and feature dimension F using first valid t
first_t = LAG_DAYS - 1
last_t = T - 2  # we predict t+1, so last input t is T-2
usable_transitions = max(0, last_t - first_t + 1)
if usable_transitions <= 0:
    raise SystemExit('Not enough days to build transitions. Check LAG_DAYS and T.')

# Build once to discover F
_dyn_lags = [stack[first_t-li, DYN_BASE_IDXS] for li in range(LAG_DAYS)]
_dyn_lags = _dyn_lags[::-1]
_dyn_lags_arr = np.stack(_dyn_lags, axis=0)
_diff_feats_arr = None
if INCLUDE_DIFFS and LAG_DAYS > 1:
    dfs = [_dyn_lags_arr[k]-_dyn_lags_arr[k-1] for k in range(1, LAG_DAYS)]
    _diff_feats_arr = np.stack(dfs, axis=0).reshape(-1, _dyn_lags_arr.shape[-2], _dyn_lags_arr.shape[-1])
_burn_now = (stack[first_t, B_BURN] > 0).astype('uint8')
_static_feats = stack[first_t, STATIC_IDXS]
_dyn_flat = _dyn_lags_arr.reshape(LAG_DAYS*len(DYN_BASE_IDXS), H, W)
_parts = [_dyn_flat]
if _diff_feats_arr is not None and _diff_feats_arr.size:
    _parts.append(_diff_feats_arr)
_parts.append(_burn_now[None, ...])
_parts.append(_static_feats)
_feats_full = np.concatenate(_parts, axis=0)
F = int(_feats_full.shape[0])
del _dyn_lags, _dyn_lags_arr, _diff_feats_arr, _burn_now, _static_feats, _dyn_flat, _parts, _feats_full

pts_per_transition = H * W
N_total = usable_transitions * pts_per_transition
print(f'Preparing memmaps for features: N={N_total} F={F} (transitions={usable_transitions}, pts/transition={pts_per_transition})')

# Prepare memmaps on disk (absolute paths; robust to locked files on Windows)
MMAP_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, 'memmap'))
os.makedirs(MMAP_DIR, exist_ok=True)

X_MEMMAP_PATH = os.path.abspath(os.path.join(MMAP_DIR, 'X_anyburn.dat'))
Y_ANY_MEMMAP_PATH = os.path.abspath(os.path.join(MMAP_DIR, 'y_any.dat'))
Y_IGN_MEMMAP_PATH = os.path.abspath(os.path.join(MMAP_DIR, 'y_ign.dat'))

def _create_memmap_safe(path, dtype, shape):
    try:
        return np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    except OSError as e:
        base, ext = os.path.splitext(path)
        alt = f"{base}_{np.random.randint(1_000_000_000)}{ext}"
        print(f'Memmap path busy or invalid, using fallback: {alt}')
        return np.memmap(alt, dtype=dtype, mode='w+', shape=shape)

X = _create_memmap_safe(X_MEMMAP_PATH, dtype='float32', shape=(N_total, F))
y_any = _create_memmap_safe(Y_ANY_MEMMAP_PATH, dtype='uint8', shape=(N_total,))
y_ign = _create_memmap_safe(Y_IGN_MEMMAP_PATH, dtype='uint8', shape=(N_total,))

any_counts = []  # positives per transition
write_ptr = 0

for t in range(first_t, last_t + 1):
    # Build features for day t
    dyn_lags = [stack[t-li, DYN_BASE_IDXS] for li in range(LAG_DAYS)]
    dyn_lags = dyn_lags[::-1]
    dyn_lags_arr = np.stack(dyn_lags, axis=0)

    diff_feats_arr = None
    if INCLUDE_DIFFS and LAG_DAYS > 1:
        dfs = [dyn_lags_arr[k]-dyn_lags_arr[k-1] for k in range(1, LAG_DAYS)]
        diff_feats_arr = np.stack(dfs, axis=0).reshape(-1, H, W)

    burn_now = (stack[t, B_BURN] > 0).astype('uint8')
    static_feats = stack[t, STATIC_IDXS]

    dyn_flat = dyn_lags_arr.reshape(LAG_DAYS*len(DYN_BASE_IDXS), H, W)
    parts = [dyn_flat]
    if diff_feats_arr is not None and diff_feats_arr.size:
        parts.append(diff_feats_arr)
    parts.append(burn_now[None, ...])
    parts.append(static_feats)

    feats_full = np.concatenate(parts, axis=0)
    F_chk = feats_full.shape[0]
    if F_chk != F:
        raise SystemExit(f'Feature dimension changed at t={t}: {F_chk} vs expected {F}')

    # Labels for t+1
    burn_next = (stack[t+1, B_BURN] > 0).astype('uint8')
    any_burn = burn_next.copy()
    if DILATE_LABELS and any_burn.any():
        any_burn = binary_dilation(any_burn, structure=structure).astype('uint8')
    new_ignition = ((burn_now == 0) & (burn_next == 1)).astype('uint8')

    # Flatten chunk and write to memmaps
    X_chunk = feats_full.reshape(F, -1).T.astype('float32', copy=False)
    y_any_chunk = any_burn.reshape(-1)
    y_ign_chunk = new_ignition.reshape(-1)

    end_ptr = write_ptr + X_chunk.shape[0]
    X[write_ptr:end_ptr] = X_chunk
    y_any[write_ptr:end_ptr] = y_any_chunk
    y_ign[write_ptr:end_ptr] = y_ign_chunk

    any_counts.append(int(y_any_chunk.sum()))
    write_ptr = end_ptr


X.flush(); y_any.flush(); y_ign.flush()

print('Feature memmaps ready:', X.shape, y_any.shape, y_ign.shape)
print('ANY_BURN positives total:', int(y_any.sum()), 'ratio:', float(y_any.mean()))
print('NEW_IGN  positives total:', int(y_ign.sum()), 'ratio:', float(y_ign.mean()))

initial_train_trans = int(0.8 * usable_transitions)
train_trans = initial_train_trans
while train_trans < usable_transitions - 1 and sum(any_counts[train_trans:]) == 0:
    train_trans -= 1
if train_trans <= 0:
    train_trans = initial_train_trans

train_pts = train_trans * pts_per_transition


X_train = X[:train_pts]; X_test = X[train_pts:]
y_any_train = y_any[:train_pts]; y_any_test = y_any[train_pts:]
y_ign_train = y_ign[:train_pts]; y_ign_test = y_ign[train_pts:]

print(f'Train transitions: {train_trans}  Test transitions: {usable_transitions-train_trans}')
print('ANY_BURN train pos:', int(y_any_train.sum()), 'test pos:', int(y_any_test.sum()))
print('IGNITION train pos:', int(y_ign_train.sum()), 'test pos:', int(y_ign_test.sum()))

TRAIN_IGNITION_MODEL = int(y_ign_train.sum()) >= MIN_TRAIN_POS_IGNITION
if not TRAIN_IGNITION_MODEL:
    print(f'Skip ignition model (<{MIN_TRAIN_POS_IGNITION} train positives).')


feature_rows = [None] * usable_transitions  
any_counts = any_counts 
anyburn_rows = None  


import numpy as np, time
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

MAX_TRAIN_SAMPLES = int(8e5)   
NEG_PER_POS      = 4          
RANDOM_STATE     = 42
MAX_TEST_EVAL    = 200_000    
BATCH_PRED       = 200_000     

RF_PARAMS  = dict(
    n_estimators=250,        
    max_depth=20,              
    min_samples_leaf=2,      
    max_features='sqrt',
    n_jobs=-1,
    class_weight='balanced_subsample',
    max_samples=0.5,          
    random_state=RANDOM_STATE,
)
HGB_PARAMS = dict(max_depth=None, learning_rate=0.1, max_iter=400, random_state=RANDOM_STATE)
FAST_MODEL = 'RF'

def downsample(Xd, yd, max_neg_per_pos):
    pos_idx = np.where(yd == 1)[0]
    neg_idx = np.where(yd == 0)[0]
    if pos_idx.size == 0:
        return Xd[:0], yd[:0]
    rng = np.random.RandomState(RANDOM_STATE)
    target_neg = min(len(neg_idx), max_neg_per_pos * len(pos_idx))
    neg_sel = rng.choice(neg_idx, size=target_neg, replace=False)
    keep = np.concatenate([pos_idx, neg_sel])

    
    if keep.size > MAX_TRAIN_SAMPLES:
        keep = rng.choice(keep, size=MAX_TRAIN_SAMPLES, replace=False)

    keep_sorted = np.sort(keep)
    X_small = np.asarray(Xd[keep_sorted])
    y_small = np.asarray(yd[keep_sorted])
    perm = rng.permutation(X_small.shape[0])
    return X_small[perm], y_small[perm]

def fit_eval(name, Xtr, ytr, Xte, yte):
    clf = HistGradientBoostingClassifier(**HGB_PARAMS) if FAST_MODEL.upper()=='HGB' else RandomForestClassifier(**RF_PARAMS)
    Xtr_ds, ytr_ds = downsample(Xtr, ytr, NEG_PER_POS)
    print(f'[{name}] original:{Xtr.shape} pos={int(ytr.sum())} ({float(ytr.mean()):.5f})')
    print(f'[{name}] sampled :{Xtr_ds.shape} pos={int(ytr_ds.sum())} ({float(ytr_ds.mean()):.3f})')
    t0 = time.time()
    clf.fit(Xtr_ds, ytr_ds)
    print(f'[{name}] fit time: {time.time()-t0:.1f}s')

    rng = np.random.RandomState(RANDOM_STATE)
    pos = np.where(yte==1)[0]
    neg = np.where(yte==0)[0]
    n_pos_eval = min(len(pos), MAX_TEST_EVAL//2)
    n_neg_eval = min(len(neg), MAX_TEST_EVAL - n_pos_eval, max(1000, 3*n_pos_eval))
    sel_pos = pos[:n_pos_eval]
    sel_neg = rng.choice(neg, size=n_neg_eval, replace=False) if n_neg_eval>0 else np.array([], dtype=int)
    idx_eval = np.unique(np.sort(np.concatenate([sel_pos, sel_neg]))) if (n_pos_eval+n_neg_eval)>0 else np.array([], dtype=int)

    Xev = Xte[idx_eval]
    yev = yte[idx_eval]

    def proba_for_one_local(model, Xc):
        out = np.empty(Xc.shape[0], dtype=np.float32)
        for s in range(0, Xc.shape[0], BATCH_PRED):
            e = min(s+BATCH_PRED, Xc.shape[0])
            if hasattr(model, 'predict_proba'):
                pp = model.predict_proba(Xc[s:e])
                if hasattr(model, 'classes_') and 1 in list(model.classes_):
                    idx1 = int(np.where(model.classes_ == 1)[0][0])
                    out[s:e] = pp[:, idx1]
                else:
                    out[s:e] = 0.0
            elif hasattr(model, 'decision_function'):
                from sklearn.preprocessing import MinMaxScaler
                z = model.decision_function(Xc[s:e]).reshape(-1,1)
                out[s:e] = MinMaxScaler().fit_transform(z).ravel()
            else:
                out[s:e] = 0.0
        return out

    prob = proba_for_one_local(clf, Xev).astype('float32')
    from sklearn.metrics import f1_score
    ths = np.linspace(0.1, 0.9, 17)
    f1s = [(th, f1_score(yev, (prob>=th).astype('uint8'), zero_division=0)) for th in ths]
    thr = max(f1s, key=lambda x:x[1])[0]
    roc = roc_auc_score(yev, prob) if len(np.unique(yev))>1 else float('nan')
    pr  = average_precision_score(yev, prob) if len(np.unique(yev))>1 else float('nan')
    return clf, float(thr), dict(roc_auc=roc, pr_auc=pr, f1=max(f1s,key=lambda x:x[1])[1])

MODEL_ANY, THR_ANY, METRICS_ANY = fit_eval('ANY_BURN', X_train, y_any_train, X_test, y_any_test)
if TRAIN_IGNITION_MODEL:
    MODEL_IGN, THR_IGN, METRICS_IGN = fit_eval('IGNITION', X_train, y_ign_train, X_test, y_ign_test)


# ...existing code...
# Balanced sampling without large concatenations
import time
if 'MODEL_ANY' in globals() and MODEL_ANY is not None:
    print('MODEL_ANY already trained in previous cell; skipping duplicate training.')
else:
    rng = np.random.RandomState(RANDOM_STATE)

    # Positive and negative indices from memmaps (train view)
    pos_idx = np.where(y_any_train == 1)[0]
    neg_idx = np.where(y_any_train == 0)[0]

    n_pos = len(pos_idx)
    n_neg = min(len(neg_idx), NEG_PER_POS * n_pos)

    if n_pos == 0:
        raise RuntimeError('No positive samples in y_any_train; cannot train ANY_BURN model. Adjust split or sampling.')

    neg_sel = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([pos_idx, neg_sel])

    if keep.size > MAX_TRAIN_SAMPLES:
        keep = rng.choice(keep, size=MAX_TRAIN_SAMPLES, replace=False)

    keep_sorted = np.sort(keep)
    X_tr_bal = np.asarray(X_train[keep_sorted])
    y_tr_bal = np.asarray(y_any_train[keep_sorted])
    perm = rng.permutation(X_tr_bal.shape[0])
    X_tr_bal = X_tr_bal[perm]
    y_tr_bal = y_tr_bal[perm]

    MODEL_ANY = RandomForestClassifier(**RF_PARAMS)
    t0 = time.time()
    MODEL_ANY.fit(X_tr_bal, y_tr_bal)
    print(f'[ANY_BURN] fit time: {time.time()-t0:.1f}s')

    te_pos = np.where(y_any_test == 1)[0]
    te_neg = np.where(y_any_test == 0)[0]
    te_pos = te_pos[:MAX_TEST_EVAL//2]
    te_neg = te_neg[:min(len(te_neg), MAX_TEST_EVAL - len(te_pos), 3*len(te_pos)+1000)]
    sel_te = np.concatenate([te_pos, te_neg])
    sel_te.sort()
    X_tune = X_test[sel_te]
    y_tune = y_any_test[sel_te]

    def proba_for_one(model, Xc, batch=BATCH_PRED):
        out = np.empty(Xc.shape[0], dtype='float32')
        for s in range(0, Xc.shape[0], batch):
            e = min(s+batch, Xc.shape[0])
            pp = model.predict_proba(Xc[s:e])
            if hasattr(model, 'classes_') and 1 in list(model.classes_):
                idx1 = int(np.where(model.classes_ == 1)[0][0])
                out[s:e] = pp[:, idx1]
            else:
                out[s:e] = 0.0
        return out

    prob_tune = proba_for_one(MODEL_ANY, X_tune).astype('float32')
    from sklearn.metrics import f1_score
    ths = np.linspace(0.1, 0.9, 17)
    f1s = []
    for th in ths:
        pred = (prob_tune >= th).astype('uint8')
        f1s.append(f1_score(y_tune, pred, zero_division=0))
    THR_ANY = float(ths[int(np.argmax(f1s))])

    import joblib, os
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': MODEL_ANY, 'thr': THR_ANY, 'train_transitions': int(train_trans)}, 'models/anyburn_rf.joblib')
    print('Saved -> models/anyburn_rf.joblib (thr=', THR_ANY, ')')
  
import matplotlib.pyplot as plt
from imageio.v2 import imread, mimsave
import os
import numpy as np
import joblib


def proba_for_one(model, Xc):
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(Xc)
        if p.ndim == 2:
            if p.shape[1] == 2:
                try:
                    idx1 = int(np.where(model.classes_ == 1)[0][0])
                except Exception:
                    idx1 = 1
                return p[:, idx1]
            if p.shape[1] == 1:
                only = int(getattr(model, 'classes_', np.array([1]))[0])
                return np.full(Xc.shape[0], 1.0 if only == 1 else 0.0, dtype='float32')
        return p.ravel().astype('float32', copy=False)
    if hasattr(model, 'decision_function'):
        df = model.decision_function(Xc).astype('float32', copy=False)
        mn, mx = float(df.min()), float(df.max())
        return (df - mn) / (mx - mn + 1e-9)
    pred = model.predict(Xc).astype('float32', copy=False)
    return pred

PRED_TARGET = 'ANY_BURN'  
HORIZON = 1


g_model, g_thr = None, None
if PRED_TARGET.upper()=='ANY_BURN':
    if 'MODEL_ANY' in globals() and MODEL_ANY is not None:
        g_model, g_thr = MODEL_ANY, THR_ANY if 'THR_ANY' in globals() else 0.5
    else:
        if os.path.exists('models/anyburn_rf.joblib'):
            bundle = joblib.load('models/anyburn_rf.joblib')
            g_model = bundle['model']
            g_thr = float(bundle.get('thr', 0.5))
            
            if 'train_trans' not in globals():
                train_trans = int(bundle.get('train_transitions', 0))
            print('Loaded model from models/anyburn_rf.joblib')
        else:
            raise SystemExit('No in-memory model and no saved model bundle found.')
elif PRED_TARGET.upper()=='IGNITION' and 'MODEL_IGN' in globals() and MODEL_IGN is not None:
    g_model, g_thr = MODEL_IGN, THR_IGN if 'THR_IGN' in globals() else 0.5
else:
    raise SystemExit('Invalid PRED_TARGET or model not available.')


first_test_t = (LAG_DAYS - 1) + train_trans
last_input_t = T - 1 - HORIZON
start_t = min(max(first_test_t, LAG_DAYS-1), last_input_t)
end_t = last_input_t + 1 
print(f'GIF over test inputs t in [{start_t}, {last_input_t}] (inclusive)')

os.makedirs(FIG_DIR, exist_ok=True)
gif_tmp = os.path.join(FIG_DIR, 'gif_tmp')
os.makedirs(gif_tmp, exist_ok=True)
frames = []

first_frame_png = os.path.join(FIG_DIR, 'first_frame_debug.png')
first_saved = False

for t in range(start_t, end_t):

    dyn_lags = []
    for lag in range(LAG_DAYS):
        dyn_lags.append(stack[t-lag, DYN_BASE_IDXS])
    dyn_lags_arr = np.stack(dyn_lags, axis=0)
    dfs = [dyn_lags_arr[li]-dyn_lags_arr[li-1] for li in range(1, LAG_DAYS)]
    diff_feats_arr = np.stack(dfs, axis=0).reshape(-1, H, W) if dfs else []
    burn_now = (stack[t, B_BURN] > 0).astype('uint8')
    static_feats = stack[t, STATIC_IDXS]
    dyn_flat = dyn_lags_arr.reshape(LAG_DAYS*len(DYN_BASE_IDXS), H, W)
    parts = [dyn_flat]
    if len(diff_feats_arr): parts.append(diff_feats_arr)
    parts.append(burn_now[None, ...])
    parts.append(static_feats)
    feats_full = np.concatenate(parts, axis=0)
    X_t = feats_full.reshape(feats_full.shape[0], -1).T
    prob = proba_for_one(g_model, X_t.astype('float32', copy=False))
   
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    risk = prob.reshape(H, W)

    base = stack[t, B_NDVI]
    
    try:
        vmin_base, vmax_base = np.nanpercentile(base,5), np.nanpercentile(base,95)
    except Exception:
        vmin_base, vmax_base = float(np.nanmin(base)), float(np.nanmax(base))
    if not np.isfinite(vmin_base) or not np.isfinite(vmax_base) or vmin_base == vmax_base:
        vmin_base, vmax_base = float(np.nanmin(base)), float(np.nanmax(base))
        if not np.isfinite(vmin_base) or not np.isfinite(vmax_base) or vmin_base == vmax_base:
            vmin_base, vmax_base = 0.0, 1.0

    if t == start_t:
        above_thr = int((prob >= float(g_thr) if g_thr is not None else prob >= 0.5).sum())
        p5, p95 = np.nanpercentile(prob, 5), np.nanpercentile(prob, 95)
        print(f'[Diag t={t}] risk min={float(risk.min()):.3f} max={float(risk.max()):.3f} mean={float(risk.mean()):.3f}  >=thr count={above_thr}  p5={p5:.3f} p95={p95:.3f}')

    try:
        r5, r95 = np.nanpercentile(risk, 5), np.nanpercentile(risk, 95)
    except Exception:
        r5, r95 = 0.0, 1.0
    if not np.isfinite(r5) or not np.isfinite(r95) or r5 == r95:
        r5, r95 = 0.0, 1.0
    risk_stretch = np.clip((risk - r5) / (r95 - r5 + 1e-9), 0.0, 1.0)

    thr_plot = float(g_thr) if g_thr is not None else 0.5
    pred_mask = (risk >= thr_plot).astype(float)
    pred_count = int(pred_mask.sum())

    fig = plt.figure(figsize=(14,7))

    ax1 = plt.subplot(2,2,1)
    im1 = ax1.imshow(base, cmap='Greens', vmin=vmin_base, vmax=vmax_base)
    ax1.set_title(f'NDVI t={t}')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.ax.tick_params(labelsize=7)
    cbar1.set_label('NDVI', fontsize=8)

    ax2 = plt.subplot(2,2,2)
    im2 = ax2.imshow(risk, cmap='inferno', vmin=0, vmax=1)
    ax2.set_title('Risk (probability 0–1)')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.ax.tick_params(labelsize=7)
    cbar2.set_label('Risk (0–1)', fontsize=8)

    ax3 = plt.subplot(2,2,3)
    im3 = ax3.imshow(risk_stretch, cmap='inferno', vmin=0, vmax=1)
    ax3.set_title('Risk (stretched 5–95%)')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)
    cbar3.ax.tick_params(labelsize=7)
    cbar3.set_label('Stretched risk', fontsize=8)

    ax4 = plt.subplot(2,2,4)
    ax4.imshow(base, cmap='Greens', vmin=vmin_base, vmax=vmax_base)
    ax4.imshow(np.ma.masked_where(pred_mask==0, pred_mask), cmap='autumn', alpha=0.6)
    ax4.set_title(f'Predicted mask (thr={thr_plot:.2f}, count={pred_count})')
    ax4.axis('off')

    out_png = os.path.join(gif_tmp, f'frame_{t:04d}.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=110)

    if not first_saved:
        plt.savefig(first_frame_png, dpi=120)
        print('Saved first frame preview ->', first_frame_png)
        first_saved = True

    plt.close(fig)
    frames.append(imread(out_png))

GIF_PATH = os.path.join(FIG_DIR, f'test_risk_h{HORIZON}_{PRED_TARGET.lower()}.gif')
if len(frames) == 0:
    print('No frames to write — check start/end indices (start_t > last_input_t).')
else:
    mimsave(GIF_PATH, frames, fps=3)
    print('Saved GIF ->', GIF_PATH, '| frames:', len(frames))

import os, re, json, pathlib, numpy as np, rasterio
from glob import glob

if 'stack_files' not in globals() or not stack_files:
    pattern = STACK_PATTERN if 'STACK_PATTERN' in globals() else os.path.join(OUTPUT_DIR, 'Uttarakhand_stack_*.tif')
    stack_files = sorted(glob(pattern))
    if not stack_files:
        raise SystemExit('No stacks found — run earlier export/discovery cells first.')

T_guess = len(stack_files)
if T_guess == 0:
    raise SystemExit('No stack files available.')


with rasterio.open(stack_files[0]) as src0:
    B, H, W = src0.count, src0.height, src0.width
print('Detected shape: T=?, B,H,W=', B, H, W)


MMAP_DIR = os.path.join(OUTPUT_DIR, 'memmap')
pathlib.Path(MMAP_DIR).mkdir(parents=True, exist_ok=True)
STACK_MMAP_PATH = os.path.join(MMAP_DIR, 'stack_mmap.dat')
print('Creating memmap at:', STACK_MMAP_PATH)
mm = np.memmap(STACK_MMAP_PATH, dtype='float32', mode='w+', shape=(T_guess, B, H, W))


for i, p in enumerate(stack_files):
    with rasterio.open(p) as src:
        arr = src.read().astype('float32', copy=False)
        if arr.shape != (B, H, W):
            raise SystemExit(f'Shape mismatch at {p}: {arr.shape} != {(B,H,W)}')
        mm[i] = arr
        if (i+1) % 5 == 0 or i == T_guess-1:
            print(f'Wrote {i+1}/{T_guess} days')

del mm
stack = np.memmap(STACK_MMAP_PATH, dtype='float32', mode='r+', shape=(T_guess, B, H, W))
T = T_guess
print('Memmap stack ready with shape:', tuple(stack.shape))

if 'dates' not in globals() or not dates:
    date_re = re.compile(r'(\d{4}-\d{2}-\d{2})')
    dates = []
    for p in stack_files:
        m = date_re.search(os.path.basename(p))
        dates.append(m.group(1) if m else os.path.basename(p))
    print('Derived dates count:', len(dates))


rng = np.random.RandomState(RANDOM_STATE)


pos_idx = np.where(y_any_train == 1)[0]
neg_idx = np.where(y_any_train == 0)[0]

n_pos = len(pos_idx)
n_neg = min(len(neg_idx), NEG_PER_POS * n_pos)

if n_pos == 0:
    raise RuntimeError('No positive samples in y_any_train; cannot train ANY_BURN model. Adjust split or sampling.')

neg_sel = rng.choice(neg_idx, size=n_neg, replace=False)
keep = np.concatenate([pos_idx, neg_sel])
rng.shuffle(keep)

if keep.size > MAX_TRAIN_SAMPLES:
    keep = keep[:MAX_TRAIN_SAMPLES]

X_tr_bal = X_train[keep]
y_tr_bal = y_any_train[keep]


MODEL_ANY = RandomForestClassifier(**RF_PARAMS)
MODEL_ANY.fit(X_tr_bal, y_tr_bal)


te_pos = np.where(y_any_test == 1)[0]
te_neg = np.where(y_any_test == 0)[0]
te_neg = rng.choice(te_neg, size=min(3*len(te_pos)+1000, len(te_neg)), replace=False) if len(te_pos) > 0 else te_neg[:5000]
sel_te = np.concatenate([te_pos, te_neg])
rng.shuffle(sel_te)
X_tune = X_test[sel_te]
y_tune = y_any_test[sel_te]

def proba_for_one(model, Xc):
    if hasattr(model, 'predict_proba'):
        pp = model.predict_proba(Xc)
        if hasattr(model, 'classes_') and 1 in list(model.classes_):
            idx1 = int(np.where(model.classes_ == 1)[0][0])
            return pp[:, idx1]
        else:
            return np.zeros(Xc.shape[0], dtype=np.float32)
    elif hasattr(model, 'decision_function'):
        from sklearn.preprocessing import MinMaxScaler
        z = model.decision_function(Xc).reshape(-1,1)
        return MinMaxScaler().fit_transform(z).ravel()
    else:
        return np.zeros(Xc.shape[0], dtype=np.float32)

prob_tune = proba_for_one(MODEL_ANY, X_tune).astype('float32')
from sklearn.metrics import f1_score
ths = np.linspace(0.1, 0.9, 17)
f1s = []
for th in ths:
    pred = (prob_tune >= th).astype('uint8')
    f1s.append(f1_score(y_tune, pred, zero_division=0))
THR_ANY = float(ths[int(np.argmax(f1s))])


import joblib, os
os.makedirs('models', exist_ok=True)
joblib.dump({'model': MODEL_ANY, 'thr': THR_ANY, 'train_transitions': int(train_trans)}, 'models/anyburn_rf.joblib')
print('Saved -> models/anyburn_rf.joblib (thr=', THR_ANY, ')')


# Inference and GIF (test period only) — ANY_BURN, batched
import os, numpy as np, joblib, matplotlib.pyplot as plt
from imageio.v2 import imread, mimsave  # use PIL to read PNGs as RGB
from PIL import Image

# Safe probability for class 1 even if model trained on a single class
def proba_for_one(model, Xc):
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(Xc)
        if p.ndim == 2:
            if p.shape[1] == 2:
                try:
                    idx1 = int(np.where(model.classes_ == 1)[0][0])
                except Exception:
                    idx1 = 1
                return p[:, idx1]
            if p.shape[1] == 1:
                only = int(getattr(model, 'classes_', np.array([1]))[0])
                return np.full(Xc.shape[0], 1.0 if only == 1 else 0.0, dtype='float32')
        return p.ravel().astype('float32', copy=False)
    if hasattr(model, 'decision_function'):
        df = model.decision_function(Xc).astype('float32', copy=False)
        mn, mx = float(df.min()), float(df.max())
        return (df - mn) / (mx - mn + 1e-9)
    pred = model.predict(Xc).astype('float32', copy=False)
    return pred

bundle = joblib.load('models/anyburn_rf.joblib')
MODEL_ANY = bundle['model']
THR_ANY = float(bundle['thr'])
train_trans = int(bundle.get('train_transitions', 0))

HORIZON = 1
first_test_t = (LAG_DAYS - 1) + train_trans
last_input_t = T - 1 - HORIZON
start_t = min(max(first_test_t, LAG_DAYS-1), last_input_t)
end_t = last_input_t + 1  
print(f'Generating test GIF for t in [{start_t}, {last_input_t}] (inclusive)')

os.makedirs(FIG_DIR, exist_ok=True)
gif_tmp = os.path.join(FIG_DIR, 'gif_tmp'); os.makedirs(gif_tmp, exist_ok=True)
frames = []
BATCH = 200_000

for t in range(start_t, end_t):
    dyn_lags = [stack[t-li, DYN_BASE_IDXS] for li in range(LAG_DAYS)]
    dyn_lags_arr = np.stack(dyn_lags, axis=0)
    dfs = [dyn_lags_arr[li]-dyn_lags_arr[li-1] for li in range(1, LAG_DAYS)]
    diff_feats_arr = np.stack(dfs, axis=0).reshape(-1, H, W) if dfs else []
    burn_now = (stack[t, B_BURN] > 0).astype('uint8')
    static_feats = stack[t, STATIC_IDXS]
    dyn_flat = dyn_lags_arr.reshape(LAG_DAYS*len(DYN_BASE_IDXS), H, W)
    parts = [dyn_flat]
    if len(diff_feats_arr): parts.append(diff_feats_arr)
    parts.append(burn_now[None, ...])
    parts.append(static_feats)
    feats_full = np.concatenate(parts, axis=0)

    X_t_flat = feats_full.reshape(feats_full.shape[0], -1).T.astype('float32', copy=False)
    prob = np.empty(X_t_flat.shape[0], dtype='float32')
    for s in range(0, X_t_flat.shape[0], BATCH):
        e = min(s+BATCH, X_t_flat.shape[0])
        prob[s:e] = proba_for_one(MODEL_ANY, X_t_flat[s:e])
  
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    risk = prob.reshape(H, W)

    base = stack[t, B_NDVI]
    
    try:
        vmin_base, vmax_base = np.nanpercentile(base,5), np.nanpercentile(base,95)
    except Exception:
        vmin_base, vmax_base = float(np.nanmin(base)), float(np.nanmax(base))
    if not np.isfinite(vmin_base) or not np.isfinite(vmax_base) or vmin_base == vmax_base:
        vmin_base, vmax_base = float(np.nanmin(base)), float(np.nanmax(base))
        if not np.isfinite(vmin_base) or not np.isfinite(vmax_base) or vmin_base == vmax_base:
            vmin_base, vmax_base = 0.0, 1.0

    
    if t == start_t:
        above_thr = int((prob >= float(THR_ANY) if THR_ANY is not None else prob >= 0.5).sum())
        print(f'[Diag t={t}] risk min={float(risk.min()):.3f} max={float(risk.max()):.3f} mean={float(risk.mean()):.3f}  >=thr count={above_thr}')

    plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,3,1)
    im1 = ax1.imshow(base, cmap='Greens', vmin=vmin_base, vmax=vmax_base)
    ax1.set_title(f'NDVI t={t}')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.ax.tick_params(labelsize=7)
    cbar1.set_label('NDVI', fontsize=8)

    ax2 = plt.subplot(1,3,2)
    im2 = ax2.imshow(risk, cmap='inferno', vmin=0, vmax=1)
    ax2.set_title('Risk (probability)')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.ax.tick_params(labelsize=7)
    cbar2.set_label('Risk (0–1)', fontsize=8)

    actual = (stack[t+HORIZON, B_BURN] > 0).astype(float)
    ax3 = plt.subplot(1,3,3)
    ax3.imshow(base, cmap='Greens', vmin=vmin_base, vmax=vmax_base)
    ax3.imshow(np.ma.masked_where(actual==0, actual), cmap='autumn', alpha=0.6)
    ax3.set_title(f'Actual t+{HORIZON}')
    ax3.axis('off')

    out_png = os.path.join(gif_tmp, f'frame_{t:04d}.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close()


    if not 'first_saved' in globals() or first_saved is False:
        try:
            prob_arr = np.asarray(prob).reshape(-1)
            uniq_rounded = np.unique(np.round(prob_arr, 3))
            print(f'prob stats t={t}: min={prob_arr.min():.4f}, max={prob_arr.max():.4f}, mean={prob_arr.mean():.4f}, std={prob_arr.std():.4f}, unique≈{len(uniq_rounded)} (rounded 3dp)')
            print('MODEL_ANY.classes_ =', getattr(MODEL_ANY, 'classes_', None))
           
            try:
                print('X_t_flat shape:', getattr(globals(), 'X_t_flat', np.array([])).shape)
                if 'X_t_flat' in globals():
                    col_std = np.nanstd(X_t_flat, axis=0)
                    print(f'feature std t={t}: min={np.min(col_std):.3e}, median={np.median(col_std):.3e}, max={np.max(col_std):.3e}')
                    if np.all(col_std < 1e-9):
                        print('WARNING: All features nearly constant across pixels at this t; risk will be flat.')
            except Exception as fe:
                print('Feature diag error:', fe)
           
            try:
                import matplotlib.pyplot as plt
                hist_png = os.path.join(FIG_DIR, f'risk_hist_t{t:04d}.png')
                plt.figure(figsize=(4,3)); plt.hist(prob_arr, bins=50, range=(0,1)); plt.title(f'Risk histogram t={t}'); plt.tight_layout(); plt.savefig(hist_png); plt.close()
                print('Saved risk histogram ->', hist_png)
            except Exception as he:
                print('Histogram save error:', he)
        except Exception as de:
            print('Diag error:', de)
 
    arr_rgb = np.array(Image.open(out_png).convert('RGB'))
    frames.append(arr_rgb)

GIF_PATH = os.path.join(FIG_DIR, 'test_risk_anyburn.gif')
if len(frames) == 0:
    print('No frames to write — check start/end indices (start_t > last_input_t).')
else:
    
    mimsave(GIF_PATH, frames, fps=3)
    print('Saved GIF ->', GIF_PATH, '| frames:', len(frames))


import os, glob
import numpy as np
from imageio.v2 import imread, mimsave

png_dir = os.path.join(FIG_DIR, 'gif_tmp')
png_paths = sorted(glob.glob(os.path.join(png_dir, 'frame_*.png')))
print(f'Re-encoding GIF from {len(png_paths)} PNG frames in {png_dir!r}')

frames_rgb = []
for p in png_paths:
    arr = imread(p)
  
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
       
        a = np.asarray(arr, dtype=np.float32)
        a = np.clip(a, 0, 1)
        arr = (a * 255).astype(np.uint8)
    frames_rgb.append(arr)

out_gif = os.path.join(FIG_DIR, 'test_risk_anyburn_rgb.gif')
if frames_rgb:
    mimsave(out_gif, frames_rgb, fps=3)
    print('Saved RGB GIF ->', out_gif, '| frames:', len(frames_rgb))
else:
    print('No frames found to encode. Ensure the previous GIF cell created PNGs into figures/gif_tmp.')
