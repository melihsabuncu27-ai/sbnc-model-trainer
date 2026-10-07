# ============================================================
# SBNC OTOMATİK EĞİTİM — GitHub Actions (102 Feature)
# ============================================================
import os, sys, json, time
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, optimizers
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_auc_score
from datetime import datetime

# ============================================================
# AYARLAR
# ============================================================
with open("config.json") as f:
    CFG = json.load(f)

SYMBOLS = CFG["symbols"]
VERSION = f"v4.{int(time.time())}"
GLOBAL_MEDIAN = 0.007338

CURRENT_AUC = 0.76
CURRENT_WALK = 0.68

print(f"🚀 Eğitim başlıyor: {VERSION}")
print(f"📅 {datetime.now()}")


# ============================================================
# BINANCE PUBLIC DATA API — Multi-Timeframe
# ============================================================
def fetch_klines(symbol, interval="1h", limit=35000):
    base_url = "https://data-api.binance.vision/api/v3/klines"
    all_k = []
    end_time = None
    
    while len(all_k) < limit:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time:
            params["endTime"] = end_time
        
        try:
            resp = requests.get(base_url, params=params, timeout=15)
            if resp.status_code != 200:
                break
            klines = resp.json()
            if not klines:
                break
            
            all_k = klines + all_k
            end_time = klines[0][0] - 1
            time.sleep(0.03)
        except Exception:
            break
    
    if not all_k:
        return None
    
    df = pd.DataFrame(all_k, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("time")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open","high","low","close","volume"]]


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    dn = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + up / (dn + 1e-9))


def macd(close, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sig = ml.ewm(span=signal, adjust=False).mean()
    return ml, sig, ml - sig


def atr(high, low, close, period=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ============================================================
# FEATURE FONKSİYONLARI
# ============================================================
def add_vol_features(df):
    df = df.copy()
    lr = np.log(df["close"]).diff()
    for w in [6, 12, 24, 48, 96]:
        df[f"rv_{w}"] = lr.rolling(w).std()
        df[f"absr_{w}"] = lr.abs().rolling(w).mean()
    df["rv_sq_24"] = np.sqrt((lr**2).rolling(24).sum())
    hl = np.log(df["high"] / df["low"])
    df["park_24"] = np.sqrt((hl**2).rolling(24).mean() / (4 * np.log(2)))
    co = np.log(df["close"] / df["open"])
    df["gk_24"] = np.sqrt((0.5*hl**2 - (2*np.log(2)-1)*co**2).rolling(24).mean())
    df["vol_of_vol"] = lr.rolling(24).std().rolling(24).std()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["hl_ma24"] = df["hl_range"].rolling(24).mean()
    df["vol_ma"] = df["volume"].pct_change().rolling(24).std()
    d = df["close"].diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100/(1 + up/(dn + 1e-9))
    return df


def add_advanced_features(df):
    df = df.copy()
    lr = np.log(df["close"]).diff()
    ar = lr.abs()
    df["rv_d"] = lr.rolling(24).std()
    df["rv_w"] = lr.rolling(24*7).std()
    df["rv_m"] = lr.rolling(24*30).std()
    df["rv_d_w"] = df["rv_d"] / (df["rv_w"] + 1e-9)
    df["rv_w_m"] = df["rv_w"] / (df["rv_m"] + 1e-9)
    df["bipower"] = (ar * ar.shift(1)).rolling(24).mean() * (np.pi/2)
    df["jump"] = np.maximum(df["rv_d"]**2 - df["bipower"], 0)
    df["jump_r"] = df["jump"] / (df["rv_d"]**2 + 1e-9)
    df["skew_24"] = lr.rolling(24).skew()
    df["kurt_24"] = lr.rolling(24).kurt()
    df["mom_6"] = np.log(df["close"] / df["close"].shift(6))
    df["mom_24"] = np.log(df["close"] / df["close"].shift(24))
    df["mom_96"] = np.log(df["close"] / df["close"].shift(96))
    df["trend_12"] = df["close"].rolling(12).mean() / df["close"] - 1
    df["trend_24"] = df["close"].rolling(24).mean() / df["close"] - 1
    df["vol_mom"] = df["volume"].pct_change(24).replace([np.inf,-np.inf], np.nan)
    return df


def add_technical_features(df):
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["stoch_j"] = 3 * df["stoch_k"] - 2 * df["stoch_d"]
    pdm = h.diff(); mdm = -l.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()],
                   axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    pdi = 100 * (pdm.rolling(14).mean() / atr14)
    mdi = 100 * (mdm.rolling(14).mean() / atr14)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    df["adx"] = dx.rolling(14).mean()
    df["plus_di"] = pdi; df["minus_di"] = mdi
    tk = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kj = (h.rolling(26).max() + l.rolling(26).min()) / 2
    df["ichi_tenkan"] = tk / c - 1
    df["ichi_kijun"] = kj / c - 1
    df["ichi_senkou_a"] = (((tk + kj) / 2).shift(26)) / c - 1
    df["ichi_senkou_b"] = (((h.rolling(52).max() + l.rolling(52).min()) / 2)
                            .shift(26)) / c - 1
    df["ichi_cloud"] = (df["ichi_senkou_a"] + df["ichi_senkou_b"]) / 2
    hl2 = (h + l) / 2
    up = hl2 + 3 * atr14; lo = hl2 - 3 * atr14
    st = pd.Series(0.0, index=df.index); dr = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if c.iloc[i] > up.iloc[i-1]: dr.iloc[i] = 1
        elif c.iloc[i] < lo.iloc[i-1]: dr.iloc[i] = -1
        else: dr.iloc[i] = dr.iloc[i-1]
        st.iloc[i] = lo.iloc[i] if dr.iloc[i] == 1 else up.iloc[i]
    df["supertrend"] = st / c - 1
    df["supertrend_dir"] = dr
    typ = (h + l + c) / 3
    rm = typ * v
    pf = rm.where(typ > typ.shift(), 0).rolling(14).sum()
    nf = rm.where(typ < typ.shift(), 0).rolling(14).sum()
    df["mfi"] = 100 - 100 / (1 + pf / (nf + 1e-9))
    df["williams_r"] = -100 * (hi14 - c) / (hi14 - lo14 + 1e-9)
    mfm = ((c - l) - (h - c)) / (h - l + 1e-9)
    mfv = mfm * v
    df["cmf"] = mfv.rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
    ema20 = c.ewm(span=20).mean(); atr20 = tr.rolling(20).mean()
    df["keltner_pos"] = (c - (ema20 - 2*atr20)) / (4*atr20 + 1e-9)
    dcU = h.rolling(20).max(); dcL = l.rolling(20).min()
    df["donchian_pos"] = (c - dcL) / (dcU - dcL + 1e-9)
    return df


def add_advanced_technical(df):
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    hi20 = h.rolling(20).max(); lo20 = l.rolling(20).min()
    diff = hi20 - lo20 + 1e-9
    df["fib_pos"] = (c - lo20) / diff
    df["fib_near_236"] = np.abs(df["fib_pos"] - 0.236)
    df["fib_near_382"] = np.abs(df["fib_pos"] - 0.382)
    df["fib_near_618"] = np.abs(df["fib_pos"] - 0.618)
    dh = h.resample("D").max().shift(1).reindex(df.index, method="ffill")
    dl = l.resample("D").min().shift(1).reindex(df.index, method="ffill")
    dcl = c.resample("D").last().shift(1).reindex(df.index, method="ffill")
    pp = (dh + dl + dcl) / 3
    r1 = 2 * pp - dl; s1 = 2 * pp - dh
    df["pivot_pos"] = (c - s1) / (r1 - s1 + 1e-9)
    df["pivot_dist"] = (c - pp) / (pp + 1e-9)
    typ = (h + l + c) / 3
    vwap = (typ * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-9)
    vwap_std = (typ - vwap).rolling(24).std()
    df["vwap_dist"] = (c - vwap) / (vwap + 1e-9)
    df["vwap_band_1"] = (c - vwap) / (2 * vwap_std + 1e-9)
    df["vwap_band_2"] = (c - vwap) / (4 * vwap_std + 1e-9)
    return df


def add_time_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(float)
    df["is_month_end"] = (df.index.day >= 28).astype(float)
    return df


def compute_mtf(sym, d1h):
    try:
        d4h = fetch_klines(sym, "4h", 10000)
        d1d = fetch_klines(sym, "1d", 2000)
        if d4h is None or d1d is None: return None
        f4 = pd.DataFrame(index=d4h.index)
        f4["h4_rsi"] = rsi(d4h["close"])
        f4["h4_macd"] = macd(d4h["close"])[0]
        f4["h4_macd_hist"] = macd(d4h["close"])[2]
        f4["h4_atr_norm"] = atr(d4h["high"], d4h["low"], d4h["close"]) / d4h["close"]
        f4["h4_ema_trend"] = (d4h["close"].ewm(span=12).mean()
                              / d4h["close"].ewm(span=26).mean() - 1)
        f1 = pd.DataFrame(index=d1d.index)
        f1["d1_rsi"] = rsi(d1d["close"])
        f1["d1_macd"] = macd(d1d["close"])[0]
        f1["d1_atr_norm"] = atr(d1d["high"], d1d["low"], d1d["close"]) / d1d["close"]
        f1["d1_trend"] = (d1d["close"] > d1d["close"].rolling(20).mean()).astype(float)
        return pd.concat([f4.reindex(d1h.index, method="ffill"),
                          f1.reindex(d1h.index, method="ffill")], axis=1)
    except Exception:
        return None


def add_ext_mtf(sym, d1h):
    try:
        dfs = {}
        for tf in ["15m","30m","2h","3h"]:
            d = fetch_klines(sym, tf, 5000)
            if d is not None: dfs[tf] = d
        if not dfs: return None
        f = pd.DataFrame(index=d1h.index)
        if "15m" in dfs:
            d = dfs["15m"]
            f["m15_rsi"] = rsi(d["close"]).reindex(d1h.index, method="ffill")
            f["m15_macd"] = macd(d["close"])[0].reindex(d1h.index, method="ffill")
            f["m15_trend"] = (d["close"] > d["close"].rolling(20).mean()).astype(float).reindex(d1h.index, method="ffill")
        if "30m" in dfs:
            d = dfs["30m"]
            f["m30_rsi"] = rsi(d["close"]).reindex(d1h.index, method="ffill")
            f["m30_macd"] = macd(d["close"])[0].reindex(d1h.index, method="ffill")
            f["m30_trend"] = (d["close"] > d["close"].rolling(20).mean()).astype(float).reindex(d1h.index, method="ffill")
        if "2h" in dfs:
            d = dfs["2h"]
            f["h2_rsi"] = rsi(d["close"]).reindex(d1h.index, method="ffill")
            f["h2_trend"] = (d["close"] > d["close"].ewm(span=12).mean()).astype(float).reindex(d1h.index, method="ffill")
        if "3h" in dfs:
            d = dfs["3h"]
            f["h3_rsi"] = rsi(d["close"]).reindex(d1h.index, method="ffill")
            f["h3_trend"] = (d["close"] > d["close"].ewm(span=12).mean()).astype(float).reindex(d1h.index, method="ffill")
        tc = [c for c in f.columns if "trend" in c]
        if tc: f["mtf_trend_sum"] = f[tc].sum(axis=1)
        return f
    except Exception:
        return None


def add_ext_mtf_v2(sym, d1h):
    try:
        d1d = fetch_klines(sym, "1d", 2000)
        if d1d is None: return None
        d1w = d1d.resample("W").agg({"open":"first","high":"max","low":"min",
                                      "close":"last","volume":"sum"}).dropna()
        d1M = d1d.resample("ME").agg({"open":"first","high":"max","low":"min",
                                       "close":"last","volume":"sum"}).dropna()
        if len(d1w) < 5 or len(d1M) < 3: return None
        f = pd.DataFrame(index=d1h.index)
        f["w1_rsi"] = rsi(d1w["close"]).reindex(d1h.index, method="ffill")
        f["w1_trend"] = (d1w["close"] > d1w["close"].rolling(4).mean()).astype(float).reindex(d1h.index, method="ffill")
        f["w1_mom"] = (d1w["close"] / d1w["close"].shift(4) - 1).reindex(d1h.index, method="ffill")
        f["m1_rsi"] = rsi(d1M["close"]).reindex(d1h.index, method="ffill")
        f["m1_trend"] = (d1M["close"] > d1M["close"].rolling(3).mean()).astype(float).reindex(d1h.index, method="ffill")
        f["m1_mom"] = (d1M["close"] / d1M["close"].shift(3) - 1).reindex(d1h.index, method="ffill")
        return f
    except Exception:
        return None


def add_div(sym, d1h):
    try:
        d4h = fetch_klines(sym, "4h", 10000)
        d1d = fetch_klines(sym, "1d", 2000)
        if d4h is None: return None
        f = pd.DataFrame(index=d1h.index)
        r1 = rsi(d1h["close"])
        r4 = rsi(d4h["close"]).reindex(d1h.index, method="ffill")
        rd = rsi(d1d["close"]).reindex(d1h.index, method="ffill") if d1d is not None else r1
        f["rsi_div_1h_4h"] = r1 - r4
        f["rsi_div_1h_1d"] = r1 - rd
        f["rsi_div_4h_1d"] = r4 - rd
        m1 = macd(d1h["close"])[0]
        m4 = macd(d4h["close"])[0].reindex(d1h.index, method="ffill")
        md = macd(d1d["close"])[0].reindex(d1h.index, method="ffill") if d1d is not None else m1
        f["macd_div_1h_4h"] = (m1 - m4) / d1h["close"]
        f["macd_div_1h_1d"] = (m1 - md) / d1h["close"]
        f["macd_div_4h_1d"] = (m4 - md) / d1h["close"]
        return f
    except Exception:
        return None


def add_cross_tf_adv(sym, d1h):
    try:
        d4h = fetch_klines(sym, "4h", 10000)
        d1d = fetch_klines(sym, "1d", 2000)
        if d4h is None or d1d is None: return None
        f = pd.DataFrame(index=d1h.index)
        mom_1h = np.log(d1h["close"] / d1h["close"].shift(24))
        mom_4h = np.log(d4h["close"] / d4h["close"].shift(6)).reindex(d1h.index, method="ffill")
        f["mtf_mom_div_1h_4h"] = mom_1h - mom_4h
        vol_1h = d1h["volume"].pct_change(24)
        vol_4h = d4h["volume"].pct_change(6).reindex(d1h.index, method="ffill")
        f["mtf_vol_div"] = vol_1h - vol_4h
        rsi_1h = rsi(d1h["close"])
        rsi_4h = rsi(d4h["close"]).reindex(d1h.index, method="ffill")
        rsi_1d = rsi(d1d["close"]).reindex(d1h.index, method="ffill")
        f["mtf_rsi_align"] = ((rsi_1h > 50).astype(int)
                              + (rsi_4h > 50).astype(int)
                              + (rsi_1d > 50).astype(int))
        tr_1h = pd.concat([d1h["high"]-d1h["low"],
                            (d1h["high"]-d1h["close"].shift()).abs(),
                            (d1h["low"]-d1h["close"].shift()).abs()],
                           axis=1).max(axis=1)
        atr_1h = tr_1h.rolling(14).mean()
        f["mtf_trend_strength"] = d1h["close"].diff(24).abs() / (atr_1h + 1e-9)
        bo = (d1h["close"] > d1h["high"].rolling(20).max().shift(1)).astype(float)
        t4 = (d4h["close"] > d4h["close"].rolling(20).mean()).astype(float)
        f["mtf_breakout_conf"] = bo * t4.reindex(d1h.index, method="ffill")
        r1h = d1h["high"].rolling(20).max(); s1h = d1h["low"].rolling(20).min()
        f["mtf_sr_pos_1h"] = (d1h["close"] - s1h) / (r1h - s1h + 1e-9)
        r4h = d4h["high"].rolling(20).max().reindex(d1h.index, method="ffill")
        s4h = d4h["low"].rolling(20).min().reindex(d1h.index, method="ffill")
        f["mtf_sr_pos_4h"] = (d1h["close"] - s4h) / (r4h - s4h + 1e-9)
        return f
    except Exception:
        return None


def normalize_per_symbol(df, window=720, min_periods=168):
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            m = df[c].rolling(window, min_periods=min_periods).mean()
            s = df[c].rolling(window, min_periods=min_periods).std()
            df[c] = (df[c] - m) / (s + 1e-9)
    return df


# ============================================================
# CROSS-ASSET (BTC/ETH korelasyon)
# ============================================================
print("\n📊 Cross-asset için BTC/ETH verisi...")
btc_df = fetch_klines("BTCUSDT", "1h", 35000)
eth_df = fetch_klines("ETHUSDT", "1h", 35000)

btc_ret = np.log(btc_df["close"]).diff() if btc_df is not None else None
eth_ret = np.log(eth_df["close"]).diff() if eth_df is not None else None


# ============================================================
# VERİ HAZIRLA (Her coin için)
# ============================================================
print("\n📊 Feature'lar hazırlanıyor...\n")
X_list, yd_list, yv_list, r_list = [], [], [], []
FEATS = None

for sym in SYMBOLS:
    try:
        print(f"  🔄 {sym}...", end=" ")
        df = fetch_klines(sym, "1h", 35000)
        if df is None or len(df) < 5000:
            print("yetersiz veri")
            continue
        
        # Feature pipeline
        df = add_time_features(df)
        df = add_vol_features(df)
        df = add_advanced_features(df)
        df = add_technical_features(df)
        df = add_advanced_technical(df)
        
        # MTF sözlükleri
        m = compute_mtf(sym, df)
        if m is not None: df = df.join(m, how="left")
        m2 = add_ext_mtf(sym, df)
        if m2 is not None: df = df.join(m2, how="left")
        m3 = add_ext_mtf_v2(sym, df)
        if m3 is not None: df = df.join(m3, how="left")
        m4 = add_div(sym, df)
        if m4 is not None: df = df.join(m4, how="left")
        m5 = add_cross_tf_adv(sym, df)
        if m5 is not None: df = df.join(m5, how="left")
        
        # Cross-asset
        if btc_ret is not None:
            f_cross = pd.DataFrame(index=df.index)
            coin_ret = np.log(df["close"]).diff()
            f_cross["btc_corr_24"] = coin_ret.rolling(24).corr(btc_ret)
            f_cross["btc_corr_168"] = coin_ret.rolling(168).corr(btc_ret)
            if eth_ret is not None:
                f_cross["eth_corr_24"] = coin_ret.rolling(24).corr(eth_ret)
            f_cross["rel_ret_24"] = coin_ret.rolling(24).sum() - btc_ret.rolling(24).sum()
            df = df.join(f_cross, how="left")
        
        # NaN handling
        df = df.dropna(subset=["close", "volume"])
        for c in df.columns:
            if df[c].isna().any():
                df[c] = df[c].ffill().bfill().fillna(0)
        
        # Hedef
        lr = np.log(df["close"]).diff()
        fwd = np.log(df["close"].shift(-24) / df["close"])
        fv = lr.rolling(24).std().shift(-24)
        df["direction"] = (fwd > 0).astype(int)
        df["high_vol"] = (fv > GLOBAL_MEDIAN).astype(int)
        df["ret_fwd"] = fwd
        df = df.dropna().iloc[::24].copy()
        
        FEATURES = [c for c in df.columns
                    if c not in ["direction","high_vol","ret_fwd"]
                    and np.issubdtype(df[c].dtype, np.number)]
        
        df_norm = normalize_per_symbol(df[FEATURES].copy())
        
        X_list.append(df_norm.values.astype("float32"))
        yd_list.append(df["direction"].values.astype("int32"))
        yv_list.append(df["high_vol"].values.astype("int32"))
        r_list.append(df["ret_fwd"].values.astype("float32"))
        FEATS = FEATURES
        print(f"✅ {len(df)} örnek, {len(FEATURES)} feature")
    except Exception as e:
        print(f"❌ {e}")

if not X_list:
    print("\n❌ HİÇ VERİ ÇEKİLEMEDİ")
    sys.exit(1)

X_all = np.nan_to_num(np.concatenate(X_list,0), nan=0., posinf=0., neginf=0.)
yd_all = np.concatenate(yd_list, 0)
yv_all = np.concatenate(yv_list, 0)
r_all = np.concatenate(r_list, 0)
print(f"\n✅ Veri: {X_all.shape}, {len(FEATS)} feature")


# ============================================================
# SEKANS + SPLIT
# ============================================================
SEQ_LEN = 48
split = int(len(X_all) * 0.7)

def make_seq(X, yd, yv, r, seq_len=48):
    n = len(X)
    Xs = np.zeros((n - seq_len, seq_len, X.shape[1]), dtype="float32")
    for i in range(seq_len, n):
        Xs[i - seq_len] = X[i - seq_len:i]
    return Xs, yd[seq_len:], yv[seq_len:], r[seq_len:]

X_seq, yd_seq, yv_seq, r_seq = make_seq(X_all, yd_all, yv_all, r_all, SEQ_LEN)
seq_split = split - SEQ_LEN

X_tr = X_seq[:seq_split]; X_te = X_seq[seq_split:]
yd_tr, yd_te = yd_seq[:seq_split], yd_seq[seq_split:]
yv_tr, yv_te = yv_seq[:seq_split], yv_seq[seq_split:]


# ============================================================
# LSTM
# ============================================================
print("\n🧠 LSTM eğitiliyor...")

def build_lstm(seq_len, n_feat):
    inp = layers.Input(shape=(seq_len, n_feat))
    x = layers.LSTM(96, return_sequences=True)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(48)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    o_d = layers.Dense(1, activation="sigmoid", name="direction")(x)
    o_v = layers.Dense(1, activation="sigmoid", name="vol")(x)
    m = Model(inp, [o_d, o_v])
    m.compile(optimizer=optimizers.Adam(5e-4),
              loss={"direction":"binary_crossentropy","vol":"binary_crossentropy"},
              loss_weights={"direction":1.0,"vol":0.3},
              metrics={"direction":["accuracy",AUC(name="auc")],
                       "vol":["accuracy",AUC(name="auc")]})
    return m

tf.keras.utils.set_random_seed(42)
model_lstm = build_lstm(SEQ_LEN, X_all.shape[1])

model_lstm.fit(
    X_tr, {"direction":yd_tr, "vol":yv_tr},
    validation_data=(X_te, {"direction":yd_te, "vol":yv_te}),
    epochs=30, batch_size=128, verbose=2,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_direction_auc", mode="max",
                                patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_direction_auc", mode="max",
                                    factor=0.5, patience=3, min_lr=1e-5),
    ])

p_lstm, _ = model_lstm.predict(X_te, verbose=0)
p_lstm = p_lstm.flatten()
auc_lstm = roc_auc_score(yd_te, p_lstm)
print(f"\n✅ LSTM AUC: {auc_lstm:.4f}")

n_q = len(yd_te) // 4
aucs = []
for i in range(4):
    s = i*n_q; e = (i+1)*n_q if i<3 else len(yd_te)
    a = roc_auc_score(yd_te[s:e], p_lstm[s:e])
    aucs.append(a)
min_walk = min(aucs)
print(f"✅ Walk-Forward min: {min_walk:.4f}")


# ============================================================
# KALİTE KONTROLÜ
# ============================================================
passed = (auc_lstm > CURRENT_AUC - 0.02 and
          min_walk > CURRENT_WALK - 0.02 and
          auc_lstm > 0.70 and
          min_walk > 0.62)

print(f"\n{'='*60}")
print(f"KALİTE KONTROLÜ")
print(f"{'='*60}")
print(f"  Yeni AUC    : {auc_lstm:.4f}  (min: {CURRENT_AUC - 0.02:.4f})")
print(f"  Yeni Walk   : {min_walk:.4f}  (min: {CURRENT_WALK - 0.02:.4f})")
print(f"  Geçti mi?   : {'✅' if passed else '❌'}")
print(f"{'='*60}")

if not passed:
    print("\n⚠️ Model kaliteyi geçemedi, deploy edilmeyecek")
    sys.exit(1)


# ============================================================
# TFLITE
# ============================================================
print("\n📦 TFLite'a çevriliyor...")

converter = tf.lite.TFLiteConverter.from_keras_model(model_lstm)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False
tflite_lstm = converter.convert()


# MLP
print("\n🧠 MLP eğitiliyor...")

def build_mlp(n_feat):
    inp = layers.Input(shape=(n_feat,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    o_d = layers.Dense(1, activation="sigmoid", name="direction")(x)
    o_v = layers.Dense(1, activation="sigmoid", name="vol")(x)
    m = Model(inp, [o_d, o_v])
    m.compile(optimizer=optimizers.Adam(5e-4),
              loss={"direction":"binary_crossentropy","vol":"binary_crossentropy"},
              loss_weights={"direction":1.0,"vol":0.3},
              metrics={"direction":["accuracy",AUC(name="auc")],
                       "vol":["accuracy",AUC(name="auc")]})
    return m

tf.keras.utils.set_random_seed(42)
model_mlp = build_mlp(X_all.shape[1])
model_mlp.fit(
    X_all[:split], {"direction":yd_all[:split], "vol":yv_all[:split]},
    validation_data=(X_all[split:], {"direction":yd_all[split:], "vol":yv_all[split:]}),
    epochs=30, batch_size=256, verbose=2,
    callbacks=[callbacks.EarlyStopping(monitor="val_direction_auc", mode="max",
                                        patience=6, restore_best_weights=True)])

converter = tf.lite.TFLiteConverter.from_keras_model(model_mlp)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_mlp = converter.convert()


# ============================================================
# KAYDET
# ============================================================
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

with open(f"{OUT_DIR}/model_lstm.tflite", "wb") as f:
    f.write(tflite_lstm)
with open(f"{OUT_DIR}/model_mlp.tflite", "wb") as f:
    f.write(tflite_mlp)

params = {
    "version": VERSION,
    "trained_at": datetime.now().isoformat(),
    "features": FEATS,
    "feature_count": len(FEATS),
    "seq_len": 48,
    "w_lstm": 0.5,
    "w_mlp": 0.5,
    "threshold": 0.78,
    "metrics": {
        "auc_lstm": float(auc_lstm),
        "walk_forward_min": float(min_walk),
    },
}
with open(f"{OUT_DIR}/params.json", "w") as f:
    json.dump(params, f, indent=2)

print(f"\n✅ Model kaydedildi: {OUT_DIR}/")
print(f"   - model_lstm.tflite ({len(tflite_lstm)/1024:.1f} KB)")
print(f"   - model_mlp.tflite ({len(tflite_mlp)/1024:.1f} KB)")
print(f"   - params.json ({len(FEATS)} feature)")
print(f"\n🎉 EĞİTİM BAŞARILI: {VERSION}")
