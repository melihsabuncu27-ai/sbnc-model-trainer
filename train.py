# ============================================================
# SBNC OTOMATİK EĞİTİM — GitHub Actions
# ============================================================
import os, sys, json, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, optimizers
from tensorflow.keras.metrics import AUC
from sklearn.metrics import roc_auc_score
from datetime import datetime
import requests

# AYARLAR
with open("config.json") as f:
    CFG = json.load(f)

SYMBOLS = CFG["symbols"]
VERSION = f"v4.{int(time.time())}"
GLOBAL_MEDIAN = 0.007338

print(f"🚀 Eğitim başlıyor: {VERSION}")
print(f"📅 {datetime.now()}")

client = Client()

def fetch_klines(symbol, interval="1h", limit=35000):
    """Binance PUBLIC DATA API — coğrafi kısıtlama yok"""
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
                print(f"  ⚠️ {symbol}: HTTP {resp.status_code}")
                break
            klines = resp.json()
            if not klines:
                break
            
            all_k = klines + all_k
            end_time = klines[0][0] - 1
            time.sleep(0.05)
        except Exception as e:
            print(f"  ⚠️ {symbol}: {e}")
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

def add_features(df):
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
    
    df["mom_6"] = np.log(df["close"] / df["close"].shift(6))
    df["mom_24"] = np.log(df["close"] / df["close"].shift(24))
    df["mom_96"] = np.log(df["close"] / df["close"].shift(96))
    
    d = df["close"].diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100/(1 + up/(dn + 1e-9))
    
    df["skew_24"] = lr.rolling(24).skew()
    df["kurt_24"] = lr.rolling(24).kurt()
    
    df["rv_d"] = lr.rolling(24).std()
    df["rv_w"] = lr.rolling(24*7).std()
    df["rv_m"] = lr.rolling(24*30).std()
    df["rv_d_w"] = df["rv_d"] / (df["rv_w"] + 1e-9)
    df["rv_w_m"] = df["rv_w"] / (df["rv_m"] + 1e-9)
    
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(float)
    
    return df

def normalize_per_symbol(df, window=720, min_periods=168):
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            m = df[c].rolling(window, min_periods=min_periods).mean()
            s = df[c].rolling(window, min_periods=min_periods).std()
            df[c] = (df[c] - m) / (s + 1e-9)
    return df

print("\n📊 Veri hazırlanıyor...")
X_list, yd_list, yv_list, r_list = [], [], [], []
FEATS = None

for sym in SYMBOLS:
    try:
        df = fetch_klines(sym, "1h", 35000)
        if len(df) < 5000: continue
        
        df = add_features(df).dropna()
        for c in df.columns:
            if df[c].isna().any(): df[c] = df[c].ffill().bfill().fillna(0)
        
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
    except Exception as e:
        print(f"  ❌ {sym}: {e}")

X_all = np.nan_to_num(np.concatenate(X_list,0), nan=0., posinf=0., neginf=0.)
yd_all = np.concatenate(yd_list, 0)
yv_all = np.concatenate(yv_list, 0)
r_all = np.concatenate(r_list, 0)
print(f"✅ Veri: {X_all.shape}, {len(FEATS)} feature")

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

print("\n🧠 Model eğitiliyor...")

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

passed = auc_lstm > 0.73 and min_walk > 0.65

print(f"\n{'='*60}")
print(f"EĞİTİM SONUCU")
print(f"{'='*60}")
print(f"Version        : {VERSION}")
print(f"LSTM AUC       : {auc_lstm:.4f}")
print(f"Walk-Fwd min   : {min_walk:.4f}")
print(f"Durum          : {'✅ GEÇTİ' if passed else '❌ BAŞARISIZ'}")
print(f"{'='*60}")

if not passed:
    print("\n⚠️ Model kaliteyi geçemedi, deploy edilmeyecek")
    sys.exit(1)

converter = tf.lite.TFLiteConverter.from_keras_model(model_lstm)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False
tflite_lstm = converter.convert()

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
print(f"   - params.json")
print(f"\n🎉 EĞİTİM BAŞARILI: {VERSION}")
