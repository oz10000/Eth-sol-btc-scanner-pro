
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h"]
LOOKBACK_DAYS = 60
SCAN_LOOKBACK = 30
MC_ITER = 200
DATA_CACHE = {}
ATR_CACHE = {}
PIDELTA_CACHE = {}

TENSION_QUANTILE_SCAN = 0.85
SCAN_K_VALUES = [3,5,8,13]
MIN_FUTURE_VELAS = 50

PARAM_GRID = {
    'tp_atr':[1,2,3,5,8],
    'sl_atr':[1,2,3,5],
    'atr_window':[7,14,21],
    'tension_quantile':[0.5,0.6,0.7,0.8,0.9],
    'pidelta_window':[5,8,13,21]
}

# ============================================================
# FUNCIONES PRINCIPALES
def fetch_klines(symbol, interval, days=LOOKBACK_DAYS):
    key = f"{symbol}_{interval}"
    if key in DATA_CACHE:
        return DATA_CACHE[key]
    end = int(time.time() * 1000)
    start = end - days*24*60*60*1000
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol":symbol,"interval":interval,"startTime":start,"endTime":end,"limit":1500}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","num_trades","taker_base_vol","taker_quote_vol","ignore"
        ])
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        df = df[["open","high","low","close","volume"]]
        DATA_CACHE[key] = df
        return df
    except:
        return None

def tension_235(series):
    ema2 = series.ewm(span=2).mean()
    ema3 = series.ewm(span=3).mean()
    ema5 = series.ewm(span=5).mean()
    return (ema2-ema3).abs() + (ema3-ema5).abs()

def scan_symbol_tf(symbol, tf):
    df = fetch_klines(symbol, tf, days=SCAN_LOOKBACK)
    if df is None or len(df)<100: return []
    price = df['close']
    tension = tension_235(price)
    threshold = tension.quantile(TENSION_QUANTILE_SCAN)
    high_points = tension[tension>=threshold].index
    signals = []
    for dt in high_points:
        idx = price.index.get_loc(dt)
        if idx + MIN_FUTURE_VELAS >= len(price):
            continue
        for k in [3,5,8,13]:
            if idx+k>=len(price):
                continue
            fut = price.iloc[idx+k]-price.iloc[idx]
            if fut>0:
                signals.append({'Symbol':symbol,'TF':tf,'open_time':dt,'Direction':'LONG','score':fut})
            elif fut<0:
                signals.append({'Symbol':symbol,'TF':tf,'open_time':dt,'Direction':'SHORT','score':-fut})
    return signals

def run_scanner():
    all_signals=[]
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            sigs=scan_symbol_tf(sym, tf)
            all_signals.extend(sigs)
    df=pd.DataFrame(all_signals)
    if not df.empty:
        df=df.sort_values('score',ascending=False)
        df.to_csv("escaneo_filtrado.txt", index=False, sep='\t')  # TXT separado por tab
    return df

# ============================================================
# MAIN
if __name__=="__main__":
    print("🚀 Ejecutando ScannerPro (TXT)...")
    df_signals = run_scanner()
    if df_signals.empty:
        print("⚠️ No se generaron señales")
    else:
        print(f"✅ Escáner completado. Guardado en escaneo_filtrado.txt")
