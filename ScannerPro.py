import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
EXCHANGE = "multi"
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "8h", "12h", "1d"]
LOOKBACK_CANDLES = 300
TP_PCT = 0.02
SL_PCT = 0.01

ASSETS_FILE = "assets.csv"
BASE_SYMBOLS = ["BTC", "ETH", "SOL"]

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
DATA_CACHE = {}

# Mapeo de categorías por timeframe
CATEGORY_MAP = {
    '1m': 'Scalp', '3m': 'Scalp', '5m': 'Scalp', '15m': 'Scalp',
    '30m': 'Intraday', '1h': 'Intraday', '4h': 'Intraday',
    '8h': 'Swing', '12h': 'Swing', '1d': 'Swing'
}

# ============================================================
# FUNCIONES DE DESCARGA (KuCoin, Crypto.com, CoinGecko)
# ============================================================
def fetch_klines_kucoin(symbol, timeframe, limit=500):
    tf_map = {
        '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1hour', '2h': '2hour', '4h': '4hour', '6h': '6hour', '8h': '8hour',
        '12h': '12hour', '1d': '1day', '1w': '1week'
    }
    interval = tf_map.get(timeframe)
    if not interval:
        return None
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {'symbol': symbol, 'type': interval, 'limit': limit}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if data['code'] != '200000':
            return None
        candles = data['data']
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
        df.set_index('timestamp', inplace=True)
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df = df[['open','high','low','close','volume']].sort_index()
        print(f"   📥 KuCoin {symbol} {timeframe}: {len(df)} velas")
        return df
    except Exception as e:
        print(f"   ⚠️ KuCoin error {symbol}: {e}")
        return None

def fetch_klines_cryptocom(symbol, timeframe, limit=500):
    tf_map = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h',
        '12h': '12h', '1d': '1d', '1w': '1w'
    }
    interval = tf_map.get(timeframe)
    if not interval:
        return None
    url = "https://api.crypto.com/exchange/v1/public/get-candlestick"
    params = {'instrument_name': symbol, 'timeframe': interval}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if data['code'] != 0:
            return None
        candles = data['result']['data']
        df = pd.DataFrame(candles)
        for col in ['t', 'o', 'h', 'l', 'c', 'v']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
        df = df[['open','high','low','close','volume']].sort_index()
        print(f"   📥 Crypto.com {symbol} {timeframe}: {len(df)} velas")
        return df
    except Exception as e:
        print(f"   ⚠️ Crypto.com error {symbol}: {e}")
        return None

def fetch_klines_coingecko(coin_id, timeframe, days=90):
    if timeframe in ['1m','5m','15m','30m','1h']:
        days = 2
    else:
        days = 30
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {'vs_currency': 'usd', 'days': days}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['volume'] = 0
        print(f"   📥 CoinGecko {coin_id}: {len(df)} velas OHLC")
        return df
    except Exception as e:
        print(f"   ⚠️ CoinGecko error {coin_id}: {e}")
        return None

def fetch_klines(symbol, timeframe):
    key = f"{symbol}_{timeframe}"
    if key in DATA_CACHE:
        return DATA_CACHE[key]

    df = None

    if '_' in symbol:
        kucoin_sym = symbol.replace('_', '-')
        cryptocom_sym = symbol
    else:
        base = symbol
        kucoin_sym = f"{base}-USDT"
        cryptocom_sym = f"{base}_USDT"

    # 1. KuCoin
    df = fetch_klines_kucoin(kucoin_sym, timeframe, limit=LOOKBACK_CANDLES)
    if df is not None and len(df) > 50:
        DATA_CACHE[key] = df
        return df

    # 2. Crypto.com
    df = fetch_klines_cryptocom(cryptocom_sym, timeframe, limit=LOOKBACK_CANDLES)
    if df is not None and len(df) > 50:
        DATA_CACHE[key] = df
        return df

    # 3. CoinGecko
    coingecko_ids = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
        'BNB': 'binancecoin', 'DOGE': 'dogecoin', 'ADA': 'cardano',
        'DOT': 'polkadot', 'LINK': 'chainlink', 'MATIC': 'matic-network',
        'AVAX': 'avalanche-2', 'UNI': 'uniswap', 'ATOM': 'cosmos',
        'XRP': 'ripple', 'LTC': 'litecoin', 'BCH': 'bitcoin-cash'
    }
    if symbol in coingecko_ids:
        df = fetch_klines_coingecko(coingecko_ids[symbol], timeframe)
        if df is not None and len(df) > 20:
            DATA_CACHE[key] = df
            return df

    print(f"   ⚠️ No se pudieron obtener datos para {symbol} {timeframe}")
    return None

# ============================================================
# NORMALIZACIÓN Y TENSION 2-3-5
# ============================================================
def normalize(series, window=50):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)

def tension_235(series):
    ema2 = series.ewm(span=2).mean()
    ema3 = series.ewm(span=3).mean()
    ema5 = series.ewm(span=5).mean()
    return (ema2 - ema3).abs() + (ema3 - ema5).abs()

# ============================================================
# EDGE PORCENTUAL Y HITRATE
# ============================================================
def compute_edge_pct(price, tension, k, quantile=0.85):
    """
    Retorna edge porcentual y hitrate para puntos con tensión > cuantil.
    edge = (precio futuro - precio actual) / precio actual
    """
    mask = tension > tension.quantile(quantile)
    future_ret = (price.shift(-k) - price) / price
    edge = future_ret[mask].mean()
    hitrate = (future_ret[mask] > 0).mean()
    return edge, hitrate

# ============================================================
# TIEMPO MEDIO HASTA ALCANZAR EL TARGET (TAU)
# ============================================================
def time_to_target(price, idx, target_pct, max_lookahead=50):
    base = price.iloc[idx]
    target_price = base * (1 + target_pct) if target_pct > 0 else base * (1 + target_pct)  # target_pct negativo
    for i in range(1, max_lookahead):
        if idx + i >= len(price):
            break
        current = price.iloc[idx + i]
        if target_pct > 0 and current >= target_price:
            return i
        if target_pct < 0 and current <= target_price:
            return i
    return np.nan

def compute_tau(price, mask, target_pct, max_lookahead=50):
    idxs = np.where(mask)[0]
    tau_list = []
    for idx in idxs:
        tau = time_to_target(price, idx, target_pct, max_lookahead)
        if not np.isnan(tau):
            tau_list.append(tau)
    return np.nanmean(tau_list) if tau_list else np.nan

# ============================================================
# PIDelta (para correlación)
# ============================================================
def compute_pidelta(price, window=20):
    returns = price.pct_change().fillna(0)
    P_struct = returns.rolling(window).mean()
    P_hist = returns.ewm(span=window).mean()
    return P_struct - P_hist

def compute_corr(pidelta1, pidelta2):
    return pidelta1.corr(pidelta2) if len(pidelta1) > 1 else 0

# ============================================================
# ANÁLISIS INDIVIDUAL POR SÍMBOLO Y TIMEFRAME
# ============================================================
def analyze_symbol_tf(symbol, tf, display_symbol=None):
    df = fetch_klines(symbol, tf)
    if df is None or len(df) < 100:
        print(f"   ⚠️ {display_symbol or symbol} {tf}: datos insuficientes ({len(df) if df is not None else 0} velas)")
        return None, None

    price = df['close']
    volume = df['volume']
    S = normalize(price)
    T = tension_235(S)
    pidelta = compute_pidelta(price)

    k_values = [1, 2, 3, 5, 8, 13, 21]
    best_score = -np.inf
    best_result = None

    for k in k_values:
        edge_pct, hitrate = compute_edge_pct(price, T, k, quantile=0.85)

        # Monte Carlo para Z-score
        mc_edges = []
        for _ in range(30):
            perm = np.random.permutation(S.values)
            T_mc = tension_235(pd.Series(perm, index=S.index))
            mc_e, _ = compute_edge_pct(price, T_mc, k)
            mc_edges.append(mc_e)
        mc_mean = np.nanmean(mc_edges)
        mc_std = np.nanstd(mc_edges)
        Z = (edge_pct - mc_mean) / (mc_std + 1e-9)

        mask = T > T.quantile(0.85)
        tau = compute_tau(price, mask, edge_pct, max_lookahead=50)

        if np.isnan(tau) or tau == 0:
            continue

        # Score = |Z| * |edge_pct| / tau
        score = abs(Z) * abs(edge_pct) / (tau + 1e-9)

        if score > best_score:
            best_score = score
            best_result = {
                'Symbol': display_symbol if display_symbol else symbol,
                'TF': tf,
                'Categoria': CATEGORY_MAP.get(tf, 'Otro'),
                'k': k,
                'Edge_%': edge_pct,
                'HitRate': hitrate,
                'Z': Z,
                'Tau': tau,
                'Score': score,
                'Precio_actual': price.iloc[-1],
                'TP_est': price.iloc[-1] * (1 + TP_PCT),
                'SL_est': price.iloc[-1] * (1 - SL_PCT),
                'PIDelta_mean': pidelta.mean(),
                'PIDelta_std': pidelta.std(),
                'Volume_mean': volume.mean(),
                'Volume_std': volume.std()
            }

    return best_result, pidelta

# ============================================================
# ESCANEO POR TIMEFRAME
# ============================================================
def scan_timeframe(tf):
    if not os.path.exists(ASSETS_FILE):
        print(f"⚠️ Archivo {ASSETS_FILE} no encontrado. Usando lista por defecto.")
        assets = ['BTC', 'ETH', 'SOL', 'BNB', 'DOGE']
    else:
        assets_df = pd.read_csv(ASSETS_FILE)
        assets = assets_df[assets_df['active'] == 1]['symbol'].tolist()
        print(f"📋 Activos activos: {assets}")

    results = []
    pideltas = {}

    # Obtener PIDelta de activos base para correlación
    base_pideltas = {}
    for base in BASE_SYMBOLS:
        _, pid = analyze_symbol_tf(base, tf, display_symbol=base)
        if pid is not None:
            base_pideltas[base] = pid

    for asset in assets:
        print(f"🔍 Analizando {asset} {tf}...")
        sys.stdout.flush()

        result, pid = analyze_symbol_tf(asset, tf, display_symbol=asset)

        if result:
            # Calcular correlaciones con activos base
            for base in BASE_SYMBOLS:
                if base in base_pideltas and pid is not None:
                    common_idx = pid.index.intersection(base_pideltas[base].index)
                    if len(common_idx) > 10:
                        corr = compute_corr(pid.loc[common_idx], base_pideltas[base].loc[common_idx])
                    else:
                        corr = np.nan
                    result[f'Corr_{base}'] = corr
                else:
                    result[f'Corr_{base}'] = np.nan

            results.append(result)
            print(f"   ✅ {asset} {tf} | Edge: {result['Edge_%']:.4%} | HitRate: {result['HitRate']:.2%} | Score: {result['Score']:.4f} | Corr BTC: {result.get('Corr_BTC', 0):.2f}")
        else:
            print(f"   ⚠️ {asset} {tf}: no se obtuvieron suficientes datos")

        time.sleep(0.2)

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('Score', ascending=False)
        # Guardar archivo de este timeframe (opcional)
        df_results.to_csv(f"escaneo_{tf}.txt", index=False, sep='\t')
        print(f"✅ {len(df_results)} registros guardados para {tf}.")
    else:
        print(f"❌ No se generaron resultados para {tf}.")

    return df_results

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀  SISTEMA DE ESCANEO MULTI-EXCHANGE (KuCoin + Crypto.com + CoinGecko)")
    print("📊  Análisis por retorno porcentual futuro | Clasificación Scalp/Intraday/Swing")
    print("=" * 70)
    sys.stdout.flush()

    all_results = []
    for tf in TIMEFRAMES:
        print(f"\n⏰ Timeframe: {tf} ({CATEGORY_MAP.get(tf, 'Otro')})")
        df_tf = scan_timeframe(tf)
        if not df_tf.empty:
            all_results.append(df_tf)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        # Reordenar columnas para mejor lectura
        cols = ['Symbol', 'TF', 'Categoria', 'k', 'Edge_%', 'HitRate', 'Z', 'Tau', 'Score',
                'Precio_actual', 'TP_est', 'SL_est', 'PIDelta_mean', 'PIDelta_std',
                'Volume_mean', 'Volume_std', 'Corr_BTC', 'Corr_ETH', 'Corr_SOL']
        combined = combined[[c for c in cols if c in combined.columns]]
        combined.to_csv("escaneo_completo.txt", index=False, sep='\t')
        print("\n✅ Archivo combinado generado: 'escaneo_completo.txt'")
    else:
        # Crear archivo vacío para que el artifact no falle
        pd.DataFrame(columns=['Symbol','TF','Categoria','Edge_%','HitRate','Z','Tau','Score',
                              'Precio_actual','TP_est','SL_est','Corr_BTC','Corr_ETH','Corr_SOL'])\
          .to_csv("escaneo_completo.txt", index=False, sep='\t')
        print("⚠️ No se generaron datos. Archivo vacío creado.")

    print("\n✅ Proceso completado.")
