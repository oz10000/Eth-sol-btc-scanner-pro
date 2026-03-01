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
EXCHANGE = "cryptocom"  # Usamos Crypto.com como fuente principal
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "8h", "12h", "1d"]
LOOKBACK_CANDLES = 300   # Número de velas a descargar
TP_PCT = 0.02            # Take profit % (para estimación)
SL_PCT = 0.01            # Stop loss % (para estimación)

# Archivo de activos (CSV con columnas 'symbol' y 'active')
ASSETS_FILE = "assets.csv"

# Activos base para correlación
BASE_SYMBOLS = ["BTC", "ETH", "SOL"]  # En formato corto

# Headers para simular navegador
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# Cache para datos
DATA_CACHE = {}

# ============================================================
# FUNCIONES DE DESCARGA (Crypto.com + fallback CoinGecko)
# ============================================================
def fetch_candles_cryptocom(symbol, timeframe, limit=500):
    """
    symbol: formato 'BTC_USDT'
    timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d
    """
    tf_map = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '4h', '8h': '8h', '12h': '12h', '1d': '1d'
    }
    interval = tf_map.get(timeframe, '1h')
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
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
        return df
    except Exception as e:
        print(f"Error Crypto.com {symbol}: {e}")
        return None

def fetch_ohlc_coingecko(coin_id, vs_currency='usd', days=30):
    """
    CoinGecko fallback (solo cierre, no high/low fiables)
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {'vs_currency': vs_currency, 'days': days}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['volume'] = 0
        return df
    except:
        return None

def fetch_klines(symbol, timeframe):
    """
    Wrapper: intenta Crypto.com primero, si no, CoinGecko.
    symbol: puede ser 'BTC_USDT' o 'bitcoin'
    """
    key = f"{symbol}_{timeframe}"
    if key in DATA_CACHE:
        return DATA_CACHE[key]

    df = None
    # Si el símbolo tiene guion bajo, es formato Crypto.com
    if '_' in symbol:
        df = fetch_candles_cryptocom(symbol, timeframe, limit=LOOKBACK_CANDLES)
    else:
        # Si no, asumimos que es ID de CoinGecko
        df = fetch_ohlc_coingecko(symbol, days=LOOKBACK_CANDLES//24)  # aprox
    if df is not None and len(df) > 50:
        DATA_CACHE[key] = df
        return df
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
# EDGE Y HITRATE PARA UN HORIZONTE k
# ============================================================
def compute_edge(price, tension, k, quantile=0.85):
    """
    Retorna (edge, hitrate) para puntos con tensión > cuantil.
    edge: retorno medio futuro (en precio)
    hitrate: proporción de veces que el retorno es positivo
    """
    mask = tension > tension.quantile(quantile)
    future_ret = price.shift(-k) - price
    edge = future_ret[mask].mean()
    hitrate = (future_ret[mask] > 0).mean()
    return edge, hitrate

# ============================================================
# TIEMPO MEDIO HASTA ALCANZAR EL TARGET (TAU)
# ============================================================
def time_to_target(price, idx, target, max_lookahead=50):
    """
    Calcula cuántas velas tarda en alcanzar un target de precio.
    target positivo: subida, negativo: bajada.
    """
    base = price.iloc[idx]
    for i in range(1, max_lookahead):
        if idx + i >= len(price):
            break
        move = price.iloc[idx + i] - base
        if target > 0 and move >= target:
            return i
        if target < 0 and move <= target:
            return i
    return np.nan

def compute_tau(price, mask, target, max_lookahead=50):
    """
    Calcula el tiempo medio hasta alcanzar target para los índices en mask.
    """
    idxs = np.where(mask)[0]
    tau_list = []
    for idx in idxs:
        tau = time_to_target(price, idx, target, max_lookahead)
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
# BACKTEST INDIVIDUAL PARA UN SÍMBOLO Y TIMEFRAME
# ============================================================
def analyze_symbol_tf(symbol, tf, display_symbol=None):
    """
    symbol: puede ser 'BTC_USDT' o 'bitcoin'
    display_symbol: nombre para mostrar (ej. 'BTC')
    """
    df = fetch_klines(symbol, tf)
    if df is None or len(df) < 100:
        return None, None

    price = df['close']
    volume = df['volume']
    S = normalize(price)
    T = tension_235(S)
    pidelta = compute_pidelta(price)

    # Probar distintos horizontes k
    k_values = [1, 2, 3, 5, 8, 13, 21]
    best_score = -np.inf
    best_result = None

    for k in k_values:
        edge, hitrate = compute_edge(price, T, k, quantile=0.85)

        # Monte Carlo para Z-score
        mc_edges = []
        for _ in range(50):
            perm = np.random.permutation(S.values)
            T_mc = tension_235(pd.Series(perm, index=S.index))
            mc_e, _ = compute_edge(price, T_mc, k)
            mc_edges.append(mc_e)
        mc_mean = np.nanmean(mc_edges)
        mc_std = np.nanstd(mc_edges)
        Z = (edge - mc_mean) / (mc_std + 1e-9)

        # Tiempo medio hasta TP (usando edge como target)
        mask = T > T.quantile(0.85)
        tau = compute_tau(price, mask, edge, max_lookahead=50)

        if np.isnan(tau) or tau == 0:
            continue

        # Score: combinación de Z, edge y tau
        score = abs(Z) * abs(edge) / (tau + 1e-9)

        if score > best_score:
            best_score = score
            best_result = {
                'Symbol': display_symbol if display_symbol else symbol,
                'TF': tf,
                'k': k,
                'Edge': edge,
                'HitRate': hitrate,
                'Z': Z,
                'Tau': tau,
                'Score': score,
                'TP_est': price.iloc[-1] * (1 + TP_PCT),
                'SL_est': price.iloc[-1] * (1 - SL_PCT),
                'PIDelta_mean': pidelta.mean(),
                'PIDelta_std': pidelta.std(),
                'Volume_mean': volume.mean(),
                'Volume_std': volume.std()
            }

    return best_result, pidelta

# ============================================================
# ESCANEO COMPLETO
# ============================================================
def scan_all(timeframe):
    # Leer activos desde CSV
    if not os.path.exists(ASSETS_FILE):
        print(f"⚠️ Archivo {ASSETS_FILE} no encontrado. Usando lista por defecto.")
        assets = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
    else:
        assets_df = pd.read_csv(ASSETS_FILE)
        assets = assets_df[assets_df['active'] == 1]['symbol'].tolist()
        print(f"📋 Activos activos: {assets}")

    results = []
    pideltas = {}

    # Obtener PIDelta de activos base para correlación
    base_pideltas = {}
    for base in BASE_SYMBOLS:
        base_symbol = f"{base}_USDT"
        _, pid = analyze_symbol_tf(base_symbol, timeframe, display_symbol=base)
        if pid is not None:
            base_pideltas[base] = pid

    for asset in assets:
        print(f"🔍 Analizando {asset} {timeframe}...")
        sys.stdout.flush()

        # Determinar símbolo de búsqueda
        # Si es formato 'BTC', convertimos a 'BTC_USDT'
        if '_' not in asset and asset in BASE_SYMBOLS:
            search_sym = f"{asset}_USDT"
            display = asset
        else:
            search_sym = asset
            display = asset

        result, pid = analyze_symbol_tf(search_sym, timeframe, display_symbol=display)

        if result:
            # Calcular correlaciones con activos base
            for base in BASE_SYMBOLS:
                if base in base_pideltas and pid is not None:
                    corr = compute_corr(pid, base_pideltas[base])
                    result[f'Corr_{base}'] = corr
                else:
                    result[f'Corr_{base}'] = np.nan

            results.append(result)
            print(f"   ✅ {display} {timeframe} | Edge: {result['Edge']:.4f} | HitRate: {result['HitRate']:.2%} | Score: {result['Score']:.4f} | Corr BTC: {result.get('Corr_BTC', 0):.2f}")
        else:
            print(f"   ⚠️ {display} {timeframe}: no se obtuvieron suficientes datos")

        time.sleep(0.2)  # Pequeña pausa para no saturar

    # Convertir a DataFrame
    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("❌ No se generaron resultados.")
        return df_results

    # Ordenar y guardar
    df_results = df_results.sort_values('Score', ascending=False)
    df_results.to_csv("escaneo_filtrado.txt", index=False, sep='\t')
    print(f"✅ Escaneo completado. {len(df_results)} registros guardados.")

    # Mostrar tops
    print("\n=== TOP 10 LONG (mayor Score) ===")
    top_long = df_results[df_results['Edge'] > 0].head(10)
    print(top_long[['Symbol', 'TF', 'Edge', 'HitRate', 'Score', 'Corr_BTC']].to_string())

    print("\n=== TOP 10 SHORT (menor Edge, más negativo) ===")
    top_short = df_results[df_results['Edge'] < 0].sort_values('Edge').head(10)
    print(top_short[['Symbol', 'TF', 'Edge', 'HitRate', 'Score', 'Corr_BTC']].to_string())

    # Top 3 señales seguras (con menor correlación con BTC/ETH/SOL)
    df_results['MaxCorr'] = df_results[[f'Corr_{b}' for b in BASE_SYMBOLS if f'Corr_{b}' in df_results.columns]].max(axis=1)
    safe_signals = df_results[df_results['MaxCorr'] < 0.85].sort_values('Score', ascending=False).head(3)

    print("\n=== TOP 3 SEÑALES MÁS SEGURAS (baja correlación con base) ===")
    print(safe_signals[['Symbol', 'TF', 'Edge', 'HitRate', 'Score', 'MaxCorr']].to_string())

    return df_results

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🚀 SISTEMA DE ESCANEO CON RETORNO FUTURO (basado en edge)")
    print("=" * 60)
    # Escanear para cada timeframe (o elegir uno)
    for tf in TIMEFRAMES:
        print(f"\n⏰ Timeframe: {tf}")
        scan_all(tf)
    print("\n✅ Proceso completado. Archivo generado: 'escaneo_filtrado.txt'")
