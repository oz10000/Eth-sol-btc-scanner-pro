import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from itertools import product
import warnings
import sys

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
SYMBOLS = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'solana': 'SOL'
}

CRYPTOCOM_SYMBOLS = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h"]
LOOKBACK_DAYS = 30
SCAN_LOOKBACK = 20
MC_ITER = 200
DATA_CACHE = {}

# Parámetros del escáner
TENSION_QUANTILE_SCAN = 0.5
SCAN_K_VALUES = [3, 5, 8, 13]
MIN_FUTURE_VELAS = 20

# Grid de optimización
PARAM_GRID = {
    'tp_atr': [1, 2, 3, 5, 8],
    'sl_atr': [1, 2, 3, 5],
    'atr_window': [7, 14, 21],
    'tension_quantile': [0.5, 0.6, 0.7, 0.8, 0.9],
    'pidelta_window': [5, 8, 13, 21]
}

# Parámetros por defecto para backtest de señales top
DEFAULT_TP_ATR = 3
DEFAULT_SL_ATR = 2
DEFAULT_ATR_WINDOW = 14

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# ============================================================
# FUNCIONES DE DESCARGA (CoinGecko + Crypto.com)
# ============================================================
def fetch_ohlc_coingecko(coin_id, vs_currency='usd', days=LOOKBACK_DAYS):
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

def fetch_market_chart_coingecko(coin_id, vs_currency='usd', days=LOOKBACK_DAYS):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        df = prices.merge(volumes, on='timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Aproximar high/low como el precio (no ideal)
        df['high'] = df['price']
        df['low'] = df['price']
        df['open'] = df['price'].shift(1).fillna(df['price'])
        df['close'] = df['price']
        return df
    except:
        return None

def fetch_candles_cryptocom(instrument_name, timeframe='1m', limit=500):
    tf_map = {'1m':'1m','5m':'5m','15m':'15m','30m':'30m','1h':'1h'}
    interval = tf_map.get(timeframe, '1h')
    url = "https://api.crypto.com/exchange/v1/public/get-candlestick"
    params = {'instrument_name': instrument_name, 'timeframe': interval}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if data['code'] != 0:
            return None
        candles = data['result']['data']
        df = pd.DataFrame(candles)
        for col in ['t','o','h','l','c','v']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
        df = df[['open','high','low','close','volume']].sort_index()
        return df
    except:
        return None

def fetch_klines(symbol, tf):
    key = f"{symbol}_{tf}"
    if key in DATA_CACHE:
        return DATA_CACHE[key]

    df = None
    # Priorizar Crypto.com (datos OHLC reales)
    if symbol in SYMBOLS:
        cryptocom_sym = f"{SYMBOLS[symbol]}_USDT"
    else:
        cryptocom_sym = symbol

    df = fetch_candles_cryptocom(cryptocom_sym, tf, limit=500)
    if df is not None and len(df) > 50:
        DATA_CACHE[key] = df
        return df

    # Si falla, usar CoinGecko (aproximado)
    if symbol in SYMBOLS:
        df = fetch_market_chart_coingecko(symbol, days=SCAN_LOOKBACK)
        if df is not None and len(df) > 50:
            DATA_CACHE[key] = df
            return df
        df = fetch_ohlc_coingecko(symbol, days=SCAN_LOOKBACK)
        if df is not None and len(df) > 50:
            DATA_CACHE[key] = df
            return df

    return None

# ============================================================
# INDICADORES
# ============================================================
def get_atr(symbol, tf, window):
    df = fetch_klines(symbol, tf)
    if df is None or len(df) < window + 10:
        return None
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def tension_235(series):
    ema2 = series.ewm(span=2).mean()
    ema3 = series.ewm(span=3).mean()
    ema5 = series.ewm(span=5).mean()
    return (ema2 - ema3).abs() + (ema3 - ema5).abs()

# ============================================================
# ESCÁNER
# ============================================================
def scan_symbol_tf(symbol_id, symbol_display, tf):
    df = fetch_klines(symbol_id, tf)
    if df is None or len(df) < 100:
        return []

    price = df['close']
    tension = tension_235(price)
    threshold = tension.quantile(TENSION_QUANTILE_SCAN)
    high_tension_points = tension[tension >= threshold].index

    signals = []
    for dt in high_tension_points:
        try:
            idx = price.index.get_loc(dt)
        except:
            continue
        if idx + MIN_FUTURE_VELAS >= len(price):
            continue
        for k in SCAN_K_VALUES:
            if idx + k >= len(price):
                continue
            future_ret = price.iloc[idx + k] - price.iloc[idx]
            if future_ret > 0:
                signals.append({
                    'Symbol': symbol_display,
                    'TF': tf,
                    'open_time': dt,
                    'Direction': 'LONG',
                    'tension': tension.loc[dt],
                    'edge': future_ret / price.iloc[idx],
                    'winrate': 1.0,
                    'k': k,
                    'score': future_ret,
                    'entry_price': price.iloc[idx],
                    'future_price_k': price.iloc[idx + k]
                })
            elif future_ret < 0:
                signals.append({
                    'Symbol': symbol_display,
                    'TF': tf,
                    'open_time': dt,
                    'Direction': 'SHORT',
                    'tension': tension.loc[dt],
                    'edge': -future_ret / price.iloc[idx],
                    'winrate': 1.0,
                    'k': k,
                    'score': -future_ret,
                    'entry_price': price.iloc[idx],
                    'future_price_k': price.iloc[idx + k]
                })
    return signals

def run_scanner():
    all_signals = []
    for coin_id, display in SYMBOLS.items():
        for tf in TIMEFRAMES:
            print(f"🔍 Escaneando {display} {tf}...")
            sys.stdout.flush()
            sigs = scan_symbol_tf(coin_id, display, tf)
            all_signals.extend(sigs)

    for sym in CRYPTOCOM_SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"🔍 Escaneando {sym} {tf}...")
            sys.stdout.flush()
            sigs = scan_symbol_tf(sym, sym, tf)
            all_signals.extend(sigs)

    df_signals = pd.DataFrame(all_signals)
    if len(df_signals) > 0:
        df_signals = df_signals.sort_values('score', ascending=False)
        df_signals.to_csv("escaneo_filtrado.txt", index=False, sep='\t')
        print(f"✅ Escáner completado. {len(df_signals)} señales guardadas.")
    else:
        print("⚠️ No se generaron señales. Creando archivo vacío.")
        pd.DataFrame(columns=['Symbol','TF','open_time','Direction','tension','edge','winrate','k','score','entry_price','future_price_k'])\
          .to_csv("escaneo_filtrado.txt", index=False, sep='\t')
    sys.stdout.flush()
    return df_signals

# ============================================================
# BACKTEST DE UNA SEÑAL (con stops basados en ATR)
# ============================================================
def backtest_signal(symbol, tf, signal_time, direction, tp_atr, sl_atr, atr_window, max_lookahead=30):
    df = fetch_klines(symbol, tf)
    if df is None:
        return None
    atr_series = get_atr(symbol, tf, atr_window)
    if atr_series is None:
        return None
    try:
        idx = df.index.get_loc(signal_time, method='nearest')
    except:
        return None
    if idx < atr_window or idx >= len(df) - 1:
        return None
    entry_price = df['close'].iloc[idx]
    atr_val = atr_series.iloc[idx]
    if pd.isna(atr_val) or atr_val == 0:
        return None

    if direction == 'LONG':
        target = entry_price + tp_atr * atr_val
        stop = entry_price - sl_atr * atr_val
    else:
        target = entry_price - tp_atr * atr_val
        stop = entry_price + sl_atr * atr_val

    for i in range(1, min(max_lookahead, len(df)-idx-1)):
        high = df['high'].iloc[idx+i]
        low = df['low'].iloc[idx+i]
        if direction == 'LONG':
            if high >= target:
                ret = (target - entry_price) / entry_price
                return {'exit_type': 'TP', 'bars': i, 'return': ret}
            if low <= stop:
                ret = (stop - entry_price) / entry_price
                return {'exit_type': 'SL', 'bars': i, 'return': ret}
        else:
            if low <= target:
                ret = (entry_price - target) / entry_price
                return {'exit_type': 'TP', 'bars': i, 'return': ret}
            if high >= stop:
                ret = (entry_price - stop) / entry_price
                return {'exit_type': 'SL', 'bars': i, 'return': ret}
    final_price = df['close'].iloc[min(idx+max_lookahead, len(df)-1)]
    ret = (final_price - entry_price) / entry_price if direction == 'LONG' else (entry_price - final_price) / entry_price
    return {'exit_type': 'NONE', 'bars': max_lookahead, 'return': ret}

# ============================================================
# MÉTRICAS DE PORTAFOLIO
# ============================================================
def portfolio_metrics(trades, daily_corr):
    if len(trades) < 1:  # Cambiado de 3 a 1 para permitir resultados
        return None, None, None, None, -np.inf
    df_trades = pd.DataFrame(trades)
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_trades['date'] = df_trades['exit_time'].dt.date
    daily_returns = df_trades.groupby('date')['return'].sum().sort_index()
    if len(daily_returns) == 0:
        return None, None, None, None, -np.inf
    total_return = daily_returns.sum()
    sharpe = daily_returns.mean() / (daily_returns.std() + 1e-9)
    equity = daily_returns.cumsum()
    peak = equity.expanding().max()
    dd = (equity - peak).min()
    max_dd = abs(dd)
    returns_array = df_trades['return'].values
    n = len(returns_array)
    perm_returns = []
    for _ in range(MC_ITER):
        sign_perm = np.random.choice([-1, 1], size=n)
        perm_total = (returns_array * sign_perm).sum()
        perm_returns.append(perm_total)
    perm_returns = np.array(perm_returns)
    z_score = (total_return - perm_returns.mean()) / (perm_returns.std() + 1e-9)
    score = total_return * sharpe * (1 - max_dd) * (1 - daily_corr) * (1 + max(0, z_score)/3)
    return total_return, sharpe, max_dd, z_score, score

def get_daily_correlation(symbols):
    return 0.5  # Simplificado

# ============================================================
# OPTIMIZACIÓN DE PARÁMETROS (con mínimo de trades = 1)
# ============================================================
def optimize_parameters(signals_df):
    if signals_df.empty:
        print("⚠️ No hay señales para optimizar. Creando archivo vacío.")
        pd.DataFrame(columns=list(PARAM_GRID.keys())+['num_trades','total_return','sharpe','max_drawdown','z_score','score'])\
          .to_csv("optimization_results.txt", index=False, sep='\t')
        return None

    time_col = 'open_time'
    signals_df[time_col] = pd.to_datetime(signals_df[time_col])
    if 'tension' not in signals_df.columns:
        signals_df['tension'] = 1.0

    daily_corr = get_daily_correlation(SYMBOLS)
    print(f"📊 Correlación media diaria: {daily_corr:.3f}")

    results = []
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    combinations = list(product(*param_values))
    total = len(combinations)
    print(f"🔍 Evaluando {total} combinaciones (mínimo 1 trade)...")

    for idx, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        tp_atr, sl_atr, atr_win, tens_q, pid_win = params.values()

        threshold = signals_df['tension'].quantile(tens_q)
        filtered = signals_df[signals_df['tension'] >= threshold]
        if filtered.empty:
            continue

        trades = []
        for _, sig in filtered.iterrows():
            # Determinar símbolo para fetch
            symbol_key = None
            for coin_id, display in SYMBOLS.items():
                if display == sig['Symbol']:
                    symbol_key = coin_id
                    break
            if symbol_key is None:
                symbol_key = sig['Symbol'] if '_' in sig['Symbol'] else f"{sig['Symbol']}_USDT"

            res = backtest_signal(
                symbol=symbol_key,
                tf=sig['TF'],
                signal_time=sig[time_col],
                direction=sig['Direction'],
                tp_atr=tp_atr,
                sl_atr=sl_atr,
                atr_window=atr_win
            )
            if res:
                trades.append({
                    'return': res['return'],
                    'exit_time': sig[time_col] + timedelta(minutes=res['bars'] * 5),
                    'symbol': sig['Symbol'],
                    'direction': sig['Direction'],
                    'exit_type': res['exit_type']
                })

        n_trades = len(trades)
        if idx % 100 == 0:
            print(f"   Progreso: {idx}/{total} - {params} -> trades={n_trades}")
        if n_trades < 1:  # Permitimos al menos 1 trade
            continue

        total_ret, sharpe, mdd, z, score = portfolio_metrics(trades, daily_corr)
        if score == -np.inf:
            continue

        results.append({**params, 'num_trades': n_trades, 'total_return': total_ret,
                        'sharpe': sharpe, 'max_drawdown': mdd, 'z_score': z, 'score': score})
        print(f"   ✅ {params} -> score={score:.4f} (trades={n_trades})")

    if not results:
        print("❌ No se encontraron combinaciones válidas. Creando archivo vacío.")
        pd.DataFrame(columns=list(PARAM_GRID.keys())+['num_trades','total_return','sharpe','max_drawdown','z_score','score'])\
          .to_csv("optimization_results.txt", index=False, sep='\t')
        return None

    best = max(results, key=lambda x: x['score'])
    print("\n🏆 Mejor combinación:")
    for k, v in best.items():
        print(f"   {k}: {v}")
    df_opt = pd.DataFrame(results)
    df_opt.to_csv("optimization_results.txt", index=False, sep='\t')
    print("✅ Optimización guardada en 'optimization_results.txt'")
    return best

# ============================================================
# BACKTEST DE SEÑALES TOP (con parámetros fijos)
# ============================================================
def backtest_top_signals(signals_df, top_n=3):
    if signals_df.empty:
        print("No hay señales para backtest.")
        return

    long_signals = signals_df[signals_df['Direction'] == 'LONG'].sort_values('score', ascending=False).head(top_n)
    short_signals = signals_df[signals_df['Direction'] == 'SHORT'].sort_values('score', ascending=False).head(top_n)
    top_signals = pd.concat([long_signals, short_signals])

    print("\n📈 BACKTEST DE SEÑALES TOP (con parámetros por defecto TP={} SL={} ATR={})".format(
        DEFAULT_TP_ATR, DEFAULT_SL_ATR, DEFAULT_ATR_WINDOW))
    print("-" * 80)

    for _, sig in top_signals.iterrows():
        # Determinar símbolo
        symbol_key = None
        for coin_id, display in SYMBOLS.items():
            if display == sig['Symbol']:
                symbol_key = coin_id
                break
        if symbol_key is None:
            symbol_key = sig['Symbol'] if '_' in sig['Symbol'] else f"{sig['Symbol']}_USDT"

        res = backtest_signal(
            symbol=symbol_key,
            tf=sig['TF'],
            signal_time=sig['open_time'],
            direction=sig['Direction'],
            tp_atr=DEFAULT_TP_ATR,
            sl_atr=DEFAULT_SL_ATR,
            atr_window=DEFAULT_ATR_WINDOW
        )

        if res:
            print(f"{sig['Symbol']} {sig['TF']} {sig['Direction']} | "
                  f"Entrada: {sig['entry_price']:.2f} | "
                  f"Salida: {res['exit_type']} en {res['bars']} velas | "
                  f"Retorno: {res['return']*100:.2f}%")
        else:
            print(f"{sig['Symbol']} {sig['TF']} {sig['Direction']} | "
                  f"Entrada: {sig['entry_price']:.2f} | "
                  f"Backtest no disponible (datos insuficientes)")

    print("-" * 80)

# ============================================================
# TOP SEÑALES (solo información)
# ============================================================
def analyze_top_signals(signals_df, best_params):
    if signals_df.empty:
        print("No hay señales para analizar.")
        return
    long_signals = signals_df[signals_df['Direction'] == 'LONG'].sort_values('score', ascending=False).head(3)
    short_signals = signals_df[signals_df['Direction'] == 'SHORT'].sort_values('score', ascending=False).head(3)
    top_signals = pd.concat([long_signals, short_signals])

    print("\n📊 TOP 3 SEÑALES LONG Y TOP 3 SHORT (según score del escáner)")
    for _, sig in top_signals.iterrows():
        print(f"{sig['Symbol']} {sig['TF']} {sig['Direction']} | "
              f"Score: {sig['score']:.2f} | Tensión: {sig['tension']:.2f} | "
              f"Entrada: {sig['entry_price']:.2f}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🚀 SISTEMA INTEGRADO CON COINGECKO + CRYPTO.COM")
    sys.stdout.flush()
    signals = run_scanner()
    best_params = optimize_parameters(signals)
    if best_params is None and not signals.empty:
        best_params = {'tp_atr': DEFAULT_TP_ATR, 'sl_atr': DEFAULT_SL_ATR,
                       'atr_window': DEFAULT_ATR_WINDOW, 'pidelta_window': 13,
                       'tension_quantile': 0.8}
    analyze_top_signals(signals, best_params if best_params else {})
    backtest_top_signals(signals, top_n=3)  # Nuevo backtest
    print("\n✅ Proceso completado. Archivos: 'escaneo_filtrado.txt', 'optimization_results.txt'")
    sys.stdout.flush()
