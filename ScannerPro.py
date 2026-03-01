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
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h"]
LOOKBACK_DAYS = 60
SCAN_LOOKBACK = 30
MC_ITER = 200
DATA_CACHE = {}
ATR_CACHE = {}
PIDELTA_CACHE = {}

# Parámetros del escáner (ajustados para generar señales)
TENSION_QUANTILE_SCAN = 0.5          # Reducido de 0.85 para capturar más puntos
MIN_WINRATE = 0.55
SCAN_K_VALUES = [3, 5, 8, 13]
MIN_FUTURE_VELAS = 20                 # Reducido de 50 para requerir menos velas hacia adelante

# Grid de optimización
PARAM_GRID = {
    'tp_atr': [1, 2, 3, 5, 8],
    'sl_atr': [1, 2, 3, 5],
    'atr_window': [7, 14, 21],
    'tension_quantile': [0.5, 0.6, 0.7, 0.8, 0.9],
    'pidelta_window': [5, 8, 13, 21]
}

# ============================================================
# DESCARGA DE DATOS
# ============================================================
def fetch_klines(symbol, interval, days=LOOKBACK_DAYS):
    key = f"{symbol}_{interval}"
    if key in DATA_CACHE:
        return DATA_CACHE[key]

    end = int(time.time() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "startTime": start, "endTime": end, "limit": 1500}
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
        print(f"   📥 {symbol} {interval}: {len(df)} velas descargadas")
        sys.stdout.flush()
        return df
    except Exception as e:
        print(f"   ❌ Error descargando {symbol} {interval}: {e}")
        sys.stdout.flush()
        return None

def get_atr(symbol, tf, window):
    key = (symbol, tf, window)
    if key in ATR_CACHE:
        return ATR_CACHE[key]
    df = fetch_klines(symbol, tf)
    if df is None or len(df) < window:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_series = tr.rolling(window).mean()
    ATR_CACHE[key] = atr_series
    return atr_series

def get_pidelta(symbol, tf, window):
    key = (symbol, tf, window)
    if key in PIDELTA_CACHE:
        return PIDELTA_CACHE[key]
    df = fetch_klines(symbol, tf)
    if df is None or len(df) < window:
        return None
    price = df["close"]
    rets = price.pct_change().fillna(0)
    P_struct = rets.rolling(window).mean()
    P_hist = rets.ewm(span=window).mean()
    pidelta = P_struct - P_hist
    PIDELTA_CACHE[key] = pidelta
    return pidelta

# ============================================================
# ESCÁNER
# ============================================================
def tension_235(series):
    ema2 = series.ewm(span=2).mean()
    ema3 = series.ewm(span=3).mean()
    ema5 = series.ewm(span=5).mean()
    return (ema2 - ema3).abs() + (ema3 - ema5).abs()

def scan_symbol_tf(symbol, tf):
    df = fetch_klines(symbol, tf, days=SCAN_LOOKBACK)
    if df is None or len(df) < 100:
        print(f"   ⚠️ {symbol} {tf}: datos insuficientes ({len(df) if df is not None else 0} velas)")
        return []
    price = df['close']
    tension = tension_235(price)
    threshold = tension.quantile(TENSION_QUANTILE_SCAN)
    high_tension_points = tension[tension >= threshold].index
    print(f"   📊 {symbol} {tf}: umbral tensión={threshold:.6f}, puntos alta tensión={len(high_tension_points)}")
    sys.stdout.flush()

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
                    'Symbol': symbol, 'TF': tf, 'open_time': dt, 'Direction': 'LONG',
                    'tension': tension.loc[dt], 'edge': future_ret / price.iloc[idx],
                    'winrate': 1.0, 'k': k, 'score': future_ret
                })
            elif future_ret < 0:
                signals.append({
                    'Symbol': symbol, 'TF': tf, 'open_time': dt, 'Direction': 'SHORT',
                    'tension': tension.loc[dt], 'edge': -future_ret / price.iloc[idx],
                    'winrate': 1.0, 'k': k, 'score': -future_ret
                })
    print(f"   ✅ {symbol} {tf}: {len(signals)} señales generadas")
    sys.stdout.flush()
    return signals

def run_scanner():
    all_signals = []
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"🔍 Escaneando {sym} {tf}...")
            sys.stdout.flush()
            sigs = scan_symbol_tf(sym, tf)
            all_signals.extend(sigs)
    df_signals = pd.DataFrame(all_signals)
    # Siempre guardar archivo, aunque esté vacío (con encabezados)
    if len(df_signals) > 0:
        df_signals = df_signals.sort_values('score', ascending=False)
        df_signals.to_csv("escaneo_filtrado.txt", index=False, sep='\t')
        print(f"✅ Escáner completado. {len(df_signals)} señales guardadas en 'escaneo_filtrado.txt'.")
    else:
        print("⚠️ No se generaron señales. Creando archivo vacío con encabezados.")
        pd.DataFrame(columns=['Symbol','TF','open_time','Direction','tension','edge','winrate','k','score'])\
          .to_csv("escaneo_filtrado.txt", index=False, sep='\t')
    sys.stdout.flush()
    return df_signals

# ============================================================
# BACKTEST DE UNA SEÑAL
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
                return {'exit_type': 'TP', 'bars': i, 'return': ret, 'exit_price': target}
            if low <= stop:
                ret = (stop - entry_price) / entry_price
                return {'exit_type': 'SL', 'bars': i, 'return': ret, 'exit_price': stop}
        else:
            if low <= target:
                ret = (entry_price - target) / entry_price
                return {'exit_type': 'TP', 'bars': i, 'return': ret, 'exit_price': target}
            if high >= stop:
                ret = (entry_price - stop) / entry_price
                return {'exit_type': 'SL', 'bars': i, 'return': ret, 'exit_price': stop}
    final_idx = min(idx+max_lookahead, len(df)-1)
    final_price = df['close'].iloc[final_idx]
    ret = (final_price - entry_price) / entry_price if direction=='LONG' else (entry_price - final_price) / entry_price
    return {'exit_type': 'NONE', 'bars': final_idx - idx, 'return': ret, 'exit_price': final_price}

# ============================================================
# MÉTRICAS DE PORTAFOLIO
# ============================================================
def portfolio_metrics(trades, daily_corr):
    if len(trades) < 3:
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
    daily_data = {}
    for sym in symbols:
        df = fetch_klines(sym, '1d', days=LOOKBACK_DAYS)
        if df is not None:
            daily_data[sym] = df['close'].pct_change().dropna()
    common_dates = None
    for sym in symbols:
        if sym in daily_data:
            if common_dates is None:
                common_dates = set(daily_data[sym].index)
            else:
                common_dates = common_dates.intersection(daily_data[sym].index)
    if not common_dates:
        return 0.5
    common_dates = sorted(common_dates)
    rets_df = pd.DataFrame({sym: daily_data[sym].loc[common_dates] for sym in symbols if sym in daily_data})
    corr_matrix = rets_df.corr()
    off_diag = []
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            if symbols[i] in corr_matrix and symbols[j] in corr_matrix:
                off_diag.append(abs(corr_matrix.loc[symbols[i], symbols[j]]))
    return np.mean(off_diag) if off_diag else 0.5

# ============================================================
# OPTIMIZACIÓN DE PARÁMETROS
# ============================================================
def optimize_parameters(signals_df):
    if signals_df.empty:
        print("⚠️ No hay señales para optimizar. Creando archivo vacío.")
        pd.DataFrame(columns=list(PARAM_GRID.keys())+['num_trades','total_return','sharpe','max_drawdown','z_score','score'])\
          .to_csv("optimization_results.txt", index=False, sep='\t')
        return None

    time_col = 'open_time' if 'open_time' in signals_df.columns else signals_df.columns[0]
    signals_df[time_col] = pd.to_datetime(signals_df[time_col])
    if 'tension' not in signals_df.columns:
        signals_df['tension'] = 1.0

    daily_corr = get_daily_correlation(SYMBOLS)
    print(f"📊 Correlación media diaria entre activos: {daily_corr:.3f}")
    sys.stdout.flush()

    results = []
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    combinations = list(product(*param_values))
    total = len(combinations)
    print(f"🔍 Evaluando {total} combinaciones...")
    sys.stdout.flush()

    for idx, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        tp_atr, sl_atr, atr_win, tens_q, pid_win = params.values()

        threshold = signals_df['tension'].quantile(tens_q)
        filtered = signals_df[signals_df['tension'] >= threshold]
        if filtered.empty:
            continue

        trades = []
        fallos = 0
        for _, sig in filtered.iterrows():
            df = fetch_klines(sig['Symbol'], sig['TF'])
            if df is None:
                fallos += 1
                continue
            try:
                idx_s = df.index.get_loc(sig[time_col], method='nearest')
            except:
                fallos += 1
                continue
            if idx_s + 30 >= len(df):
                fallos += 1
                continue

            res = backtest_signal(
                symbol=sig['Symbol'], tf=sig['TF'], signal_time=sig[time_col],
                direction=sig['Direction'], tp_atr=tp_atr, sl_atr=sl_atr, atr_window=atr_win
            )
            if res:
                trades.append({
                    'return': res['return'], 'exit_time': df.index[min(idx_s+res['bars'], len(df)-1)],
                    'symbol': sig['Symbol'], 'direction': sig['Direction'], 'exit_type': res['exit_type']
                })
            else:
                fallos += 1

        n_trades = len(trades)
        if idx % 100 == 0:
            print(f"   Progreso: {idx}/{total} - Última: {params} -> trades={n_trades}")
            sys.stdout.flush()
        if n_trades < 3:
            continue

        total_ret, sharpe, mdd, z, score = portfolio_metrics(trades, daily_corr)
        if score == -np.inf:
            continue

        results.append({**params, 'num_trades': n_trades, 'total_return': total_ret,
                        'sharpe': sharpe, 'max_drawdown': mdd, 'z_score': z, 'score': score})
        print(f"   ✅ {params} -> score={score:.4f} (trades={n_trades})")
        sys.stdout.flush()

    # Siempre guardar archivo, aunque esté vacío
    if not results:
        print("❌ No se encontraron combinaciones válidas. Creando archivo vacío.")
        pd.DataFrame(columns=list(PARAM_GRID.keys())+['num_trades','total_return','sharpe','max_drawdown','z_score','score'])\
          .to_csv("optimization_results.txt", index=False, sep='\t')
        return None

    best = max(results, key=lambda x: x['score'])
    print("\n🏆 Mejor combinación:")
    for k, v in best.items():
        print(f"   {k}: {v}")
    sys.stdout.flush()
    df_opt = pd.DataFrame(results)
    df_opt.to_csv("optimization_results.txt", index=False, sep='\t')
    print("✅ Optimización guardada en 'optimization_results.txt'")
    sys.stdout.flush()
    return best

# ============================================================
# TOP SEÑALES
# ============================================================
def analyze_top_signals(signals_df, best_params):
    if signals_df.empty:
        print("No hay señales para analizar.")
        return
    long_signals = signals_df[signals_df['Direction'] == 'LONG'].sort_values('score', ascending=False).head(3)
    short_signals = signals_df[signals_df['Direction'] == 'SHORT'].sort_values('score', ascending=False).head(3)
    top_signals = pd.concat([long_signals, short_signals])

    print("\n📊 TOP 3 SEÑALES LONG Y TOP 3 SHORT")
    for _, sig in top_signals.iterrows():
        print(f"{sig['Symbol']} {sig['TF']} {sig['Direction']} | Score: {sig['score']:.6f} | Tensión: {sig['tension']:.4f}")
    sys.stdout.flush()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🚀 SISTEMA INTEGRADO: ESCÁNER + OPTIMIZACIÓN + TOP SEÑALES")
    sys.stdout.flush()
    signals = run_scanner()
    best_params = optimize_parameters(signals)
    if best_params is None and not signals.empty:
        # Si no hay resultados de optimización, usamos valores por defecto solo para top señales
        best_params = {'tp_atr': 3, 'sl_atr': 2, 'atr_window': 14, 'pidelta_window': 13, 'tension_quantile': 0.8}
    analyze_top_signals(signals, best_params if best_params else {})
    print("\n✅ Proceso completado. Archivos: 'escaneo_filtrado.txt', 'optimization_results.txt'")
    sys.stdout.flush()
