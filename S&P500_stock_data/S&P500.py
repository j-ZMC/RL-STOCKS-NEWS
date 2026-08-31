import yfinance as yf
import pandas as pd
import sqlite3
import time

# 1. Obtener la lista de las 500 empresas directamente desde un CSV actualizado sin usar Wikipedia
def obtener_tickers_sp500():
    print("Obteniendo lista de las 500 empresas principales...")
    # URL pública con la lista actualizada del S&P 500
    url = "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/sp500.csv"
    try:
        df = pd.read_csv(url)
        # Adaptar tickers con punto (ej: BRK.B -> BRK-B) para yfinance
        tickers = [str(symbol).replace('.', '-') for symbol in df['symbol'].tolist()]
        print(f"✓ Se obtuvieron {len(tickers)} tickers exitosamente.")
        return tickers
    except Exception as e:
        print(f"[X] Error al obtener el listado de tickers: {e}")
        return []

# 2. Indicadores Técnicos
def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calcular_macd(series, fast=12, slow=26):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

# --- EJECUCIÓN PRINCIPAL ---

tickers_500 = obtener_tickers_sp500()

if tickers_500:
    # Base de datos SQLite
    db_name = "sp500_siglo21.db"
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS market_data")
    conn.commit()

    # Parámetros para procesar las 500 empresas en lotes eficientes
    TAMANO_LOTE = 50
    PAUSA_SEGUNDOS = 1.0
    lotes = [tickers_500[i:i + TAMANO_LOTE] for i in range(0, len(tickers_500), TAMANO_LOTE)]

    print(f"\nProcesando {len(tickers_500)} empresas en {len(lotes)} lotes...\n")

    todos_los_dataframes = []

    for idx, lote in enumerate(lotes, start=1):
        print(f"--- Procesando Lote {idx}/{len(lotes)} ({len(lote)} empresas) ---")
        try:
            # Descarga grupal masiva desde yfinance
            datos_lote = yf.download(
                tickers=lote,
                start="2000-01-01",
                group_by="ticker",
                threads=True,
                progress=False
            )
            
            filas_lote = []

            for ticker in lote:
                try:
                    # Validar si el ticker existe en la respuesta
                    if ticker not in datos_lote.columns.levels[0]:
                        continue

                    df = datos_lote[ticker].copy().dropna(subset=['Close'])

                    if df.empty:
                        continue

                    # Aplanar MultiIndex de columnas si existe
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    # Indicadores
                    df['rsi'] = calcular_rsi(df['Close'])
                    df['macd'] = calcular_macd(df['Close'])

                    # Formato de salida
                    df = df.reset_index()
                    df['ticker'] = ticker

                    df = df.rename(columns={
                        'Date': 'timestamp',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })

                    df_limpio = df[['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']]
                    filas_lote.append(df_limpio)

                except Exception as e:
                    print(f"  [!] Error formateando {ticker}: {e}")

            if filas_lote:
                df_final_lote = pd.concat(filas_lote, ignore_index=True).dropna()
                
                # Insertar en base de datos SQLite gradualmente
                df_final_lote.to_sql("market_data", conn, if_exists="append", index=False)
                todos_los_dataframes.append(df_final_lote)
                
                print(f"  [✓] Lote {idx} guardado en la BD ({len(df_final_lote)} filas).")

        except Exception as e:
            print(f"  [X] Falló la descarga del Lote {idx}: {e}")

        time.sleep(PAUSA_SEGUNDOS)

    # Exportar el acumulado total a un archivo CSV
    if todos_los_dataframes:
        print("\nGenerando archivo CSV consolidado...")
        df_completo = pd.concat(todos_los_dataframes, ignore_index=True)
        archivo_csv = "sp500_market_data.csv"
        df_completo.to_csv(archivo_csv, index=False)
        print(f"[✓] Éxito: Exportado archivo CSV '{archivo_csv}' con {len(df_completo)} registros.")

    conn.close()
    print(f"\n¡Listo! Base de datos '{db_name}' y archivo CSV creados correctamente.")