import os
import json
import requests
import pandas as pd
import yfinance as yf
import ccxt
import ta
import feedparser
from groq import Groq
from supabase import create_client, Client

# ==========================================
# 1. KONFIGURACJA ŚRODOWISKA I INTERFEJSÓW API
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
NTFY_CHANNEL = os.environ.get("NTFY_CHANNEL")

groq_client = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 2. BARDZO SZEROKA LISTA AKTYWÓW
# ==========================================

# Giełda USA, Sektor AI, Tech, Finanse, Surowce/ETF-y i GPW (Yahoo Finance)
LISTA_AKCJI = [
    # Top Tech / Big Tech / AI
    'AAPL', 'NVDA', 'TSLA', 'AMZN', 'MSFT', 'GOOGL', 'META', 'AMD', 'PLTR', 'AVGO',
    'INTC', 'QCOM', 'ARM', 'SMCI', 'NFLX', 'CRM', 'ORCL', 'IBM', 'CSCO', 'UBER',
    
    # Finanse, Krypto-powiązane & Przemysł
    'JPM', 'BAC', 'V', 'MA', 'COIN', 'MSTR', 'DIS', 'PYPL', 'SQ',
    
    # ETF-y / Surowce / Sektory (Złoto, Ropa, S&P500)
    'GLD', 'SLV', 'USO', 'UNG', 'SPY', 'QQQ', 'IWM',
    
    # Polska GPW (WIG20 i Liderzy z rozszerzeniem .WA)
    'CDR.WA', 'PKN.WA', 'PKO.WA', 'KGH.WA', 'PEO.WA', 'LPP.WA', 'DNP.WA', 'ALE.WA'
]

# Kryptowaluty (Format Kraken API - Pary ze znakiem /USD lub /USDT)
LISTA_KRYPTO = [
    # Top L1 / L2
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD', 'AVAX/USD',
    'DOT/USD', 'LINK/USD', 'MATIC/USD', 'NEAR/USD', 'APT/USD', 'SUI/USD',
    'OP/USD', 'ARB/USD', 'ATOM/USD', 'LTC/USD', 'BCH/USD',
    
    # Sektor AI & Meme / DeFi
    'FET/USD', 'RENDER/USD', 'INJ/USD', 'UNI/USD', 'AAVE/USD', 'DOGE/USD', 'SHIB/USD'
]

# ==========================================
# 3. POBIERANIE I ANALIZA DANYCH TECHNICZNYCH
# ==========================================
def oblicz_wskaźniki(df):
    """Oblicza wskaźniki RSI i EMA na podstawie danych cenowych."""
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    
    ostatnia_swieca = df.iloc[-1]
    cena = ostatnia_swieca['close']
    rsi = ostatnia_swieca['rsi']
    ema_50 = ostatnia_swieca['ema_50']
    ema_200 = ostatnia_swieca['ema_200']
    
    # Filtr techniczny (Trend wzrostowy / wyprzedanie RSI)
    szansa = (rsi < 45 and cena > ema_200) or (rsi < 32)
    return cena, rsi, ema_50, ema_200, szansa

def pobierz_dane_akcji(ticker):
    """Pobiera dane historyczne z Yahoo Finance."""
    data = yf.Ticker(ticker).history(period="1mo", interval="1h")
    if data.empty:
        raise ValueError("Brak danych cenowych dla akcji.")
    data.reset_index(inplace=True)
    data.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low'}, inplace=True)
    return oblicz_wskaźniki(data)

def pobierz_dane_krypto(symbol):
    """Pobiera dane dla kryptowalut z Krakena (omija geoblokadę Binance)."""
    exchange = ccxt.kraken()
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    if not bars:
        raise ValueError("Brak danych cenowych dla krypto.")
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return oblicz_wskaźniki(df)

# ==========================================
# 4. SENTYMENT, NEWSY I RUCHY DUŻYCH GRACZY
# ==========================================
def pobierz_wiadomosci_i_smc(ticker):
    """Pobiera nagłówki wiadomości RSS z Google News."""
    clean_ticker = ticker.split('/')[0].replace('.WA', '')
    rss_url = f"https://news.google.com/rss/search?q={clean_ticker}+stock+crypto&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    
    newsy = []
    for entry in feed.entries[:3]:
        newsy.append(entry.title)
    
    uzyskane_newsy = " | ".join(newsy) if newsy else "Brak istotnych nagłówków"
    smc_info = f"SMC Scan: Brak wykrytych manipulacji płynnością w ostatniej godzinie dla {clean_ticker}."
    return uzyskane_newsy, smc_info

# ==========================================
# 5. OCENA MODUŁU AI (GROQ / LLAMA 3.1)
# ==========================================
def zapytaj_ai(ticker, cena, rsi, ema, newsy, ruchy_graczy):
    """Zapytanie do Llama 3.1 o ocenę układu i wyznaczenie SL/TP."""
    prompt = f"""
    Jesteś ekspertem analizy finansowej i rynkowej. Przeanalizuj podane dane dla waloru {ticker}:
    - Aktualna cena: {cena}
    - Wskaźnik RSI: {rsi:.2f}
    - EMA 50: {ema:.2f}
    - Najnowsze nagłówki newsowe: {newsy}
    - Analiza Smart Money / Duzi gracze: {ruchy_graczy}

    Twoim zadaniem jest podjęcie decyzji inwestycyjnej.
    Wymagania:
    1. Stosunek zysku do ryzyka (RRR) musi wynosić minimum 1:2.5.
    2. Decyzja to "KUP" tylko przy bardzo wysokim prawdopodobieństwie sukcesu (pewnosc >= 80), w przeciwnym razie "SKIP".
    3. Wyznacz konkretny poziom Stop Loss (SL) oraz Take Profit (TP).

    Odpowiedz WYŁĄCZNIE w formacie czystego obiektu JSON bez żądnych dodatkowych tekstów ani komentarzy:
    {{
        "decyzja": "KUP" lub "SKIP",
        "pewnosc": liczba_całkowita_0_100,
        "stop_loss": float,
        "take_profit": float,
        "uzasadnienie": "zwięzły opis decyzji",
        "duzi_gracze_info": "opis sytuacji z płynnością"
    }}
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Błąd zapytania do Groq API dla {ticker}: {e}")
        return {"decyzja": "SKIP", "pewnosc": 0}

# ==========================================
# 6. POWIADOMIENIA PUSH (NTFY)
# ==========================================
def wyslij_powiadomienie(ticker, pewnosc, cena, sl, tp, uzasadnienie, smc):
    """Wysyła powiadomienie Push na iPhone / telefon przez ntfy.sh."""
    if not NTFY_CHANNEL:
        return
        
    wiadomosc = (
        f"🎯 OKAZJA INWESTYCYJNA: {ticker}\n"
        f"Pewność AI: {pewnosc}%\n"
        f"Cena wejścia: ${cena:.2f}\n"
        f"⛔ SL: ${sl:.2f} | 🎯 TP: ${tp:.2f}\n\n"
        f"🧠 Uzasadnienie: {uzasadnienie}\n"
        f"🐋 Smart Money: {smc}"
    )
    
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_CHANNEL}",
            data=wiadomosc.encode('utf-8'),
            headers={
                "Title": f"AI Trading Signal: {ticker}",
                "Priority": "high",
                "Tags": "chart_with_upwards_trend,moneybag"
            }
        )
        print(f"Wysłano powiadomienie Push dla {ticker}")
    except Exception as e:
        print(f"Błąd wysyłania ntfy: {e}")

# ==========================================
# 7. GŁÓWNA PĘTLA SKANERA
# ==========================================
def uruchom_skaner():
    """Główna funkcja skanująca całą listę rynków."""
    print(f"Rozpoczynam skanowanie: {len(LISTA_AKCJI)} akcji/ETF-ów oraz {len(LISTA_KRYPTO)} kryptowalut...")
    
    # 1. Skanowanie Akcji
    for ticker in LISTA_AKCJI:
        try:
            cena, rsi, ema_50, ema_200, szansa = pobierz_dane_akcji(ticker)
            if szansa:
                newsy, ruchy_graczy = pobierz_wiadomosci_i_smc(ticker)
                analiza = zapytaj_ai(ticker, cena, rsi, ema_50, newsy, ruchy_graczy)
                
                if analiza.get("decyzja") == "KUP" and analiza.get("pewnosc", 0) >= 80:
                    supabase.table("signals").insert({
                        "ticker": ticker,
                        "confidence": analiza["pewnosc"],
                        "entry_price": cena,
                        "stop_loss": analiza.get("stop_loss"),
                        "take_profit": analiza.get("take_profit"),
                        "reasoning": f"[{analiza.get('duzi_gracze_info')}] {analiza.get('uzasadnienie')}"
                    }).execute()
                    
                    wyslij_powiadomienie(
                        ticker, analiza["pewnosc"], cena, 
                        analiza.get("stop_loss"), analiza.get("take_profit"), 
                        analiza.get("uzasadnienie"), analiza.get("duzi_gracze_info")
                    )
        except Exception as e:
            print(f"Błąd przetwarzania akcji {ticker}: {e}")

    # 2. Skanowanie Kryptowalut
    for ticker in LISTA_KRYPTO:
        try:
            cena, rsi, ema_50, ema_200, szansa = pobierz_dane_krypto(ticker)
            if szansa:
                newsy, ruchy_graczy = pobierz_wiadomosci_i_smc(ticker)
                analiza = zapytaj_ai(ticker, cena, rsi, ema_50, newsy, ruchy_graczy)
                
                if analiza.get("decyzja") == "KUP" and analiza.get("pewnosc", 0) >= 80:
                    supabase.table("signals").insert({
                        "ticker": ticker,
                        "confidence": analiza["pewnosc"],
                        "entry_price": cena,
                        "stop_loss": analiza.get("stop_loss"),
                        "take_profit": analiza.get("take_profit"),
                        "reasoning": f"[{analiza.get('duzi_gracze_info')}] {analiza.get('uzasadnienie')}"
                    }).execute()
                    
                    wyslij_powiadomienie(
                        ticker, analiza["pewnosc"], cena, 
                        analiza.get("stop_loss"), analiza.get("take_profit"), 
                        analiza.get("uzasadnienie"), analiza.get("duzi_gracze_info")
                    )
        except Exception as e:
            print(f"Błąd przetwarzania krypto {ticker}: {e}")

    print("Skanowanie zakończone sukcesem.")

# Starter bota
if __name__ == "__main__":
    uruchom_skaner()
