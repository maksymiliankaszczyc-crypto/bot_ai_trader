import os
import json
import re
import datetime
import yfinance as yf
import ccxt
from groq import Groq
from supabase import create_client, Client

# ==========================================
# 1. INICJALIZACJA KLIENTÓW API
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Kraken CCXT dla rynku krypto
kraken = ccxt.kraken()

# ==========================================
# 2. LISTY WALORÓW DO SKANOWANIA
# ==========================================
LISTA_AKCJI = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "PLTR",
    "NFLX", "COIN", "DIS", "BA", "BAC", "JPM", "V", "MA", "PYPL", "ORCL",
    "U", "HOOD", "RBLX", "SHOP", "NET", "SNOW", "SPOT", "UBER", "ABNB", "MARA",
    "RIOT", "CLSK", "MSTR", "GLD", "SLV", "USO", "UNG", "QQQ", "SPY", "IWM"
]

LISTA_KRYPTO = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "DOGE/USD", "AVAX/USD",
    "LINK/USD", "DOT/USD", "SUI/USD", "NEAR/USD", "APT/USD", "LTC/USD", "BCH/USD",
    "UNI/USD", "ATOM/USD", "POL/USD", "FIL/USD", "SHIB/USD", "PEPE/USD", "FET/USD"
]

# ==========================================
# 3. POBIERANIE DANYCH RYNKOWYCH I WSKAŹNIKÓW
# ==========================================
def oblicz_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def oblicz_ema(prices, period=50):
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price * k) + (ema * (1 - k))
    return ema

def pobierz_dane_akcji(ticker):
    try:
        data = yf.Ticker(ticker).history(period="3mo", interval="1d")
        if data.empty or len(data) < 15:
            return None
        prices = data["Close"].tolist()
        volumes = data["Volume"].tolist()
        cena = prices[-1]
        rsi = oblicz_rsi(prices)
        ema50 = oblicz_ema(prices, 50)
        
        # Pobieranie ostatnich wiadomości
        news_items = yf.Ticker(ticker).news
        newsy = []
        if news_items:
            for item in news_items[:3]:
                title = item.get("title") or item.get("content", {}).get("title", "")
                if title:
                    newsy.append(title)
        
        avg_vol = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else volumes[-1]
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        smart_money = f"Wolumen dzisiejszy wynosi {vol_ratio:.2f}x średniego wolumenu z 10 dni."

        return {
            "ticker": ticker,
            "cena": cena,
            "rsi": rsi,
            "ema50": ema50,
            "newsy": " | ".join(newsy) if newsy else "Brak kluczowych newsów w tej chwili.",
            "ruchy_graczy": smart_money
        }
    except Exception as e:
        print(f"Błąd przetwarzania akcji {ticker}: {e}")
        return None

def pobierz_dane_krypto(symbol):
    try:
        ohlcv = kraken.fetch_ohlcv(symbol, timeframe='1d', limit=60)
        if not ohlcv or len(ohlcv) < 15:
            return None
        prices = [x[4] for x in ohlcv]
        volumes = [x[5] for x in ohlcv]
        cena = prices[-1]
        rsi = oblicz_rsi(prices)
        ema50 = oblicz_ema(prices, 50)

        avg_vol = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else volumes[-1]
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        smart_money = f"Wolumen 24h wynosi {vol_ratio:.2f}x średniej z 10 dni."

        return {
            "ticker": symbol,
            "cena": cena,
            "rsi": rsi,
            "ema50": ema50,
            "newsy": "Silna zmienność rynkowa krypto, obserwacja przepływów kapitału.",
            "ruchy_graczy": smart_money
        }
    except Exception as e:
        print(f"Błąd przetwarzania krypto {symbol}: {e}")
        return None

# ==========================================
# 4. MODUŁ AI (GROQ / LLAMA 3.3 70B)
# ==========================================
def zapytaj_ai(ticker, cena, rsi, ema, newsy, ruchy_graczy):
    prompt = f"""
Jesteś ekspertem analizy finansowej i rynkowej. Przeanalizuj podane dane dla waloru {ticker}:
- Aktualna cena: {cena}
- Wskaźnik RSI: {rsi:.2f}
- EMA 50: {ema:.2f}
- Najnowsze nagłówki newsowe: {newsy}
- Analiza Smart Money / Duzi gracze: {ruchy_graczy}

Twoim zadaniem jest podjąć decyzję inwestycyjną.
Zwróć WYŁĄCZNIE czysty obiekt JSON bez żadnego tekstu przed ani po obiekcie i bez znaczników markdown. Format JSON:
{{
  "decyzja": "KUP" lub "CZEKAJ",
  "pewnosc": liczba_od_0_do_100,
  "entry_price": {cena},
  "stop_loss": sugerowany_sl_float,
  "take_profit": sugerowany_tp_float,
  "reasoning": "Zwięzła analiza po polsku (maksymalnie 2 zdania)"
}}
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        raw_text = response.choices[0].message.content.strip()

        # Usunięcie opcjonalnych znaczników ```json ... ```
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text).strip()

        return json.loads(raw_text)

    except Exception as e:
        print(f"Błąd zapytania do Groq API dla {ticker}: {e}")
        return {"decyzja": "CZEKAJ", "pewnosc": 0, "reasoning": "Błąd API"}

# ==========================================
# 5. GŁÓWNA PĘTLA BOTA I ZAPIS DO SUPABASE
# ==========================================
def main():
    print(f"Rozpoczynam skanowanie: {len(LISTA_AKCJI)} akcji oraz {len(LISTA_KRYPTO)} kryptowalut...")
    wykryte_sygnaly = []

    # Skanowanie Akcji
    for ticker in LISTA_AKCJI:
        dane = pobierz_dane_akcji(ticker)
        if dane:
            analiza = zapytaj_ai(dane["ticker"], dane["cena"], dane["rsi"], dane["ema50"], dane["newsy"], dane["ruchy_graczy"])
            
            # Próg pewności >= 80% do wygenerowania sygnału
            if analiza.get("decyzja") == "KUP" and analiza.get("pewnosc", 0) >= 80:
                sygnal = {
                    "ticker": dane["ticker"],
                    "confidence": analiza.get("pewnosc"),
                    "entry_price": analiza.get("entry_price", dane["cena"]),
                    "stop_loss": analiza.get("stop_loss"),
                    "take_profit": analiza.get("take_profit"),
                    "reasoning": analiza.get("reasoning"),
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                wykryte_sygnaly.append(sygnal)

    # Skanowanie Krypto
    for symbol in LISTA_KRYPTO:
        dane = pobierz_dane_krypto(symbol)
        if dane:
            analiza = zapytaj_ai(dane["ticker"], dane["cena"], dane["rsi"], dane["ema50"], dane["newsy"], dane["ruchy_graczy"])
            
            if analiza.get("decyzja") == "KUP" and analiza.get("pewnosc", 0) >= 80:
                sygnal = {
                    "ticker": dane["ticker"],
                    "confidence": analiza.get("pewnosc"),
                    "entry_price": analiza.get("entry_price", dane["cena"]),
                    "stop_loss": analiza.get("stop_loss"),
                    "take_profit": analiza.get("take_profit"),
                    "reasoning": analiza.get("reasoning"),
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                wykryte_sygnaly.append(sygnal)

    # Zapis wykrytych sygnałów do bazy Supabase
    if wykryte_sygnaly:
        print(f"Wykryto {len(wykryte_sygnaly)} wysokoprawdopodobnych sygnałów! Zapisuję w Supabase...")
        try:
            supabase.table("signals").insert(wykryte_sygnaly).execute()
            print("Sygnały zostały pomyślnie dodane do bazy danych.")
        except Exception as e:
            print(f"Błąd zapisu do Supabase: {e}")
    else:
        print("Skanowanie zakończone. Brak sygnałów spełniających próg pewności >= 80%.")

if __name__ == "__main__":
    main()
