import os
import json
import requests
import pandas as pd
import yfinance as yf
import ccxt
import feedparser
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from groq import Groq
from supabase import create_client, Client

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
NTFY_CHANNEL = os.environ.get("NTFY_CHANNEL")

groq_client = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

KRYPTO = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
AKCJE = ['NVDA', 'TSLA', 'AAPL', 'AMD']

def pobierz_wiadomosci_i_smc(ticker):
    clean_ticker = ticker.split('/')[0]
    url_news = f"https://news.google.com/rss/search?q={clean_ticker}+when:1d&hl=pl&gl=PL&ceid=PL:pl"
    feed_news = feedparser.parse(url_news)
    naglowki = [entry.title for entry in feed_news.entries[:3]]
    
    query_whales = f"{clean_ticker}+(investor+OR+whale+OR+fund+OR+bought+OR+SEC+13F)"
    url_whales = f"https://news.google.com/rss/search?q={query_whales}+when:3d&hl=en&gl=US&ceid=US:en"
    feed_whales = feedparser.parse(url_whales)
    ruch_duzych_graczy = [entry.title for entry in feed_whales.entries[:3]]
    
    return " | ".join(naglowki) if naglowki else "Brak", " | ".join(ruch_duzych_graczy) if ruch_duzych_graczy else "Brak"

def pobierz_dane_krypto(symbol):
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return oblicz_wskaźniki(df)

def pobierz_dane_akcji(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1mo", interval="1h")
    df.reset_index(inplace=True)
    df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low'}, inplace=True)
    return oblicz_wskaźniki(df)

def oblicz_wskaźniki(df):
    df['ema200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    cena, rsi, ema = df['close'].iloc[-1], df['rsi'].iloc[-1], df['ema200'].iloc[-1]
    return (cena > ema) and (rsi < 68), cena, rsi, ema

def zapytaj_ai(ticker, cena, rsi, ema, newsy, ruchy_graczy):
    prompt = f"""
    Jesteś traderem Smart Money Concepts / Minervini. Przeanalizuj {ticker}:
    - Cena: {cena}, RSI: {rsi:.2f}, 200 EMA: {ema:.2f}
    - Newsy: "{newsy}"
    - Duzi gracze / Fundusze: "{ruchy_graczy}"

    Czy to okazja na KUPNO (1h - 7 dni)? Wyznacz SL i TP (RRR min. 1:2.5).
    Zwróć WYŁĄCZNIE JSON:
    {{
        "decyzja": "KUP" lub "CZEKAJ",
        "pewnosc": liczba_0_100,
        "stop_loss": cena,
        "take_profit": cena,
        "duzi_gracze_info": "1 zdanie co robią fundusze/wieloryby",
        "uzasadnienie": "Max 2 zdania po polsku"
    }}
    """
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def wyslij_powiadomienie(ticker, pewnosc, cena, sl, tp, uzasadnienie, duzi_gracze):
    tekst = f" Cena: {cena}\n SL: {sl} | TP: {tp}\n Pewność: {pewnosc}%\n Duzi gracze: {duzi_gracze}\n Powód: {uzasadnienie}"
    requests.post(f"https://ntfy.sh/{NTFY_CHANNEL}", data=tekst.encode('utf-8'), headers={"Title": f" OKAZJA: KUP {ticker}"})

def uruchom_skaner():
    wszystkie = [(t, 'krypto') for t in KRYPTO] + [(s, 'akcje') for s in AKCJE]
    for ticker, typ in wszystkie:
        try:
            szansa, cena, rsi, ema = pobierz_dane_krypto(ticker) if typ == 'krypto' else pobierz_dane_akcji(ticker)
            if szansa:
                newsy, ruchy_graczy = pobierz_wiadomosci_i_smc(ticker)
                analiza = zapytaj_ai(ticker, cena, rsi, ema, newsy, ruchy_graczy)
                if analiza.get("decyzja") == "KUP" and analiza.get("pewnosc", 0) >= 80:
                    supabase.table("signals").insert({
                        "ticker": ticker, "action": "KUP", "confidence": analiza["pewnosc"],
                        "entry_price": cena, "stop_loss": analiza.get("stop_loss"),
                        "take_profit": analiza.get("take_profit"),
                        "reasoning": f"[{analiza.get('duzi_gracze_info')}] {analiza.get('uzasadnienie')}"
                    }).execute()
                    wyslij_powiadomienie(ticker, analiza["pewnosc"], cena, analiza.get("stop_loss"), analiza.get("take_profit"), analiza.get("uzasadnienie"), analiza.get("duzi_gracze_info"))
        except Exception as e:
            print(f"Błąd {ticker}: {e}")

if __name__ == "__main__":
    uruchom_skaner()
