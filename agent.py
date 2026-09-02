import os
import sys
import json
import re
import time
import datetime
import pandas as pd
import ta
import yfinance as yf
import ccxt
from groq import Groq
from supabase import create_client, Client

# ==========================================
# 1. WALIDACJA ZMIENNYCH ŚRODOWISKOWYCH
# ==========================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

brakujace = [nazwa for nazwa, wartosc in [
    ("GROQ_API_KEY", GROQ_API_KEY),
    ("SUPABASE_URL", SUPABASE_URL),
    ("SUPABASE_KEY", SUPABASE_KEY),
] if not wartosc]

if brakujace:
    print(f"BŁĄD: brak wymaganych zmiennych środowiskowych: {', '.join(brakujace)}")
    sys.exit(1)

groq_client = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
kraken = ccxt.kraken()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
OPOZNIENIE_GROQ = float(os.environ.get("OPOZNIENIE_GROQ", "1.2"))
OPOZNIENIE_YFINANCE = float(os.environ.get("OPOZNIENIE_YFINANCE", "0.5"))
MAX_PROBY_GROQ = 3

# ==========================================
# 2. STRATEGIE TECHNICZNE — KONFIGURACJA
#    Włącz/wyłącz dowolną strategię ustawiając True/False.
#    WAGI muszą sumować się do 100 przy wszystkich włączonych
#    (jeśli wyłączysz część, reszta i tak przeskaluje się poprawnie).
# ==========================================
STRATEGIE = {
    "ema_crossover": True,   # trend: szybka EMA vs wolna EMA
    "macd": True,            # momentum trendu
    "rsi": True,             # wykupienie/wyprzedanie
    "bollinger": True,       # skrajności ceny względem zmienności
    "wolumen": True,         # potwierdzenie ruchu wolumenem
    "adx": True,             # siła trendu (+DI/-DI)
    "stochastic": True,      # oscylator stochastyczny
    "williams_r": True,      # wykupienie/wyprzedanie (Williams %R)
    "cci": True,             # Commodity Channel Index
    "mfi": True,             # Money Flow Index (RSI ważony wolumenem)
    "obv": True,             # On-Balance Volume (akumulacja/dystrybucja)
    "psar": True,            # Parabolic SAR (kierunek trendu)
}

WAGI = {
    "ema_crossover": 12,
    "macd": 12,
    "rsi": 10,
    "bollinger": 8,
    "wolumen": 8,
    "adx": 8,
    "stochastic": 8,
    "williams_r": 6,
    "cci": 6,
    "mfi": 8,
    "obv": 6,
    "psar": 8,
}

EMA_SZYBKA_OKRES = 20
EMA_WOLNA_OKRES = 50
RSI_OKRES = 14
BOLLINGER_OKRES = 20
ADX_OKRES = 14
STOCH_OKRES = 14
WILLIAMS_OKRES = 14
CCI_OKRES = 20
MFI_OKRES = 14
OBV_SMA_OKRES = 20

# Minimalna liczba świec potrzebna do policzenia wszystkich wskaźników bez błędów
MIN_SWIEC = max(EMA_WOLNA_OKRES, ADX_OKRES, BOLLINGER_OKRES) + 20

# Próg "technical score" (0-100) poniżej którego walor NIE trafia do AI.
PROG_TECHNICZNY = int(os.environ.get("PROG_TECHNICZNY", "55"))

# Próg pewności AI (0-100), od którego generujemy sygnał KUP
PROG_PEWNOSCI_AI = int(os.environ.get("PROG_PEWNOSCI_AI", "80"))

# ==========================================
# 3. LISTY WALORÓW DO SKANOWANIA
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
# 4. WSKAŹNIKI TECHNICZNE (biblioteka `ta`)
# ==========================================
def oblicz_wskazniki(df):
    """
    Przyjmuje DataFrame z kolumnami: open, high, low, close, volume.
    Dokłada kolumny wszystkich wskaźników używanych przez strategie.
    """
    df = df.copy()

    # --- Trend ---
    df["ema_szybka"] = ta.trend.ema_indicator(df["close"], window=EMA_SZYBKA_OKRES)
    df["ema_wolna"] = ta.trend.ema_indicator(df["close"], window=EMA_WOLNA_OKRES)

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=ADX_OKRES)
    df["adx"] = adx_ind.adx()
    df["adx_pos"] = adx_ind.adx_pos()
    df["adx_neg"] = adx_ind.adx_neg()

    psar_ind = ta.trend.PSARIndicator(df["high"], df["low"], df["close"])
    df["psar"] = psar_ind.psar()

    # --- Momentum ---
    df["rsi"] = ta.momentum.rsi(df["close"], window=RSI_OKRES)

    stoch_ind = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=STOCH_OKRES)
    df["stoch_k"] = stoch_ind.stoch()
    df["stoch_d"] = stoch_ind.stoch_signal()

    df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=WILLIAMS_OKRES)

    cci_ind = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=CCI_OKRES)
    df["cci"] = cci_ind.cci()

    # --- Zmienność ---
    bb = ta.volatility.BollingerBands(df["close"], window=BOLLINGER_OKRES)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    # --- Wolumen ---
    avg_vol = df["volume"].rolling(10).mean()
    df["vol_ratio"] = df["volume"] / avg_vol

    df["mfi"] = ta.volume.MFIIndicator(
        df["high"], df["low"], df["close"], df["volume"], window=MFI_OKRES
    ).money_flow_index()

    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["obv_sma"] = df["obv"].rolling(OBV_SMA_OKRES).mean()

    return df


def oblicz_technical_score(wiersz):
    """
    Liczy złożony wynik 0-100 na podstawie wszystkich aktywnych strategii.
    Zwraca (wynik_procentowy, lista_opisow) - opisy trafiają do promptu AI.
    """
    punkty = 0.0
    maks_mozliwych = 0.0
    opisy = []

    def aktywna(klucz):
        return STRATEGIE.get(klucz, False)

    if aktywna("ema_crossover"):
        w = WAGI["ema_crossover"]
        maks_mozliwych += w
        if pd.notna(wiersz["ema_szybka"]) and pd.notna(wiersz["ema_wolna"]):
            if wiersz["ema_szybka"] > wiersz["ema_wolna"]:
                punkty += w
                opisy.append(f"EMA{EMA_SZYBKA_OKRES} powyżej EMA{EMA_WOLNA_OKRES} - trend wzrostowy")
            else:
                opisy.append(f"EMA{EMA_SZYBKA_OKRES} poniżej EMA{EMA_WOLNA_OKRES} - trend spadkowy")

    if aktywna("macd"):
        w = WAGI["macd"]
        maks_mozliwych += w
        if pd.notna(wiersz["macd"]) and pd.notna(wiersz["macd_signal"]):
            if wiersz["macd"] > wiersz["macd_signal"] and wiersz["macd_hist"] > 0:
                punkty += w
                opisy.append("MACD powyżej linii sygnałowej - momentum wzrostowe")
            else:
                opisy.append("MACD poniżej linii sygnałowej - brak momentum wzrostowego")

    if aktywna("rsi"):
        w = WAGI["rsi"]
        maks_mozliwych += w
        rsi_val = wiersz["rsi"]
        if pd.notna(rsi_val):
            if 30 <= rsi_val <= 65:
                punkty += w
                opisy.append(f"RSI {rsi_val:.1f} w zdrowym zakresie")
            elif rsi_val < 30:
                punkty += w * 0.5
                opisy.append(f"RSI {rsi_val:.1f} - wyprzedanie, możliwe odbicie")
            else:
                opisy.append(f"RSI {rsi_val:.1f} - wykupienie, ryzyko korekty")

    if aktywna("bollinger"):
        w = WAGI["bollinger"]
        maks_mozliwych += w
        if pd.notna(wiersz["bb_low"]) and pd.notna(wiersz["bb_high"]):
            if wiersz["close"] <= wiersz["bb_low"] * 1.02:
                punkty += w
                opisy.append("Cena blisko dolnej wstęgi Bollingera - potencjalne odbicie")
            elif wiersz["close"] >= wiersz["bb_high"]:
                opisy.append("Cena powyżej górnej wstęgi Bollingera - wykupienie")
            else:
                punkty += w * 0.5
                opisy.append("Cena w środkowym zakresie wstęg Bollingera")

    if aktywna("wolumen"):
        w = WAGI["wolumen"]
        maks_mozliwych += w
        vol_ratio = wiersz.get("vol_ratio")
        if pd.notna(vol_ratio):
            if vol_ratio > 1.3:
                punkty += w
                opisy.append(f"Wolumen {vol_ratio:.2f}x średniej z 10 dni - potwierdzenie ruchu")
            else:
                opisy.append(f"Wolumen {vol_ratio:.2f}x średniej - brak wyraźnego potwierdzenia")

    if aktywna("adx"):
        w = WAGI["adx"]
        maks_mozliwych += w
        adx_val = wiersz.get("adx")
        if pd.notna(adx_val):
            if adx_val > 25 and wiersz["adx_pos"] > wiersz["adx_neg"]:
                punkty += w
                opisy.append(f"ADX {adx_val:.1f} - silny trend wzrostowy potwierdzony")
            elif adx_val > 25:
                opisy.append(f"ADX {adx_val:.1f} - silny trend, ale spadkowy")
            else:
                punkty += w * 0.4
                opisy.append(f"ADX {adx_val:.1f} - trend słaby/boczny")

    if aktywna("stochastic"):
        w = WAGI["stochastic"]
        maks_mozliwych += w
        k, d = wiersz.get("stoch_k"), wiersz.get("stoch_d")
        if pd.notna(k) and pd.notna(d):
            if k < 20:
                punkty += w
                opisy.append(f"Stochastic {k:.1f} - wyprzedanie, możliwe odbicie")
            elif k > d and k < 80:
                punkty += w * 0.7
                opisy.append(f"Stochastic %K powyżej %D ({k:.1f}) - momentum wzrostowe")
            elif k > 80:
                opisy.append(f"Stochastic {k:.1f} - wykupienie")
            else:
                opisy.append(f"Stochastic {k:.1f} - neutralnie")

    if aktywna("williams_r"):
        w = WAGI["williams_r"]
        maks_mozliwych += w
        wr = wiersz.get("williams_r")
        if pd.notna(wr):
            if wr < -80:
                punkty += w
                opisy.append(f"Williams %R {wr:.1f} - wyprzedanie")
            elif wr > -20:
                opisy.append(f"Williams %R {wr:.1f} - wykupienie")
            else:
                punkty += w * 0.5
                opisy.append(f"Williams %R {wr:.1f} - neutralnie")

    if aktywna("cci"):
        w = WAGI["cci"]
        maks_mozliwych += w
        cci_val = wiersz.get("cci")
        if pd.notna(cci_val):
            if -100 <= cci_val <= 100:
                punkty += w * 0.6
                opisy.append(f"CCI {cci_val:.1f} - zakres neutralny")
            elif cci_val < -100:
                punkty += w
                opisy.append(f"CCI {cci_val:.1f} - wyprzedanie, możliwe odbicie")
            else:
                opisy.append(f"CCI {cci_val:.1f} - wykupienie, ryzyko korekty")

    if aktywna("mfi"):
        w = WAGI["mfi"]
        maks_mozliwych += w
        mfi_val = wiersz.get("mfi")
        if pd.notna(mfi_val):
            if mfi_val < 20:
                punkty += w
                opisy.append(f"MFI {mfi_val:.1f} - wyprzedanie wolumenowe")
            elif mfi_val > 80:
                opisy.append(f"MFI {mfi_val:.1f} - wykupienie wolumenowe")
            else:
                punkty += w * 0.6
                opisy.append(f"MFI {mfi_val:.1f} - przepływ kapitału neutralny")

    if aktywna("obv"):
        w = WAGI["obv"]
        maks_mozliwych += w
        obv_val, obv_sma = wiersz.get("obv"), wiersz.get("obv_sma")
        if pd.notna(obv_val) and pd.notna(obv_sma):
            if obv_val > obv_sma:
                punkty += w
                opisy.append("OBV powyżej średniej - akumulacja (kapitał napływa)")
            else:
                opisy.append("OBV poniżej średniej - dystrybucja (kapitał odpływa)")

    if aktywna("psar"):
        w = WAGI["psar"]
        maks_mozliwych += w
        psar_val = wiersz.get("psar")
        if pd.notna(psar_val):
            if wiersz["close"] > psar_val:
                punkty += w
                opisy.append("Cena powyżej Parabolic SAR - trend wzrostowy")
            else:
                opisy.append("Cena poniżej Parabolic SAR - trend spadkowy")

    wynik = round((punkty / maks_mozliwych) * 100) if maks_mozliwych else 0
    return wynik, opisy


# ==========================================
# 5. POBIERANIE DANYCH RYNKOWYCH
# ==========================================
def pobierz_dane_akcji(ticker):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y", interval="1d")
        if hist.empty or len(hist) < MIN_SWIEC:
            print(f"[{ticker}] Za mało danych historycznych, pomijam.")
            return None

        df = pd.DataFrame({
            "open": hist["Open"],
            "high": hist["High"],
            "low": hist["Low"],
            "close": hist["Close"],
            "volume": hist["Volume"],
        })
        df = oblicz_wskazniki(df)
        ostatni = df.iloc[-1]

        wynik_techniczny, opisy_strategii = oblicz_technical_score(ostatni)

        newsy = []
        try:
            news_items = tk.news
            if news_items:
                for item in news_items[:3]:
                    title = item.get("title") or item.get("content", {}).get("title", "")
                    if title:
                        newsy.append(title)
        except Exception as e:
            print(f"[{ticker}] Nie udało się pobrać newsów: {e}")

        return {
            "ticker": ticker,
            "cena": float(ostatni["close"]),
            "rsi": float(ostatni["rsi"]) if pd.notna(ostatni["rsi"]) else 50.0,
            "ema50": float(ostatni["ema_wolna"]) if pd.notna(ostatni["ema_wolna"]) else float(ostatni["close"]),
            "newsy": " | ".join(newsy) if newsy else "Brak kluczowych newsów w tej chwili.",
            "wynik_techniczny": wynik_techniczny,
            "opisy_strategii": opisy_strategii,
        }
    except Exception as e:
        print(f"Błąd przetwarzania akcji {ticker}: {e}")
        return None


def pobierz_dane_krypto(symbol):
    try:
        ohlcv = kraken.fetch_ohlcv(symbol, timeframe="1d", limit=200)
        if not ohlcv or len(ohlcv) < MIN_SWIEC:
            print(f"[{symbol}] Za mało danych z Kraken, pomijam.")
            return None

        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df = oblicz_wskazniki(df)
        ostatni = df.iloc[-1]

        wynik_techniczny, opisy_strategii = oblicz_technical_score(ostatni)

        return {
            "ticker": symbol,
            "cena": float(ostatni["close"]),
            "rsi": float(ostatni["rsi"]) if pd.notna(ostatni["rsi"]) else 50.0,
            "ema50": float(ostatni["ema_wolna"]) if pd.notna(ostatni["ema_wolna"]) else float(ostatni["close"]),
            "newsy": "Silna zmienność rynkowa krypto, obserwacja przepływów kapitału.",
            "wynik_techniczny": wynik_techniczny,
            "opisy_strategii": opisy_strategii,
        }
    except ccxt.BadSymbol:
        print(f"[{symbol}] Para niedostępna na Kraken, pomijam.")
        return None
    except Exception as e:
        print(f"Błąd przetwarzania krypto {symbol}: {e}")
        return None


# ==========================================
# 6. MODUŁ AI (GROQ)
# ==========================================
def zapytaj_ai(dane):
    opisy_tekst = "\n".join(f"- {o}" for o in dane["opisy_strategii"])

    prompt = f"""
Jesteś ekspertem analizy finansowej i rynkowej. Przeanalizuj dane dla waloru {dane['ticker']}:

- Aktualna cena: {dane['cena']}
- Wskaźnik RSI: {dane['rsi']:.2f}
- EMA{EMA_WOLNA_OKRES}: {dane['ema50']:.2f}
- Najnowsze nagłówki newsowe: {dane['newsy']}
- Wynik analizy technicznej (0-100, złożony z {sum(1 for k in STRATEGIE if STRATEGIE[k])} strategii): {dane['wynik_techniczny']}
- Szczegóły poszczególnych strategii:
{opisy_tekst}

Ten walor przeszedł już wstępny filtr techniczny (próg {PROG_TECHNICZNY}/100), więc dane wskazują na pewien potencjał,
ale to Ty podejmujesz ostateczną decyzję inwestycyjną na podstawie całości kontekstu, w tym ewentualnych sprzecznych sygnałów
między strategiami (np. trend wzrostowy przy jednoczesnym wykupieniu).

Zwróć WYŁĄCZNIE czysty obiekt JSON bez żadnego tekstu przed ani po obiekcie i bez znaczników markdown. Format JSON:
{{
  "decyzja": "KUP" lub "CZEKAJ",
  "pewnosc": liczba_od_0_do_100,
  "entry_price": liczba_float,
  "stop_loss": liczba_float,
  "take_profit": liczba_float,
  "reasoning": "Zwięzła analiza po polsku (maksymalnie 2 zdania)"
}}
"""

    ostatni_blad = None
    for proba in range(1, MAX_PROBY_GROQ + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            raw_text = response.choices[0].message.content.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text).strip()

            wynik = json.loads(raw_text)
            if "decyzja" not in wynik or "pewnosc" not in wynik:
                raise ValueError(f"Odpowiedź AI nie zawiera wymaganych pól: {wynik}")

            wynik.setdefault("entry_price", dane["cena"])
            return wynik

        except json.JSONDecodeError as e:
            ostatni_blad = e
            print(f"[{dane['ticker']}] Próba {proba}/{MAX_PROBY_GROQ}: AI zwróciło niepoprawny JSON: {e}")
        except Exception as e:
            ostatni_blad = e
            print(f"[{dane['ticker']}] Próba {proba}/{MAX_PROBY_GROQ}: błąd zapytania do Groq: {e}")

        time.sleep(OPOZNIENIE_GROQ * proba)

    return {
        "decyzja": "BLAD_API",
        "pewnosc": 0,
        "reasoning": f"Nie udało się uzyskać odpowiedzi od AI: {ostatni_blad}"
    }


# ==========================================
# 7. GŁÓWNA PĘTLA BOTA I ZAPIS DO SUPABASE
# ==========================================
def przetworz_walor(dane):
    if dane["wynik_techniczny"] < PROG_TECHNICZNY:
        return None, False, True

    analiza = zapytaj_ai(dane)
    time.sleep(OPOZNIENIE_GROQ)

    if analiza.get("decyzja") == "BLAD_API":
        return None, True, False

    if analiza.get("decyzja") == "KUP" and analiza.get("pewnosc", 0) >= PROG_PEWNOSCI_AI:
        sygnal = {
            "ticker": dane["ticker"],
            "confidence": analiza.get("pewnosc"),
            "entry_price": analiza.get("entry_price", dane["cena"]),
            "stop_loss": analiza.get("stop_loss"),
            "take_profit": analiza.get("take_profit"),
            "reasoning": analiza.get("reasoning"),
            "technical_score": dane["wynik_techniczny"],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        return sygnal, False, False

    return None, False, False


def main():
    aktywne = [k for k, v in STRATEGIE.items() if v]
    print(f"Rozpoczynam skanowanie: {len(LISTA_AKCJI)} akcji oraz {len(LISTA_KRYPTO)} kryptowalut...")
    print(f"Model AI: {GROQ_MODEL} | Próg techniczny: {PROG_TECHNICZNY} | Próg pewności AI: {PROG_PEWNOSCI_AI}")
    print(f"Aktywne strategie ({len(aktywne)}): {aktywne}")

    wykryte_sygnaly = []
    liczba_bledow_api = 0
    liczba_pominietych = 0

    for ticker in LISTA_AKCJI:
        dane = pobierz_dane_akcji(ticker)
        time.sleep(OPOZNIENIE_YFINANCE)
        if dane:
            sygnal, byl_blad, pominiety = przetworz_walor(dane)
            if byl_blad:
                liczba_bledow_api += 1
            if pominiety:
                liczba_pominietych += 1
            if sygnal:
                wykryte_sygnaly.append(sygnal)

    for symbol in LISTA_KRYPTO:
        dane = pobierz_dane_krypto(symbol)
        time.sleep(OPOZNIENIE_YFINANCE)
        if dane:
            sygnal, byl_blad, pominiety = przetworz_walor(dane)
            if byl_blad:
                liczba_bledow_api += 1
            if pominiety:
                liczba_pominietych += 1
            if sygnal:
                wykryte_sygnaly.append(sygnal)

    if wykryte_sygnaly:
        print(f"Wykryto {len(wykryte_sygnaly)} wysokoprawdopodobnych sygnałów! Zapisuję w Supabase...")
        try:
            supabase.table("signals").insert(wykryte_sygnaly).execute()
            print("Sygnały zostały pomyślnie dodane do bazy danych.")
        except Exception as e:
            print(f"Błąd zapisu do Supabase: {e}")
    else:
        print(f"Skanowanie zakończone. Brak sygnałów spełniających progi (techniczny={PROG_TECHNICZNY}, AI={PROG_PEWNOSCI_AI}).")

    print(f"Pominiętych przez filtr techniczny (oszczędzone zapytania do AI): {liczba_pominietych}")
    if liczba_bledow_api > 0:
        print(f"UWAGA: {liczba_bledow_api} walorów nie udało się ocenić z powodu błędów API.")


if __name__ == "__main__":
    main()
