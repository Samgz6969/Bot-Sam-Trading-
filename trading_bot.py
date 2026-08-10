"""
Bot de Trading Automatico Multi-Moneda - Binance - Paper Trading
==================================================================

Estrategia: Cruce de Medias Moviles (SMA Crossover)
  - Senal de COMPRA: la media corta cruza por ENCIMA de la media larga (cruce dorado)
  - Senal de VENTA:  la media corta cruza por DEBAJO de la media larga (cruce de la muerte)

Gestion de riesgo / margen de ganancias:
  - TAKE_PROFIT_PCT: cierra la posicion automaticamente si la ganancia alcanza este %
  - STOP_LOSS_PCT:   cierra la posicion automaticamente si la perdida alcanza este %

Monedas operadas (cada una con su propio balance virtual independiente):
  BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT

Extras de esta version:
  - DATOS EN TIEMPO REAL VIA WEBSOCKET: en vez de preguntarle a Binance cada
    tantos segundos, el bot mantiene una conexion permanente (WebSocket) y
    Binance le avisa apenas hay un cambio de precio o de vela -- practicamente
    al mismo instante que ocurre en Binance, sin el retraso de las consultas
    periodicas (polling).
  - Cada moneda puede operar en su PROPIA temporalidad (5m/15m/1h/4h/1d),
    elegible en vivo desde el panel web, sin reiniciar el bot.
  - Red de seguridad: si el WebSocket llegase a fallar o desconectarse, un
    "vigilante" (watchdog) detecta la inactividad y recurre automaticamente
    a una consulta REST de respaldo, para que el bot nunca se quede ciego.
  - Un hilo de trading independiente POR MONEDA para la carga inicial de
    historial (cada una con su propia cuenta de paper trading, historial de
    precios e historial de operaciones); los ticks en vivo llegan por el
    WebSocket compartido.
  - Notificaciones a Telegram en cada compra/venta y resumen periodico.
  - Comandos manuales de Telegram (/monedas, /abiertas, /cerradas, /balance).
  - Panel web (Flask) estilo Binance: velas reales en vivo por moneda con
    marcadores de compra/venta, %cambio 24h, balance operando vs disponible,
    boveda (fondos fuera del alcance del bot) y el balance total combinado.
  - Fondo animado tipo "cyberpunk" (lluvia de codigo) puramente decorativo.
  - Endpoint JSON /api/state para que el frontend se actualice sin recargar
    la pagina.

IMPORTANTE:
  - Este bot corre en modo PAPER TRADING: no envia ordenes reales, solo simula
    con balance virtual, usando datos de mercado REALES de Binance (API
    publica, no requiere API key para leer precios).
  - No es asesoria financiera. El trading conlleva riesgo real de perdida de
    capital. Ningun bot garantiza ganancias.

Requisitos:
    pip install requests pandas flask websocket-client
"""

import time
import csv
import os
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests
import websocket
from flask import Flask, jsonify, render_template_string, request


# ------------------------------------------------------------------
# CONFIGURACION -- ajusta estos parametros a tu gusto
# ------------------------------------------------------------------
@dataclass
class Config:
    symbols: list = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    ])
    timeframe: str = "1h"           # Temporalidad de las velas: 1m, 5m, 15m, 1h, 4h, 1d...
    short_window: int = 20          # Periodo de la media movil corta
    long_window: int = 50           # Periodo de la media movil larga
    take_profit_pct: float = 0.03   # Margen de ganancia objetivo: 3%
    stop_loss_pct: float = 0.015    # Perdida maxima aceptada: 1.5%
    trade_fraction: float = 0.95    # Fraccion del balance a usar en cada compra
    fee_pct: float = 0.001          # Comision simulada de Binance (~0.1%)
    initial_balance_usdt: float = 100.0   # Balance virtual inicial POR MONEDA
    candles_lookback: int = 200     # Cuantas velas historicas descarga por REST al iniciar / respaldo
    price_history_maxlen: int = 150  # Puntos que se guardan para la grafica en vivo
    log_dir: str = "logs"

    # --- Red de seguridad (watchdog) por si el WebSocket se cae ---
    watchdog_check_seconds: int = 20    # cada cuanto revisa el vigilante si hay datos recientes
    watchdog_stale_seconds: int = 60    # si no llega nada del WebSocket en este tiempo, usa REST de respaldo

    # --- Telegram ---
    telegram_enabled: bool = True
    telegram_bot_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "PON_AQUI_TU_TOKEN")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "PON_AQUI_TU_CHAT_ID")
    telegram_summary_seconds: int = 1800  # resumen periodico por moneda (30 min)

    # --- Panel web ---
    web_enabled: bool = True
    web_host: str = "0.0.0.0"
    web_port: int = int(os.environ.get("PORT", 5000))  # Render asigna el puerto via variable PORT


cfg = Config()

# Temporalidades que el usuario puede elegir para cada moneda desde el panel web,
# igual que el selector de Binance (1D = 24 horas, es la misma vela).
ALLOWED_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
TIMEFRAME_LABELS = {"5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D (24h)"}


# ------------------------------------------------------------------
# NOTIFICADOR DE TELEGRAM
# ------------------------------------------------------------------
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, enabled: bool, log):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled and bool(token) and "PON_AQUI" not in token
        self.log = log

    def send(self, text: str):
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            requests.post(url, data={"chat_id": self.chat_id, "text": text}, timeout=10)
        except Exception as e:
            self.log.warning(f"No se pudo enviar mensaje a Telegram: {e}")


# ------------------------------------------------------------------
# COMANDOS DE TELEGRAM -- permite consultar el bot manualmente
# escribiendole por Telegram (monedas, posiciones, historial, balance).
# Usa "long polling" contra la API de Telegram (getUpdates), sin
# necesidad de configurar un webhook publico.
# ------------------------------------------------------------------
class TelegramCommandBot:
    def __init__(self, token: str, chat_id: str, enabled: bool, config: Config, log):
        self.token = token
        self.chat_id = str(chat_id)
        self.enabled = enabled and bool(token) and "PON_AQUI" not in token
        self.cfg = config
        self.log = log
        self.offset = None

    def _send(self, text: str):
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            requests.post(url, data={"chat_id": self.chat_id, "text": text}, timeout=10)
        except Exception as e:
            self.log.warning(f"No se pudo responder por Telegram: {e}")

    def poll_loop(self):
        if not self.enabled:
            return
        self.log.info("Escuchando comandos de Telegram...")
        while True:
            try:
                self._poll_once()
            except Exception as e:
                self.log.warning(f"Error en polling de comandos de Telegram: {e}")
                time.sleep(5)

    def _poll_once(self):
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        params = {"timeout": 20}
        if self.offset is not None:
            params["offset"] = self.offset
        resp = requests.get(url, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        for update in data.get("result", []):
            self.offset = update["update_id"] + 1
            msg = update.get("message") or update.get("edited_message")
            if not msg:
                continue
            if str(msg.get("chat", {}).get("id", "")) != self.chat_id:
                continue  # ignora mensajes de cualquier chat que no sea el configurado
            text = (msg.get("text") or "").strip()
            if text:
                self._handle_command(text)

    def _handle_command(self, text: str):
        cmd = text.lower().split()[0]
        try:
            if cmd in ("/monedas", "/symbols"):
                self._cmd_monedas()
            elif cmd in ("/abiertas", "/posiciones", "/open"):
                self._cmd_abiertas()
            elif cmd in ("/cerradas", "/historial", "/closed"):
                self._cmd_cerradas()
            elif cmd in ("/balance", "/saldo"):
                self._cmd_balance()
            elif cmd in ("/start", "/ayuda", "/help"):
                self._cmd_ayuda()
            else:
                self._send("No reconozco ese comando. Envia /ayuda para ver la lista completa.")
        except Exception as e:
            self.log.warning(f"Error procesando comando '{text}': {e}")
            self._send("Ocurrio un error consultando esa informacion. Intenta de nuevo en un momento.")

    def _cmd_ayuda(self):
        self._send(
            "🤖 Comandos disponibles:\n"
            "/monedas - lista las monedas activas y su precio actual\n"
            "/abiertas - operaciones abiertas ahora mismo\n"
            "/cerradas - ultimas operaciones cerradas\n"
            "/balance - balance total, operando, disponible y boveda\n"
            "/ayuda - muestra este mensaje"
        )

    def _cmd_monedas(self):
        snap = shared_state.snapshot()
        lines = ["📋 Monedas activas:"]
        for sym in self.cfg.symbols:
            d = snap["symbols"].get(sym, {})
            price = d.get("current_price", 0.0)
            pct = d.get("pnl_total_pct", 0.0)
            lines.append(f"• {sym}: ${price:,.4f}  (PnL: {pct:+.2f}%)")
        self._send("\n".join(lines))

    def _cmd_abiertas(self):
        lines = ["📈 Operaciones abiertas:"]
        found = False
        snap = shared_state.snapshot()
        for sym, bot in bots_by_symbol.items():
            with bot.account_lock:
                in_pos = bot.account.in_position
                entry_price = bot.account.entry_price
                qty = bot.account.position_qty
            if not in_pos:
                continue
            found = True
            current_price = snap["symbols"].get(sym, {}).get("current_price", entry_price)
            pnl_pct = (current_price / entry_price - 1) * 100 if entry_price else 0.0
            lines.append(
                f"• {sym}: entrada {entry_price:.4f} | actual {current_price:.4f} "
                f"| cantidad {qty:.6f} | PnL flotante {pnl_pct:+.2f}%"
            )
        if not found:
            self._send("No hay ninguna operacion abierta en este momento.")
            return
        self._send("\n".join(lines))

    def _cmd_cerradas(self):
        closed = []
        for sym, bot in bots_by_symbol.items():
            with bot.account_lock:
                history = list(bot.account.trade_history)
            for t in history:
                if str(t["action"]).startswith("VENTA"):
                    closed.append((sym, t))
        if not closed:
            self._send("Todavia no hay operaciones cerradas.")
            return
        closed.sort(key=lambda x: x[1].get("time_unix", 0), reverse=True)
        lines = ["📜 Ultimas operaciones cerradas:"]
        for sym, t in closed[:10]:
            lines.append(
                f"• {sym} | {t['action']} a {t['price']:.4f} | "
                f"PnL: {t['pnl_usdt']} USDT ({t['pnl_pct']}%)"
            )
        self._send("\n".join(lines))

    def _cmd_balance(self):
        snap = shared_state.snapshot()
        t = snap["total"]
        self._send(
            "💰 Balance general\n"
            f"Equity total (operando + disponible): ${t['equity']:.2f}\n"
            f"Dinero operando ahora: ${t['invested_usdt']:.2f}\n"
            f"Dinero disponible (billetera): ${t['available_usdt']:.2f}\n"
            f"Boveda (fuera del alcance del bot): ${t['vault_usdt']:.2f}\n"
            f"PnL total: {t['pnl_total']:+.2f} USDT ({t['pnl_total_pct']:+.2f}%)\n"
            f"Trades totales: {t['total_trades']} | Win rate: {t['win_rate_pct']:.1f}%"
        )


# ------------------------------------------------------------------
# ESTADO COMPARTIDO (para el panel web) -- protegido por un lock
# Guarda el estado de CADA moneda por separado, mas el total combinado.
# ------------------------------------------------------------------
class SharedState:
    def __init__(self, symbols, initial_balance_per_symbol: float, history_maxlen: int):
        self.lock = threading.Lock()
        self.initial_balance_per_symbol = initial_balance_per_symbol
        self.history_maxlen = history_maxlen
        self.symbols_data = {
            s: {
                "symbol": s,
                "current_price": 0.0,
                "prev_price": 0.0,
                "pct_change_24h": 0.0,
                "in_position": False,
                "entry_price": 0.0,
                "equity": initial_balance_per_symbol,
                "balance_usdt": initial_balance_per_symbol,
                "invested_usdt": 0.0,
                "pnl_total": 0.0,
                "pnl_total_pct": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "avg_pnl_per_trade": 0.0,
                "last_update": "",
                "trade_history": [],
                "candles": [],
                "status": "iniciando",
                "timeframe": cfg.timeframe,
            }
            for s in symbols
        }

    def update(self, symbol: str, **kwargs):
        with self.lock:
            self.symbols_data[symbol].update(kwargs)

    def push_price(self, symbol: str, price: float):
        with self.lock:
            d = self.symbols_data[symbol]
            d["prev_price"] = d["current_price"]
            d["current_price"] = round(float(price), 6)

    def push_candles(self, symbol: str, candles: list):
        with self.lock:
            self.symbols_data[symbol]["candles"] = candles[-self.history_maxlen:]

    def snapshot(self) -> dict:
        with self.lock:
            symbols_out = {}
            total_equity = 0.0
            total_trades = 0
            total_wins = 0
            total_invested = 0.0
            total_available = 0.0
            for s, d in self.symbols_data.items():
                sd = dict(d)
                symbols_out[s] = sd
                total_equity += d["equity"]
                total_trades += d["total_trades"]
                total_wins += d["winning_trades"]
                total_invested += d.get("invested_usdt", 0.0)
                total_available += d.get("balance_usdt", 0.0)

            total_initial = self.initial_balance_per_symbol * len(self.symbols_data)
            total_pnl = total_equity - total_initial
            total_pnl_pct = ((total_equity / total_initial) - 1) * 100 if total_initial else 0.0
            total_win_rate = (total_wins / total_trades * 100) if total_trades else 0.0

            return {
                "symbols": symbols_out,
                "total": {
                    "num_assets": len(self.symbols_data),
                    "initial_balance": round(total_initial, 2),
                    "equity": round(total_equity, 2),
                    "invested_usdt": round(total_invested, 2),
                    "available_usdt": round(total_available, 2),
                    "vault_usdt": vault.snapshot(),
                    "pnl_total": round(total_pnl, 2),
                    "pnl_total_pct": round(total_pnl_pct, 2),
                    "total_trades": total_trades,
                    "win_rate_pct": round(total_win_rate, 2),
                },
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }


shared_state = SharedState(cfg.symbols, cfg.initial_balance_usdt, cfg.price_history_maxlen)


# ------------------------------------------------------------------
# BOVEDA -- dinero que el bot NUNCA puede usar para operar.
# El usuario mueve fondos aqui manualmente desde el panel web; mientras
# esten en la boveda quedan completamente fuera del alcance del bot.
# ------------------------------------------------------------------
class Vault:
    def __init__(self):
        self.lock = threading.Lock()
        self.balance = 0.0

    def deposit(self, amount: float):
        with self.lock:
            self.balance = round(self.balance + amount, 2)

    def withdraw(self, amount: float) -> bool:
        with self.lock:
            if amount <= 0 or amount > self.balance + 1e-9:
                return False
            self.balance = round(self.balance - amount, 2)
            return True

    def snapshot(self) -> float:
        with self.lock:
            return round(self.balance, 2)


vault = Vault()
# Se llena en __main__ una vez que existen las instancias de cada bot.
# Permite que el panel web y los comandos de Telegram accedan a cada
# PaperAccount (para transferencias y consultas) por simbolo.
bots_by_symbol: dict = {}
# Traduce "BTCUSDT" (formato de Binance) -> "BTC/USDT" (formato interno).
binance_symbol_to_slash: dict = {}
# Instancia unica del feed de WebSocket compartido por todas las monedas.
# Se crea en __main__ una vez que existen los bots.
ws_feed = None


# ------------------------------------------------------------------
# FEED DE WEBSOCKET DE BINANCE -- UNA sola conexion compartida para TODAS
# las monedas (streams combinados). Binance empuja los datos apenas cambian,
# en vez de que el bot tenga que estar preguntando cada tantos segundos.
# ------------------------------------------------------------------
class BinanceWebSocketFeed:
    WS_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self, log):
        self.log = log
        self.ws = None
        self._id_lock = threading.Lock()
        self._next_req_id = 1
        self._stop = False

    def _new_id(self) -> int:
        with self._id_lock:
            req_id = self._next_req_id
            self._next_req_id += 1
            return req_id

    def _desired_streams(self) -> list:
        """Arma la lista de streams que deberiamos tener activos AHORA MISMO,
        segun la temporalidad actual de cada moneda. Se recalcula cada vez que
        el WebSocket (re)conecta, para que una reconexion siempre termine con
        exactamente las suscripciones correctas (incluidos cambios de
        temporalidad que hayan ocurrido mientras estuvo desconectado)."""
        streams = []
        for bot in bots_by_symbol.values():
            sym = bot.binance_symbol.lower()
            streams.append(f"{sym}@kline_{bot.timeframe}")
            streams.append(f"{sym}@ticker")
        return streams

    def _send(self, method: str, streams: list):
        if not streams or self.ws is None:
            return
        try:
            self.ws.send(json.dumps({"method": method, "params": streams, "id": self._new_id()}))
        except Exception as e:
            self.log.warning(f"No se pudo enviar {method} al WebSocket: {e}")

    def resubscribe_symbol_timeframe(self, old_stream: str, new_stream: str):
        """Usado cuando el usuario cambia la temporalidad de una moneda desde el panel."""
        self._send("UNSUBSCRIBE", [old_stream])
        self._send("SUBSCRIBE", [new_stream])

    def _on_open(self, ws):
        streams = self._desired_streams()
        self.log.info(f"WebSocket de Binance conectado. Suscribiendo a {len(streams)} streams...")
        self._send("SUBSCRIBE", streams)

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except Exception:
            return
        if "result" in data and "id" in data:
            return  # confirmacion de un SUBSCRIBE/UNSUBSCRIBE, no es un evento de mercado
        event_type = data.get("e")
        symbol_raw = data.get("s")
        if not event_type or not symbol_raw:
            return
        symbol = binance_symbol_to_slash.get(symbol_raw)
        bot = bots_by_symbol.get(symbol) if symbol else None
        if not bot:
            return
        try:
            if event_type == "kline":
                bot.on_kline_tick(data["k"])
            elif event_type == "24hrTicker":
                bot.on_ticker_tick(data)
        except Exception as e:
            self.log.warning(f"[{symbol}] Error procesando tick de WebSocket: {e}", exc_info=True)

    def _on_error(self, ws, error):
        self.log.warning(f"WebSocket de Binance: error de conexion: {error}")

    def _on_close(self, ws, code, msg):
        self.log.warning(f"WebSocket de Binance cerrado (codigo={code}, motivo={msg}).")

    def start(self):
        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        thread = threading.Thread(target=self._run_forever_loop, daemon=True)
        thread.start()

    def _run_forever_loop(self):
        """Bucle propio de reconexion (ademas del que trae la libreria), para
        garantizar que si la conexion se cae por CUALQUIER motivo, se reintenta
        indefinidamente en vez de dejar al bot sin datos en vivo para siempre."""
        while not self._stop:
            try:
                self.ws.run_forever(ping_interval=60, ping_timeout=10)
            except Exception as e:
                self.log.warning(f"WebSocket de Binance: excepcion en run_forever: {e}")
            if self._stop:
                break
            self.log.warning("WebSocket de Binance desconectado. Reintentando en 5s...")
            time.sleep(5)



# ------------------------------------------------------------------
# CUENTA DE PAPER TRADING (balance y posicion simulados) -- una por moneda
# ------------------------------------------------------------------
@dataclass
class PaperAccount:
    balance_usdt: float
    position_qty: float = 0.0
    entry_price: float = 0.0
    in_position: bool = False
    trade_history: list = field(default_factory=list)

    def buy(self, price: float, fraction: float, fee_pct: float) -> dict:
        spend = self.balance_usdt * fraction
        fee = spend * fee_pct
        qty = (spend - fee) / price
        self.balance_usdt -= spend
        self.position_qty = qty
        self.entry_price = price
        self.in_position = True
        return self._log_trade("COMPRA", price, qty, fee)

    def sell(self, price: float, fee_pct: float, reason: str) -> dict:
        proceeds = self.position_qty * price
        fee = proceeds * fee_pct
        pnl = proceeds - fee - (self.position_qty * self.entry_price)
        pnl_pct = (price / self.entry_price - 1) * 100
        self.balance_usdt += proceeds - fee
        entry = self._log_trade(f"VENTA ({reason})", price, self.position_qty, fee, pnl, pnl_pct)
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.in_position = False
        return entry

    def equity(self, current_price: float) -> float:
        if self.in_position:
            return self.balance_usdt + self.position_qty * current_price
        return self.balance_usdt

    def stats(self) -> dict:
        sells = [t for t in self.trade_history if t["pnl_usdt"] != ""]
        total = len(sells)
        wins = [t for t in sells if float(t["pnl_usdt"]) > 0]
        losses = [t for t in sells if float(t["pnl_usdt"]) <= 0]
        win_rate = (len(wins) / total * 100) if total else 0.0
        avg_pnl = (sum(float(t["pnl_usdt"]) for t in sells) / total) if total else 0.0
        return {
            "total_trades": total,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "avg_pnl_per_trade": round(avg_pnl, 4),
        }

    def _log_trade(self, action, price, qty, fee, pnl=None, pnl_pct=None) -> dict:
        ts = datetime.now(timezone.utc)
        entry = {
            "timestamp": ts.isoformat(),
            "time_unix": int(ts.timestamp()),
            "action": action,
            "price": round(price, 6),
            "qty": round(qty, 6),
            "fee": round(fee, 4),
            "pnl_usdt": round(pnl, 4) if pnl is not None else "",
            "pnl_pct": round(pnl_pct, 3) if pnl_pct is not None else "",
            "balance_after": round(self.balance_usdt, 2),
        }
        self.trade_history.append(entry)
        return entry


# ------------------------------------------------------------------
# ESTRATEGIA: CRUCE DE MEDIAS MOVILES
# ------------------------------------------------------------------
class SmaCrossStrategy:
    def __init__(self, short_window: int, long_window: int):
        self.short_window = short_window
        self.long_window = long_window

    def compute_signal(self, df: pd.DataFrame) -> str:
        df = df.copy()
        df["sma_short"] = df["close"].rolling(self.short_window).mean()
        df["sma_long"] = df["close"].rolling(self.long_window).mean()

        if df["sma_short"].isna().iloc[-2:].any():
            return "hold"

        prev_short, prev_long = df["sma_short"].iloc[-2], df["sma_long"].iloc[-2]
        curr_short, curr_long = df["sma_short"].iloc[-1], df["sma_long"].iloc[-1]

        crossed_up = prev_short <= prev_long and curr_short > curr_long
        crossed_down = prev_short >= prev_long and curr_short < curr_long

        if crossed_up:
            return "buy"
        if crossed_down:
            return "sell"
        return "hold"


# ------------------------------------------------------------------
# BOT DE UNA MONEDA -- se crea una instancia por cada simbolo
# ------------------------------------------------------------------
class SymbolTradingBot:
    # Se usa data-api.binance.vision en vez de api.binance.com porque Binance
    # bloquea (HTTP 451) las peticiones REST desde IPs de EE.UU. Este es el
    # espejo publico de solo lectura de mercado, sin esa restriccion. El
    # WebSocket (stream.binance.com) es un servicio distinto -- no deberia
    # tener el mismo bloqueo geografico porque solo transmite datos publicos
    # de mercado, pero no hay forma de confirmarlo sin probarlo en produccion.
    # Por eso el "watchdog" de abajo cae de vuelta al REST si el WebSocket no
    # trae datos: aunque el WebSocket llegara a fallar, el bot sigue operando.
    BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
    BINANCE_TICKER24H_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"

    def __init__(self, symbol: str, config: Config, notifier: TelegramNotifier):
        self.symbol = symbol
        self.binance_symbol = symbol.replace("/", "")
        self.cfg = config
        self.timeframe = config.timeframe        # temporalidad PROPIA de esta moneda (editable en vivo)
        self.timeframe_lock = threading.Lock()    # protege self.timeframe de carreras con el panel web
        self.strategy = SmaCrossStrategy(config.short_window, config.long_window)
        self.account = PaperAccount(balance_usdt=config.initial_balance_usdt)
        self.account_lock = threading.Lock()  # protege balance_usdt de carreras con transferencias a la boveda
        self.telegram = notifier
        self.candles = []              # velas OHLC mantenidas en vivo por el WebSocket
        self.candles_lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc)
        self.last_tick_at = None       # ultima vez que llego un dato del WebSocket (para el watchdog)
        self._last_summary_at = datetime.now(timezone.utc)
        self._setup_logging()
        self._setup_csv()

    def set_timeframe(self, timeframe: str) -> bool:
        if timeframe not in ALLOWED_TIMEFRAMES:
            return False
        with self.timeframe_lock:
            if self.timeframe == timeframe:
                return True
            old_timeframe = self.timeframe
            self.timeframe = timeframe

        old_stream = f"{self.binance_symbol.lower()}@kline_{old_timeframe}"
        new_stream = f"{self.binance_symbol.lower()}@kline_{timeframe}"
        if ws_feed is not None:
            ws_feed.resubscribe_symbol_timeframe(old_stream, new_stream)

        try:
            self.backfill()  # recarga el historial (y recalcula la senal) con la nueva temporalidad
        except Exception as e:
            self.log.warning(f"[{self.symbol}] No se pudo recargar el historial tras cambiar de temporalidad: {e}")

        self.log.info(f"[{self.symbol}] Temporalidad cambiada a {timeframe}")
        self.telegram.send(f"⏱️ [{self.symbol}] Temporalidad cambiada a {TIMEFRAME_LABELS.get(timeframe, timeframe)}")
        return True

    def _setup_logging(self):
        self.log = logging.getLogger(f"trading_bot.{self.binance_symbol}")

    def _setup_csv(self):
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.cfg.log_dir, f"{self.binance_symbol}_trades.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "action", "price", "qty", "fee",
                                  "pnl_usdt", "pnl_pct", "balance_after"])

    def _append_csv(self, entry: dict):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            row = {k: v for k, v in entry.items() if k != "time_unix"}
            writer.writerow(row.values())

    def fetch_24h_change_pct(self) -> float:
        """Consulta el % de cambio en 24h directamente desde Binance (el mismo dato que
        muestra la propia app/web de Binance). Se usa como respaldo inicial; en vivo lo
        actualiza el stream @ticker del WebSocket (on_ticker_tick)."""
        response = requests.get(
            self.BINANCE_TICKER24H_URL, params={"symbol": self.binance_symbol}, timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return float(data["priceChangePercent"])

    def check_risk_management(self, current_price: float):
        if not self.account.in_position:
            return
        change_pct = (current_price / self.account.entry_price) - 1

        if change_pct >= self.cfg.take_profit_pct:
            with self.account_lock:
                entry = self.account.sell(current_price, self.cfg.fee_pct, reason="take-profit")
            self._append_csv(entry)
            msg = f"[{self.symbol}] TAKE PROFIT ejecutado a {current_price:.4f} ({change_pct*100:.2f}%)"
            self.log.info(f"[TP] {msg}")
            self.telegram.send(f"🎯 {msg}")

        elif change_pct <= -self.cfg.stop_loss_pct:
            with self.account_lock:
                entry = self.account.sell(current_price, self.cfg.fee_pct, reason="stop-loss")
            self._append_csv(entry)
            msg = f"[{self.symbol}] STOP LOSS ejecutado a {current_price:.4f} ({change_pct*100:.2f}%)"
            self.log.info(f"[SL] {msg}")
            self.telegram.send(f"🛑 {msg}")

    # ------------------------------------------------------------------
    # NUCLEO comun: recibe una tabla de velas + el precio actual y ejecuta
    # TODO el pipeline de trading (gestion de riesgo, senal, actualizacion
    # del panel). Lo llaman tanto los ticks del WebSocket como el respaldo
    # REST -- asi el bot se comporta identico sin importar de donde vino
    # el dato.
    # ------------------------------------------------------------------
    def _process_tick(self, df: pd.DataFrame, current_price: float):
        now = datetime.now(timezone.utc)
        shared_state.push_price(self.symbol, current_price)

        candles_df = df.tail(self.cfg.price_history_maxlen)
        display_candles = [
            {"time": int(row.time), "open": round(row.open, 6), "high": round(row.high, 6),
             "low": round(row.low, 6), "close": round(row.close, 6)}
            for row in candles_df.itertuples()
        ]
        shared_state.push_candles(self.symbol, display_candles)

        self.check_risk_management(current_price)

        signal = self.strategy.compute_signal(df)

        if signal == "buy" and not self.account.in_position:
            with self.account_lock:
                entry = self.account.buy(current_price, self.cfg.trade_fraction, self.cfg.fee_pct)
            self._append_csv(entry)
            msg = f"[{self.symbol}] COMPRA a {current_price:.4f}"
            self.log.info(f"[BUY] {msg}")
            self.telegram.send(f"🟢 {msg}")

        elif signal == "sell" and self.account.in_position:
            with self.account_lock:
                entry = self.account.sell(current_price, self.cfg.fee_pct, reason="senal-cruce")
            self._append_csv(entry)
            msg = (f"[{self.symbol}] VENTA (senal) a {current_price:.4f} | "
                   f"PnL: {entry['pnl_usdt']} USDT ({entry['pnl_pct']}%)")
            self.log.info(f"[SELL] {msg}")
            self.telegram.send(f"🔴 {msg}")

        self._push_account_state(current_price, now)

    # ------------------------------------------------------------------
    # Publica equity/PnL/balance/posicion actualizados en el panel. Se llama
    # en CADA tick (vela cerrada o tick de precio del WebSocket) para que el
    # panel web nunca muestre datos atrasados -- incluye el caso en que un
    # take-profit/stop-loss se ejecuta fuera de un cierre de vela.
    # ------------------------------------------------------------------
    def _push_account_state(self, current_price: float, now=None):
        now = now or datetime.now(timezone.utc)
        equity = self.account.equity(current_price)
        pnl_total = equity - self.cfg.initial_balance_usdt
        pnl_total_pct = (equity / self.cfg.initial_balance_usdt - 1) * 100
        stats = self.account.stats()
        invested_usdt = round(equity - self.account.balance_usdt, 2)  # dinero "operando" ahora mismo

        shared_state.update(
            self.symbol,
            in_position=self.account.in_position,
            entry_price=round(self.account.entry_price, 6),
            equity=round(equity, 2),
            balance_usdt=round(self.account.balance_usdt, 2),
            invested_usdt=invested_usdt,
            pnl_total=round(pnl_total, 2),
            pnl_total_pct=round(pnl_total_pct, 2),
            last_update=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            trade_history=list(reversed(self.account.trade_history[-30:])),
            status="operando",
            timeframe=self.timeframe,
            **stats,
        )

        if (now - self._last_summary_at).total_seconds() >= self.cfg.telegram_summary_seconds:
            self._last_summary_at = now
            self.telegram.send(
                f"📊 Resumen {self.symbol}\n"
                f"Precio: {current_price:.4f}\n"
                f"Equity: {equity:.2f} USDT\n"
                f"PnL total: {pnl_total:+.2f} USDT ({pnl_total_pct:+.2f}%)\n"
                f"Trades: {stats['total_trades']} | Win rate: {stats['win_rate_pct']:.1f}%"
            )

    # ------------------------------------------------------------------
    # RESPALDO REST -- carga historial completo (arranque, cambio de
    # temporalidad, o si el watchdog detecta que el WebSocket dejo de
    # enviar datos). Siempre termina ejecutando el pipeline de trading,
    # para que el bot nunca deje de operar aunque el WebSocket falle.
    # ------------------------------------------------------------------
    def backfill(self):
        with self.timeframe_lock:
            current_timeframe = self.timeframe
        params = {"symbol": self.binance_symbol, "interval": current_timeframe,
                  "limit": self.cfg.candles_lookback}
        response = requests.get(self.BINANCE_KLINES_URL, params=params, timeout=10)
        response.raise_for_status()
        raw = response.json()
        candles = [
            {"time": int(row[0] // 1000), "open": float(row[1]), "high": float(row[2]),
             "low": float(row[3]), "close": float(row[4])}
            for row in raw
        ]
        if not candles:
            raise ValueError("Binance devolvio una lista de velas vacia")

        with self.candles_lock:
            self.candles = candles

        try:
            pct = self.fetch_24h_change_pct()
            shared_state.update(self.symbol, pct_change_24h=round(pct, 2))
        except Exception as e:
            self.log.warning(f"[{self.symbol}] No se pudo obtener el %cambio 24h: {e}")

        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        current_price = float(candles[-1]["close"])
        self._process_tick(df, current_price)
        self.last_tick_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # TICKS EN VIVO del WebSocket (llamados desde BinanceWebSocketFeed)
    # ------------------------------------------------------------------
    def on_kline_tick(self, k: dict):
        """k = objeto 'k' de un evento kline de Binance:
        {t: apertura(ms), o,h,l,c: precios, x: si la vela ya cerro, ...}"""
        self.last_tick_at = datetime.now(timezone.utc)
        t = int(k["t"] // 1000)
        o, h, l, c = float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"])

        with self.candles_lock:
            if self.candles and self.candles[-1]["time"] == t:
                self.candles[-1] = {"time": t, "open": o, "high": h, "low": l, "close": c}
            else:
                self.candles.append({"time": t, "open": o, "high": h, "low": l, "close": c})
                if len(self.candles) > self.cfg.candles_lookback:
                    self.candles = self.candles[-self.cfg.candles_lookback:]
            candles_snapshot = list(self.candles)

        # Aun no hay suficiente historial para calcular la SMA larga con confianza
        if len(candles_snapshot) < self.cfg.long_window + 2:
            shared_state.push_price(self.symbol, c)
            shared_state.push_candles(self.symbol, candles_snapshot[-self.cfg.price_history_maxlen:])
            return

        df = pd.DataFrame(candles_snapshot)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        self._process_tick(df, current_price=c)

    def on_ticker_tick(self, data: dict):
        """data = evento 24hrTicker de Binance ('P' = %cambio 24h, 'c' = ultimo precio).
        Este es el tick que llega mas seguido (varias veces por segundo), asi que aqui
        es donde el take-profit / stop-loss reacciona AL INSTANTE, sin esperar a que
        cierre la siguiente vela -- y donde el equity/PnL en vivo se mantiene fresco
        aunque no haya ninguna vela cerrando en ese momento."""
        self.last_tick_at = datetime.now(timezone.utc)
        try:
            pct = float(data["P"])
        except (KeyError, TypeError, ValueError):
            pct = None
        if pct is not None:
            shared_state.update(self.symbol, pct_change_24h=round(pct, 2))

        try:
            price = float(data["c"])
        except (KeyError, TypeError, ValueError):
            return
        shared_state.push_price(self.symbol, price)
        try:
            self.check_risk_management(price)
        except Exception as e:
            self.log.warning(f"[{self.symbol}] Error en check_risk_management via tick de WebSocket: {e}")
        self._push_account_state(price)

    def run(self):
        self.log.info(
            f"Iniciando bot PAPER TRADING | {self.symbol} | {self.timeframe} | "
            f"SMA({self.cfg.short_window}/{self.cfg.long_window}) | "
            f"TP {self.cfg.take_profit_pct*100:.1f}% | SL {self.cfg.stop_loss_pct*100:.1f}% | "
            f"Balance inicial: {self.account.balance_usdt:.2f} USDT"
        )
        self.telegram.send(
            f"🤖 Bot iniciado (paper trading, datos en vivo por WebSocket)\n{self.symbol} | {self.timeframe}\n"
            f"Balance inicial: {self.account.balance_usdt:.2f} USDT"
        )
        for attempt in range(5):
            try:
                self.backfill()
                self.log.info(f"[{self.symbol}] Historial inicial cargado ({len(self.candles)} velas). "
                               f"A partir de ahora recibe datos en vivo por WebSocket.")
                break
            except Exception as e:
                self.log.warning(f"[{self.symbol}] Fallo la carga inicial (intento {attempt + 1}/5): {e}")
                time.sleep(5)
        else:
            self.log.error(f"[{self.symbol}] No se pudo cargar el historial inicial tras varios intentos.")
            shared_state.update(self.symbol, status="error: no se pudo conectar a Binance al iniciar")

        # Vigilante: si el WebSocket nunca llega a traer datos (o deja de traerlos),
        # esto lo detecta y recurre a REST para que el bot nunca se quede operando a ciegas.
        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    def _watchdog_loop(self):
        while True:
            time.sleep(self.cfg.watchdog_check_seconds)
            stale = (
                self.last_tick_at is None
                or (datetime.now(timezone.utc) - self.last_tick_at).total_seconds() > self.cfg.watchdog_stale_seconds
            )
            if not stale:
                continue
            self.log.warning(
                f"[{self.symbol}] Sin datos del WebSocket en mas de "
                f"{self.cfg.watchdog_stale_seconds}s. Usando REST de respaldo..."
            )
            try:
                self.backfill()
            except Exception as e:
                self.log.warning(f"[{self.symbol}] Fallo el respaldo REST del vigilante: {e}")


# ------------------------------------------------------------------
# PANEL WEB (Flask) -- estilo Binance, con grafica en vivo por moneda
# ------------------------------------------------------------------
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' rx='22' fill='%2305070a'/%3E%3Crect x='4' y='4' width='92' height='92' rx='18' fill='none' stroke='%2300e5ff' stroke-width='3'/%3E%3Ctext x='50' y='45' font-family='Arial, sans-serif' font-size='30' font-weight='800' text-anchor='middle' fill='%2300e5ff'%3EBT1%3C/text%3E%3Ctext x='50' y='75' font-family='Arial, sans-serif' font-size='16' font-weight='700' text-anchor='middle' fill='%2300e676'%3EAI%3C/text%3E%3C/svg%3E">
<title>BT 1 Intelligent</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg: #05070a;
    --panel: #0d1117ee;
    --border: #1c2733;
    --green: #00e676;
    --red: #ff3b5c;
    --cyan: #00e5ff;
    --muted: #7d8899;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    background: var(--bg);
    color: #e8edf3;
    margin: 0;
    padding: 18px;
    position: relative;
    overflow-x: hidden;
  }
  canvas#matrixBg {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: -1;
    opacity: 0.35;
  }
  h1 {
    font-size: 1.4em;
    margin: 4px 0 4px;
    text-shadow: 0 0 10px rgba(0,229,255,0.4);
  }
  .subtitle { color: var(--muted); font-size: 0.85em; margin-bottom: 18px; }

  .total-bar {
    display: flex; flex-wrap: wrap; gap: 12px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 22px;
    backdrop-filter: blur(6px);
    box-shadow: 0 0 20px rgba(0,229,255,0.08);
  }
  .total-item { flex: 1; min-width: 140px; }
  .total-item .label { font-size: 0.72em; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .total-item .value { font-size: 1.5em; font-weight: 700; margin-top: 4px; }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    backdrop-filter: blur(6px);
  }
  .card-head {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 6px;
  }
  .card-head .sym { font-weight: 700; font-size: 1.05em; }
  .price-row { display: flex; align-items: baseline; gap: 8px; margin-bottom: 8px; }
  .price { font-size: 1.4em; font-weight: 700; }
  .arrow-up { color: var(--green); }
  .arrow-down { color: var(--red); }
  .pct-badge {
    padding: 2px 8px; border-radius: 6px; font-size: 0.85em; font-weight: 700;
  }
  .pct-pos { background: rgba(0,230,118,0.15); color: var(--green); }
  .pct-neg { background: rgba(255,59,92,0.15); color: var(--red); }
  canvas.chart { width: 100% !important; height: 130px !important; }
  .chart-container { width: 100%; height: 160px; margin: 4px 0 2px; }
  .stats-row {
    display: flex; flex-wrap: wrap; justify-content: space-between; margin-top: 10px;
    font-size: 0.78em; color: var(--muted); row-gap: 6px; column-gap: 10px;
  }
  .stats-row span b { color: #e8edf3; }
  .pnl-inline.pnl-pos { color: var(--green); }
  .pnl-inline.pnl-neg { color: var(--red); }
  .tf-row {
    display: flex; align-items: center; gap: 8px; margin: 6px 0 2px;
  }
  .tf-label { font-size: 0.72em; color: var(--muted); }
  .tf-select {
    background: #0a0f16; color: var(--cyan); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 8px; font-size: 0.78em; font-weight: 700;
    flex: 1;
  }
  .tf-select:focus { outline: none; border-color: var(--cyan); }
  .wallet-row {
    display: flex; justify-content: space-between; margin-top: 8px;
    font-size: 0.78em; color: var(--muted); border-top: 1px solid var(--border);
    padding-top: 8px;
  }
  .wallet-row b { color: #e8edf3; }
  .vault-btns { display: flex; gap: 8px; margin-top: 8px; }
  .vault-btn {
    flex: 1; font-size: 0.72em; padding: 6px 4px; border-radius: 6px;
    border: 1px solid var(--border); background: transparent; color: var(--muted);
    cursor: pointer; transition: 0.15s;
  }
  .vault-btn:hover { border-color: var(--cyan); color: var(--cyan); }
  .vault-btn.out:hover { border-color: var(--red); color: var(--red); }
  .pos-pill {
    font-size: 0.72em; padding: 2px 7px; border-radius: 20px;
    border: 1px solid var(--border);
  }
  .pos-open { color: var(--cyan); border-color: var(--cyan); }
  .pos-closed { color: var(--muted); }
  .updated { color: var(--muted); font-size: 0.75em; margin-top: 20px; text-align: center; }
</style>
</head>
<body>
  <canvas id="matrixBg"></canvas>

  <h1>🤖 BT 1 Intelligent</h1>
  <div class="subtitle">Paper trading &middot; datos reales de Binance &middot; se actualiza solo cada 1s</div>

  <div class="total-bar" id="totalBar"></div>

  <div class="grid" id="cardsGrid"></div>

  <div class="updated" id="updatedFooter">Cargando...</div>

<script>
// ---------- Fondo cyberpunk: lluvia de codigo ----------
const bgCanvas = document.getElementById('matrixBg');
const bgCtx = bgCanvas.getContext('2d');
let cols, drops;
function resizeBg() {
  bgCanvas.width = window.innerWidth;
  bgCanvas.height = window.innerHeight;
  cols = Math.floor(bgCanvas.width / 16);
  drops = Array(cols).fill(1);
}
window.addEventListener('resize', resizeBg);
resizeBg();
const chars = "01アイウエオカキクケコ$€₿ΞΣΔ".split("");
function drawMatrix() {
  bgCtx.fillStyle = "rgba(5,7,10,0.08)";
  bgCtx.fillRect(0, 0, bgCanvas.width, bgCanvas.height);
  bgCtx.fillStyle = "#00e5ff";
  bgCtx.font = "14px monospace";
  for (let i = 0; i < drops.length; i++) {
    const text = chars[Math.floor(Math.random() * chars.length)];
    bgCtx.fillStyle = Math.random() > 0.94 ? "#00e676" : "rgba(0,229,255,0.7)";
    bgCtx.fillText(text, i * 16, drops[i] * 16);
    if (drops[i] * 16 > bgCanvas.height && Math.random() > 0.975) drops[i] = 0;
    drops[i]++;
  }
}
setInterval(drawMatrix, 60);

// ---------- Panel en vivo ----------
const charts = {};
const TIMEFRAMES = __TIMEFRAMES_JSON__;  // [[valor, etiqueta], ...] inyectado desde el backend
const COLORS = { up: "#00e676", down: "#ff3b5c" };

function fmt(n, d=2) { return Number(n).toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d}); }

function ensureCard(symbol) {
  if (document.getElementById('card-' + symbol)) return;
  const grid = document.getElementById('cardsGrid');
  const el = document.createElement('div');
  el.className = 'card';
  el.id = 'card-' + symbol;
  el.innerHTML = `
    <div class="card-head">
      <span class="sym">${symbol}</span>
      <span class="pos-pill" id="pos-${symbol}">--</span>
    </div>
    <div class="price-row">
      <span class="price" id="price-${symbol}">--</span>
      <span id="arrow-${symbol}"></span>
      <span class="pct-badge" id="pct24h-${symbol}" title="Cambio 24h (mercado real de Binance)">--</span>
    </div>
    <div class="tf-row">
      <span class="tf-label">Temporalidad:</span>
      <select class="tf-select" id="tf-${symbol}" onchange="changeTimeframe('${symbol}', this.value)"></select>
    </div>
    <div class="chart-container" id="chartc-${symbol}"></div>
    <div class="stats-row">
      <span>Trades: <b id="trades-${symbol}">0</b></span>
      <span>Win rate: <b id="winrate-${symbol}">0%</b></span>
      <span>Equity: <b id="equity-${symbol}">$0</b></span>
      <span>PnL bot: <b class="pnl-inline" id="pnlbot-${symbol}">0%</b></span>
    </div>
    <div class="wallet-row">
      <span>Operando: <b id="invested-${symbol}">$0</b></span>
      <span>Disponible: <b id="available-${symbol}">$0</b></span>
    </div>
    <div class="vault-btns">
      <button class="vault-btn" onclick="doTransfer('${symbol}','to_vault')">→ Boveda</button>
      <button class="vault-btn out" onclick="doTransfer('${symbol}','to_wallet')">← Boveda</button>
    </div>
  `;
  grid.appendChild(el);

  const tfSelect = document.getElementById('tf-' + symbol);
  TIMEFRAMES.forEach(([value, label]) => {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    tfSelect.appendChild(opt);
  });

  const container = document.getElementById('chartc-' + symbol);
  const chart = LightweightCharts.createChart(container, {
    layout: { background: { color: 'transparent' }, textColor: '#7d8899', fontSize: 10 },
    grid: {
      vertLines: { color: 'rgba(255,255,255,0.04)' },
      horzLines: { color: 'rgba(255,255,255,0.04)' },
    },
    timeScale: { timeVisible: true, secondsVisible: false, borderColor: '#1c2733' },
    rightPriceScale: { borderColor: '#1c2733' },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    autoSize: true,
  });
  const series = chart.addCandlestickSeries({
    upColor: COLORS.up, downColor: COLORS.down, borderVisible: false,
    wickUpColor: COLORS.up, wickDownColor: COLORS.down,
  });
  charts[symbol] = { chart, series };
}

function buildMarkers(tradeHistory) {
  const markers = tradeHistory
    .filter(t => t.time_unix)
    .map(t => {
      const isBuy = t.action.startsWith('COMPRA');
      return {
        time: t.time_unix,
        position: isBuy ? 'belowBar' : 'aboveBar',
        color: isBuy ? COLORS.up : COLORS.down,
        shape: isBuy ? 'arrowUp' : 'arrowDown',
        text: isBuy ? 'COMPRA' : `VENTA ${t.pnl_pct !== '' ? (t.pnl_pct >= 0 ? '+' : '') + t.pnl_pct + '%' : ''}`,
      };
    })
    .sort((a, b) => a.time - b.time);
  return markers;
}

function updateCard(symbol, d) {
  ensureCard(symbol);
  const up = d.current_price >= d.prev_price;
  document.getElementById('price-' + symbol).textContent = '$' + fmt(d.current_price, d.current_price < 10 ? 4 : 2);
  document.getElementById('price-' + symbol).style.color = up ? COLORS.up : COLORS.down;
  document.getElementById('arrow-' + symbol).innerHTML = up
    ? '<span class="arrow-up">▲</span>' : '<span class="arrow-down">▼</span>';

  const pct24hEl = document.getElementById('pct24h-' + symbol);
  const p24 = d.pct_change_24h || 0;
  pct24hEl.textContent = (p24 >= 0 ? '+' : '') + fmt(p24, 2) + '%';
  pct24hEl.className = 'pct-badge ' + (p24 >= 0 ? 'pct-pos' : 'pct-neg');

  const posEl = document.getElementById('pos-' + symbol);
  posEl.textContent = d.in_position ? 'POSICION ABIERTA' : 'SIN POSICION';
  posEl.className = 'pos-pill ' + (d.in_position ? 'pos-open' : 'pos-closed');

  document.getElementById('trades-' + symbol).textContent = d.total_trades;
  document.getElementById('winrate-' + symbol).textContent = fmt(d.win_rate_pct, 1) + '%';
  document.getElementById('equity-' + symbol).textContent = '$' + fmt(d.equity, 2);
  document.getElementById('invested-' + symbol).textContent = '$' + fmt(d.invested_usdt, 2);
  document.getElementById('available-' + symbol).textContent = '$' + fmt(d.balance_usdt, 2);

  const pnlBotEl = document.getElementById('pnlbot-' + symbol);
  pnlBotEl.textContent = (d.pnl_total_pct >= 0 ? '+' : '') + fmt(d.pnl_total_pct, 2) + '%';
  pnlBotEl.className = 'pnl-inline ' + (d.pnl_total_pct >= 0 ? 'pnl-pos' : 'pnl-neg');

  const tfSelect = document.getElementById('tf-' + symbol);
  if (tfSelect && document.activeElement !== tfSelect && d.timeframe && tfSelect.value !== d.timeframe) {
    tfSelect.value = d.timeframe;
  }

  if (d.candles && d.candles.length) {
    charts[symbol].series.setData(d.candles);
    charts[symbol].series.setMarkers(buildMarkers(d.trade_history || []));
  }
}

async function changeTimeframe(symbol, timeframe) {
  try {
    const res = await fetch('/api/timeframe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, timeframe }),
    });
    const data = await res.json();
    if (!data.ok) {
      alert(data.error || 'No se pudo cambiar la temporalidad.');
      return;
    }
    refresh();
  } catch (e) {
    alert('Error de conexion al cambiar la temporalidad.');
  }
}

async function doTransfer(symbol, direction) {
  const label = direction === 'to_vault' ? 'a la boveda' : 'de la boveda a ' + symbol;
  const raw = prompt(`Cuanto USDT quieres mover ${label}?`);
  if (raw === null) return;
  const amount = parseFloat(raw);
  if (isNaN(amount) || amount <= 0) {
    alert('Monto invalido.');
    return;
  }
  try {
    const res = await fetch('/api/vault/transfer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, direction, amount }),
    });
    const data = await res.json();
    if (!data.ok) {
      alert(data.error || data.message || 'No se pudo completar la transferencia.');
      return;
    }
    refresh();
  } catch (e) {
    alert('Error de conexion al intentar transferir.');
  }
}

function updateTotal(t) {
  const bar = document.getElementById('totalBar');
  const pos = t.pnl_total_pct >= 0;
  bar.innerHTML = `
    <div class="total-item"><div class="label">Balance total (${t.num_assets} activos)</div>
      <div class="value">$${fmt(t.equity, 2)}</div></div>
    <div class="total-item"><div class="label">Operando</div>
      <div class="value" style="color:var(--cyan)">$${fmt(t.invested_usdt, 2)}</div></div>
    <div class="total-item"><div class="label">Disponible</div>
      <div class="value">$${fmt(t.available_usdt, 2)}</div></div>
    <div class="total-item"><div class="label">Boveda (fuera del bot)</div>
      <div class="value" style="color:var(--muted)">$${fmt(t.vault_usdt, 2)}</div></div>
    <div class="total-item"><div class="label">PnL total</div>
      <div class="value" style="color:${pos ? 'var(--green)' : 'var(--red)'}">
        ${pos ? '+' : ''}$${fmt(t.pnl_total, 2)} (${pos ? '+' : ''}${fmt(t.pnl_total_pct, 2)}%)</div></div>
    <div class="total-item"><div class="label">Trades totales</div>
      <div class="value">${t.total_trades}</div></div>
    <div class="total-item"><div class="label">% Ganadoras (global)</div>
      <div class="value">${fmt(t.win_rate_pct, 1)}%</div></div>
  `;
}

async function refresh() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    for (const [symbol, d] of Object.entries(data.symbols)) {
      try {
        updateCard(symbol, d);
      } catch (cardErr) {
        console.error('Error actualizando tarjeta de ' + symbol + ':', cardErr);
      }
    }
    try {
      updateTotal(data.total);
    } catch (totalErr) {
      console.error('Error actualizando total:', totalErr);
    }
    document.getElementById('updatedFooter').textContent = 'Ultima actualizacion: ' + data.generated_at;
  } catch (e) {
    console.error('Error actualizando panel:', e);
    document.getElementById('updatedFooter').textContent = 'Error al actualizar (revisa la consola)';
  }
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""

# Inyecta la lista de temporalidades permitidas (definida en Python, fuente unica de verdad)
# dentro del JS del panel, para que el <select> de cada moneda siempre coincida con lo que
# el backend realmente acepta.
DASHBOARD_TEMPLATE = DASHBOARD_TEMPLATE.replace(
    "__TIMEFRAMES_JSON__",
    json.dumps([[tf, TIMEFRAME_LABELS.get(tf, tf)] for tf in ALLOWED_TIMEFRAMES]),
)

flask_app = Flask(__name__)


@flask_app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)


@flask_app.route("/api/state")
def api_state():
    return jsonify(shared_state.snapshot())


def transfer_to_vault(bot: "SymbolTradingBot", amount: float):
    if amount <= 0:
        return False, "El monto debe ser mayor a 0"
    with bot.account_lock:
        if amount > bot.account.balance_usdt + 1e-9:
            return False, f"Fondos insuficientes en {bot.symbol} (disponible: {bot.account.balance_usdt:.2f} USDT)"
        bot.account.balance_usdt = round(bot.account.balance_usdt - amount, 2)
    vault.deposit(amount)
    return True, f"Se movieron {amount:.2f} USDT de {bot.symbol} a la boveda"


def transfer_from_vault(bot: "SymbolTradingBot", amount: float):
    if amount <= 0:
        return False, "El monto debe ser mayor a 0"
    if not vault.withdraw(amount):
        return False, f"Fondos insuficientes en la boveda (disponible: {vault.snapshot():.2f} USDT)"
    with bot.account_lock:
        bot.account.balance_usdt = round(bot.account.balance_usdt + amount, 2)
    return True, f"Se movieron {amount:.2f} USDT de la boveda a {bot.symbol}"


@flask_app.route("/api/vault/transfer", methods=["POST"])
def api_vault_transfer():
    data = request.get_json(force=True, silent=True) or {}
    symbol = data.get("symbol")
    direction = data.get("direction")  # "to_vault" | "to_wallet"
    try:
        amount = round(float(data.get("amount", 0)), 2)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Monto invalido"}), 400

    bot = bots_by_symbol.get(symbol)
    if not bot:
        return jsonify({"ok": False, "error": "Moneda no encontrada"}), 404

    if direction == "to_vault":
        ok, msg = transfer_to_vault(bot, amount)
    elif direction == "to_wallet":
        ok, msg = transfer_from_vault(bot, amount)
    else:
        return jsonify({"ok": False, "error": "Direccion invalida"}), 400

    return jsonify({"ok": ok, "message": msg, "vault_balance": vault.snapshot()}), (200 if ok else 400)


@flask_app.route("/api/timeframe", methods=["POST"])
def api_set_timeframe():
    data = request.get_json(force=True, silent=True) or {}
    symbol = data.get("symbol")
    timeframe = data.get("timeframe")

    bot = bots_by_symbol.get(symbol)
    if not bot:
        return jsonify({"ok": False, "error": "Moneda no encontrada"}), 404

    if timeframe not in ALLOWED_TIMEFRAMES:
        return jsonify({"ok": False, "error": f"Temporalidad invalida. Usa una de: {', '.join(ALLOWED_TIMEFRAMES)}"}), 400

    bot.set_timeframe(timeframe)
    shared_state.update(symbol, timeframe=timeframe)
    return jsonify({"ok": True, "symbol": symbol, "timeframe": timeframe})


def run_web_dashboard(host: str, port: int):
    flask_app.run(host=host, port=port, debug=False, use_reloader=False)


# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root_log = logging.getLogger("trading_bot")

    shared_notifier = TelegramNotifier(
        cfg.telegram_bot_token, cfg.telegram_chat_id, cfg.telegram_enabled, root_log
    )

    bots = [SymbolTradingBot(symbol, cfg, shared_notifier) for symbol in cfg.symbols]
    bots_by_symbol.update({b.symbol: b for b in bots})
    binance_symbol_to_slash.update({b.binance_symbol: b.symbol for b in bots})

    ws_feed = BinanceWebSocketFeed(root_log)
    ws_feed.start()
    root_log.info("Feed de WebSocket en tiempo real iniciado (stream.binance.com).")

    command_bot = TelegramCommandBot(
        cfg.telegram_bot_token, cfg.telegram_chat_id, cfg.telegram_enabled, cfg, root_log
    )
    if command_bot.enabled:
        cmd_thread = threading.Thread(target=command_bot.poll_loop, daemon=True)
        cmd_thread.start()

    if cfg.web_enabled:
        web_thread = threading.Thread(
            target=run_web_dashboard, args=(cfg.web_host, cfg.web_port), daemon=True
        )
        web_thread.start()
        print(f"Panel web disponible en http://<ip-del-servidor>:{cfg.web_port}  "
              f"(o http://localhost:{cfg.web_port} si corres localmente)")

    root_log.info(f"Monedas activas: {', '.join(cfg.symbols)} | Balance inicial por moneda: "
                   f"{cfg.initial_balance_usdt:.2f} USDT | Balance total inicial: "
                   f"{cfg.initial_balance_usdt * len(cfg.symbols):.2f} USDT")

    threads = []
    for bot in bots:
        t = threading.Thread(target=bot.run, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1.5)  # escalona el arranque para no golpear la API de Binance de una vez

    for t in threads:
        t.join()
