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
  - Datos de mercado via REST: cada moneda consulta Binance periodicamente
    (precio/%24h cada pocos segundos, velas completas cada minuto) usando el
    espejo publico data-api.binance.vision, sin depender de una conexion
    persistente. Arquitectura simple y robusta a proposito: menos piezas que
    puedan fallar.
  - Cada moneda puede operar en su PROPIA temporalidad (5m/15m/1h/4h/1d),
    elegible en vivo desde el panel web, sin reiniciar el bot.
  - Alertas automaticas a Telegram: cualquier error o advertencia que ocurra
    en el bot se manda tambien a Telegram con fecha/hora exacta, ademas de
    quedar en los logs. Tambien avisa cuando la conexion inicial con Binance
    fue exitosa.
  - Un hilo de trading independiente POR MONEDA (cada una con su propia
    cuenta de paper trading, historial de precios e historial de operaciones).
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
    pip install requests pandas flask gunicorn
"""

import time
import csv
import os
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request


# ------------------------------------------------------------------
# CONFIGURACION -- ajusta estos parametros a tu gusto
# ------------------------------------------------------------------
@dataclass
class Config:
    symbols: list = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
    ])
    timeframe: str = "1h"           # Temporalidad de las velas: 1m, 5m, 15m, 1h, 4h, 1d...
    short_window: int = 20          # Periodo de la media movil corta
    long_window: int = 50           # Periodo de la media movil larga
    take_profit_pct: float = 0.03   # Objetivo base: 3%
    stop_loss_pct: float = 0.015    # Riesgo maximo base: 1.5%
    trade_fraction: float = 0.25    # Maximo 25% del capital disponible por posicion
    risk_per_trade_pct: float = 0.01 # Riesgo teorico maximo por trade sobre equity
    max_position_fraction: float = 0.25 # Tope duro de exposicion por moneda
    max_total_exposure_fraction: float = 0.50 # Tope de exposicion simultanea de toda la cartera
    fee_pct: float = 0.001          # Comision simulada de Binance (~0.1%)
    slippage_pct: float = 0.0005    # 0.05% de slippage simulado en cada ejecucion
    min_trade_usdt: float = 5.0     # Evita compras/polvo demasiado pequeno
    signal_gap_pct: float = 0.001   # Filtra cruces SMA casi planos (0.10%)
    rsi_period: int = 14
    rsi_buy_min: float = 50.0
    rsi_sell_max: float = 50.0
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    atr_take_mult: float = 3.0
    max_history_trades: int = 5000
    initial_balance_usdt: float = 100.0   # Balance virtual inicial POR MONEDA
    candles_lookback: int = 200     # Cuantas velas historicas descarga por REST al iniciar / respaldo
    price_history_maxlen: int = 150  # Puntos que se guardan para la grafica en vivo
    log_dir: str = "logs"

    # --- Cadencia de las consultas REST a Binance ---
    poll_seconds: int = 5                # cada cuanto refresca precio/%24h via REST (liviano)
    full_backfill_seconds: int = 60      # cada cuanto recarga velas completas + recalcula la senal

    # --- Telegram ---
    telegram_enabled: bool = True
    telegram_bot_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "PON_AQUI_TU_TOKEN")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "PON_AQUI_TU_CHAT_ID")
    telegram_summary_seconds: int = 1800  # resumen periodico por moneda (30 min)

    # --- Panel web ---
    web_enabled: bool = True
    web_host: str = "0.0.0.0"
    web_port: int = int(os.environ.get("PORT", 5000))  # Render asigna el puerto via variable PORT
    web_api_token: str = os.environ.get("WEB_API_TOKEN", "")  # Protege operaciones mutables del panel


cfg = Config()

# Temporalidades que el usuario puede elegir para cada moneda desde el panel web,
# igual que el selector de Binance (1D = 24 horas, es la misma vela).
ALLOWED_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
TIMEFRAME_LABELS = {"5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D (24h)"}

# Icono/color por moneda para el panel web. Se usa el simbolo de moneda oficial de
# Unicode cuando existe (₿, Ξ, ₳...) en vez de logos de marca, para no depender de
# ningun logotipo con derechos de autor.
COIN_BADGES = {
    "BTC": {"glyph": "₿", "color": "#f7931a"},
    "ETH": {"glyph": "Ξ", "color": "#627eea"},
    "SOL": {"glyph": "◎", "color": "#14f195"},
    "BNB": {"glyph": "B", "color": "#f0b90b"},
    "XRP": {"glyph": "✕", "color": "#4285f4"},
    "ADA": {"glyph": "₳", "color": "#0033ad"},
}
DEFAULT_BADGE = {"glyph": "●", "color": "#7d8899"}


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
            # OJO: este logger es distinto al que escucha TelegramLogHandler a proposito.
            # Si usara el mismo, un fallo al enviar a Telegram generaria un log, que el
            # handler intentaria reenviar a Telegram, que fallaria de nuevo... bucle infinito.
            _notifier_internal_log.warning(f"No se pudo enviar mensaje a Telegram: {e}")


_notifier_internal_log = logging.getLogger("trading_bot._notifier_internal")


# ------------------------------------------------------------------
# ALERTAS AUTOMATICAS A TELEGRAM -- cualquier WARNING o ERROR que se registre
# en cualquier parte del bot llega tambien a Telegram, con fecha/hora exacta
# y el mensaje, para no tener que revisar los logs de Render manualmente.
# Los mensajes repetidos se agrupan (no se manda el mismo error cada pocos
# segundos si se repite) para no saturar el chat.
# ------------------------------------------------------------------
class TelegramLogHandler(logging.Handler):
    def __init__(self, notifier: TelegramNotifier, throttle_seconds: int = 300):
        super().__init__(level=logging.WARNING)
        self.notifier = notifier
        self.throttle_seconds = throttle_seconds
        self._last_sent_at = {}

    def emit(self, record: logging.LogRecord):
        if record.name == "trading_bot._notifier_internal":
            return  # evita el bucle infinito descrito arriba en TelegramNotifier.send
        try:
            message = record.getMessage()
        except Exception:
            return

        key = f"{record.levelname}:{record.name}:{message[:100]}"
        now = time.time()
        last = self._last_sent_at.get(key)
        if last is not None and (now - last) < self.throttle_seconds:
            return
        self._last_sent_at[key] = now

        emoji = "🔴" if record.levelno >= logging.ERROR else "⚠️"
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        text = f"{emoji} [{record.levelname}] {ts}\n{message}"
        try:
            self.notifier.send(text)
        except Exception:
            pass  # un fallo aqui nunca debe romper el logging normal del bot


# Cuenta cuantos bots faltan por confirmar su arranque, para mandar UN solo
# mensaje consolidado al terminar en vez de 6 mensajes sueltos -- y para que
# ese mensaje diga la VERDAD sobre cuales monedas conectaron y cuales no.
_startup_lock = threading.Lock()
_startup_pending = set()
_startup_failed = set()
_startup_notifier = None


def _init_startup_tracking(symbols: list, notifier: TelegramNotifier):
    global _startup_notifier
    with _startup_lock:
        _startup_pending.clear()
        _startup_pending.update(symbols)
        _startup_failed.clear()
        _startup_notifier = notifier


def _report_startup_result(symbol: str, ok: bool):
    with _startup_lock:
        _startup_pending.discard(symbol)
        if not ok:
            _startup_failed.add(symbol)
        remaining = len(_startup_pending)
        failed = set(_startup_failed)
        notifier = _startup_notifier
    if remaining > 0 or notifier is None:
        return
    if not failed:
        notifier.send("✅ Conexion exitosa con Binance. El bot esta operando con todas las monedas configuradas.")
    else:
        ok_list = ", ".join(s for s in cfg.symbols if s not in failed)
        notifier.send(
            f"⚠️ Arranque con problemas: {', '.join(sorted(failed))} NO pudieron conectar con Binance "
            f"tras varios intentos (seguiran reintentando solas). El resto si conecto bien: {ok_list or 'ninguna'}."
        )


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
                "price_updated_at": 0,  # unix seconds del ultimo cambio de precio REAL (no del ultimo refresco de panel)
                "pct_change_24h": 0.0,
                "in_position": False,
                "entry_price": 0.0,
                "equity": initial_balance_per_symbol,
                "balance_usdt": initial_balance_per_symbol,
                "invested_usdt": 0.0,
                "floating_pnl_usdt": 0.0,
                "floating_pnl_pct": 0.0,
                "pnl_total": 0.0,
                "pnl_total_pct": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "avg_pnl_per_trade": 0.0,
                "last_update": "",
                "trade_history": [],
                "closed_trades": [],
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
            new_price = round(float(price), 6)
            if new_price != d["current_price"]:
                d["prev_price"] = d["current_price"]
                d["current_price"] = new_price
                d["price_updated_at"] = int(time.time())
            # Si el precio nuevo es IGUAL al ultimo conocido, no tocamos prev_price:
            # asi la flecha/color no parpadea en verde por defecto en cada tick sin
            # cambio real (Binance manda ticks varias veces por segundo aunque el
            # precio no se mueva ni un centavo).

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
# Serializa nuevas entradas para imponer un limite de exposicion global.
portfolio_trade_lock = threading.Lock()


# ------------------------------------------------------------------
# CUENTA DE PAPER TRADING (balance y posicion simulados) -- una por moneda
# ------------------------------------------------------------------
@dataclass
class PaperAccount:
    balance_usdt: float
    position_qty: float = 0.0
    entry_price: float = 0.0
    in_position: bool = False
    entry_notional_usdt: float = 0.0
    trade_history: list = field(default_factory=list)

    def buy(self, price: float, fraction: float, fee_pct: float, slippage_pct: float = 0.0,
            min_trade_usdt: float = 0.0) -> dict:
        if price <= 0 or not math.isfinite(price):
            raise ValueError("Precio de compra invalido")
        if self.in_position:
            raise RuntimeError("Ya existe una posicion abierta")
        fraction = max(0.0, min(1.0, float(fraction)))
        spend = self.balance_usdt * fraction
        if spend < min_trade_usdt:
            raise ValueError("Balance insuficiente para el tamano minimo de operacion")
        fill_price = price * (1.0 + max(0.0, slippage_pct))
        fee = spend * fee_pct
        qty = (spend - fee) / fill_price
        if qty <= 0:
            raise ValueError("Cantidad de compra invalida")
        self.balance_usdt -= spend
        self.position_qty = qty
        self.entry_price = fill_price
        self.entry_notional_usdt = spend
        self.in_position = True
        return self._log_trade("COMPRA", fill_price, qty, fee)

    def sell(self, price: float, fee_pct: float, reason: str, slippage_pct: float = 0.0) -> dict:
        if not self.in_position or self.position_qty <= 0 or self.entry_price <= 0:
            raise RuntimeError("No existe una posicion abierta")
        if price <= 0 or not math.isfinite(price):
            raise ValueError("Precio de venta invalido")
        fill_price = price * (1.0 - max(0.0, slippage_pct))
        proceeds = self.position_qty * fill_price
        fee = proceeds * fee_pct
        # El coste real incluye la comision pagada en la entrada. El notional
        # reservado fue entry_notional_usdt, por lo que no debemos contar la
        # comision de compra como "ganancia" fantasma.
        cost_basis = self.entry_notional_usdt
        pnl = proceeds - fee - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0
        qty = self.position_qty
        self.balance_usdt += proceeds - fee
        entry = self._log_trade(f"VENTA ({reason})", fill_price, qty, fee, pnl, pnl_pct)
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.entry_notional_usdt = 0.0
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
        gross_profit = sum(max(0.0, float(t["pnl_usdt"])) for t in sells)
        gross_loss = sum(min(0.0, float(t["pnl_usdt"])) for t in sells)
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else (float("inf") if gross_profit else 0.0)
        return {
            "total_trades": total,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "avg_pnl_per_trade": round(avg_pnl, 4),
            "profit_factor": round(profit_factor, 3) if math.isfinite(profit_factor) else "inf",
        }

    def _log_trade(self, action, price, qty, fee, pnl=None, pnl_pct=None) -> dict:
        ts = datetime.now(timezone.utc)
        entry = {
            "timestamp": ts.isoformat(),
            "time_unix": int(ts.timestamp()),
            "action": action,
            "price": round(price, 8),
            "qty": round(qty, 8),
            "fee": round(fee, 6),
            "pnl_usdt": round(pnl, 6) if pnl is not None else "",
            "pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else "",
            "balance_after": round(self.balance_usdt, 6),
        }
        self.trade_history.append(entry)
        return entry


# ------------------------------------------------------------------
# ESTRATEGIA: CRUCE DE MEDIAS MOVILES
# ------------------------------------------------------------------
class SmaCrossStrategy:
    """Cruce SMA con filtros sencillos para reducir ruido:
    - distancia minima entre medias
    - RSI como confirmacion de momentum
    - pendiente de la SMA larga como filtro de tendencia
    """
    def __init__(self, short_window: int, long_window: int, signal_gap_pct: float = 0.001,
                 rsi_period: int = 14, rsi_buy_min: float = 50.0, rsi_sell_max: float = 50.0):
        if short_window <= 0 or long_window <= short_window:
            raise ValueError("short_window debe ser > 0 y long_window debe ser mayor")
        self.short_window = short_window
        self.long_window = long_window
        self.signal_gap_pct = max(0.0, signal_gap_pct)
        self.rsi_period = max(2, rsi_period)
        self.rsi_buy_min = rsi_buy_min
        self.rsi_sell_max = rsi_sell_max

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0.0)
        return rsi

    def compute_signal(self, df: pd.DataFrame) -> str:
        if len(df) < max(self.long_window + 2, self.rsi_period + 2):
            return "hold"
        work = df.copy()
        work["sma_short"] = work["close"].rolling(self.short_window).mean()
        work["sma_long"] = work["close"].rolling(self.long_window).mean()
        work["rsi"] = self._rsi(work["close"], self.rsi_period)

        prev = work.iloc[-2]
        curr = work.iloc[-1]
        if any(pd.isna(x) for x in (prev.sma_short, prev.sma_long, curr.sma_short, curr.sma_long, curr.rsi)):
            return "hold"

        crossed_up = prev.sma_short <= prev.sma_long and curr.sma_short > curr.sma_long
        crossed_down = prev.sma_short >= prev.sma_long and curr.sma_short < curr.sma_long
        gap = abs(curr.sma_short - curr.sma_long) / curr.sma_long if curr.sma_long else 0.0

        long_sma_slope = curr.sma_long - prev.sma_long
        if crossed_up and gap >= self.signal_gap_pct and curr.rsi >= self.rsi_buy_min and long_sma_slope >= 0:
            return "buy"
        if crossed_down and gap >= self.signal_gap_pct and curr.rsi <= self.rsi_sell_max and long_sma_slope <= 0:
            return "sell"
        return "hold"


# ------------------------------------------------------------------
# BOT DE UNA MONEDA -- se crea una instancia por cada simbolo
# ------------------------------------------------------------------
class SymbolTradingBot:
    # Se usa data-api.binance.vision en vez de api.binance.com porque Binance
    # bloquea (HTTP 451) las peticiones REST desde IPs de EE.UU. Este es el
    # espejo publico de solo lectura de mercado, sin esa restriccion.
    BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
    BINANCE_TICKER24H_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"

    def __init__(self, symbol: str, config: Config, notifier: TelegramNotifier):
        self.symbol = symbol
        self.binance_symbol = symbol.replace("/", "")
        self.cfg = config
        self.timeframe = config.timeframe        # temporalidad PROPIA de esta moneda (editable en vivo)
        self.timeframe_lock = threading.Lock()    # protege self.timeframe de carreras con el panel web
        self.strategy = SmaCrossStrategy(config.short_window, config.long_window, config.signal_gap_pct, config.rsi_period, config.rsi_buy_min, config.rsi_sell_max)
        self.account = PaperAccount(balance_usdt=config.initial_balance_usdt)
        self.account_lock = threading.RLock()  # protege TODA la cuenta, permitiendo snapshots anidados seguros
        self.telegram = notifier
        self.candles = []              # velas OHLC, recargadas periodicamente via REST
        self.candles_lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc)
        self._last_summary_at = datetime.now(timezone.utc)
        self._setup_logging()
        self._setup_csv()

    def set_timeframe(self, timeframe: str) -> bool:
        if timeframe not in ALLOWED_TIMEFRAMES:
            return False
        with self.timeframe_lock:
            if self.timeframe == timeframe:
                return True
            self.timeframe = timeframe

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
        muestra la propia app/web de Binance)."""
        response = requests.get(
            self.BINANCE_TICKER24H_URL, params={"symbol": self.binance_symbol}, timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return float(data["priceChangePercent"])

    def fetch_current_price(self) -> float:
        response = requests.get(
            "https://data-api.binance.vision/api/v3/ticker/price",
            params={"symbol": self.binance_symbol}, timeout=10
        )
        response.raise_for_status()
        price = float(response.json()["price"])
        if price <= 0 or not math.isfinite(price):
            raise ValueError("Precio actual invalido")
        return price

    def _execution_prices(self, current_price: float):
        """Precios teoricos de ejecucion del paper engine (slippage incluido)."""
        return (
            current_price * (1.0 + self.cfg.slippage_pct),
            current_price * (1.0 - self.cfg.slippage_pct),
        )

    def check_risk_management(self, current_price: float):
        with self.account_lock:
            if not self.account.in_position:
                return None
            entry_price = self.account.entry_price
            change_pct = (current_price / entry_price) - 1 if entry_price else 0.0
            # TP/SL son disparadores, no precios garantizados. El fill se degrada
            # con slippage para que el paper trading no sea artificialmente optimista.
            if change_pct >= self.cfg.take_profit_pct:
                entry = self.account.sell(current_price, self.cfg.fee_pct, reason="take-profit",
                                          slippage_pct=self.cfg.slippage_pct)
                reason = "TAKE PROFIT"
            elif change_pct <= -self.cfg.stop_loss_pct:
                entry = self.account.sell(current_price, self.cfg.fee_pct, reason="stop-loss",
                                          slippage_pct=self.cfg.slippage_pct)
                reason = "STOP LOSS"
            else:
                return None
        self._append_csv(entry)
        msg = f"[{self.symbol}] {reason} ejecutado | fill {entry['price']:.6f} | PnL {entry['pnl_usdt']:+.4f} USDT ({entry['pnl_pct']:+.3f}%)"
        self.log.info(msg)
        self.telegram.send(("🎯 " if reason == "TAKE PROFIT" else "🛑 ") + msg)
        return entry

    def _execute_signal(self, signal: str, current_price: float):
        if signal == "buy":
            with portfolio_trade_lock:
                snap = shared_state.snapshot()
                total_equity = snap["total"]["equity"]
                total_invested = snap["total"]["invested_usdt"]
                global_room = max(0.0, total_equity * self.cfg.max_total_exposure_fraction - total_invested)
                with self.account_lock:
                    if self.account.in_position:
                        return None
                    equity = self.account.equity(current_price)
                    risk_budget = equity * self.cfg.risk_per_trade_pct
                    stop_distance = max(self.cfg.stop_loss_pct, 1e-6)
                    risk_fraction = min(self.cfg.max_position_fraction,
                                        risk_budget / max(equity * stop_distance, 1e-9))
                    fraction = min(self.cfg.trade_fraction, risk_fraction,
                                   global_room / max(equity, 1e-9))
                    if equity * fraction < self.cfg.min_trade_usdt:
                        return None
                    entry = self.account.buy(current_price, fraction, self.cfg.fee_pct,
                                             self.cfg.slippage_pct, self.cfg.min_trade_usdt)
            self._append_csv(entry)
            msg = f"[{self.symbol}] COMPRA | fill {entry['price']:.6f} | exposicion {entry['qty'] * entry['price']:.2f} USDT"
            self.log.info(msg)
            self.telegram.send(f"🟢 {msg}")
            return entry

        if signal == "sell":
            with self.account_lock:
                if not self.account.in_position:
                    return None
                entry = self.account.sell(current_price, self.cfg.fee_pct, reason="senal-cruce",
                                          slippage_pct=self.cfg.slippage_pct)
            self._append_csv(entry)
            msg = f"[{self.symbol}] VENTA (senal) | fill {entry['price']:.6f} | PnL {entry['pnl_usdt']:+.4f} USDT ({entry['pnl_pct']:+.3f}%)"
            self.log.info(msg)
            self.telegram.send(f"🔴 {msg}")
            return entry
        return None

    def _process_tick(self, df: pd.DataFrame, current_price: float, evaluate_signal: bool = True,
                       status: str = "operando"):
        now = datetime.now(timezone.utc)
        shared_state.push_price(self.symbol, current_price)

        candles_df = df.tail(self.cfg.price_history_maxlen)
        display_candles = [
            {"time": int(row.time), "open": round(row.open, 8), "high": round(row.high, 8),
             "low": round(row.low, 8), "close": round(row.close, 8)}
            for row in candles_df.itertuples()
        ]
        shared_state.push_candles(self.symbol, display_candles)

        risk_event = self.check_risk_management(current_price)

        # Si TP/SL acaba de cerrar una posicion, no permitimos que la misma
        # actualizacion vuelva a abrir otra inmediatamente por una senal vieja.
        if evaluate_signal and risk_event is None:
            signal = self.strategy.compute_signal(df)
            self._execute_signal(signal, current_price)

        self._push_account_state(current_price, now, status=status)

    # ------------------------------------------------------------------
    # Publica equity/PnL/balance/posicion actualizados en el panel. Se llama
    # en cada refresco de precio o vela cerrada para que el panel web nunca
    # muestre datos atrasados -- incluye el caso en que un take-profit/stop-loss
    # se ejecuta entre backfills completos.
    # ------------------------------------------------------------------
    def _build_closed_trades(self, limit: int = 20) -> list:
        """Empareja cada COMPRA con su VENTA correspondiente para mostrar el
        historial en el panel: precio de compra, precio de venta, PnL $ y %."""
        closed = []
        pending_buy = None
        for t in self.account.trade_history:
            if t["action"] == "COMPRA":
                pending_buy = t
            elif str(t["action"]).startswith("VENTA") and pending_buy is not None:
                closed.append({
                    "buy_price": pending_buy["price"],
                    "sell_price": t["price"],
                    "pnl_usdt": t["pnl_usdt"],
                    "pnl_pct": t["pnl_pct"],
                    "reason": t["action"],
                    "time_unix": t.get("time_unix"),
                    "timestamp": t["timestamp"],
                })
                pending_buy = None
        return list(reversed(closed[-limit:]))

    def _push_account_state(self, current_price: float, now=None, status: str = "operando"):
        now = now or datetime.now(timezone.utc)
        with self.account_lock:
            equity = self.account.equity(current_price)
            balance_usdt = self.account.balance_usdt
            in_position = self.account.in_position
            entry_price = self.account.entry_price
            qty = self.account.position_qty
            history = list(self.account.trade_history[-30:])
            stats = self.account.stats()
        pnl_total = equity - self.cfg.initial_balance_usdt
        pnl_total_pct = (equity / self.cfg.initial_balance_usdt - 1) * 100 if self.cfg.initial_balance_usdt else 0.0
        invested_usdt = round(equity - balance_usdt, 2)

        if in_position and entry_price:
            floating_pnl_usdt = round(qty * (current_price - entry_price), 2)
            floating_pnl_pct = round((current_price / entry_price - 1) * 100, 2)
        else:
            floating_pnl_usdt = 0.0
            floating_pnl_pct = 0.0

        shared_state.update(
            self.symbol,
            in_position=in_position,
            entry_price=round(entry_price, 8),
            equity=round(equity, 2),
            balance_usdt=round(balance_usdt, 2),
            invested_usdt=invested_usdt,
            floating_pnl_usdt=floating_pnl_usdt,
            floating_pnl_pct=floating_pnl_pct,
            pnl_total=round(pnl_total, 2),
            pnl_total_pct=round(pnl_total_pct, 2),
            last_update=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            trade_history=list(reversed(history)),
            closed_trades=self._build_closed_trades(),
            status=status,
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
                f"Trades: {stats['total_trades']} | Win rate: {stats['win_rate_pct']:.1f}% | PF: {stats['profit_factor']}"
            )


    # ------------------------------------------------------------------
    # Carga/recarga el historial completo de velas via REST (arranque, cambio
    # de temporalidad, o el refresco periodico completo). Siempre termina
    # ejecutando el pipeline de trading (senal, TP/SL, estado del panel).
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
        try:
            current_price = self.fetch_current_price()
        except Exception:
            current_price = float(candles[-1]["close"])
        # La ultima vela REST suele estar abierta. Nunca usamos su cierre parcial
        # para fabricar una señal: eso evita decisiones inconsistentes/repainting.
        signal_df = df.iloc[:-1].copy() if len(df) > self.cfg.long_window + 2 else df.copy()
        self._process_tick(signal_df, current_price, evaluate_signal=len(signal_df) >= self.cfg.long_window + 2,
                            status="operando")

    def _refresh_price_rest(self):
        """Actualizacion ligera de precio y %24h (sin recargar todo el historial de velas).
        Se llama cada cfg.poll_seconds para que el panel y Telegram no se queden con un
        precio atrasado respecto a Binance, sin pagar el costo de un backfill completo
        cada vez."""
        price = self.fetch_current_price()
        shared_state.push_price(self.symbol, price)
        try:
            pct = self.fetch_24h_change_pct()
            shared_state.update(self.symbol, pct_change_24h=round(pct, 2))
        except Exception as e:
            self.log.warning(f"[{self.symbol}] REST %24h fallo: {e}")
        try:
            self.check_risk_management(price)
        except Exception as e:
            self.log.warning(f"[{self.symbol}] Error TP/SL en refresco de precio: {e}")
        self._push_account_state(price, status="operando")

    def run(self):
        self.log.info(
            f"Iniciando bot PAPER TRADING | {self.symbol} | {self.timeframe} | "
            f"SMA({self.cfg.short_window}/{self.cfg.long_window}) | "
            f"TP {self.cfg.take_profit_pct*100:.1f}% | SL {self.cfg.stop_loss_pct*100:.1f}% | "
            f"Balance inicial: {self.account.balance_usdt:.2f} USDT"
        )
        self.telegram.send(
            f"🤖 Bot iniciado (paper trading)\n{self.symbol} | {self.timeframe}\n"
            f"Balance inicial: {self.account.balance_usdt:.2f} USDT"
        )
        backfill_ok = False
        last_error = None
        for attempt in range(5):
            try:
                self.backfill()
                self.log.info(f"[{self.symbol}] Historial inicial cargado ({len(self.candles)} velas).")
                backfill_ok = True
                break
            except Exception as e:
                last_error = e
                self.log.warning(f"[{self.symbol}] Fallo la carga inicial (intento {attempt + 1}/5): {e}")
                time.sleep(5)
        if not backfill_ok:
            self.log.error(f"[{self.symbol}] No se pudo cargar el historial inicial tras varios intentos.")
            shared_state.update(self.symbol, status=f"error: {last_error}" if last_error else "error: no se pudo conectar a Binance al iniciar")
        _report_startup_result(self.symbol, backfill_ok)

        # Bucle principal: refresco liviano de precio cada poll_seconds, y recarga
        # completa de velas + recalculo de la senal cada full_backfill_seconds. Esto
        # mantiene el proceso vivo indefinidamente (ver nota en start_services sobre
        # por que esto es imprescindible bajo el arranque clasico sin Gunicorn).
        last_full_backfill_at = datetime.now(timezone.utc) if backfill_ok else None
        consecutive_failures = 0
        while True:
            time.sleep(self.cfg.poll_seconds)
            try:
                self._refresh_price_rest()
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                # IMPORTANTE: si esto falla, el status debe reflejarlo de inmediato --
                # antes este error se registraba en el log pero el panel se quedaba
                # congelado mostrando el ultimo estado bueno (o "iniciando" si nunca
                # tuvo ninguno), sin avisar que algo esta fallando de verdad.
                shared_state.update(
                    self.symbol,
                    status=f"error: {e}" if consecutive_failures == 1 else f"error: {e} (x{consecutive_failures})",
                )
                self.log.error(f"[{self.symbol}] Fallo el refresco de precio (fallo consecutivo #{consecutive_failures}): {e}")

            now = datetime.now(timezone.utc)
            need_full = (
                last_full_backfill_at is None
                or (now - last_full_backfill_at).total_seconds() >= self.cfg.full_backfill_seconds
            )
            if need_full:
                try:
                    self.backfill()
                    last_full_backfill_at = datetime.now(timezone.utc)
                except Exception as e:
                    shared_state.update(self.symbol, status=f"error: {e}")
                    self.log.error(f"[{self.symbol}] Fallo el backfill periodico: {e}")


# ------------------------------------------------------------------
# BACKTEST LOCAL -- utilitario determinista para validar la estrategia sin
# depender de Binance ni de datos futuros. Se usa tambien en las pruebas internas.
# ------------------------------------------------------------------
def backtest_ohlc(candles: list, config: Config = None) -> dict:
    config = config or cfg
    if len(candles) < max(config.long_window + 5, config.rsi_period + 5):
        raise ValueError("No hay suficientes velas para el backtest")
    strategy = SmaCrossStrategy(config.short_window, config.long_window, config.signal_gap_pct,
                                config.rsi_period, config.rsi_buy_min, config.rsi_sell_max)
    account = PaperAccount(config.initial_balance_usdt)
    df = pd.DataFrame(candles)
    for i in range(config.long_window + 2, len(df)):
        hist = df.iloc[:i + 1].copy()
        price = float(hist.iloc[-1]["close"])
        signal = strategy.compute_signal(hist)
        if account.in_position:
            move = price / account.entry_price - 1
            if move >= config.take_profit_pct:
                account.sell(price, config.fee_pct, "take-profit", config.slippage_pct)
            elif move <= -config.stop_loss_pct:
                account.sell(price, config.fee_pct, "stop-loss", config.slippage_pct)
        if signal == "buy" and not account.in_position:
            equity = account.equity(price)
            risk_fraction = min(config.max_position_fraction,
                                config.risk_per_trade_pct / max(config.stop_loss_pct, 1e-6))
            fraction = min(config.trade_fraction, risk_fraction)
            if equity * fraction >= config.min_trade_usdt:
                account.buy(price, fraction, config.fee_pct, config.slippage_pct, config.min_trade_usdt)
        elif signal == "sell" and account.in_position:
            account.sell(price, config.fee_pct, "senal-cruce", config.slippage_pct)
    final_equity = account.equity(float(df.iloc[-1]["close"]))
    stats = account.stats()
    return {
        "initial_equity": round(config.initial_balance_usdt, 6),
        "final_equity": round(final_equity, 6),
        "return_pct": round((final_equity / config.initial_balance_usdt - 1) * 100, 4),
        "max_drawdown_pct": round(_max_drawdown_from_backtest(candles, config), 4),
        "trades": stats["total_trades"],
        "win_rate_pct": stats["win_rate_pct"],
        "profit_factor": stats["profit_factor"],
    }

def _max_drawdown_from_backtest(candles: list, config: Config) -> float:
    # Curva simplificada usando el mismo motor de señales; no usa datos futuros.
    strategy = SmaCrossStrategy(config.short_window, config.long_window, config.signal_gap_pct,
                                config.rsi_period, config.rsi_buy_min, config.rsi_sell_max)
    account = PaperAccount(config.initial_balance_usdt)
    peak = account.balance_usdt
    max_dd = 0.0
    df = pd.DataFrame(candles)
    for i in range(config.long_window + 2, len(df)):
        hist = df.iloc[:i + 1].copy()
        price = float(hist.iloc[-1]["close"])
        signal = strategy.compute_signal(hist)
        if account.in_position:
            move = price / account.entry_price - 1
            if move >= config.take_profit_pct or move <= -config.stop_loss_pct:
                account.sell(price, config.fee_pct, "risk", config.slippage_pct)
        if signal == "buy" and not account.in_position:
            fraction = min(config.trade_fraction, config.max_position_fraction)
            if account.equity(price) * fraction >= config.min_trade_usdt:
                account.buy(price, fraction, config.fee_pct, config.slippage_pct, config.min_trade_usdt)
        elif signal == "sell" and account.in_position:
            account.sell(price, config.fee_pct, "signal", config.slippage_pct)
        equity = account.equity(price)
        peak = max(peak, equity)
        if peak:
            max_dd = max(max_dd, (peak - equity) / peak * 100)
    return max_dd


# ------------------------------------------------------------------
# PANEL WEB (Flask) -- estilo Binance, con grafica en vivo por moneda
# ------------------------------------------------------------------
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/png" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALQAAAC0CAYAAAA9zQYyAACLRklEQVR42uz96dNlS3beh/1WZu59znnnt+bp1p3n7ttzszF1A6RAkSIpWaQEUuBgChQFhR0yRUtWKOQIWZZky7IVDlsKUrJJUJxJCaIETgCaABoNNHq6Pdwe7jzfujUP7zycc/bOTH9YK3Ofgv8BfaiNaNzb1VX1nrN37sy1nvUM8OB6cD24HlwPrgfXg+vB9eB6cD24HlwPrgfXg+vB9eB6cD24HlwPrgfXg+vB9eB6cD24HlwPrgfXg+vB9eB6cD24HlwPrgfXg+vB9eB6cD24HlwPrgfXg+vB9eB6cD24HlwPrgfXg+vB9eB6cD24HlwPrgfXg+vB9eB6cD24HlwPrgfXg+vB9eB6cD24HlwPrgfXg+vB9eB6cD24HlwPrgfX/+Ivsf88uB5cD64H14Prf3E79Oc+/9MvkzM5Z3AeL+BE/50kQCQCkgFx9Q86B4kE6K8FBETo7VccCZwjZwESOWccgiBkB04cpIwTTwSQhOBwJFKGLOCdJ5IRwCPEFO1vT3jvyTEj3hFzxjuHE4gx4byDnEFEvxdAtrMo67+IiB1NguiXw3lHiuAFcEJMGQfEnBa+u+hfrZ8aLx5xkFMEPMkJxATe6efve5zLSBJiBlImOQgipHJIuqwfMELOkSyC857UR5wM912/uX4nETtYc9b7W/5rKs8KElm/N4lyG/R7C3n46XprxP5s/XMJ+y3EDB6HeP29yZ5HlkzfRZwLZCJkR85Rf0aGRERwIB79ciBO7GPn+mxy+Zw5I+JIMQOx/Hj97t4T9QYSCGSXiCmRU8aJPruYIIB7Xrx+UWxhIPqV8SCuJceEc4JkIeaE9w4hQbKFIQ4vDjK4WsAkRJzdSFdvphO9QU4cWcCJkAHnvN1cTxAh2mJtsr4M4j2tNOSUyFn/bh8cMSdd8M6TU0K8w3tBcibbQsg54Zyzm8bwSJw9+ZzJCM45RDLOvoT+70D2OCf6amVIOeJzA87ukziS05+lD9ohzuFEkMaTYkR8oEFIOeumAfpzRPTzAVkSfcoE3yI5k4K9dLqq7X2MSLl/Odtid/Y9oq16fSbihK7v9UXLEbLgRXBeiGl4qUUg5UwW3aCcoPfP2WZiz1AEIplGnH4W0Rc648B5csqQbduyzaSLPcEHBEdKyV7ErI8hCyknxIGkTEr6zDy6obqcAKGPPc45fYYp6Z8Rhzjw3un6Ed11Q8ox0Sec19evy9kWrz5kybog+l7fHoC+i/aW6/vlfWae+3LL9YcJ9W1Nscf7QLK33XkhpUhMCUJDSokYwbtgN97e1Jxxkkk5kaP+TF1TDqEn2tLULxNtAWdirztAyuCdQ0To+jmZjMeTRXfYHAVvLxYCuY/EHPHe68sYe8R5MsJ83iOgix5HFyOZTAi6WHLK6CHlyWQkR2IqW2UkB/1MkiHGiA8Bh6OPPV48HVlfVIQcI13fkcl62kTRz+49Kfb6bILQ9729eELG62kSO3t59bRJqdedUQTJiSSZhL7c2e4nOZFytsWt+2IfIwlPKC+9A5Keuin1pJTIEcQ7hEjsEiJ6cuQEwQXboTN9P9MTJ4P+9fozUk56MPXJNkMh5V53nQQpDZtL3/f6MqRMdo6cdBNIOZFiJIluFkEEJ+Jxorsd9mDAQU56E8pxlUu54ezDZN0p7GwoO56uj4yIRySTvdc/J7p7pZxxtvBTsqPblQNVb57e3EzM2d5A3V2yYC+a7kCSbMGL1GMroze2Hus54yToDSNh+wuu/hndpUTEdhx9kcV5UsqI0/Ih2c6Sc0S8x5Hr0emcqy94OfLFaekjzpFTRnCIs3uWdQfzTjeS4RTJxNTjvEOsjsguIznr93AehyflhHMBrTyyfTLRky5b2SB2X+vn8biUySQt5bRWImds98u1LPM+aCVki15S1nInpvqExena8KLPKuN10QvE3NsurLt9jImcwDu99zFqmaSntt0n9Pvp5pAJzv460VWVgCxaDol4Ukq1EhBET76UEplyjOvbnvLwlLX2ZeGoLzuJnlV1F3FWr9bayJFTX+vUUsM5cl10utYEEW+/x3a5WtnZQ0zJjnZ0MeZEzFFfKqeLQRdWJNvbVV4+cqnJ7c87tH604zpbPUfO+nKVXSTp35PF/l5y/Q5Qbqr+bMlZdzgrnzKQY7JTIpOykLIQU2+nTtASxo7PlLXWzDFZn2BHttW+Uj+O3seejPiAOIgpkrPoIu8jKWlnoyewHvHidMGRspVoUgtnKadTto0h2Z+2Z58zpKivcspRnzO2ozqHc2mo5evGos83JbTOtWeqzyfSx976L6sARfTlR19825dw4nRt2X2VPCwLybqYs60NEbGX3A3FQ+0UxenNROru7L0H50nlTbEGT8TbQ3KINFaX6YOsL4QVOSnb3yq5NmNiJUUpwHNOtaZO9sPFBz3qsj4wsmgjUBsG3fVzfcn0JCg3V9BjqS5KfXN1MTptKPSXvd1cO8azHuQ56WIQ3WPrC6fNi/5dYruYZF0k+vlEyzfbKZ1zWsLFqPWu6HeCjHfaZJXvrA8222nlcD5gfzFS3vfSEFkZgJUl2MulP89eXjsFdKfLOAelLfXO63Pyzu5z1pJCBCRb2aY/SzcWLQV0sdsJJZ5MxEmy3dtbP+KszrcNJNv9wu65ozaJyf5ZGt6YF+6HlU3eCU7sPJLy3OzEdVgxVQ/hpEdCtg6S3vqN8igTKUVbrNhxJ5Ag9kPtq7usbS165/RNktKha7OYc7bvajuR1VHOKaJRdvEkue5cCHivnyll0aM8O0UmcunUXd1FxBoYbws0207nvN60lBIp6QsYU0+MPRktdbAu2qPHoreXTJEbKzHKA0i6hTjBSh672XZG5RT1OBUhSbT7mwjO1wXpcLRBd94s5T7psxDJ+tlsIce+0xcXq5PLTmuNsRTkQMoJ6IjWnzjv7zshdEFKfVVLESNkvDgSidhrn1BKxJijbXm108a7oE2hOFKMONFFllO2kqq2g7W0yVE3ihi1AnA+GNAgivaIYl+Q7TSzbjRnRTi8t1/XU8SVekWcfp1oH1jQuhqrc/UPSP2B5WZlEhL0TctE3UUzFd0oR4g4X99wsYeekzXmrrwdmT4l3R3KkS6Z3Ee8OGJKpHJY2e7unNhnKy+YohClg4+pIBj6s52Xuvh1Ny0P3+pTJ7YraD2aMbREtJkrJ89QfEitf2NSKNM7Z1WtHeMk7VyzPTQ7Sh1iL0hBZBQqJBu0VqBOsd3YmnXnBO8c3mmfoKWTVEgypUQfa2NAzHrEi/UQOeV68pZNopRkyeoqZ6dbzPrCFzTF4fVkshcLgziTVthWUpamj3qylSo2LZaghpClVBpxqd1ALvdGnKEaC1Vo+TUv9/39ghC8112jPKdsi7A8Mwdk5xGy1kJ20/LCkad/WVb4xnb6nMEP76O9kAZx5QKRYV1x1O7aC8E7fQOlwDxDXa+1nz6wWpOVv6fsE84aLcUxDAGALulRJYaJK2YK3movxBHtezvbPfq8+PJpeaKNWimZyhdT/LVsBHrKR20UU1Y8XvE+PS5FGz7baPR8LF8iYSWbNWqI1cILDVpp+hjq3FoLiyHFBiMWSFOb02DloiEIBTYVhWALHlzhQC92shoObke9964OmLWGpZ4mOWfwDpdYuKd678hiB34pacvebie3c9orWBlWmsVahkgaGm9cbUDB2VpLuIzuFDFqYyJZbGehlgXOsOlyecFqKK1NnS0wJ05RATsyCvacDRqKVlqQk5UwVul7R5JcECRrKnPZ1IYJvbh6NMaYQIL1do5kO5yINwgn4bIoppkzOUV9+DFr+Wo/u9Ts2LFeX2ZrcpIWivbipfoIFtkDYviMliH6+1PKRNuRsqFHupvbIrfaVqxWLN9ZnCM7sUVDLZNyWbzJ7mOKpJjs5+b6QunLqkd1+e8Fs47JMP16R6XiwaVEERlqfVehPCmwUynHrVGNBqNmFv+/aJszbDy2+Gxl1bslOdc+qEx2tFb3tipLiZrrwndWHlUA3cAAXfuC0xpNYShtrJIV7eVt1R1RjxndbZKUo9LVXaZ8S+dEP65N2lLWFy5Z2aEd9zCrSuWlET80bAVIKOVJRSDy8JC8Gw4nq73Fai7rDsmGRDkvFfpxNvTIOeFtEYqgx1ce6nCssSzHecpSX/CM1KMZhJh1ouednWA6NbFGWe5DcAqKkxde+NIElgYIq+mtgwCvTZoMZXFtKAMQyvTNXvYyrFp8+RTj1eepp4AOybA+BMNxnZUBFXtET0xFLZIOrRabMRuolQWq9w3bGKROBlPWk7vsFQq6pKHRTpEyIs6kYaJoCE+OWoqmXEq/tDjntBIq47w4vHiDw/Svd9ZRJoNZyMkAbqkdszhHgThTrU1yhZl06qg1X/DDpKs0jlqz5fuQEu3CrX72Tgco5cGXViXL0NSUmhFbDIY7O3sJYozgnI1MrdYUq2/tDU82Ts+lyTOYqCxEZ0cvVoPmegqVls/q3QKJWa0piA4gYhwWarZdy+lDrJ+xrIiyhAombDCAIgi6k4vIAqdM6FM0rNYtdP463CinTUqx3heRbF1IGV4ZrGl4e7YepBzzYkiHsxIgZSuB0GfEffDbgOcrpj5MQbV88vaSW9NNBpdrWVlbTBuK1RdCbDZh/VJ9eWwtidNNN5NxrgwrypQq5/rfXYFXstSmkTKKNo5FwRzzwtjbWW1aSwenE7mUoh1FsaIddWJUdjurvXSK6+viQLTWjwYpIVpvlwco1rQpXk4pRutOmusIudw2/bsKBquduL3I1pxiO2yOZaomOiyTcrNdPZUUYrM6ktJgoRNBUegtFbw0xgVYTL+biFdOhNWszgcbQwf9mVai6a9Dzr3CnGVzqF0YhoroCxtjJtt4ejhU9GXwrrJrKmRWpndOxOrfUo7Y6WeokGTjMCStq/T5a++SbKorTsuWnKN++lReJNvMspB7LYWwxr1Ol1MixQSpL4/N6nWx5z00mjoI0n+GsmOkJFjvoB/M2U5kR4t+lvI2lZtjyIBgi1NoChyWk+6U4ul6bZC888MQISl3wFmDVEEjsZq3lA+SDTC0ekns6BO3QK7Rm+JdaQ2ylU1louXqgEXsoZfJntjupxNBbegEuxdpwJrLSeHcQvVsZVESEL2jtmhdrUf7PlI2jeHTanngEbIPtZZVXoOz72ClijU+pWTTTdtBchUFSVEHTKVfcYY8lBIulalc/cwsfGctK8qGI+LqIA0rScrhriStWrpqI1+meGVC64aSRtEm450kxbOzDcS0sdW/zFWykjVRdjLnXBCMNACKdkrWosp6hmRDpMoyKdBHqfuSMZucK1Oc0qh5xJqFAgvhdPeJKRnUY2+8lMWdK9ZbjrAC9+hLYVi1E8QmdAPJSaynNTxV9IVLtuij/X6RhfFQKhCcVsneSx23lYY3lbH1wjEtIkiwFZEMH8fqaxkq0my7YBLIrvZ3w8JMuYL95cgsJBzdthzOeQPUqSWOLPTwBWIquzoiesSL0Pfp/vq+TFxDMLTAaxOaYt0RrYAa4MbKy1LehyKLbmC/2RQx28ZU0B4p/U1ONs5fKAntOZdGzhleX3s+rOk2ZMfbG5YNIctSSkiFOuvmkQ1lcX5hc9LXPpUSqawTrY3LCHlhBCwDujF04ArxOJLhwNgQJNUSJIlBXFkqAO6dUTTrZMkebHbEPDARdJTt7DP0leKIc7U8qFMkW5Di/DD4yVTUI9uNraSY0n7Y8b549KacBxzUsNXsbE5u3JNcJp9kcu7rjpLL1oeyRrF7pni1x8aOBAn2c40bu8jDHbaLihQoNm+3SajIQ4wRfEZK7enBBaUelGloKaMQJQmVpZyNmlsgU8gL4JVtGmVKWJq3gklbUyp6dNVFXxZw/c1WMpRTqLA4lS4cbem6oQtIGZ91GTsgOFvs9VTMA9PONkdrEOwUXkRsssJ2Uv8Hg+eMNllm51Zx6TESo/J6y5gyL4yDZZjBFzy6sNPKW61vjrcyg0orzMkYa6X5wduEqOx4pc53lfxVbqQ2lgXf9pVNCdkwdqtvZUBiclKCU+VGZ4bSwqZrzvgSyiew8bATxAVrYsrJoy+HDm9YgLCGe+q8t4UiFQaTSoLSRglXgRz7fW4BKtNOXpxT0hRafuneEOp4WbFcK/SdI8rQbYbQ4J1bGG6IDc2GnVLEGVEs3g/NlpM79dpPGZokAiF4fPB14ZaSxZER68WwMqNAiFLKmjIky1FZhCnjZXixKPx8q+/LYM57X6fPlCFM0lGe7TppmMPjBziJZG9dro1P2dm0zunrFJEy+MgDCpCMoJOtGxXvjLVmO64fsM2MkKPRJHM5+pyB8WIvRq67WilV6lDC5wrgSzmiK/TIsEBE+dXJeLci1rC5whuR2unHpCSbPNAnbPfT3dd5ryPfBVw6Fygsp4UJXF8btYSiGzFFfTFSHgg4VifHVMbjovyMARuzo1tqiVPZjt4v4Mb6UmZRqml20Kdu4VWjThyFRHBii00PfCeOmPrKfCyokBSUCxkGWgapBe+NF2IUBtsoF4UEzkqZ0ruVmYIzeYfYS6RIkAoLUlJagrWD9QQszaM4pyesg6CTrIIXlkmGGHy0gAQw1GGVMil6rNSi3l5ZL+VYzUZYcQs3PlZikMu6WzvjZTgbwceYK3GblCnjg7JQc05GIzUWWU56kLlgO5ryJsosSqeheUG14Q161Lc8F1w3LYzNjdfsg6svqbNdOhMr2F+QkcXGRB+UW4AbtakV408ImWRchWS4f4plw1DUoDRFFYYsUKOdW36Y3+txXSaPUfuHofuz32Pj+2FKWHb8wnMZWpuctGzVJj4PVX2ZXpaxtQ3rck5ISmTb7AqkWuYIpEjK+vLEZIqc8jPF4FAGaDQZ+CwLtbzzjhj7im44W13e1ElK+fWElBMkb6ynZAQaB4XZVt6IGG2R2TGXssqNytQoR0VH6n1M9pabTCr29mCsTE7WNNru5JxUHNt5qcdfsu1RXOEcyEL5WZQaugvnaG+9c0QUJ3fWpNa+uEzvCsMt5toUphiHLl4GaEhH3sZCQ7m4OokzspZN8rwhK3rCFu6LrwMEammkPYsSoJyNhrWViTHV8r4w9WJWJAUrUXQEnO1elEkltWQbqOG5fods087acdfpqU1gnZGxhIoV1yl/Qdyt1LAikpiS3dcCaSZjFQp9yqYuUQg3GhzsXCk5XH2DJJeJoDX01hi6inQsYs46idQypbASU6UFhFwUFSZBKtyBQpIvjZ1b2DEwSVQ2PDK7RRglVWRC16o3OMiOepsIequts8sL+jKtnaVSJAtM5OqO7Art1I4ibDImxpHIGXIPrmFBlVGQ51yVDQqPKdkql9KggviyIEFU3keqZdmw01G6eLdwrqK7ZEGACuKRsxKldNImVaOZjUNcWa3ldKiTMCqdtmJN2XZ8g03L+DfbwkzJTtpMpdCWFysnsQ1lYP+Bcp7FDWy7wnYL3oY85IH8ZDy7Wh4wEJqKTvA+QCHb4rSmWmqPILZhZasitfwQGQhH2YSOkiFkq/vt26f65up3Sinr1LRMW2JMpHLDbdwrTmrjVXjPig2jO7TNr8uudP/YtLQ3il6IYdvOtpHC7CtEn6ITy6I3mIK5Ukjqrna3tQPVis+O4qzohKhEp8CJFfWQAfcsb7ROuV3t0r33QxlBJkukHjtOfpc4dWCqlfuDE8RrPZiN1lh2xVxfMOOuLAh4F+mSuXbzQwOZU1IBcuEX1xoyVnKQOEfstdZU3adUNU+BHH2Qodwq8wIcIYTKDixs5WSIjyt9TtYxey6ELjKulDRlqRYUQryeLIWvUU5sVyh3xqERfcmLWq1wWGKKBu8sEPsFO0ULf8W+kxtOiZALKag8WCu8E8MumezIdmUB25eqyhNReSgu1xl9NoiqcPD017XB8n6BdVuOdJN1iR1H5WhKVodJGTKw0BeK027Y/q4i8UJENYxll9OzrpYczpQoaXjMpkaPBh6lKhRI5WR0enNTlUK5Ki/KJuKsJasTaziLqNbEAX6hiSu0vlg1RnraJchea8aK8YojexsaVOJWJaSb9s5e4vJrhV+Fw/lcFTgpF/X4wtDM+iFBCCrnNyzfWW+lE0HvB1GtKyeW98TYDcptV8bhigi5qlMyxKIwGDP0OeNdsI0t32cRUxpLpRCoMFidAOQ+JqXkQcGecyKUaVf5hZyVCC71mB5KDW+Ya0rZyDz2BsvQTCUR41Ibmd46bYz/7GyEnZLV6zmSUqxHlO6mVPRCSrNoeHA5ZaLNZzARbMq2O2dbiCaPLEOLWlOifNpceM9J0Z1k+HThpYj9TMluoIwWvgADUQkyeF933pxZeJGS0QYKDVIWiElWBLnCOrTvUnZH+31KIZAqLat18wIDrVBksb/LlSZJQFIZQ/sy4EYYNJK+NK0pEgzxwemL4bLUZrCQyAXlhEuyXTnH0psuDF9kgUGnTbQSxaISwgxV8gWEkIXJq51G3krUAjTkgZanzFCn/KMqlEafY8jlKF2Y5pRyaWDgcd8Wn40kLnmYIhY+ax3Xmt+ElE9Z9IBmIUB5CZwu+jqBWiRsG2+2z1LHuZSyA4PlSvlgSpJylOoDDcMUKRfFjL3dySaPfoG1LQPXWje0XIF9vzC1zBQyj8mJCj+5qOVTKUsW2H8mbfLe24shA4Rm31PUF8A0fLpb9gzk9piKOj8N00REb6eJC8gFDbFm24faAAoBkWgSLMETVNluEjvdTKK95HZ/jRdSBLwpRhM4U9mAIeiunokGaZbxv2LlCuSI2iuUEbqVCilFXMVRFCb12H0owxyodIJcBdWD2FKsp/ACwVeGVVrYPcwro+5qUhlNYlgp5AqhifNFaVUnVAXLrBq2BRql2gO42nyR1dOmYJmlKcxpweAlD+OKwiwz7fiwS0ppclL9HkUs4E1NXFXjCNkNIuBqwWDYaFHCWCFv5Yl1/8oWMpgw/67Bz0CEElzdrZ34SpAv2C5ZjAMidXpYDH8g06VoD8/qci+D4r08m0IbEJ3CFgZjMXuQUopY+ZMXCg4pTTtJlS9lNuAWCEnO1fllAUi0IUuVAiEVux4a4GJYIzb97U2kkIoUK9tk1T4nCMkDnY71XR6azGRC6fvlglpPi30vsSloyCYJ15tjO13lW8cBR02pTmY8gzRLxayRIsstzY2QSTERQtC9KJVxyoK8KLnBV8KYacjAi1C8OA2j2lyYVkZCyoZn4mozR1UGFRqoTtEU1TCTllLWUAgzi6dPeWGphillrCqGhCi2LnVn110w2SApgUtVeEpVtEvFhnOWuiHo3x112keuXhcDo8PbkV10gEYwX6ACFA6NCx7JrpZCheeREH35irBCgsKykvDGp04GwRUxrCu6yxi1zIipDB+Hz7bg0lQdngxiFCf4VD5DpnH6wjmPTVeL+5Wx/MRsDrwMDlC2MydJdUiWUq7EuEK8csFD0pM76NuUF4goCmMV7RZlapOl6t6qDnFB4e3LFm1cXLGasNBEY+7riN2JWzjmHRjrzok3Wb4uHB8ESWXBCRgVU6d1uutSTxRXPRq8C3VgoV4cpYlww2DFWHzliaQYDW4Kw4Ir/JbKFXDD8AHdSHOB5nCm+UtaUuRhJO7F24tgo6nMAtnJ2mYj1pdxs0KVSY9g11SCk9aVzpojY+HV8Xga2HIMggwtDxKOYH1/BrzCYEas93XTcAvU4LJivfGoTGRr6JcLXkfhhanHYOVQmHYq9Bioo1oaKGrV2aRU+xSzXTCBQConSsoLjhalrFtwkwCzWdMVFeqqp3Aw8n0MsCoMN6MS7E0dCv6hceQ+qp/KZVIt+gta4YbGwXixhU+iNbW7D/asO745GCkRykjdC7RVfcsF78rU0shGRuoZvFaNVlkX2KKUyqiJIlUulNNw91LxuvCmlzdJjLUBeuQbLB1jHLjLC3yM6irFML0ru3uWhRMoD+LeMi6nKKDFERcGD4WkU1nqhQ9e9by5koQqd8J6juwGhKXI1lxWbzlXyUuDMRALXPjCq3BO773u5lpqhIL6FHTFrBzKAi1suzq1KP4vC9ZkZb/z1tuUxriM3lMV85ZhVCYMI+3an+gNi3pYuWCS9LIQXeETmIWAL/zofF9dN3BmB/xaMjZIMV1hFXLq36XyrQG96VMhxfiqbYxJKtZcFoFJTRZeDuM5l5NAFlUoCtm5WhebwsI31Q5rwfBNjd1Mw6enTTBTqaG8Gk40dLcR3b1YhKyr354+XMmKRWN87MoXteMtmSqdck+yNmkVynRNtYGoukgbLnjrE7CeoY6RjVaqC7zg11rSIOBsniAo0OFcrt/TO70PfsF9SsqJkTIe45NLRlK0QYd+11BIEDnj8jCU8XHghystoHgd2r1aGAwVMlS2Blwn174gitWmIpCMh1zYadnZW2cLJA/q26Lvq+poPwwKyP1gE5Zs8uOGBqFK/g1yYWFnFSmUxUGMO0ivXNXz6Q8NCyQjKRibsel0R64UUjvCpRob2iIp/ngFR/eOnJ2xwxQtyCb7wRQ2znvd1cy6AWfmhgsWZLYp2gTrPo+xShiqoHx2dSGKeCNpiRL3jRYqeaDIVlWR9yoMKDiX6JGdk9X1vnQqTlEmIBvfwalNbPUJdIWiEIvRjZ1AbnixvVOhsbNFJ9F0hQwndGmq6yTTayOaTbgasw1OookVDADwxm3vY7QGXbWrPjQLGsgiFtFBDZXZ6IonaLWgSwLBlbc+JluMaYGoNBy2ToauuMtd9a5TUllnQzNfldEpCxKzWa2aPZUJXZ0bdtbKx3ADYlGMI8kQ7Vh0Rnv0tlBzKYGcN45v0B1MnHE6bIH7hZLJnHeq1rjWsgNOnWMPLgz0TnMfUqQgWRXoDItOVW3tXLJagyrOFXELMk5DdYqCQxY69sxAXF8kEuWB6ZbL/S6nihvI+IqXe+NrlClqP1hzYZCeKZBy48tKsWmlM1+/YoMWq+tScLr7OkMntM5XrNsVxFyKXVxxzFL+ThF2xJyJ0WyEs6NLiWhlRZ+1ge5zWiCMFdWSMyazq2VTyvqZC6GpcPSLTURItSbS9zp4Z5o0FY+qEica+ypUmQ5SuBS5DkCKWrfYGShbS6mFLvjKgyg7bvm9pfdI3miL0Xi/3tVpllSSd8GPnW43wVdnIAzWcsGRzHOisuCN85Fk0E/FlLVhs8+vUFM0jZqvlg4iQprPbHCjdXUsEiXX4GJHjJ0d66alI9ehj46K3XBvsu0qC9wGpRkUmw8xzYXiuM7+t7zApxDvqxnPwAIYcCRHQ069Nr/Gasu2oMuCJHY6Q/FedaAFBYoZn7VeCEBjdaw3iwJFQozlZ0MbFnSh5U1M2h/TZyGFTN9boxwdqe+tTBC6lKpyv3CbVShcmr1cXZ3qHNIcolIe1OaGckStHYxkXmyh0FPRlBG+Lvgqw2EwXFz0TwDU29kgrFTgwDyYimA+0UWqHisrzEjcxvhLSWsp8QU79tbIKOHe+0YRDENdrBSkd/opYzkNtDirzqYDqGXTqIp4DNBTjhhJaxAWSNSFR9Hu2WQueWuwKMaN0SBHRz/vF/zzzPUpK0TlFnbZwepAyNnISVLqyCK1zLV8KbMrK8htERVDjGgmOQmXxKbryazcwBU67IIHhdooJBy9DiiMu+77SJzPFeMxpyqXsu4ndZBTeNW5qs2zseEk6heIKRKj0FtpqU5Y4E19FE0sklyqMC5V4T34uNSvW2HcZGWHVg/B2dRFsdRiHmL1q/E2HGWrL6hCNNql1Lp68UiodlLZmrmiOC4dswzyp5iy1lo2wKxdevEusx3NlWmcd+A9zjUk54kn15Ezp9VE0j5jNhSEDNGGMNVpXYYWrBzbXoYb5KoHtbltih7l5EzuE6mbkY6mpOMjODhCZjOCdzRmu1Aaw0wmNy1LTz7F4ShwNJsjoamuorJAdq87rwxmk0lyFd6ksnuV8quQ7M1RtKBSzuXqF5eLrWYys8ysY2tnQy7Tgdd/upTwkhl5B3duInev0ThPYM7FC0tMRoKIeh+6VLjwhpAlnTe42tO44eU046BZVL/snOBoKrz93jbVp7+qU5yhLXbvF/AUqkMXlX4Qy4CuePo5RxjsuQw6Kf9ejztXOSdSyETmVZHKHF8WzBMr1CODZ0POFRkZCDJ+IPontJkrI04j/rjizeGE5JTAn53uzD6M2I89p37/T+K/8KPsHc7qjZRc5YC1Cy6MOFedf3KFEou6vI5ZHdXVKDupx5rLCel6mM1g/4j5nbvMP7xO994VDj+8hts/YtKO8G1Lzpl5yhw3gRN/+A+wfeE8W9t7tM6h/bY5CRXos4oWxEzJF9xOzTm/yiDJ+KxCCkFhVI+iG9qwLUBr5v6qQ6KkC1i0lPBSFje0zjMHZJ7Z/oW/yujuNXJKPHJxzP/2538P7VIZ6Oh/KkVi0dDLSjs97Z1xvHK1C0uxpxmN+LVfucIrr3zIaGldp30WPwLQ52RWacU4xygYxfWpELkSVh4vCC1iJhSTF8mxSsSHTkRdP72TAT82eAVjWw1vjiAG8wy5Ife7+Liq2ra/O2r3iqfK8wsc6H3QDdUNu1Y0h8ssjj4l+rUV/PPP8s5OYv9QqiRLKgZpsQqyaI01GKfkhXq31r8LO3VhlXobhbmc8eMWWVrCndxk9Phlxj/yKZaPD8lXb3D44vc5fOllwv4xS01LkyPbL36H91/6Nud+7k8SPvVZrrz8On4205wYelW5O6dTOadGN857CF4NHZ3DNR7vHa5x+FZPgyZA4x2tFxqfacBOCk8jQhCzbBOHy5lg/+6LefxwINB4z8Fxx9VZort2Bd56nTNNw/zogE+88AQrJyAd3kJSj/QziHNSnEOKEBM5zrURj3V4YLBntD5DyFNhtnNMeOZ5vv6176mwgYF24Lyj6zoyZUjjbW6Rq8Imm/VyH5VIVZUzaSCvhaLKLl1/7DrVpgnVb6MC3bKw5RtclBi6+UVozxkBPxsVs0BtushULpOxrr68LHnQw6Wq8HYG53lEQn1LZ/MZ7TMvcLx+ksOrU8iNcm3r9FHrlFQmmrY1p6pPVEccJ4PMR6p1cPFhLpaypf6yhiiDpMRR7oDMyHsmjzzK+jNPcuYP/V62f+NrHHzte4ymM9bXN+muX+Ht/9N/yuW/+G9z/nM/wYcvvUw4npn9rrpoJu/UxTUIEhotrRpP9g6JQZGJKmhWuqsknQ+UvBZneLAXoTHlCKnHk2ltgugNXnOFn5MdV7b2eOvOPcanzhG/8S3Wt7aYn9hgY0X4yHOnyfND6OaQ5uR+Dl2PpA6i/jsx4mI0ckYxvR+MNvv9xOHLt2geuciVG3PeenOHyfLZOg9wpYZ2oXquRKMSY7zyKuTQQXwd5FX2ZAEZisYsWTHvvFRC9TAPyoPdVh5GkkoRdgvzNqkWTVVNXYNpiuzHgEiDaTDLW3UVLYvPG2Hd2HA+KJSGUwJSSswE2o88y72pEGe63lzUvztH7R5ydrjkFUqLDMblSaVP+p+iyrF/L3zEVBhTrnrC1c+eyijK45InzjKH96bcvL7PrWaJ1T/6hzn78z/L9JGL7M9mrJ48w+bSGu/9X/8f8Ju/wZlHLmtOzTxD9JC8KeEboKlkdZ2IOm2CnRvQnIpDl2cl1WizeKxIFiQq5Tdkh09CSIL0VoOJY3ca+d61O7x8+x7dZAm3s8Pul7+C6xP7e3tcPNtw7nxApnvmPZcg2ikcQXpPTk6Zi0ltKXK5/9kTZ47jdw/ovn0LduYsPXmZ3/ztD7m3NVOqRGJQ0tdUhGTKJnOFssehShZdT967+7wCi083KRMGQvXAXSh6MpVK+UpqF6MPFvVRKnCQyQpq/ISpmoP3Sq4x8atzAzaZi+Q+Dx5xKvUXJOhR2yeFp6r3hvlP9LMZ/akTuMefYHtrRuy0VpQq5shEsk3irDyqVFZ0J3fFcyFXAYw43X2z/C6fNqv/nTgap3W1s9G+ZCX4eHFIzBzdnXGYD1i/cJ5zP/+vcevv/s9s/8bvsLyyxvH+Hh/8F/8Fj/2H/yGbDz/Ozlvv06SAy96SqsyhqAhtY1/NcgKZlHtVQWdRroN3uqP7YkLj8JLIQQG1ko/lammlJUkvwtXtA97d2ec469CoHU2Yffdb5OvXSetrTI/2eOH5hxC/Q5zvKcd63usO3XeaQhCTlhp9QmJvU0Wt/fudObP3tvBbxzTZkc8vc9RM+Pa3r+DbMX3fgRuZIdHifKoQzoY+xhnPvk+xmDkNOQqVb6RrL+SkA5Wiii5KkcLlTTFVyRVSxt2L3sAOFwxHLQ2f9yYezThXBiVCH22CKAPFMlcDywXbLoQ+F+J6aQxdHYlO+zmjjz7LbGmD2fUjHEHnCF6hntGSo11uwORTxW+xqKeLaaErZesC18QviEkqSzdDn2EaE0ddZj7V02XU6JBBonIhU0o4yfSdcPfaDrtrI87+mZ8hN55b/9M/ZmVthe5on6v/9V/isf/s/8zs3Ak4OKJdmhBGDb51+JFnNGppGk9oHU3b0HiP94IPEIIjeEfwGn8XvNBaHewFWgetczRG3w1GpPcIfTdj/3jOW9vbXD88pndOOdcZ2h62v/oNmuNjuuVlzp1wfOqT55AQ8atrmA8C9LpTpz5C7JA+QteT+w6Jc9LeAbMP7pKuHTCaQwgt0/kx7eMP89I7u7z19j2WN04zjxHJvVXRvvJriAtZNjVmxOmmUqqGOCiRFk1uRDDYrsBdhRRV6sjF0MzS3MnAkS5st5jUkkBqLkSuxiQ5Z7DMwSKWEe90jGyK8OLfMYhEqaNgPXZ1Ipidg5iZj8ac+PQn2DrISK8NVXFK8svCZj4kv7pFDsuAqiS83E/UcWbg7RH1pLDv6My7z5fATy80jcONYLzcMl5r6TcdBzFzbXfK1lEHrjF/iuK3oWLUo9v7vHdwxOWf/Vfoj4+580v/mLWTJ7lz5Qo3//p/x5m/8Be5dzij2T2mbXuCy4QQGIdjGkm0zjH2TuvfnJDUExD792Rsg0xX3Dpz4lgG/77M4Gc3i3NOfeZJ3g0dNw6P8D7gcyb2CT+ZkK/d5OgbL3JuPGZne5ufeOEsJ1bg7reuE/zYatSwoAK3YE6tB2A+Qw72kVtb+P0pk9AirdbS/QiWHj7PP/rLb7FzkFg54egjONE0row5aJmeMSep2lKy+X6YelxJwJjipfB7XF3YobrsFjd9s2rSUJy+2gekIZpoSJAy3oMzbLVPgx8aBsP4YixjfIhi7l08hF1WgL3WzTaTl2SeeU4WtIKBrpuSH7lI+/DjHHw4N0614pLz2LO+usz0V7/E63/tG4wf+0n6JEPXLcMQo4h5q/cDC/56LtTvKmU3dzBaEtZOeM6em/Dosxv86DMbHKy1vHzvgDvHZjyZEin26hzkMt3WHh9K4uGf/VfYf+dd5i+/zObFi2x9/UVO/eQPcGsXufbNV2jFE7uO3M1Jc9vxukjuevI8krs5dB30ne6SXafstoI09GrCLYYLFxIWhhqsXTjFv/xX/vfsjDsNWUrV2odRMyK+/hr+YI9w8iRpusMf+ukn2PrOe7z3P36b9eUVnES8iQycc/hGrfmC143Bec1sbFpwa2Pz20j0fWR8cYO7/RJf/eYNJssT+qQbGlGj6aKpmJQHU9aTlaPJWVZlXGBH5oHGXBAqw1tD6nu8HwJxYGHKUyROKVULgpL1V1X8KSPBVXeIQl91OVfaY+W4LmwbyTi4qUQ0FFGnE5vG5erPXM3BHRzMpyx/7Dmmfky/u493ba3JaYWlNOX2d1/F99uM+23S0sN0NeF1gUXqis+EqydSQTmG0qeQdHRSmjrYuZG5d7XnB9++zebZO3ziM2t88rOneW8055Wb+zqiTwliR4pzQnBM726zvbbM5T/5M7zxf7vK+OCQ8fIKd3/91zj1v/mL7J4+gb+7T3Ce7Dy5KZ4GiewCOSSIYyUVxQh9IsdOWXQpIklfIFIB3RPESIqRMG7povCpn/nnmJ9e5vjOviYmmJlLcp62y9z65rdYHo3oYuRjHznNx5+9xFt/721W104zmni8SwQP3mWC10mhDxnxJp714HxSYwxfqAsaqjl+/nG+9o1ttm7sc/ahsyrAzkU2ZqP+SkQ2jxRDO5IMqWfVT7C6gg8VhMYDJpxzJWkpVtO7FM3dcQHtkAXz8QXjLlOKKEknJxl41VUMW73GBxrgfZkr5e92epSYFav+AV8Bh4iQ+p5+bYmNj3+cvTsdITl8ApeU7jpe8cjVq9x78xrtJDC7+yrSbVWDm0o1cMZRNgN0nMJj2RUzR7cwpq+uL3jvGbcty0tjNlbWmO0v86Vf2eOX/upbLN/u+dj5DbrpMf18RuynpBjpTXO3de0m4cmnOPUTP8Ys9iytr3Hw6iukt19ncukM89kRuZ+TYkeKPanvSX1H7nvoe3LsSbFX19MUa6Raikk9oA19iKnXnb5PNKMxrp0wWmk5+4lHubq7raFMfarpX368RH/1Okc/fJnVtRXmx4f8xI8/hTua0d87YrQctI72Ce8jTZMITcS3PTLq8KOEGydkkmAJZOJg4pDlQBoLcm4dTj/KP/ynb5jNWqDPJb1hwQulDGrSkKmiu/IQDZ2lxHdwX/5WzoNdsbHf1YqqEG6q2WLqyTnina++ZznZMKTCfTrOTLkC1zWl6r6sCkKVKOUFn9+8qDwoxo5FspSHgHUncHh4SPPoI3D+Moe3Z5q3HR0+Kfl2czVw/L1X6Q8Tvl1B4hFx920Cs6o8X1Ar3O8dknONXqhOnIWu6gUJHmkdqUVjiX3Ct8La+irbO0v807/7PvN3j/n4Q6eYHR9YyKZitCnDfNZxa2uXE1/4Anl9g6YZIbMZe1/7bdY2JqQ0h+kU6TqY91pW9FGbrr6HvsOVX4sZiWmAPvOQ2ZKz4EcjRhubNKubdH3m7BOXkEub7Bzsk6IOJmKMdDkxCg1HL36b5uAQCS1Lo54f+fRl7v7wHSYpsjRKLIXEUhNZanvGbUc76mnGkWac8ZOEmyT8JOGXMrIssOzIS55+FGgfPsf7VxM/+OGHbGyu1HlGskUdbQ2lytIs+tNiSjTwoQcxXK5BsOTS3ItlfZs5YkkErRkeBRgvhouGO4uppJ33lYwy+LdhfGo3kF6qjEaHKIWBNpDuNcwzFrGqG9R0UrK7U0R6OJzPOPWJj7NzIHQHicYPyuNmSTjBlNdeuYZM1sihUeOV41twdAVOPk4nxuAzn2GRYaJZShBkyCQsaExMCek7vNFhXbPAohNoRy0xn+I3f/k6f+RPPcozl07z2tVbtM2gt3QxsXPzNmuPXGby7DP0X/s67WSFwzff4uTv3UfObHB8a0u9r4OdxRHoBXrIfTLM2dHn6memOHAOOBIhCqPeE0ZjJXeJEN2Ux37sBXZ9pJ9HmqB87YSQQ8DtHLD167/B5rjh6HCfn/7cZR47H/jwpVusPzIhBJ3ONj4jvldDTCsr8EIOFuPhhBycQqs+DMGml87zm//d+xzuR06eGpHEV9psoR9r3yVVnJGqdtBqa4v6E2e0XVl08ShVgJ44QQ33+mraUiX/JYXIFkxxtEmxrx5zMXYgvropZSNii0sLnhFU74myaVfucU0mlapuHnxhhmPfOUc3O8adOcH4uee5cvUY+obeLKRi37N+qWV69R3uvr/LaGVTMVvJ5Nkxbj1y4hMnyd2xTQgdLrgaJpRdyfDwZpTulFJZpm3B0Utg7yBycPMAuTOl9U3tHbIIjXfQn+R3fu0mf+hPXubK7bscz3vFqo1H2c96do+O2Pjo89z82jeYLC2xfece3LzG4z/yUfrdfTWtkbRgMqNPOKW5GjFm0/aZ518GZtkR2xGH33yP/uouKQRyjsRuxuq5VU5/4lFe29upqFPOGpQ6Wlpn+jtf5/j7P+D044/STff47CcewY0DZ3/yE9WccpgxxBo/rbVtXjArYzj5XLEjdhxPV/gn/+SfsryyRMIaf1HJVvXyyJZymFMll5VSNC2YheYShR0XlHNVBqhrJkh1TJIhqrbYuy5AdLnCamYAbrxoVRFE4yf7YcUaid0ZX7PAcos5eFK+oOkINd9Ep2aD3Zd+7uOjY8Y/8mmO2k2O7h7SpInN8NWqdnO15davvkPsPXncQgq6g7qGjY88T3r9N9j+4t9E2rF9Pgt+XxgIDWoZT5ZoA4lE0zjCxnlGn/4xNj77BQ6ePsfeK/dopq4GkWbJtKPA/kHi6lsHPHb2BN97+ypt4xTLN2LTwdYOJy6dR5bHjKbCeDTi9t/6G/ilsW0m1DH9wIJJw9Ebe6NtKqUgTg9Z/tHPs/rP/a+4c2+HUduSnWazTI+PefLxh2Fjmb0PbxKCp++zkYCEtSzc++o3mIxbAMajFf7q3/oOf+NvfdWgR61nGyt1fTlxjaNchKk128XcURUw0kFZ32XuXN1jdW1dVfrVet6bBtRb2FJJ6DWjSyn+XW4YzJZy0A1yrCI+SdbuhJylsuOkmoJ4U2I4syuVmqAqCymj4nW0qoSiWPO5XR7omynrcKVagdsxooSIXA0PpQYU6bGVzPwkpUzs5sy84+RHP872diYfe3LItXFtNh3t8T63X71Js7SqpY0ZcPszp5mcXOfm3/tVZh++TVhathhi44nIQoBxzb0r5o/JdsdEnn0H91u/zNKTT/PQn/+3Wf6Jf54b37xDO1tAeRwEJrz5zoxPPLRE61DYzcKMJGemu3u48yeQlWXy8ZTlpSX27t5ievvG/x9BflAM5RrkU5LWtQmMTI+OuPDHH+Pw6l3cvMeNx7p5i+CXWp78/Ee5fbyn3yEWBCHiJhO4fovD73yHzaUVnPN4PNduHjI73Cf3nZnQJ3zOOKOjysICrtnp9pzzQr5uadZ8aNjYXCNbkkD59eIvnWugaq6qnMLbLtYRuU/DSVXITr+bR27IWFD+BvrwqpO7uRZVY5bBgEHraRlyUGQxu1vtvWqWoOn7Fp1DB1dTV8uZUnfnhbQGL6GeFF3X0Vy+jLv8FAfvz/DRVROYrptx9uSY7TfeY//mEaPlFZ1giaNPkfVHH4E77zG/e4XVs2e1066mJYNXRt2dc43eG/5fER/0kemVK7z9H//7PPOfJM5+7g9z5xu3GIWmtpdtcOwczJjNPMvBsbU/1ZfeUKR+OqMPDXljg3TrLm07YmV9k3Y8JiblKw/ho4vq+3xfErcIzA4OmTz9HEvPfoxr/+w7hNASbcTddT1nL59h45lLvHr7fZwL5uWc6frI6sYyR1/5dbh3j9HFCxXLXV1dZnnsSV1XLZVd6tVhn7wQj2NcZ4k167As8BJBlxHEN5o4ILZJiQPXaFBqtkO9+K/lvBBzPXjB5Oq+NZjx5AW301w87nIiOFdSlNwQR7tgolhz4Up02YJ3RDYBgNTQeleJ8yXQp0JwZlzuXFMzT6iBRFjMcagRZcksWEWEru8Zf/SjzGSNuLWv3IAIUTKddEyWJlz95ls435i+qNGSx49Ye+g8e1/6m7jjPdoTJ2reSLHHxZQSxT6rGtEI9xnEpNTjxLG0ucnx/h5v/Zf/OS/87d/D/qOr9NfmNG0YRrap4XjmCSGQpvNKplFD8p5Z7HFnTiOvvYmbjAldi+RMHztIkSgDulOsgAcxgHk0e8f8eMr6Zz/H0dwxv3fA0nhJS8AgHB8ccvkjH2fHRQ6PZzSh0UWWEvjAaN5z56XvsrI0ITRBHavESFsu4Fulf0qvJuM+L+aZGCrkqzX+0MgJ9FH52rlK58zsR7ScjNVJxSnDdFEsLEIfk5nvGMqRtPxLMVXwgOLCJIMtmDonFcfKSvRf8GU2ZbTumJ4+pVovVjvXIXpqITU1WflgL1YsBH8/CEAdQ3ppYcaarW+dWoqQY898ssz6Rz/JrVsdMhUkWDPQRyZnW+K9LbbfvUOYjJVP7AJ9TLQXzrPSzLn3+rcZLy0RfKg/wxkEmYt/xaIqoualSOV8p+zJXaTvOyYbGxxcv8L+N36Tzd/3p7h97yY05kWSBKLnuHP6gnZFwGkmMEnNY5rxWEsi8bjQqupD1DrfV7/shSkuQ6QGJa7hzBlWP/sj3L56j7adkJuAMyL8ZHPCw59+jDfv3oE0+OXFPhM2V0jvvcvsvXc4vblOaIJZAiR82+B9JsfSnQuSPC7Hyhv36PQOEyZgJ3FMiT5lmiCKNWdXXaucN/pCGdq6wbAoWzVQxLzFvyWZeWjOuZbAhX+TLF+9OGgVuVbIC97MC4TnQfZijK2+7xEXakC5d960fKnGiyFp0P2xwAmpY5iFYylmVPUllb1JilRrJK/Hynw2Jzz3HJx8iPn3pzQEhbMy9LHn5IkJuz98mX6WGS03ta6KJE4+doHZh2+Qbl+lWVrSfsBEBmSvWsU86CNlIR5OlRZFjKq2ZVGcTvMk0U5W2f/B9znzL/4seUUWovEc0ns6y3VJfafTvyQqOEoJOqCPZBPWtiHQ50Asvnk1dYtq3FKa90Kf7GYzxk8/g984xeG3X6JZWq489aPDQ5564XHaM6vcfPVDnPM2Y9DnuOZH7H/9RcLREc36mrqOmohD2XmNHuOdMvl8iGpAg5D6WDykVPJV1Rlqt+Ai6t4vYieucjO8BS0lJ4SSWiwDfdj2U6LJ7rLR8IokMNYsQ9tgSPent1j5EUrSWNllc6R6AhfCR6xedLHWncURqUYJlFTYmhsYqjFslvuTzLLZQZW3JRfhaFGPD5NQjvuOpedf4PiwJe0e4nMwE8CMbzNjN+fK2zfwkxVy0DqsixFZblk9vczOF38F108J7Yp651k2YJUWOldFC5VMJb8rtzqXfL1M73st/ZoWd+sWwUUYe+hsbCvDHKnvOnLfUc/IbBYBMTLd3WVS4SaxI1/doypRXoaJajYqgnHe6TKsPP0suztHpINjpBkZhTaTPDz9qae5cXhAFxOtOU7RRwiBsL3D0UvfZTwZVyJYaLzl9uaqgMfpieOzIyD03SFrm0HhzkpNdWSnfI5kkRpdTEx7z9HU0ZsIBHFI8ASzLC5lap9Ufe9wdWGXXJdU4zP0ROrNpEb56WbBvKA2985baBBDjuAw4RvSTZ13VnSb+U6iGpMX6E5UkWqGHxbnxQIx6L40voVU0Dw4mGudbuUKkGYzZssrbD7xEW5+eIQcCzmoJXkfO5bPNsTtLaa7HW4yHsLR45zlhy7Q7N3g4NUXWZ6M8S7gG2PUSJFa5TpWHYwqh+bLyeADHXPGpQxJDbp9OyKtrtJ7T25kcMk3C1kXIB4dI31STsaCk2bTJ+T6TSuPlAwVPUinfUYOqVoEUB1MDaIUR4pz0sY648ef4ubVmwTvcEFPwL7LnLx4kjOPneXL199HfKMQXEz0sznjpRXSa2/Qf3CFyeaGKkaMRhosQs2BlmcmEGiccLRzlz/2xz/D7/+XPwFdV/0Ba6xz2RJjj9Dy//1L/4yvfe0DJuMx896SbMVpCoFtfl2KxSN1IQKkBFbpqVLdrBj8GvQ+yJChbptFkkSoZX4eRI9qPrMYmFgtBe0oGPyOc91lFwW2Q2ZfTtSXRfxgZJNloJUmwyaLHaxmvsDR0QH+sz/KbHSK6fUt2tySumjZ1YnlNcfOW3eJbkLwjpw6C5cMrD18gdkrX4a9HZrzF3CNR5qAC03Fmhesm03qE2sfUCwZauKXScqkU6n/1HvCM08wc41OH4MZk/cgI5DQ0+3uak3bD6WUG41ws45+6w5N46uVQ8gNaZQM4UjDAza+dhlAeBGOD2eMn3qafmmFozvv0468Jg24QJ9mPPexxzhwka2DI4Jv9CWxk3alXWLv618npI4wbmiMX+2yUmW9qP4w2KnhkzA72uJn/uzn+RM/91nire/j5vvGjstDymtKpB7cxilubi3zzuvvsTxZphdHakqUnzNzGrk/iqMkD5ccQheq32IZmPQx3RdyUs7VmrSLvrTBzFar9VO2OjAxWOmK88ZJTQMDT4bgcMnDGHggXQu9BTYWt9FczRBNrlPom4VWka3W9CAxcpwSK899kp0bHXkv6ZjVxO9hxUHq2Lq2ixuPbXcM5H6KO7HJ0vKIra/9Bs14TBiN8KMW14wsZKdMBQeZk9ZazZB5WAkm1HwRxUDVODy1I05/+nNszzNNcBbK40gSWV4fkWa7HG3tIHhS7ixfJtKsr5G27sHhHmFpxXSNDqKYK1Q27eRga5/dAFVBZpph9ZnnOdg5VE1fM6r+cCx5Hnr6Id7ZukefsjLxMKva5TXaw0O23nmT5bU1QgiEJigd1DsaD41ztOIZ+UScHjFqp/zMn/59/MF/9dPMrnwDt3V9gDUx0nLUCeJs3jM5e5Zf+9WXuXr1kDMXN+jnGW9q/CqVKidzVb2bH0v21asklpO6BgsNMX69pW1hEsAawKpqdn1iiSHiV6d6OlwpUnhkIV1WRBlh9mdKyGJaUA6kPBiia0gO1RC75rfIsDvnkutiQP18ekTcOEU+9yR7L+/QzITUJ3Md7WmXA0dbe3SzjG9bJVUBXU6sPnQRbr7HwZsvceLCed1pwkgbExFcCDX+whVeSSWVD8FCKQ2h74q1Jpow4nBvj5Nf+HEmn/k0168d0o4ay2vM9PSsb7Zsv3uV2f4ho8mSCZ9Vqb66vET3/W8hR0f4jZM4CRZRY+iLh/sGyjVOQ3uWfj5Dzpyleehh9t+8TtM06h4FzGLkwqVTNOtjrr3zgZYBVo/HlFnb2GD+7W+St++wdPIkjddSw5sNWHDQeo90HbPpAY88ss4f+9N/kBc+dZLZW79FONyu2TVi1m81gDVCszphHlf56u+8SztZ1l21aEor5Ut7tSIQr+aegy9bNSgq2HvMg/NpjbkuESRFiG1ywVDm6GLHh7NBiuQhfiHWZlCqzevwMU12lROJVP0Vqmm6ZWI4p5YminlDFm92WGVXFwZPEWE27xk9+wmODyfkO/eQPDbBgacPkcnYsXVjS80AzfEJ58gry6ycP8/Rl3+DMJnQjCbqeyeFp2HWV+aJ54OJByy0RZyv3sjivYXjKBeoGY3oZnO69TUe+V//Se7mhhinhHGjyEuMtMuZpTDlyqtvKd+369QjWYDRmIlzbL34Im0wCNGaGTHuShHgp0WUqEJg0HWHTD7xcWZuxHz/iFHbmJOUI6Ypjz19kesHu+wdHBG8J0UzPRfPami499tfZnk0Yjwe0zZabrRe70s/65l3+5zZ8PzkH/gIf/BffIH1dovpD75E2800U7yLmjhsud7leI0pEc5c5Dvf3eL1l29w+tK5IVakRCIzRHWU9CaP1splY8kll9Kw7SH/J9ZccWdDGimDmAzeB0M5ZLByHVI7iumew5vh0PD2UNNha4aeGbk4BlNBtxD6nmxgocMVi8x1ZWDgbacvzvxC7DrmzZjxYx9j78YxzQycV5FooseNI/PplL2b95DW49aXWTqxwWR1iXbSMJluc+uH32BpcxM3XsK1I83n9h4JoXK7G+8M/Bd9wVCieimlSvJXSpoo2/eRg5B55M/9KfpnP8r2tQP8qKmmON18zpmzE/bffZXpvQPacaNjW6fWwJPTZ3BX3mf65musnthcmETaoy52ZSYFyzKEf+tu2zNvWlaef4E7Owd64rT682OMjFdaTl48yXc+fJ807UjeqME5M9o8if/wGvM3Xuf05irBO0ZNwyh4podHON9z7kzLF37yBb7w+cc4dzLS3/gus3s3GQWb5Ha9niTFDFLEYDtHalpYPc/f+/v/lM5KupgS0SzRNNlFpWlpsKaxOlwWXLcM1y5peHm4TamGb2KRc7bve189ZEIqGYMF18slBN3XKAo1rI41i7lmX8uQvTxYx3qrN2MtT3wxs1nAtjGxbSF5137Ve7rZlHT2IeL6w8x/uM8oByRGPJl56khNopvNWb98lhMXTrEkM2Y3PqB/8YfsvfUKO1ffw/cHjDY2cG2LH000GsOyEp0dg0lkiBFeyJYppVWxSRNJTI+PmW2u8Pif/TOkz3yOK7ePtD+wwJxEz/KGZyPM+da3Xia4gpebhVjwrK+ssPtr/5iQenxoBhoAxfFJqgtVtlIuy2CEOO975PJDyPlLHH7vDfySEpHIQj/tufTIWfrcc/fGHX0RXSL7hphhc2mZvd/6EuPU2yDF0YbAxE/5whce4kd+7DE+8swaq80B8dYrzH5wk5AirQ9ITGSJSHDkPjKbzWiWGiRIlUKFjdPs3om88eoHrK2vWC/kF/B0Zws3V7FzSpbK64pVhPltmBk7eTA+L1PsXIdfeaBXYAJl5wzjzgsUQBkgu0rYkerQoq6YWVEKL6HC24OcOy24+y8QbextdHmhMUj6ClZja1Gq5TzD5NnPMD0c4Q73VUmdpkQXyZueyUnPJM9xd95k78tf584PXmS6twWzKWHUMl5ZZbx+Aucbq1EdRSQmdWhhL6slSykNdBhqJKO39gJ5bYXJj32Ci3/opzm8dJnrtw6JSSVGkjRb3Dc9F05PeOdXfpO4fcy4DRo/Jom+61m6cAF/7Qrb3/gqm6vLNWa40m4La6/czhp7nA1FEqbA0sc+zkEXkZTwk3GdtHoXefLJS1y9cYv5dM5oNFJxQRdhssT48IB7L36dzXGLd57xaMRsesgf+gMP8/N/4ffBjdfprn+H6dY9AolRGFU/byJI40lHkd23bzC6sMZora2PuiPTnjrD13/5fQ63j9g8e3bgZVjMSDJKaKEnJ/MTUWfWXksqo02oSWfxFxyM9AfDC4u9y3mhDNYlHBbHq6UhdBayvkgGKTZeMRVHHCspZChYaux9vj8bcIir0J0/LUQMF089O3PJsaNfWWfp4vMc3OuR5Ji6DrfmWTnR0HbXOfrKb3D7u7/F4Ydvk/tjmuVVRs0IGa2blZRnHhpkY4O0sUG3vEIYNcpvlpIqMPB5q2wsJ5IoT9ovTXBrK0wuX2T0/HNw+SFudMLejQPNRi8cD4ng51w6u8TWt7/H7R++y7gZ0fcWVxYTebLMqfUVtn7xFxjFOe3o5CDaTUkHPtaYOyc1Fapgt2oE3tOtrLD0xFMcbO0S2gYJjSq3Y2RtdZn19RW+/cqrBITUKcU09T0r6yfpX3+DeOMK7Yl1Rt7TOGG0JHz+J58hXXuF4+99l0nb0khjGebJKABeVS43DuneuwNOGH90RX20jevi2glZlvn1X/4Nm2ZSDdpLEmJBNMpCLnh9HRqZ2X5eKLFSXiRpWSQAqfpyDHDzALGG+/JFzFYgmaGHW8gB1Nl7NAf2jMSoXFQLnUyWJz1EVGtkQTLIrojF1S43DJMzQ0IKsDefz8kPP09qTzG9vUuz4lm/sEI7u83+136JO9/+Ike33iU0gXZ5RJp78rxHVjdZefxJVp57htHlhxk/8SiTC2fJyyv0oSU6qeZ/uQgr60RwQH2zpc72owbahplvuDdPzO8cIdHhpaRtJfPgi1w6u8L+Sy/z5q9/ixGB3PUVBp1nx8Wnn6T/+m+w98OXOHH2nMJLvh3chgzO9VZm+BJgWlT4znF4NCU8+zTTlTXmV27STMbWjwjdbMrDj15k//CQnd0jRqHReDSUr7LkPIff/jpt31m6lWd2fMyTTyzz+GMb9K/9kInz952eTrS86HePibcOcFvHuFmk/cQ5ZMnDVKHIvp/hz5zng3f2ePUHVxgvT4gpExrThCbDn72jSwMVtoYnOWeJELqD94a166heapC9F6lBQiykiJXcFec1OiWoKDWZ0WHWI0akel2ISM3xqAGvacFLuapPFmKK7QPW+IZiL1tu1OCtqy+JMx5IznQIzWOfYGcnMt7wnD7XcvTSF7n+pV/k4MOXaUaBpY1N5odHxOjZfPYJTv7oj7Pyhd9Luvwo/co6c9ew3/dsR5XlxJiqgDcXnzuXGaJaBippyaCJHeRZr3yL7OxEGQy2U9+zNBEun17mzte/wxtf/AahF7IoqoFzzFPk5DNPMbn6Hu/9k3/A6uYmvmnwIVRhQ6HYkhLZDznfxQCz3KO+CYyf/wi7x8eqjA/GL3ZC04x57PJ5XvrBK7ojmng29T1+dZVmZ5utH36XU6OxeVoLB3t7PPv0RZruFgfXr+GkIUmspLJ03JH3pri9Y5q5ctxnqyPaSyeQPLPoD41wC2sn+Movvcfu7pTT55dNo5qI1jQWsXTlPVdMWSrBaDFtrLMhilThq69Gm5IHo3iqmqUIsJ2G15dSORV3dsODk1hkm/k8S1WxLH646gZS36hcU6RMuWD6NlnIwq5wn4X35Ay5mxHXzyCbl2malo10ixu/+Avce/GLjFxkeWOV+ayjdy3nf/8XuPRH/xj+459ie7TC9f05x/tz4r0jdXIX9ZEetG+iQTYiZrWbEZcs0i4P2dKumADrnfKY5tEmeDEmnEucPTXhXEi8/w9/nfe+9jIj3yhzRRS+68msPf4oJ9Mx7/53f5lR6hgvbTJqRozaJZxxqLMhKJhPCWRSmbJa3ETXd3QnTjJ66BEOb2/hR61+N6dEoDNnTtB4x9Xrt2lcIPW9NlRdZGV1he617+N2d2nPnMY7IXaRMydb/vmffgw5vsbk1IggAfpMnguHW0fIwZTJUcJJC22mj1N4+AzN+ph8fGwQW4+bjJlOR/zqL79MaAMYV8MZQCCF45MXCPlmk1FG11L5GcbsrHhZXtiJrXIuu7LLCzi4vsAiWE4hg4fvINxIdUEWgaKIutYM3Izil2FNVZbBTjeU3UXI3pTgDJFmJZAoGhboJDHrpqQzjzDZPAff/U3e+Z//Et3d91hdW9Y33rWc/dGf4JF/4+dwH/kEd47h1s09pkf3cBIIIeB9MH6EVP6EZNFTBcH7XKebOu0pY/sB2BdzO9W5hFoHxJwQn1hZazgZHP6t13jpV77O9rUtJsvL2oCZhrIH1h59hPPLDR/+f/6fNHv3WDlxEu8DITT4pqmpYN74MNgJxmIOuNEmZ11P88xzTJeWifEOftwqEcHrYn/i8gU+eO8qs8M5k3ZUTSlz27LSthz+4CVGbTBqaGA2n/HYU6eJx3OuHDlkfFknji6R/ZTzHzmB3z4iffeqLswI81FLeOISuGMLM43ECGHzND98ZZcP3rrJytpqHY4lQyjSkLlt6yEvDJmToWJl/O2MUms+2UXateCtWL0Ki+F7taTL5m2X74+zFdGGMJrgFQuhz0RyctWntwRqSiVESPUCLhPObMZ1KhLw1fbUG1VVzOxQ7QN65hI49djTbP323+fDv///ZuTmLK+tMDucs/HMR3j83/w38Z/+cW4eCFsvbxE7VXGP/RKpEIliXlD39xqr4Q3Q9JoMpf9u+eFm96ucJRkyyl1EfCY0iclImDQto/mM+O7b3PzKt7n9w3fwfszS0oQ0mxt8lOlDw/JDF1iTYz78hb9MvvUeaxvr+OBpRi0htEi28EnvLCwpVSf6tDB6z1beHI9alp95mnvHx/r7LTKuz7C+tsTmpOU3XnmbkDx5Fm3Q0TM6fRq/vcXx229wanWtKtnbEHj51Zv8uX/z+/RHR7r4U2Y27Thzeszf/G//MGv37mhz6oU0j8Szp2hPb5L3D7QHShAdhJWzfPW3fkCcR0Ztaz50g4/XYOouQ4PHkKleUApKIGgyDrQOLqqcbDH1dxENui9xImdVrCwKYWvsQRrss0rcdqzdpLKhShRF4WqU+WGyuOXBFqzkBVK/QAm+d8aHnc2nTDbPMn3pt7n6pX/EpBVEWqaH+zz2x36Ws3/632Jrcoobr+/ST8GFFsXTiz1vpu96Yo64kPFLgdGyYzxR9YU3ib2I4t9q4FhI5Vo/ezQlyzuhcZkm9uSjY+Ltu+y/+SEffOc19q9chz4xGk8Uk5/1tisn3No6fvMEnz+1zc9cOuT/Hq9xtQksLY/BBZwPeDMyVx9AGTaR4jgqg5gwA9PZlPTQQ+Sz5znc3kWaAEH5Nf1szsXzp9i+cYetm/dYWl4jduqI36XEifVNum/8Ok0/Z7J8Sn82Gk467zxdXgPfkvMc5xP7hzv84T/4LJuTMbO7O4xGAXqho8c9dkGBwy4a5Cn41RPs7o34rd98k6XVib6gxdKAXFVABZUoOkCdffghkiRT3f6LxVzJ3RExV9k8SALFqWFj6WErZckJodAS78ONDSYpZPw65rbkV1eO5tIYEAcpTqLGBqQFrR4L9l6lNirBjGJN1NHdG+y/+UNW15aYH+7jliY8/+/+h/jP/xFeu9kxu3eP4Ef4YMeRMsKZpzl+Gc6cX2Lz1JiJ7wnHR8y37zK9uqeLMvakrldZkflviDcfiKKWyJk+Rg6Pj+h3Dul2jjjeOWC2c8D8YIZ3QtuOkLFTgnkWiB3RZ9g4wXhtzB899Tb/zom3eOriMs//V3+MP//v/UPuHc1ZP7FGnz2+8TUpqzTOUv2fWYiM1sU+TT3hueeYjkbEfkZoW90UvOB94PTGCu9+9xVc9vr9shqwyGjEisDOm6+zPFmqSWJBBCTigxAmI2hAej2VV9dafvqnHiLfvoqbzVVoLJFufURzaQOObtbSLCahOXmO739zl6vv3+X8xVPESnwXC30yViZKVJOFqOxcuPN5sACrBvksji9KwtqCvtLsczV/3BlTTwuEUCl6FTc2vdZCmLI4IeZo9D8zmnFU08O8kJ9SzGmoUi6joroFYr24Bf9lN5g/9h3jyZjDnSOa8xd59j/4j9l55DNcfXkLmTn8qDH82JP7RN/NmZx0PPr4GufXHN0HH3LvV37Ie6++y9H1Laa7B8z2j3AWooMb0mRT2U1MsFsQDrJGn2bxSPD4dkRoGtqmqS9A4QDMSch4RFhf48nVA/6NS+/wJ899wKrscHxFePbZll/4y/8a//b/4Rc5iD3LSxOSWTRkU8gXMUUliKGE+2T54/3qGmsvfIxbR4da1nhnCV+RzROrxKMjrr1zlcZ7ct8jZGLXs3LmPPHqFWbvvsnaxgqC0JRIN3H4ENTZSRqCzxzsHfL0E2f5yFNr9G+9TWhbdQ6Nc/LFs7gmI7tHxpIEmhZWT/PiV36bSashToi3rBxXx9SVgGQQqRcN4iw05IwoR9rWml8ouSrImtOgzC9OAmK8I0k1Px4yoaDgMac66s2IJoRZfVfD42sw+cCNL4qpYsubrZutb5WFdap+D5zXjJQKVztH6nW3a8ZjuvmM9omnefQv/AfcXH+aO9+5TXATaJoqLOhmRzRr8PjTG1xambH31d/ih7/9Xa69/C7zvRlhNME1Dc47Ru0SxuuhWjOZuUx2CwE31nkPZVYx3i44ntlVOYjB4zfWWD1zio0JfGHyHn/+3Dt8Zu0e+WifFAKTtuX4yhWe/uRF/vP/5F/l//if/So5OHocSbx+l6APVxaa7BJ6KgKz+RR59nk4fYbp9au4Jqg0CkgxcnJzlduvvMNs74jxZKka0/c5s7G5weGv/hqh62oyq5g6Rz02HE1oyAJNFo444Ce/8BQr/pjZtIdWseyu8YTLZ+HwjnqCiM4awomTbN8N/OavvcLSysrA1hT121i0Xs55EHOUxAhqaKezz5wGw/z8uxiH1QLMWeCr1AFfHqj/OljJaTjmNMRHasddflOyXVUWSNklEbaOMG2y9Lu/BFL87GRBODBkWFf0oWmZTzvco0/w1L/173FdLnLnO3dowxK5EaRXxe80HXDmyRUev+iZf+u3+OYvfYlbP7iCG40ZLS+xvL5qknlZyMw2X2r7fyXFVkqnXazuC208UZ1UKRwD52A8otlcZ/3cWcbLS0zv3uXgpe/wh/4AfObUlG73COcaQy0c4yYwe+sNPvXpn+BP/cxn+Gt/93uMV0/TRejdIEVLkivzLi9Irw5TYu3Tn2HffAclBDs9E83Ys+kdb7z2Hr5PMFfrhpR6mvUNmtmUrVe+x9rSuJKsFPtOhg6oEsm7gI+wPBE++9nz5J1r+Kzsv0hHv7lOuzGB7WtqeC5ZBcgnTvKb/+Or3Ly6xWOPX6SnkMvKaa3POhYdkZW0pb/KNsRKRjEtk9o8pJ4PQ5iF9Zdr7LTt2gYA9KaaCosGhkbVMFRCNJixmihSA9Kl0krT/RZQliWYi+WiOYoOVFI1bymWB7nEunlP18+Zbpzi0T/7F9iSh9h+eYuxW9KdrBNi39FNjnjy95zhzP4V3vi//F2ufuNVQjtheXN9GOLEoZHNJcYv2lCndMZFD9yVFAGxvHNDbbwZ3wSPa1va1RXGq8u0o5EyAd9/j90rN5hv70Lf8e/9zX0mf+YMv//xE8TjHY2DsaOwSXP6d3/Av/QHn+UbL77Lm1eO8e2YmAWfMsllkoJ3VfAQYyJ2Hf7MWZqnnmF/b7/6nQD0MXH+zAbdjbtsvX9LhcPzXi3Tuo4TT5xl9sr36O9cZ3ThAs5JNQtnwVsjZ7XIjcdTnnvuBE88skz/xhGhHamdAgl3/hwuznDHU5BGIymCJ/WB3/zV7zEZN5UglY2QXzw+Sn67c5rekGXg0BSzmbxIPbCDqk+WvFDF17kS/Ev0ttbMqfYaxWEgDGJMqb/ZLWrujMAdY7aESuMKF+TC3jSxUfngHloG6pbJbFL16u9hv1Zq9cMw4qE/8efYX3mCWz+4R8OohqP280hcnfORnziL/50v8zt/6e8wn/VMVjdVFBqLdCrS9cr9ZbyEX92gmSzjmkYNZhYsGCj50CRLzFWFt2tbwmSMW5rAWG0RZH+P7v13OLh2lfneno7aXUszGuHHYz6cj/n3/8Eel3/+DM+szYh5rjFtJByRfucWq6dO8S/9C8/wX/7lb9P4JVKEFPR+BhElvFvz7EQ4ODpm9KMf4Xi8xPzmTYI3yyzjBp9ZXWbrW6/RH81pJoHcRbJ0ZN+wOp6w9dLXGbcNITTVfNI5JcFLipaU5RkFmDLjJ37qBcZ+yrxPMG5glohLK/iz52D3Q6TLIJF+NqW9dJ7XX93muy99wOr6Cl1U9KHkqMdcM0dMKJINl7bd2MKfemPIxVTSdxdPejdIhaAGcZaGUWvxXBG6ghYFKT69Vs/oQMVUzoKRfZLlXJvJSOVqiEVH5CFAyo4UVxouhtSsahxSAhqzOvzvHM/Y/CN/gv7Sp7j2vTv4vtHMw6IzXO/55OfPcesX/zZv/LV/TLuyxnh5bPgpQE8fI7ldYuncRZbOnmNpddWIUFFFpSkvyN9LKFIcauu8oLSWOXSRLHNkNCZMRszOnudoe5t4d4e2nSBtq/fLOVbW13ltz/OffvGIv/6nzuJnVyyRqidntT7ob73P5z75HM8+vsbbV+eMmhHzROXBlEqx4PnzyYSVj32CO3sHpFlHbLMJBRJLyyPWYs+br79P64PSFXKm7+dMHjqPbN1mfvU9NpeXNTfG+9oTOCmhnArfSezY3Gj55CcvkbbeU984caTOk06fpZm0yLVtddzPc2KKNOsn+Wf/w/sc7HdsrIfKfylzB6mTYSl++aZVVUuDLEJv7lbYDp5jMiHJIAYo3oMplt16ILrFoiQvkXDGpw7V1FxYNFOrNroFIy+qA/EL0J4rxblNh0g27hxij3FDFMSgUSwlh3A8m9E8+wlGL/wUV17dIhx7JOiXAEeazPnYj5zk3t/9u7z6d/4pSxsna+SvCMTYEdsRa488w/rFi7j5Poc3XuPei28w27pOn2Y6vKmKbAZftRI9lwthiYo4aNkSkKZhtHqK5Rc+w2M/9QV2do+48/Vv4o+PcW0LeGKG5fUNfvnN23z5g4affnRMnE81Ui4pXJUODli9MOXHPneJ9//Be0gY0fXV4ZBFYXM363CPP0U+f4mD69c0ZdbI9fPUc+HCGgfvXGP3xjaj8Vjr0Jzps3Dq1BmOv/ubuOMDmvUzZhNs6IZziMsEhEaExifm+0d87JMbnDsTmH9/l1HbKEjQgD/3MOlwn2Y21QXbJ8Jyy/Fhy4tfeZeV5RHJqT1xiQ2JOZNE9YHJQqayZVunZFYPZTiSCvKaauBUVYDLgjCkbLqFHSnmpShOy96SO6kLesHeq2C7xBLNaP/nB1A8JoNWzASk7MA1vNMmiIv9Z+VDu2r9RdYMl+nSGic//y9x40aiuzunCQ2pV1jmOB/z3GdPsv3Ff8QP/v4XGa+cqC9GJtN1kfGZi5x7+iOE+S5b3/wlDq68TDzc0kYzBFyxxa0Nq9rzLvp/FEtc5xZpiYmU5qSjY2Z72xxdeZ3db32Fc//qz/HIz/5xrv7Kr5HubeOdr6aC07TEf/OlbT7/8ycI8w+UKGSDSRcj7G/x/PMnaH7pVSLLNhyLdcwuJkM7ionxMx/hoM/E/UNCwd1TT5aeE+MxV775CnmeyEENcZSItMJScNz+3jcYjVsdtYu3sE1lPzpL1Q0u0YgQZc7v+X3PIcdbSN9BGEGfiMursH4Ceed9XASJiX4+J1y6xDdf2eKtt2+ytrZa/74SP1I3yAVCW0HSiiC2TJFLk1gooM42Fyfm8F9KVPP8cANBsw72ZFHULBWHzpX2KU7IUT9c8EpvLPYHycDxZFmGYl4a0XZydS41nDRa6M6i6aEfwsiFzFHfMf7UTzFtL3H41m1CClp/hsC0n3PxUxs0b77ED/7KP2Q0WTX+h8KJPcLa4x/h1KWHOHj1t7n2w18nT7doxmOa1U3bNYpNgbvPzreGBLnCvV1wjaqew5btXVJKBeZ7d/nwr//XPPLn/x0e/qP/Au/8rV/EdZ3RbBPjyYSvf7jHy/cmfOrEErHXHHNixmdH3tnm0pnTnNwMXNuagxsZu2+Q4veznvnyMu1jT7F3b0u9pWNQHk2fmGyMyXvH3H33JoFAnqkUqp91rD9+lnzrQ+LObUYrS8NiQ/AuDz/LjoN5N+PE2RU++okn6K+9SAiNoU+JdOosbj4nb90zoX1PJNIsb/LFX3+Vvs+0o1GlAmdRz7psizclLUOd8/RdqonEOQ8m5eV5uFx9OcENhudi0znB4d1gn1um0wUqXBRMOTfQl8nJGiwWQlpK4yQLNaZ3Q+RWEYQnM6pGDdSrYXkufywYvKc/r+860qnLNM9+nnvv7RKmGlXmUqI/ntKeaTg72uXb/83fIoQxZf7gnKfPjvVnPsfpx57m1pf/Dje/+Yt4OWa8vkkzXsI3gaZtaNsxo9GI0Laa9xcamibQhkDbBoIPtKFVOX/wwz+bhtCMaFr9s6Ft8U3LaHWVkUtc+Rt/iXHeZ/NzLzDvZ/oS+0AzatnN63z9egtLG+o7koC+R2JPPNhnbdzx0IVV4nSqIR1ZOQpiU635fI48/iTz1Q2m97ZwySi9KdHNZ5w6ucbWy+8x3T7WY7mP5E7txjZOnuLg5W/TxJ52PNZG0HsN+EE9NxoyLiWCdxzt7/LCZx5hZTXS7+/iWjXi6X2DO3mOdOMa/mhGnifStMcvTdjaFl576QYrK2PtnwxSMt1MXbClzzJaM33KpChDpLGpglKRvZSyIS36iw+EJu3tUl1PaSETqIQMFQR2cPSVQblRhimpNHSRBZ/ZEt9g9VEdDDiDY3zN1FadUrA8FV+bhRmO8cd+iuPZCv3dA03NihHpemKecvbyiLf/+/+B2a1tmsYhVurELjG58CwnH3qcG1/6Gxxd/TbLJzZpJyv4pqUZtbRtq7q54HFB+cfKcmsJTYtvGiQEXOF3BIPqvEe8w4WAM8jONQ2+bVWb2DS0y8ukvXvc/ZV/xOlPfRTZWKMngw/k0CLtCt+5KuRmGYmzGvhD7ImzKcRDzp6eILknBNTTo5xbGY6do3nqI+zvH5APD8gxkvqO1Hf4xrPiAze/+QpBVGkuORHnc8LqBmE+5/D17zFamhCco2kCwamqu0zqHJrtnmPPaAw//pPPkrevEiwbPedIXttEwoR47Qoyz+RZopt2NCdO8L0fbHH92jZLyxMSghcr5xg4zroB6i4dUyIVVZMbVDpl4ik1r0eZmcVmWSTXRV99qBdO0TyYDpitmNnVFZilwGwpWd2ThxByzUBhyOwuo9qcq590quJ7Ma5QrqaNxWAGp+y4vuvo184wevTTHF7boe0zOYKLmTjrmZxbhfdf4eaXv854MoJuru5BMSFrJznz/Ce59+IvcfzBt1ha3yCEMW0zoW1GeD/GhRFeGpxr9T8ScL7F+YDzKl1yIeBDq4MFr//deY9rWlwTCE1L8C3ej/Sl8LrD++AZLy+z9/JLhLUlJs89qTBh22oAfTvivXuZo36keeNdT+4g9xmZR5hNOXNiWfMEc9acbBu7xr6nXz+Bu/Aoh7fvIl1E5j2uS8Rpx+rGKunDOxxcvYcPRVmk/JO1s2forrxB2r3LaGmCt77BEpSr9MyRCR5mR0dcfuQ0jz22TnfnmjmiRVKa4c5eYn53i7yzpxj+tFdt5co6v/bld+mSEJpgA5IyBZQFMyE/GKCL5u4kC/nJxp5LZu0rJX99IaFMSw7Do2KuWLMzzkcxnhxYibkCGEGxUhs+WMRbVUAvmJXfZ3N7n552kC6JcP+bKgWXdlaO6O58OJvRPvUppv0y8e42bel8YyK5ntWTDXd/5auQY2UBptzTReH0U58i3nqL3e//MstrSzini098MAWK2ATOfIOd4KSxrMU6I12Yamqr7at20g1TUVe6aGrWoqREu7zK4eEUjqcsvfAM87evankB+OC5d9izMxuxjJj7fF/H/3Qd41ZVKC5ZUyQBQTied4THn6XzLd3ujn6mLpJFnYk2lsbc/cYPoIPcKtKTY0KaMRsnTrLztV9i7KBtWn1BnSO4UtYoRqt8Ckc/2+eFjz9JKwfMDve0V+gjffD4pQ36732f0SwSjT7aXlrm9l14+Xu3WF2d6PHe6KnWV8KZH3SklGmgqwr6MofLQg0HTXmBiKSTOUtaY8i/KQW3PS8n2thnsw4e8gopKEexFkjGzVBOqrNGL9EjPmgmSp0GmnDW5oJuIXJCsuLQ2QYGhZ2nUFxiPlpl9anPcPf2Pn6eLDY5EWNPc37MaOtDPvzu92naiYXcC6mfMzr5KJunz3Dll/4KwU8RPyGEEcE3OsiwmGVnDWF97REklCBP0e+z8IIls8rSwRALSQLKaCtBRjp1jMhYcPNjXN/hz27iTyyRbuxWeuZ0HjmOXh/OvDPX0ETOHcQ5MarNbtnFNPohMnOe0RMf5d72LsznSGhUZEskrLS0xzPuvfKexiPPza9uPmdy+mHC8R7H77/GxuqKWu6KYc56DNtCRu2A+zknNlt+5CeeIN25ii+Bq/0cOfEQ85053QfXGEUVNM9jZnLmNC++eI879444cf5k9TFJeWHjM/V+goWswSE6RBtsS8dygtjOPZQOop4vbtg064ZZz/oiUUvVeIY6A9RGvrKhvCyYNskCuSgPMW2liC8JRTkXooknJRtn28OPyYLJ7T/OB2bHM9rLHyGtPcz81h4ep7tMTHSxY+nMMkevfo9+dw/nQzV8SRE2Hv0I0yuvcHz1B4wnKxqxZq5D3peyYKQ51qGhDSOa0KoVmAuq5fOeELSMaJqR/jln5YQLVVXS+BGNa+xlaezvbAntmOA8bmOduLZMIuJPrprrvj4IjQhuoXfIPMMcZJ6QeYLUMzuc6nQsqtA2pcx8OmW+eQ45c4n9W7eQmNWCoNcybG1jjfm71+nu7SmsFXtIPSlnTly4wPHbrxC6Y8bjiVp7OcFnbZm8ZZmTM95DN5vyxHOXuHRuTH/nOj47/TldRDbOcfjmFfL2jPnUMT+C1HhY3uR//vV3kJHu/Di/kLoig2bTGrp4X+ZKTZscjvA0GPpgoUop9zrvYAgPSlaDZwb31iEjM9l4XddmjImQZegMk9XGhdJYYmslDxkYuW5hGKasw5WMM+6r6cmc2WfbAs/GJJj7EavPfI6jrZ580OkuVIzAx441d8SHr3wf346Mpmog+topVs49xN0v/WOacSC0E1xo1GzQ60L1vjHGoFQIp3ILLD6DBejIma+OKzbCZSTv/H1G6A6NcC5Ydpxmlh5+mHzyBP2V92hXV5g6j3gh9on15RErLcT5HB+1LBjAf8/23Z0KZRa49Dj2LD33SfanPWl3hxCMkWe3fqVpufe9N7R0Sr2WQDHhl1dYmky48/2vsjRutak13nMbtOQomZPefLXn00M+9rFPEOZ36PYPkBAULRkvE9OYgzevshIburkQ+8jGE2u8/cGcl1+7zdraKsksb8Xy18VoDDENBvZOvI7ESyxcFcRKHXHnslktcDWKCkWMhup+N0e8enU4EwBQgQktq3JiMQlLAzYHxUFMmuUM0Vzq/UDAvo8MZQ6SVX4z4Ne6uIS+73CnLjM69wxbb2/TZm8mi8K865lcWqe//i57731Iu7xcYZmYIisXnkKOtji8/iqj1XXwqs3zbVAzGec1VsGI8mIvlWeAGAsBJsfh+MrWTCRZCPs0InnMvWb1UYId9RSbJceJH/8x5i7STw91N/ae7AOJKWfWG1ZDTzzq8Fn1dwDSCsiYD6/uLAgdNCmrX95k/MTz7N+4Qe567Wyc+nGMTq4i93bZe/cqbWi17HFCP+9Zf/QC/dW3mF57k/Vzp/E4vBkwOoqQINvIG7rpnBMnGz720ROkW29aLnmCrsedeYyda0fEu8f0voGuJwmMzpzmV37pCsdHkZOnTM9YRK1ZX/lk+Ts1H2vRUyPr/+6MepoxH8JYhi+5mjn2hr3VSLtccuHv36Fzce0vHtL2/FyJKMbIScU0MNkx75y3RezuU584U4AnWRibZxtVZpXoSLFRtdT6edfTPPI8x/0y87sHuqB6dazvJLN2ZoWd134wUE9tchZdw9qlJ9h//7vk2TZhtETTtFpeuBESRng/QlyDSIu4FnEjnNNfc77V/80FnDOEw7e40Crd07fKGfYB8Vq+gLNyQ6eoHk/TNKT5HLl0kdWf+gI7168TYhwIj05dgp5/qKWdHpKPk0JrXSbNEy4EDg4z779/gHOtDp+AWdchl59iPl7j8PYtZTn2GemjkvVXljh+5wPS8czorboAs3Osb57m4PXvMA4Y79nhLTpDRF9oh7qLNs7RTY94+vnznF2PdLdv43tg1hOzECen2HvrDnHumXWe2TTTro3Y71v+0W+9zfLaSBepN+5GKS+l6P5kiHsrFLSa7T6UG6myOKV67Bc+Z4Evxcq34ifOgtyqeminVJMNnJUhLuch/aRgyymr3Wkhm6eajVIcSVONksjFPVMstjZTXdkxE23VgPV0YUJ7+XkObh/iZhbIniJd1+E3lhnnY3beepcwXiYn/QGx62k2zjNeGbP//ou0kzFNwZPDGN9MEN8ifgShIftAdg2aLORqwKdzDSZFweMR35BFYTrlGeuvOd/obXVGkhZPzh5x2qDtu8Dpn/0TbAfP8c1betRO5/YoA43r+MxjgbyzRZ4B0wSzRJp2uOWWazePuXn9iNAELUVSonMB//hH2d3aQabHOh3s5+RZjzhh3Ce2X37HAol6pE/0s47R2gma/pDD919htLSsI23xBKPBSoVfo5UdGUlTPvXJC8j+TeRgqoY0x3OkXeb4sOHwgx36zjM7zhzPIivnTvDtdw65dvuAlaWJagFF2XJ6tpbsw+o7adnv1OzuWHIMc6ZHee1D9LEp9BfQDlcz4UupluhzrLTmMsJzlkDsXdB0YG8UWakiWTsmvKvC2FxM9ayeVWyqsOi8mfBpPFflddif63M0LnZkNj1GzjwCq5eZ3dgikJHc4xL0fWTt7Bp7b7zM8d27hKat3I8uRlYeepJu6yqzO+/TLi0TQotvW6RpERdomjESbBF7zcRTzFsd+5115M47i34L5lqpWKo4jwT9u3K5Oc4pTh1aQtvQ95G9dsyZf+PnyC98nLtvvKH6ugiz3UNyH+m6OWfaIz77SODo2i1cL+R5gnkm9hlZXeeNd3eYS1Mb5/lsSrd+Bs48zMHN6/iUyL3+p59PaZda+ut3OLx+V8unWYSoDdza6XNMr75N3rtFOxrjF5T7xVsu5h4vWoLMZ1POnWn45LOr9NeuE3oHfSL3PXnlJDsf7DG7d0wfhb6D7IXm1Bpf+fpVRkHx+mLIWbzqeovrKNhyROiLTCq7YTe1hQ1CKFVAylXSZrbnaqJTd+dBe1o8SwpmndIQbYGtT020q29HWrTaWMCtS5imHxzYsyxEGueaMmuW6ogkkKAPwKQy8z7RXn6B2aEn7x5qhERUN/4chKUGrr/8HXwxcbRaNrebrF56hr1v/X1cOmY0OoEPY5w0BBd0IbswKDJMk5dLnl3WCZgvOHJh+1WvPaMrSmmKrZ7OunvGmDgQiE8/xdk/9i9zeOEcN994Az/vwDnaPnK8vU8Q4Xh/l5/+0REXZZ97Nw9onCfmiM/qrN+3q3zzxZfps6cxEkKfM+6hpzjuMrN7d2nEyjDRDPPl0TKHr/4Q3/UWnqlSf9eMWV7dYOf7v6K8Z69jex/sZXQ66nbmyiQuMzs64FM/eZHV5ojZ3R0CHrpIEqEPJ7j7w2vEeSI2jjiLnD434vaR8OVvfcjy0sQOvGDxfIteK6aBzMprTtnka/X+euV/VyG2DlRSsVou02qRuqtjJDInuRoS5ZIIZoqjVLuQZBtrJlR3R0MvnPgqPeJ3uR0p4G1ycq8f3tlCF8xHOrvagdYIrhjpl0+wdPEj7Fy7R+hmZPRtj6mn3dygv3eDgw+vVHGmeE/fR0aXn8a5lu33XkXaJabRE6NWZLrLJ5wvp4cRotICBA06lXO5LvLS+JUbLn6wDShITgwNaW0Nd+kSyx//GP6557h7dMTuK6/TpGhC14DfnzG7u0dDYFl2+Nf/4OMcvP5N5DCRJ7aL9T3NY6d5/dqUH7y+xWj5nCICMdM1q7iLT7N38wbMjklt0AM1RULb0h72bL97neC0RHFA1/UsXb6MHG8xv/kOa0tLuCaYn7fWy+VY9sb7zknopwd8+mNn4OZ1/FEPrYfYw/oGu9uwd/2AsbTETjPFT1w6yxdfucPdnSPOntkkW+5gdZyVoVSw/bU6ksaYh0CgtJhOTDU1EvvNxQe72BRUuRVUF4HybJJxUUSGxrSMvS2ncCjlpfhgyf1iRfF+mOK4ITBoMZZN0EFMxusXL7rEnOnnc5rHn0bcJvHW24yzQk/ZNH+T5RE7r32FPD2E0cgiByBGx5knPkZeP8vFP/Bv4UYZaRuyl+ofh/OaP+i1pFCLAPuSDsSrIiVXTUFZ/MXLTmPQiudb8mqU6Ccj3MYG3dKY/fmM/TffJB0f0gZ1/592PedPnmL7G9/A93A0P+SPfHLMxzaPuf7F62wyop9FnMvM28TyuTP82j94n/1ZYLzSkiSQYiadusC8Xefowzd0ttZHM4KMLK2tc/TeVeZb+zTjQErGZ4mZ1VPnOf7gO7jpDmH1tC5e56oaWksPPQGdg+nxMc88vcaTD0+YvXSLNmKlUILRCW69fcDsMBNGDcSe8UTw68t89R++TdMotwWv2HNesErToVwxK44VYKipXbYHF0clTe+1fXnRtb9GcLiKspWoi0WXUWciiJR04im5DMWkGHblatWkDj6+BhzmcoSXYYobflAxMK/uqAviTpEhWDHnxFGXmDz0cY62jpCjI3IzVjwVgVGL7w7Yf/dVQnFiJ5EijM9eIDJja+tDRqcfrWKD5CJJElFyVWIQLHPQa+gQTiB4XEgkb2Izt2CcU6KlHGY8Y6p2o0/mbk537w7pygFpPid40fiKmJkeH3Pi0nl4+xp7b12jbSecTHf4i3/4MvsvfZ+855iPtQGdTw9Ye/Ikr1/r+LXf+pB26bSiG0E4mHfMT17mcH9KPNjFF4NMIglok2f/3Q+02Y56APVpjp+sMh4Ftj58nfF4ZDkpg6K7OKOKdzU7Zdof8fnPP0k43qHfOkLCiNypO+xxt8L1t/eJMmGePLP5nDMXR3y4M+PFH15jZXlCYsERSRa5PSawTqk6EsWUavKBOcNYozfoT3MedtliQVD9FS1luEj2CqkqgSXeZqKYwkVMP551UBXIUgcqwWpLde0fKKRYiFARwFY9OiWPxevCSEbGXvCM7rs5ae0czfln2X3lNiNlbtef2YzGdLfeIO7fwTtHTJE8mjA5/xDNiVPs3H5TI95EedDJwnWUW5LqIlUWvdRFnb1UQ8NcF3zJURHLIy8NsDbBuYpjdQzug+C9r2aK5Mzx0RGnH77E5t6M1375q0yaJfa37/Dzf2CVTy3t8v53rrIsS3R9IvWR0ckGuXiRv/5ffZ/deWC00tBlR551TJt18qmH2bt7k9zP0cehR28YTeDODrMbt9RDI+pD7edz1h8+T96/Q3/vKsurY92V7esV1kyZ0TbOkfqe1XHiE89t0r//On7myNERY0ZWV7lz17FzZ8ayH9GZQPf05XX+0et3uHcw58L51crF0UmHK4nftvvmevoVNX1Ki/YWQ0lQSP1Fzlf9+W3XrjZ4ta+TanrufUmCKHzoVF1tMd57KH+lFC2Z2fxXh/RFPNFgGZe1DM9SKpQ8aM9NHVGCbGbznsnHf5Q0H+G296o/mUe9GBqO2HvnuzDfo2+XkNVTrFx6mH48Zhbnpo4x1MVlzQ1xpfkMutN6xUK1aXEaC6ezfCtPsJBLTT3FS80sF7/49xsfwJpESUMgZxd7uiBceO5Jzhx0/ODv/BPGEjjaP+DTF/b5C7//BDd//au004bUOpgmUjPj/Kce5+/80w/57W/dYfnieaYRsgv0UUjnLjP1S8y3dReu3XqfGY9aDt97h9T1hNAgmvJOJrB69gIHr32RkKb4sGysOoups83EG9NOnDA9OOSTL6xzdqnn+MNtxlFV530fodnkyjvHpC6QnGPWZdZWod1Y4usvvcZ40qo1nN2gVCZ7FLtk1Zay4GFXRdJmUJRsSlw9RSUPi9wVSwxj5JVqoZrLKPHIu1DN89NCeKtOCYfpesg1CtdVy9Iiy8o2MSt+deXXqZNEGUwFFxUfxQos9sxH65x87HPs3N4j9BlpXG0cfONI+zc5uPE2EgLtqXP4UxeY+QDdfME8cdFQpLxkHlyCZJ/TS5X/SLLavti12vey/DRzHbWHVIKcyDrQKNKhXr9vcpCXG9YuneWRRx4ivfohL/+dLzLqPV2Cs81t/l8/9xDh+9/n+N1DxpNl4jwSOeLhT1zgd16Z8Qv/02tMTp2iS47sFOc+ThBPPsbB7j7x+IAmqOtQ8UP2h1P2b9/FSSBFlfX3saPdOMMkwO7VV5mMRgQsM6baMBjgWf6ZHI3r+LHPPIzcuk3ey/RBSH0muRG7R2OufnhE04wVcuumPHR5mbdvHvHi6zdZWl2u4tdc1f/mJY6rCyyLWg6XMbiUQYc9sd5OfTFrr5RsYZuXeEzWc8WCUg1rq3L0K/Ox1OL6IkQbbokxChfcQqNxN2SwvE0Cja9y8TrKMR5vTbMqygLDQcU55tMpzYVPIZNLdLffonVNxR51nAmH196CNjA68ygsrTCfz3UU613l2YpEVcwUKyxn1mLm3okvLkjm6lV0ZnruDti5N8jO6wkifmFHt7KjxLn5UWC0scrG+VOsbW4y2jrg5l//FW5++w1WltaZ94mVfJW/8u8+xGN773P1WzdZGq8y7xLkIx791GlePRjxn/3VrzFtVgjSEHMgSyAmmI42kdVzHN25rWkIol5tAKNmTHdvm3h4RONLo56JMXPiwsPM7rxPPt6lWV9Tw0vvjWSvNAY1nMwEycxnU05vOJ69OGH2zjVcbIk4PSFWV3n/DszmgaaxeLUw46HHTvNf/upbbB/3PHyyGfwxSsnBwM1JefB3LltOYdstOh+5YodhsCMLcitnu2xJJZYSh1xEAlkWAukLGc4QDhsIlgoi1IKh+kQXOo6+cW5hpF12yWQ6L0zMWIYENWvQuM1dDkwe+hQHt+fkvaOa0QLgW0ee3WM232fpwmOkZkx2nmbcQtuqy2YTYDQitA2MnEYb269L8MjIm+eGg8bjWlWfiFeHIIJHgjWNLtfmT//pIejxmZ2rGK54TyPCeB6R3UP237nOjX/wVfbeuEo67lleX+Hg4JALq9v8wv/uYV7orvD+l99nPF5n1kccxzzx2fO83bf8R3/5q9yZCaOVMXMakJYsgVlMpHOP0fWOfveeYu9drNpGT2Z66xbEnuxseBUzrlli7fRptr7x6wTJ+CaA4c7OGsCcEtkC2cUL3fSYZz62xgkXObjT4ZmQoqNLmcO0xJXbEXEtXc7Q91w+09KNW776yg1WV5dJ+OokO8zohqGbGFJV4LVko3ldE1JTYf2CpVchj1XXfhmoxyWsKZvNcrXtNqBCvKsm5+X3iZMK4YVczJQrozoN3ISsO2VecFuv6ukUER9qtxqN8+rNziDPO/rRWcKp59h7/w4hRq1ns9A00B3eYO/uq+Q8J0XIHCMhEA8Drm2QJpCDQ0Igty0ysozB4HW3tYWL93oqBDsKg4bTJ1GrK13Ag1Pq8DyUP51iT4qZ7B3znDk+mNFt78OdffrtA+Y7BzhxTCYTZHXC7vY9PnP5kP/6X7/I5a3XuPLNG4yaVbp+zspqz6OffoxvvHfMf/RXfoetmUMmy3QoxySLDkaOeo8/9Qg729vIbA5NcR7KGiC0f0S3s0tjESHZ7BpWzl9GZjvMrr/GxsqSDVCc2c0uesglUo7E6GjCnB/56Fny7X3irCFJoI/Q+RHX90Zs70AjjogQ4xGPPbbC77y+zZvv3eXixVPVCzDl++mfmlBlHoGlNq6bqDVy3qmnRkG0clooA40Ml1WvWs2NzF5XnEacZCt3XfEwTyr2KKw9FnBuNWscdDKDYWBWV8fyZhSnpBJULmCDCjMyTwNVsyR8zroO9/ALzNkk7txgbMeicMjh3Xc52rmC5JmV/L2NR6fqnVZsvcrRZXhUXtSUleaiZHdIsmZluPElNo3KxFIhQQ0KqsQZqRbChbXnXUNoPKtr64gI09mMprvDz/1Y4j/6/ScYvf0Drr+6RdMs49IhD1+esPLwBf77r1znv/0fX+YwTPDjJTppcChvJEug6yNx+QI5bDC//jYu6s6c0OGPzzDf2oLZbMEDRVlvq2cvcvzhDwnxmNCsKO5scJoUbZCYAimr58nF9YanzrXsv3yVvvdk8cyT5ziscn3Hm5pE07qWxj0nzq7wj3/5++TgCCEQRWwQZVTcLDZJVZw/VnZmyc8pme9lcQ5+J0USVoVO5pwkC+gIC/Zg2Va+LDD3Enmw5a3GRgNWXUffLNY9xRjGjpNQ8/1s6pJLoxirrkw/uLd5UWQuI5YufZzpvX2aeUdoIR7eZvfeW8yP79IEY2thivBizeWohpF6rBTS1OAPPODexVhyoC1iFr0l20O9NvSBVEolC/xkI135ciBaDe69cqLjvGMSjvjc43P+7I+u8kcuzbj77Ze5d+2YlaUxy5vC5rmTvHYAf+2vfJ8vffcq480NCCNmEpTU5BrwDRnoosDJx5geTYkHezgvpNTXSGiXZky3d+9zD0q5JyytMxoF7rzzXUYjGy6VhSFOSWCitmKOjM/C8dEhz310jfXZETduHeFkhXl2dG7MvW7CwbFn1Hg8kdx3PPpQw+2jKd958xaba8vEquiWCo0tDlZyMekMZjSUGIw53WACk0kENB6uOMekqCdoSsOkrzxT1bbGmnvjZJhGFlOkMgspHBIxemygvmGppnRmK7wLWyubKUu1iSqjX6FCCtnIMM4J/XwKa5dxKw+R3r5Fm7Y4uPUhh9vXgBmhaUg1QIYKvEdXulprEiTXBNIkdist9BMpn9vVBTtMdpJ127mKwwTweQirF2fGJnZ0hVzspTJ9SvTTI04sdTz/eORPfLLh911yTO5+yK2v3KVtWh55/jR+acR7O3P+9pc+5H/45jXuTTOrJ08wzY6cvZobSiCHBi8q5eqadfzmIxzevYNLUWv5qGw45x15f498tEcofiDeEXth9dxl4uEd4u41mtUlhd3NvbQ8p0rHtC+8Mk781KfOMbu2RZo3dN4xzYGjMOb2UaDH07RCUAyFRx5e4p+9cps7ux1nTy8pkl0MzJ1GimjQpzfTIX12fZ8qY24QWBd/fAsIIlbOhzKLhmzzWjCUyZ+UsCnuI/5jaEcNbi2wXg1MLaNvUxgoH9oryXphnJ1SNGPFRM6xRrNVHrS9nmLOksezCKefpzuKHL/3PbbvvMn0aMuU1Q19L4PTkis4m+LLSRa72WF8O7jjxMoEFDcEdoqjhu5oma8lk0fJ8AptmRtmsSvL2nX3KSI5Mg6RUTPj4fUpv+dR+Ocfj3zixBFrB9v07xzhl9Y49/GHOOiF7763y2//1rv82g9vcnUv4iZLjFcDxwndlVFLLV8ipoG+T6RTl5mzxNHOW4TkYZ4gReVUhMD0zm36owNiaBTJ6YXoxyyfOsvR+1/BO8VkS/xaTe/NfghCFcd0esTzlyY8fmLEna9uMZ15DunYy4FdmXOnE2L2NWR1c3nGyokT/PKL7+MbFUwU4xgNOJIFXeDgiOTMMSklS73KWddJ39epc9EBlubNOaccapsoDvbLDIMZI0oPu3eZFUoFJuqJnHS6rMaXxl9V7mGR67g6ESwuo0UdLQuRtGY/rRyPmDT5Kncw3mD8yI9z3K/gTz3JyomHWDM8WamZqktTgnhcSAc1mK3wsyu1MNVFH8Jg+1v4GHkxw0CsDy9BM6KjZJc6Qu4Iac6IOY1MGcuUtcmchzczj53IPHFqzqNnxjx8omXV78P+Aez2HI6X2V5f5u1bU777zbf52g9v8c7NGXMRpGkJK4GIo0teqakSiOLwvqXH4XNWCm1qcKefpm/WWD73jGa8ROWF56ShmKPxGbgwt/thQ6ilDXwz4vjaqyyNWvUMcUrix7jFUZJZEwOh4fBgj5/81LMsn13CP3kGlz2z3jHNDYd9YB4zjqgvXIazp9e5etDzwfVt1lYmOpUd0ksMwcq1UcNErhr/7G1BxxoPoYOS4kBlMrYYK55tK1bxZ4awqkENZyWI4espRxb7vbL5DrEW5j5aask6ljRXm5yTmsJUe9NC7ROzWzJStzHc8OBypu8j7XiEXPs2MmuYiEEfZJv6ebUYEDtikAUYSJ3ypUaBWf9smKHzRfhZVMa5WFsOZpBGylGVCTiXaF1idZRYC3NWw5T1Zsap8ZTTkyknl6asuCmtzHC7R9y4O+Xdgxm7ez1b+0e8e2WXN28csXM0Z/egZ9oLYTSiHS0rjorudCKB7L0a6ziv9r0WMVdifYVMvPse+e4N2lQoXUV5EbX2NWvZEmOWybjZLkevXoHDuzSrY3BOG1ebfvri9JmVO3w8m9OOGu7u9nzxy9dhBmQzrIkz+qi2btmcjESE1XsTfvv1e+xNI6trS7oze18XYNHxeacm5jmJGZUHZVnK4IhYXEJLuoNGuketAkocskAswxFLUKC+NFQzTVKu6RLFSamUyYXhWVidmYw8+8Inc7G1zVUEaM6ReaipxXaDbH5llMmQeGuTCwc1cXx8zM7WFvPjA9t9BnFhFUmWnJU8RBZTk5Pc0PHKQnlseSgii2YO1ii4VLt9sclZMZUMRuXwLuElWl0XtZSxAHssgljEQS9Kxmk8Lni8eUd7b5BgzqZyHsIlo1OhAIYJOx/MTsFG7pKZTmfcu3eXo4NdTdDNg3FPLrJkJQ5bA66+Fi54Wt+wtLzMZGmZpmnU6iw06pDkHOL0JQii7kf97Ii9O9fojo5LaKCpR4yIb8d8yoNXx2gyZnljjdCOzGUqGDpjVE2nvJecstkrm2t/TirhGkxpjTW4sFmlhPOBRBpmGqKpEVgShBj1NdtndWaRFuuEOtWkNmRAUWJW7o3TpthyCUschVswmTH7pmRHgDPXmiwJbx7P2RCQZEB4ShFPYnm5NWPvaHLzXMH0YccXm0IXLCdXi4TM4LrkjAUwsPlctVYd7PNKzVwCiYo3hxkWWuJWYS16J6aJsUlTod9YrJuzOjxX53gNHR1oAWbrmr1RBMJQb4otZAsa1SlfQnJieTwi5GVS7rTrX5D7F6iyKsLzQPJyQf31xGA0x5BTXjxSUu7qAu2zEFZO4EZzYteR7bkEg7pK+E49xp0j+JZklFzE0SfBeXu5nB94FhqoaLa2BgoYKlXKCClZlEV1IoqKeYvHKDYEzsbcshCLUibNqQ70BvvkgZYsVTZYeDAJG6zcJ0LMkCSars7XGK4id/E1R4NhVJztg4tyXYWMF0/TtPTS49L9HmUD78MZwpGH6U+xthV/X1Dj8Gfuz0WsuPgi5Gxqh8r1dqIk/gW71rp4pFTputJdTvTRjr6FDIIybq25hl7U4qqMzMVVFp8rujtDUgokldB8xTBKpBhITjWVmaRMRepYrMKTihPrgg5eFe6FWpm9ejEnQKIxCHOvp6gPhDCiN8+45KJBYbn6y8kC90aj5YLCi+bs6ZwMipQ6k9B/V5aqmbXX8B65z4UjkW0RuxpMWnbjXHOuFsKBcsGlLfJN1D6iWulqca7kuJTILg9NsfH5Qx6iqu7LWhFkQV41zNaxMkR7ibywsFLVfYkz3Zgv7jrJjoZhsl8bgeJUmTVKbXECVHYvsdIklaF8NVCXuttnW8hicpw6VElUHLvs7jagtdSDVDMYjQ6PMyDfLVggFIfMEnmnpVZJo8XGz0ajdTKQomrQunljeAetIj0iDpc7dUeSmklRFTXOTiv12iimOsWewCT80dE7a9ijncwWnpq9+/+1d27HccUwDOVj3X91aSaxyHwQoOQC8pEZnALsmd27uhQJAuZfX7B6m4XfqnkTV9RtAPjEW8+y84j4+TatFV2gk8TvxbGN9Fx36KplNbuFFEixf5xc2avGPQe7qMFcFmTuRKCkHYfRUX/2dq+qrlVzrxTVoOWYe95OY6KvtK+798I1oTp5jRqfKdB64j05LZYIHWpfj+l8NAAUq/SmgGK5df0yaFiCB7fD0vuKXfwZOeEVPGK5eFZ2prPC0+HqEVDWeK/7pWPbnUGO1of5Q7uGdk1GkPORvsMe5wMYvgMdjmVrPY1tgunb58HuMD8fK77WHx+T9Q3ci+Zc0j74EUXz79u2w2ahNCawoe+15gTuQz7CrtnAjueHz+0fllLzt7+bDyX2ADv2/O2ah6zYlOxJ0/V6PzDelXhi1baT93RnpPa6sKAcYSIWL/6PQ267mR1Or/GvcurtT6CWCGg23HxSonoGERO0+OToIbPjnIJnBwTbeFWkpZ3DMsyt/px70tOflEV+340Xlil3nXELkb3BO/0fmv3RG4puWPfaaR/Tbl83bNSz+VxEfXW6TyQHp43sBroj1oJu9fgh0PUf61sOSeqe0Phsgh+6+0zl3Mxier0NJTxzzzc7fZc9fKIxYsq9T352h5JaDlvjb9+Sb9ujqDePBWrn/lEUsJzKpBNRmudcBMfXD3eSkSSiw5B20EmC3A7eGjOr8Kfl6AmX1Q64aYG0myYbsUNvqvfSE0o69hraIhKHjq2Oep9N6Ns/1f1rrXFXTodT6BTkHliGZShcQs9abv6Bow1tJHFBrFO3XluXarub2BSBM8/ZeoQn4bsPN5/91OgUAUz82V3fuVoMWvs6ZzRzi0b7yNCGjDVxT7q0o6dNZ8v50hjW6faUW3YXf+ncXt9unvPjZj/f6/bqaV0Fo+35H0iEokaFZpZrBs5X/s1fo4cUtxBmXG6fJynKVzJgGICdLhhk4sOoHUtArB+rZjs1asRwmLQnLvArVrsHR9VvlCEULfUORbrqiSPpJ/nMZ7OpR3+3Q5pzven2gOvbui12PFCCXHXoDbGnqOn0Tzc9If57/EezUAghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEL8a/4CVqSlGdM5WzwAAAAASUVORK5CYII=">
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
    margin: 0;
    text-shadow: 0 0 10px rgba(0,229,255,0.4);
  }
  .brand-row {
    display: flex; align-items: center; gap: 10px; margin: 4px 0;
  }
  .brand-logo {
    width: 38px; height: 38px; object-fit: contain;
    filter: drop-shadow(0 0 6px rgba(0,229,255,0.35));
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
  .card-head .sym { font-weight: 700; font-size: 1.05em; display: inline-flex; align-items: center; gap: 6px; }
  .coin-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%; border: 1px solid;
    font-size: 0.75em; font-weight: 800; line-height: 1;
  }
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
  .floating-row {
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(0,229,255,0.06); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 10px; margin: 6px 0; font-size: 0.78em;
  }
  .floating-label { color: var(--muted); }
  .floating-value { font-weight: 800; font-size: 1.05em; }
  .floating-value.pnl-pos { color: var(--green); }
  .floating-value.pnl-neg { color: var(--red); }
  .history-block { margin-top: 10px; border-top: 1px solid var(--border); padding-top: 8px; }
  .history-title { font-size: 0.75em; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
  .history-list { max-height: 150px; overflow-y: auto; display: flex; flex-direction: column; gap: 4px; }
  .history-empty { font-size: 0.75em; color: var(--muted); font-style: italic; }
  .history-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.72em; padding: 4px 6px; border-radius: 6px; background: rgba(255,255,255,0.02);
  }
  .history-prices { color: var(--muted); }
  .history-prices b { color: #e8edf3; }
  .history-pnl { font-weight: 700; }
  .history-pnl.pnl-pos { color: var(--green); }
  .history-pnl.pnl-neg { color: var(--red); }
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

  <div class="brand-row">
    <img class="brand-logo" src="data:image/png;base64,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" alt="BT 1 Intelligent">
    <h1>BT 1 Intelligent</h1>
  </div>
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
const lastPriceUpdate = {};  // symbol -> unix seconds del ultimo cambio de precio REAL
const TIMEFRAMES = __TIMEFRAMES_JSON__;  // [[valor, etiqueta], ...] inyectado desde el backend
const COIN_BADGES = __COIN_BADGES_JSON__;  // { "BTC": {glyph, color}, ... } inyectado desde el backend
const DEFAULT_BADGE = { glyph: "●", color: "#7d8899" };
const COLORS = { up: "#00e676", down: "#ff3b5c" };

function fmt(n, d=2) { return Number(n).toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d}); }

function ensureCard(symbol) {
  if (document.getElementById('card-' + symbol)) return;
  const grid = document.getElementById('cardsGrid');
  const el = document.createElement('div');
  el.className = 'card';
  el.id = 'card-' + symbol;
  const ticker = symbol.split('/')[0];
  const badge = COIN_BADGES[ticker] || DEFAULT_BADGE;
  el.innerHTML = `
    <div class="card-head">
      <span class="sym">
        <span class="coin-badge" style="background:${badge.color}22; color:${badge.color}; border-color:${badge.color}55;">${badge.glyph}</span>
        ${symbol}
      </span>
      <span class="pos-pill" id="pos-${symbol}">--</span>
    </div>
    <div class="price-row">
      <span class="price" id="price-${symbol}">--</span>
      <span id="arrow-${symbol}"></span>
      <span class="pct-badge" id="pct24h-${symbol}" title="Cambio 24h (mercado real de Binance)">--</span>
    </div>
    <div class="meta-row" style="display:flex;justify-content:space-between;gap:8px;font-size:11px;color:#7d8899;margin:2px 0 6px;">
      <span id="status-${symbol}">--</span>
      <span id="upd-${symbol}">--</span>
    </div>
    <div class="tf-row">
      <span class="tf-label">Temporalidad:</span>
      <select class="tf-select" id="tf-${symbol}" onchange="changeTimeframe('${symbol}', this.value)"></select>
    </div>
    <div class="chart-container" id="chartc-${symbol}"></div>
    <div class="floating-row" id="floatingrow-${symbol}" style="display:none;">
      <span class="floating-label">PnL flotante (posicion abierta):</span>
      <span class="floating-value" id="floatingpnl-${symbol}">--</span>
    </div>
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
    <div class="history-block">
      <div class="history-title">Historial de operaciones</div>
      <div class="history-list" id="history-${symbol}">
        <div class="history-empty">Sin operaciones cerradas todavia.</div>
      </div>
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

  const floatingRow = document.getElementById('floatingrow-' + symbol);
  const floatingEl = document.getElementById('floatingpnl-' + symbol);
  if (d.in_position) {
    floatingRow.style.display = 'flex';
    const fp = d.floating_pnl_usdt || 0;
    const fpct = d.floating_pnl_pct || 0;
    floatingEl.textContent = (fp >= 0 ? '+$' : '-$') + fmt(Math.abs(fp), 2) + '  (' + (fpct >= 0 ? '+' : '') + fmt(fpct, 2) + '%)';
    floatingEl.className = 'floating-value ' + (fp >= 0 ? 'pnl-pos' : 'pnl-neg');
  } else {
    floatingRow.style.display = 'none';
  }

  const historyEl = document.getElementById('history-' + symbol);
  const closedTrades = d.closed_trades || [];
  if (closedTrades.length === 0) {
    historyEl.innerHTML = '<div class="history-empty">Sin operaciones cerradas todavia.</div>';
  } else {
    historyEl.innerHTML = closedTrades.map(ct => {
      const pos = ct.pnl_usdt >= 0;
      const decimals = ct.buy_price < 10 ? 4 : 2;
      return `
        <div class="history-row">
          <span class="history-prices">Compra <b>$${fmt(ct.buy_price, decimals)}</b> → Venta <b>$${fmt(ct.sell_price, decimals)}</b></span>
          <span class="history-pnl ${pos ? 'pnl-pos' : 'pnl-neg'}">${pos ? '+' : ''}$${fmt(ct.pnl_usdt, 2)} (${pos ? '+' : ''}${fmt(ct.pnl_pct, 2)}%)</span>
        </div>`;
    }).join('');
  }

  const statusEl = document.getElementById('status-' + symbol);
  if (statusEl) statusEl.textContent = d.status || '--';
  // La "antiguedad" del precio se calcula y refresca cada segundo en el propio
  // navegador (ver updateFreshnessLabels), en vez de depender del texto que
  // manda el servidor -- asi no se confunde "el panel se actualizo" con
  // "el precio realmente cambio".
  lastPriceUpdate[symbol] = d.price_updated_at || 0;

  const tfSelect = document.getElementById('tf-' + symbol);
  if (tfSelect && document.activeElement !== tfSelect && d.timeframe && tfSelect.value !== d.timeframe) {
    tfSelect.value = d.timeframe;
  }

  if (d.candles && d.candles.length) {
    charts[symbol].series.setData(d.candles);
    charts[symbol].series.setMarkers(buildMarkers(d.trade_history || []));
  }
}

function apiHeaders() {
  let token = sessionStorage.getItem('bt1_api_token') || '';
  if (!token) {
    token = prompt('Introduce WEB_API_TOKEN para autorizar cambios/transferencias:') || '';
    if (token) sessionStorage.setItem('bt1_api_token', token);
  }
  return {
    'Content-Type': 'application/json',
    'X-API-Token': token
  };
}

async function changeTimeframe(symbol, timeframe) {
  try {
    const res = await fetch('/api/timeframe', {
      method: 'POST',
      headers: apiHeaders(),
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
      headers: apiHeaders(),
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
    const res = await fetch('/api/state', { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
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
    const footer = document.getElementById('updatedFooter');
    footer.textContent = 'Ultima actualizacion: ' + data.generated_at;
    delete footer.dataset.lastError;
  } catch (e) {
    console.error('Error actualizando panel:', e);
    const footer = document.getElementById('updatedFooter');
    // Evita spamear el mensaje cada segundo; solo actualiza cada ~5s
    const now = Date.now();
    if (!footer.dataset.lastError || now - Number(footer.dataset.lastError) > 5000) {
      footer.textContent = 'Error al actualizar (reintentando...)';
      footer.dataset.lastError = String(now);
    }
  }
}
function updateFreshnessLabels() {
  const nowSec = Math.floor(Date.now() / 1000);
  for (const [symbol, ts] of Object.entries(lastPriceUpdate)) {
    const el = document.getElementById('upd-' + symbol);
    if (!el || !ts) continue;
    const age = Math.max(0, nowSec - ts);
    let label;
    if (age < 5) label = 'precio: hace un instante';
    else if (age < 60) label = `precio: hace ${age}s`;
    else label = `precio: hace ${Math.floor(age / 60)} min`;
    el.textContent = label;
    // Si el precio lleva mucho sin cambiar, lo resaltamos para que se note
    // que puede ser un dato viejo (no confundir con "el panel no responde").
    el.style.color = age > 90 ? 'var(--red)' : '#7d8899';
  }
}

refresh();
setInterval(refresh, 1000);
setInterval(updateFreshnessLabels, 1000);
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
DASHBOARD_TEMPLATE = DASHBOARD_TEMPLATE.replace(
    "__COIN_BADGES_JSON__",
    json.dumps(COIN_BADGES),
)

flask_app = Flask(__name__)

def _api_authorized() -> bool:
    token = cfg.web_api_token.strip()
    if not token:
        # En produccion es mejor fail-closed: si el token no esta configurado,
        # las rutas que cambian dinero/estado quedan deshabilitadas.
        return False
    supplied = request.headers.get("X-API-Token", "")
    return supplied == token

def _require_api_auth():
    if not _api_authorized():
        return jsonify({"ok": False, "error": "API protegida: falta X-API-Token valido"}), 401
    return None


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
    auth_error = _require_api_auth()
    if auth_error:
        return auth_error
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
    auth_error = _require_api_auth()
    if auth_error:
        return auth_error
    data = request.get_json(force=True, silent=True) or {}
    symbol = data.get("symbol")
    timeframe = data.get("timeframe")

    bot = bots_by_symbol.get(symbol)
    if not bot:
        return jsonify({"ok": False, "error": "Moneda no encontrada"}), 404

    if timeframe not in ALLOWED_TIMEFRAMES:
        return jsonify({"ok": False, "error": f"Temporalidad invalida. Usa una de: {', '.join(ALLOWED_TIMEFRAMES)}"}), 400

    if not bot.set_timeframe(timeframe):
        return jsonify({"ok": False, "error": "No se pudo cambiar la temporalidad"}), 500
    shared_state.update(symbol, timeframe=timeframe)
    return jsonify({"ok": True, "symbol": symbol, "timeframe": timeframe})


def run_web_dashboard(host: str, port: int):
    # Solo se usa en modo "python trading_bot.py". En produccion (Render)
    # el servidor WSGI es Gunicorn y este metodo no se llama.
    flask_app.run(host=host, port=port, debug=False, use_reloader=False)


# Alias estandar para Gunicorn:  gunicorn trading_bot:app --bind 0.0.0.0:$PORT --workers 1
app = flask_app

# ------------------------------------------------------------------
# ARRANQUE UNICO DE SERVICIOS DE FONDO
# ------------------------------------------------------------------
# Se usa tanto con "python trading_bot.py" como con Gunicorn.
# El flag + lock evitan que se arranquen dos veces (por ejemplo si el
# modulo se importa mas de una vez o si hay un worker extra).
_services_started = False
_services_lock = threading.Lock()


def start_services(start_flask_thread: bool = False):
    """Inicializa bots, Telegram y (opcionalmente) el hilo de Flask.

    start_flask_thread=True solo cuando se ejecuta directamente con Python.
    Con Gunicorn, Flask lo sirve el propio Gunicorn y start_flask_thread
    debe quedar en False.
    """
    global _services_started

    with _services_lock:
        if _services_started:
            return
        _services_started = True

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    root_log = logging.getLogger("trading_bot")

    shared_notifier = TelegramNotifier(
        cfg.telegram_bot_token, cfg.telegram_chat_id, cfg.telegram_enabled, root_log
    )

    # Cualquier WARNING/ERROR registrado en cualquier parte del bot (incluidas
    # todas las submonedas, ya que sus loggers son "trading_bot.<SIMBOLO>" y por
    # lo tanto hijos de este) llega tambien a Telegram automaticamente.
    if shared_notifier.enabled:
        logging.getLogger("trading_bot").addHandler(TelegramLogHandler(shared_notifier))

    bots = [SymbolTradingBot(symbol, cfg, shared_notifier) for symbol in cfg.symbols]
    bots_by_symbol.update({b.symbol: b for b in bots})
    _init_startup_tracking(cfg.symbols, shared_notifier)

    command_bot = TelegramCommandBot(
        cfg.telegram_bot_token, cfg.telegram_chat_id, cfg.telegram_enabled, cfg, root_log
    )
    if command_bot.enabled:
        cmd_thread = threading.Thread(target=command_bot.poll_loop, daemon=True, name="telegram-commands")
        cmd_thread.start()
        root_log.info("Escuchando comandos de Telegram...")

    if start_flask_thread and cfg.web_enabled:
        web_thread = threading.Thread(
            target=run_web_dashboard,
            args=(cfg.web_host, cfg.web_port),
            daemon=True,
            name="flask-dev-server",
        )
        web_thread.start()
        print(
            f"Panel web disponible en http://<ip-del-servidor>:{cfg.web_port}  "
            f"(o http://localhost:{cfg.web_port} si corres localmente)"
        )

    root_log.info(
        f"Monedas activas: {', '.join(cfg.symbols)} | Balance inicial por moneda: "
        f"{cfg.initial_balance_usdt:.2f} USDT | Balance total inicial: "
        f"{cfg.initial_balance_usdt * len(cfg.symbols):.2f} USDT"
    )

    for bot in bots:
        t = threading.Thread(target=bot.run, daemon=True, name=f"bot-{bot.binance_symbol}")
        t.start()
        time.sleep(1.5)  # escalona el arranque para no golpear la API de Binance de una vez

    root_log.info("Todos los bots de trading han sido lanzados (proceso mantenido vivo).")


# Arranque automatico cuando el modulo se carga bajo Gunicorn (workers=1).
# Si se ejecuta como script, el bloque __main__ tambien llama a start_services.
# El flag interno evita el doble arranque.
if __name__ != "__main__" and os.environ.get("BOT_SKIP_AUTO_START") != "1":
    # Solo arrancar servicios de fondo cuando alguien importa el modulo como app WSGI.
    # No arrancar el hilo de Flask: Gunicorn es quien sirve HTTP.
    # BOT_SKIP_AUTO_START=1 permite importar el modulo en pruebas sin lanzar hilos.
    try:
        start_services(start_flask_thread=False)
    except Exception as _boot_err:
        logging.getLogger("trading_bot").exception("Fallo al arrancar servicios de fondo: %s", _boot_err)


# ------------------------------------------------------------------
if __name__ == "__main__":
    start_services(start_flask_thread=True)
    # Mantener el proceso principal vivo para siempre. Los hilos de trading
    # tambien se bloquean dentro de run(), pero esto es una red de seguridad
    # adicional por si algun dia se cambia la semantica de esos hilos.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nApagado solicitado. Saliendo...")
