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
  - Un hilo de trading independiente POR MONEDA (cada una con su propia cuenta
    de paper trading, historial de precios e historial de operaciones).
  - Notificaciones a Telegram en cada compra/venta y resumen periodico.
  - Panel web (Flask) estilo Binance: una grafica en vivo por moneda, precio
    subiendo/bajando en tiempo real, % de rentabilidad de cada moneda
    actualizandose solo, y el balance total combinado de todos los activos.
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
    pip install requests pandas flask
"""

import time
import csv
import os
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque

import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string


# ------------------------------------------------------------------
# CONFIGURACION -- ajusta estos parametros a tu gusto
# ------------------------------------------------------------------
@dataclass
class Config:
    symbols: list = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    ])
    timeframe: str = "15m"          # Temporalidad de las velas: 1m, 5m, 15m, 1h, 4h, 1d...
    short_window: int = 9           # Periodo de la media movil corta
    long_window: int = 21           # Periodo de la media movil larga
    take_profit_pct: float = 0.02   # Margen de ganancia objetivo: 2%
    stop_loss_pct: float = 0.01     # Perdida maxima aceptada: 1%
    trade_fraction: float = 0.95    # Fraccion del balance a usar en cada compra
    fee_pct: float = 0.001          # Comision simulada de Binance (~0.1%)
    initial_balance_usdt: float = 100.0   # Balance virtual inicial POR MONEDA
    poll_seconds: int = 20          # Cada cuanto revisa el mercado cada moneda (segundos)
    candles_lookback: int = 200     # Cuantas velas historicas descarga
    price_history_maxlen: int = 150  # Puntos que se guardan para la grafica en vivo
    log_dir: str = "logs"

    # --- Telegram ---
    telegram_enabled: bool = True
    telegram_bot_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "PON_AQUI_TU_TOKEN")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "PON_AQUI_TU_CHAT_ID")
    telegram_summary_every_n_cycles: int = 90  # resumen periodico por moneda

    # --- Panel web ---
    web_enabled: bool = True
    web_host: str = "0.0.0.0"
    web_port: int = int(os.environ.get("PORT", 5000))  # Render asigna el puerto via variable PORT


cfg = Config()


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
                "in_position": False,
                "entry_price": 0.0,
                "equity": initial_balance_per_symbol,
                "balance_usdt": initial_balance_per_symbol,
                "pnl_total": 0.0,
                "pnl_total_pct": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "avg_pnl_per_trade": 0.0,
                "last_update": "",
                "trade_history": [],
                "price_history": deque(maxlen=history_maxlen),
                "status": "iniciando",
            }
            for s in symbols
        }

    def update(self, symbol: str, **kwargs):
        with self.lock:
            self.symbols_data[symbol].update(kwargs)

    def push_price(self, symbol: str, timestamp: str, price: float):
        with self.lock:
            d = self.symbols_data[symbol]
            d["prev_price"] = d["current_price"]
            d["current_price"] = round(float(price), 6)
            d["price_history"].append({"t": timestamp, "p": round(float(price), 6)})

    def snapshot(self) -> dict:
        with self.lock:
            symbols_out = {}
            total_equity = 0.0
            total_trades = 0
            total_wins = 0
            for s, d in self.symbols_data.items():
                sd = dict(d)
                sd["price_history"] = list(d["price_history"])
                symbols_out[s] = sd
                total_equity += d["equity"]
                total_trades += d["total_trades"]
                total_wins += d["winning_trades"]

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
                    "pnl_total": round(total_pnl, 2),
                    "pnl_total_pct": round(total_pnl_pct, 2),
                    "total_trades": total_trades,
                    "win_rate_pct": round(total_win_rate, 2),
                },
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }


shared_state = SharedState(cfg.symbols, cfg.initial_balance_usdt, cfg.price_history_maxlen)


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
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
    # bloquea (HTTP 451) las peticiones desde IPs de EE.UU. Este es el espejo
    # publico de solo lectura de mercado, sin esa restriccion.
    BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

    def __init__(self, symbol: str, config: Config, notifier: TelegramNotifier):
        self.symbol = symbol
        self.binance_symbol = symbol.replace("/", "")
        self.cfg = config
        self.strategy = SmaCrossStrategy(config.short_window, config.long_window)
        self.account = PaperAccount(balance_usdt=config.initial_balance_usdt)
        self.telegram = notifier
        self._cycle_count = 0
        self._setup_logging()
        self._setup_csv()

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
            writer.writerow(entry.values())

    def fetch_data(self) -> pd.DataFrame:
        params = {"symbol": self.binance_symbol, "interval": self.cfg.timeframe,
                  "limit": self.cfg.candles_lookback}
        response = requests.get(self.BINANCE_KLINES_URL, params=params, timeout=10)
        response.raise_for_status()
        raw = response.json()
        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df

    def check_risk_management(self, current_price: float):
        if not self.account.in_position:
            return
        change_pct = (current_price / self.account.entry_price) - 1

        if change_pct >= self.cfg.take_profit_pct:
            entry = self.account.sell(current_price, self.cfg.fee_pct, reason="take-profit")
            self._append_csv(entry)
            msg = f"[{self.symbol}] TAKE PROFIT ejecutado a {current_price:.4f} ({change_pct*100:.2f}%)"
            self.log.info(f"[TP] {msg}")
            self.telegram.send(f"🎯 {msg}")

        elif change_pct <= -self.cfg.stop_loss_pct:
            entry = self.account.sell(current_price, self.cfg.fee_pct, reason="stop-loss")
            self._append_csv(entry)
            msg = f"[{self.symbol}] STOP LOSS ejecutado a {current_price:.4f} ({change_pct*100:.2f}%)"
            self.log.info(f"[SL] {msg}")
            self.telegram.send(f"🛑 {msg}")

    def step(self):
        df = self.fetch_data()
        current_price = float(df["close"].iloc[-1])
        now = datetime.now(timezone.utc)

        shared_state.push_price(self.symbol, now.strftime("%H:%M:%S"), current_price)

        self.check_risk_management(current_price)

        signal = self.strategy.compute_signal(df)

        if signal == "buy" and not self.account.in_position:
            entry = self.account.buy(current_price, self.cfg.trade_fraction, self.cfg.fee_pct)
            self._append_csv(entry)
            msg = f"[{self.symbol}] COMPRA a {current_price:.4f}"
            self.log.info(f"[BUY] {msg}")
            self.telegram.send(f"🟢 {msg}")

        elif signal == "sell" and self.account.in_position:
            entry = self.account.sell(current_price, self.cfg.fee_pct, reason="senal-cruce")
            self._append_csv(entry)
            msg = (f"[{self.symbol}] VENTA (senal) a {current_price:.4f} | "
                   f"PnL: {entry['pnl_usdt']} USDT ({entry['pnl_pct']}%)")
            self.log.info(f"[SELL] {msg}")
            self.telegram.send(f"🔴 {msg}")

        equity = self.account.equity(current_price)
        pnl_total = equity - self.cfg.initial_balance_usdt
        pnl_total_pct = (equity / self.cfg.initial_balance_usdt - 1) * 100
        stats = self.account.stats()

        shared_state.update(
            self.symbol,
            symbol=self.symbol,
            in_position=self.account.in_position,
            entry_price=round(self.account.entry_price, 6),
            equity=round(equity, 2),
            balance_usdt=round(self.account.balance_usdt, 2),
            pnl_total=round(pnl_total, 2),
            pnl_total_pct=round(pnl_total_pct, 2),
            last_update=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            trade_history=list(reversed(self.account.trade_history[-30:])),
            status="operando",
            **stats,
        )

        self.log.info(
            f"[{self.symbol}] Precio: {current_price:.4f} | Posicion: {'SI' if self.account.in_position else 'NO'} "
            f"| Equity: {equity:.2f} USDT | PnL total: {pnl_total:+.2f} USDT ({pnl_total_pct:+.2f}%) "
            f"| Trades: {stats['total_trades']} | Win rate: {stats['win_rate_pct']:.1f}%"
        )

        self._cycle_count += 1
        if self.cfg.telegram_summary_every_n_cycles and \
           self._cycle_count % self.cfg.telegram_summary_every_n_cycles == 0:
            self.telegram.send(
                f"📊 Resumen {self.symbol}\n"
                f"Precio: {current_price:.4f}\n"
                f"Equity: {equity:.2f} USDT\n"
                f"PnL total: {pnl_total:+.2f} USDT ({pnl_total_pct:+.2f}%)\n"
                f"Trades: {stats['total_trades']} | Win rate: {stats['win_rate_pct']:.1f}%"
            )

    def run(self):
        self.log.info(
            f"Iniciando bot PAPER TRADING | {self.symbol} | {self.cfg.timeframe} | "
            f"SMA({self.cfg.short_window}/{self.cfg.long_window}) | "
            f"TP {self.cfg.take_profit_pct*100:.1f}% | SL {self.cfg.stop_loss_pct*100:.1f}% | "
            f"Balance inicial: {self.account.balance_usdt:.2f} USDT"
        )
        self.telegram.send(
            f"🤖 Bot iniciado (paper trading)\n{self.symbol} | {self.cfg.timeframe}\n"
            f"Balance inicial: {self.account.balance_usdt:.2f} USDT"
        )
        while True:
            try:
                self.step()
            except Exception as e:
                self.log.error(f"[{self.symbol}] Error en el ciclo: {e}")
                shared_state.update(self.symbol, status=f"error: {e}")
            time.sleep(self.cfg.poll_seconds)


# ------------------------------------------------------------------
# PANEL WEB (Flask) -- estilo Binance, con grafica en vivo por moneda
# ------------------------------------------------------------------
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bot de Trading Multi-Moneda</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
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
  .stats-row {
    display: flex; justify-content: space-between; margin-top: 10px;
    font-size: 0.8em; color: var(--muted);
  }
  .stats-row span b { color: #e8edf3; }
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

  <h1>🤖 Panel de Trading Multi-Moneda</h1>
  <div class="subtitle">Paper trading &middot; datos reales de Binance &middot; se actualiza solo cada 5s</div>

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
      <span class="pct-badge" id="pct-${symbol}">--</span>
    </div>
    <canvas class="chart" id="chart-${symbol}"></canvas>
    <div class="stats-row">
      <span>Trades: <b id="trades-${symbol}">0</b></span>
      <span>Win rate: <b id="winrate-${symbol}">0%</b></span>
      <span>Equity: <b id="equity-${symbol}">$0</b></span>
    </div>
  `;
  grid.appendChild(el);

  const ctx = document.getElementById('chart-' + symbol).getContext('2d');
  charts[symbol] = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{
      data: [], borderColor: COLORS.up, borderWidth: 2, pointRadius: 0,
      tension: 0.3, fill: true,
      backgroundColor: 'rgba(0,230,118,0.08)',
    }]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { display: true, ticks: { color: '#7d8899', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
      },
    },
  });
}

function updateCard(symbol, d) {
  ensureCard(symbol);
  const up = d.current_price >= d.prev_price;
  document.getElementById('price-' + symbol).textContent = '$' + fmt(d.current_price, d.current_price < 10 ? 4 : 2);
  document.getElementById('price-' + symbol).style.color = up ? COLORS.up : COLORS.down;
  document.getElementById('arrow-' + symbol).innerHTML = up
    ? '<span class="arrow-up">▲</span>' : '<span class="arrow-down">▼</span>';

  const pctEl = document.getElementById('pct-' + symbol);
  pctEl.textContent = (d.pnl_total_pct >= 0 ? '+' : '') + fmt(d.pnl_total_pct, 2) + '%';
  pctEl.className = 'pct-badge ' + (d.pnl_total_pct >= 0 ? 'pct-pos' : 'pct-neg');

  const posEl = document.getElementById('pos-' + symbol);
  posEl.textContent = d.in_position ? 'POSICION ABIERTA' : 'SIN POSICION';
  posEl.className = 'pos-pill ' + (d.in_position ? 'pos-open' : 'pos-closed');

  document.getElementById('trades-' + symbol).textContent = d.total_trades;
  document.getElementById('winrate-' + symbol).textContent = fmt(d.win_rate_pct, 1) + '%';
  document.getElementById('equity-' + symbol).textContent = '$' + fmt(d.equity, 2);

  const chart = charts[symbol];
  chart.data.labels = d.price_history.map(p => p.t);
  chart.data.datasets[0].data = d.price_history.map(p => p.p);
  chart.data.datasets[0].borderColor = up ? COLORS.up : COLORS.red;
  chart.data.datasets[0].backgroundColor = up ? 'rgba(0,230,118,0.08)' : 'rgba(255,59,92,0.08)';
  chart.update('none');
}

function updateTotal(t) {
  const bar = document.getElementById('totalBar');
  const pos = t.pnl_total_pct >= 0;
  bar.innerHTML = `
    <div class="total-item"><div class="label">Balance total (${t.num_assets} activos)</div>
      <div class="value">$${fmt(t.equity, 2)}</div></div>
    <div class="total-item"><div class="label">Balance inicial</div>
      <div class="value" style="color:var(--muted)">$${fmt(t.initial_balance, 2)}</div></div>
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
    for (const [symbol, d] of Object.entries(data.symbols)) updateCard(symbol, d);
    updateTotal(data.total);
    document.getElementById('updatedFooter').textContent = 'Ultima actualizacion: ' + data.generated_at;
  } catch (e) {
    console.error('Error actualizando panel:', e);
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""

flask_app = Flask(__name__)


@flask_app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)


@flask_app.route("/api/state")
def api_state():
    return jsonify(shared_state.snapshot())


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
