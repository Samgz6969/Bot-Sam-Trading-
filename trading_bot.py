"""
Bot de Trading Automático - Binance - Paper Trading - Multi-Par
=================================================================

Estrategia: Cruce de Medias Moviles (SMA Crossover)
  - Senal de COMPRA: la media corta cruza por ENCIMA de la media larga (cruce dorado)
  - Senal de VENTA:  la media corta cruza por DEBAJO de la media larga (cruce de la muerte)

Gestion de riesgo / margen de ganancias:
  - TAKE_PROFIT_PCT: cierra la posicion automaticamente si la ganancia alcanza este %
  - STOP_LOSS_PCT:   cierra la posicion automaticamente si la perdida alcanza este %

Extras de esta version:
  - Opera VARIOS pares a la vez (BTC, ETH, SOL, BNB, XRP por defecto), cada uno
    con su propio balance virtual, en un hilo independiente.
  - Notificaciones a Telegram en cada compra/venta y resumen periodico, por par.
  - Panel web (Flask) con estadisticas EN VIVO por cada par: precio, equity, PnL,
    % de operaciones ganadoras, historial de trades y una grafica de precio en
    tiempo real (se actualiza sola via JavaScript, sin recargar la pagina).
    Se abre en http://<ip-del-servidor>:5000

IMPORTANTE:
  - Este bot corre en modo PAPER TRADING: no envia ordenes reales, solo simula
    con un balance virtual, usando datos de mercado reales de Binance (API
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

import pandas as pd
import requests
from flask import Flask, render_template_string, jsonify


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
    initial_balance_usdt: float = 1000.0  # Balance virtual inicial por par (paper trading)
    poll_seconds: int = 60          # Cada cuanto revisa el mercado (segundos)
    candles_lookback: int = 200     # Cuantas velas historicas descarga
    price_history_points: int = 200  # Puntos maximos guardados para la grafica en vivo
    log_dir: str = "logs"           # Carpeta donde se guarda un CSV de trades por par

    # --- Telegram ---
    telegram_enabled: bool = True
    telegram_bot_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "PON_AQUI_TU_TOKEN")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "PON_AQUI_TU_CHAT_ID")
    telegram_summary_every_n_cycles: int = 30  # resumen periodico (30 ciclos x 60s = cada 30 min)

    # --- Panel web ---
    web_enabled: bool = True
    web_host: str = "0.0.0.0"
    web_port: int = int(os.environ.get("PORT", 5000))  # Render asigna el puerto via variable PORT


# ------------------------------------------------------------------
# NOTIFICADOR DE TELEGRAM (compartido entre todos los pares)
# ------------------------------------------------------------------
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, enabled: bool, log):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled and token and "PON_AQUI" not in token
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
# Ahora guarda un diccionario por cada simbolo/par.
# ------------------------------------------------------------------
class SharedState:
    def __init__(self, symbols, max_points):
        self.lock = threading.Lock()
        self.max_points = max_points
        self.data = {
            sym: {
                "symbol": sym,
                "current_price": 0.0,
                "in_position": False,
                "entry_price": 0.0,
                "equity": 0.0,
                "balance_usdt": 0.0,
                "pnl_total": 0.0,
                "pnl_total_pct": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "avg_pnl_per_trade": 0.0,
                "last_update": "",
                "trade_history": [],
                "price_history": [],  # [{"t": "HH:MM:SS", "p": precio}, ...]
            }
            for sym in symbols
        }

    def update(self, symbol, **kwargs):
        with self.lock:
            self.data[symbol].update(kwargs)

    def push_price(self, symbol, t_label, price):
        with self.lock:
            hist = self.data[symbol]["price_history"]
            hist.append({"t": t_label, "p": round(float(price), 6)})
            if len(hist) > self.max_points:
                del hist[: len(hist) - self.max_points]

    def snapshot_all(self):
        with self.lock:
            return {k: dict(v) for k, v in self.data.items()}


# ------------------------------------------------------------------
# CUENTA DE PAPER TRADING (balance y posicion simulados)
# ------------------------------------------------------------------
@dataclass
class PaperAccount:
    balance_usdt: float
    position_qty: float = 0.0
    entry_price: float = 0.0
    in_position: bool = False
    trade_history: list = field(default_factory=list)

    def buy(self, price: float, fraction: float, fee_pct: float) -> None:
        spend = self.balance_usdt * fraction
        fee = spend * fee_pct
        qty = (spend - fee) / price
        self.balance_usdt -= spend
        self.position_qty = qty
        self.entry_price = price
        self.in_position = True
        self._log_trade("COMPRA", price, qty, fee)

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
# BOT PARA UN PAR -- cada instancia corre en su propio hilo
# ------------------------------------------------------------------
class SymbolBot:
    # Se usa data-api.binance.vision en vez de api.binance.com porque Binance
    # bloquea (HTTP 451) las peticiones desde IPs de EE.UU. Este es el espejo
    # publico de solo lectura de mercado, sin esa restriccion.
    BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

    def __init__(self, symbol: str, config: Config, telegram: TelegramNotifier,
                 shared_state: SharedState, log):
        self.symbol = symbol
        self.binance_symbol = symbol.replace("/", "")
        self.cfg = config
        self.strategy = SmaCrossStrategy(config.short_window, config.long_window)
        self.account = PaperAccount(balance_usdt=config.initial_balance_usdt)
        self.telegram = telegram
        self.shared_state = shared_state
        self.log = log
        self._cycle_count = 0
        self._setup_csv()

    def _setup_csv(self):
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.cfg.log_dir, f"trades_{self.binance_symbol}.csv")
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
        current_price = df["close"].iloc[-1]
        now = datetime.now(timezone.utc)

        self.shared_state.push_price(self.symbol, now.strftime("%H:%M:%S"), current_price)

        self.check_risk_management(current_price)

        signal = self.strategy.compute_signal(df)

        if signal == "buy" and not self.account.in_position:
            self.account.buy(current_price, self.cfg.trade_fraction, self.cfg.fee_pct)
            self._append_csv(self.account.trade_history[-1])
            msg = f"[{self.symbol}] COMPRA a {current_price:.4f}"
            self.log.info(f"[BUY] {msg}")
            self.telegram.send(f"🟢 {msg}")

        elif signal == "sell" and self.account.in_position:
            entry = self.account.sell(current_price, self.cfg.fee_pct, reason="senal-cruce")
            self._append_csv(entry)
            msg = f"[{self.symbol}] VENTA (senal) a {current_price:.4f} | PnL: {entry['pnl_usdt']} USDT ({entry['pnl_pct']}%)"
            self.log.info(f"[SELL] {msg}")
            self.telegram.send(f"🔴 {msg}")

        equity = self.account.equity(current_price)
        pnl_total = equity - self.cfg.initial_balance_usdt
        pnl_total_pct = (equity / self.cfg.initial_balance_usdt - 1) * 100
        stats = self.account.stats()

        self.shared_state.update(
            self.symbol,
            symbol=self.symbol,
            current_price=round(float(current_price), 6),
            in_position=self.account.in_position,
            entry_price=round(self.account.entry_price, 6),
            equity=round(equity, 2),
            balance_usdt=round(self.account.balance_usdt, 2),
            pnl_total=round(pnl_total, 2),
            pnl_total_pct=round(pnl_total_pct, 2),
            last_update=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            trade_history=list(reversed(self.account.trade_history[-50:])),
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
            time.sleep(self.cfg.poll_seconds)
# ------------------------------------------------------------------
# PANEL WEB (Flask) -- estilo cyberpunk, graficas en vivo con Chart.js
# ------------------------------------------------------------------
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bot de Trading — Panel en Vivo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
  :root {
    --neon-cyan: #00fff2;
    --neon-magenta: #ff00e6;
    --neon-green: #39ff88;
    --neon-red: #ff3860;
    --bg-deep: #05060a;
    --panel-bg: rgba(13, 17, 28, 0.75);
    --border-glow: rgba(0, 255, 242, 0.35);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    font-family: 'Share Tech Mono', monospace;
    color: #d8f9ff;
    background: var(--bg-deep);
    background-image:
      linear-gradient(rgba(0,255,242,0.07) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,255,242,0.07) 1px, transparent 1px),
      radial-gradient(circle at 20% 10%, rgba(255,0,230,0.18), transparent 45%),
      radial-gradient(circle at 85% 85%, rgba(0,255,242,0.16), transparent 45%);
    background-size: 42px 42px, 42px 42px, cover, cover;
    padding: 24px 16px 60px;
    animation: gridshift 18s linear infinite;
  }
  @keyframes gridshift {
    0% { background-position: 0 0, 0 0, 0 0, 0 0; }
    100% { background-position: 400px 400px, 400px 400px, 0 0, 0 0; }
  }
  h1 {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    text-align: center;
    letter-spacing: 2px;
    font-size: clamp(1.3em, 4vw, 2em);
    color: #fff;
    text-shadow: 0 0 8px var(--neon-cyan), 0 0 22px var(--neon-cyan), 0 0 2px #fff;
    margin: 0 0 6px;
  }
  .subtitle {
    text-align: center;
    color: #8fdfe8;
    font-size: 0.85em;
    margin-bottom: 26px;
    letter-spacing: 1px;
  }
  .summary {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    justify-content: center;
    margin-bottom: 30px;
  }
  .summary .stat {
    background: var(--panel-bg);
    border: 1px solid var(--border-glow);
    border-radius: 12px;
    padding: 12px 20px;
    min-width: 140px;
    text-align: center;
    box-shadow: 0 0 14px rgba(0,255,242,0.12) inset, 0 0 10px rgba(0,255,242,0.15);
    backdrop-filter: blur(6px);
  }
  .summary .stat .label { font-size: 0.68em; color: #7fbfca; text-transform: uppercase; letter-spacing: 1px;}
  .summary .stat .value { font-family:'Orbitron',sans-serif; font-size: 1.25em; font-weight: 700; margin-top: 4px; }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 20px;
    max-width: 1400px;
    margin: 0 auto;
  }
  .card {
    background: var(--panel-bg);
    border: 1px solid var(--border-glow);
    border-radius: 16px;
    padding: 18px 20px 10px;
    box-shadow: 0 0 24px rgba(0,255,242,0.08), 0 0 2px rgba(0,255,242,0.5) inset;
    backdrop-filter: blur(6px);
    transition: box-shadow 0.3s ease;
  }
  .card:hover { box-shadow: 0 0 30px rgba(255,0,230,0.25), 0 0 2px rgba(0,255,242,0.6) inset; }
  .card-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 8px; }
  .card-head .sym { font-family:'Orbitron',sans-serif; font-size:1.15em; font-weight:700; color:#fff; text-shadow: 0 0 6px var(--neon-magenta); }
  .card-head .price { font-size: 1.05em; color: var(--neon-cyan); text-shadow: 0 0 6px var(--neon-cyan); }
  .mini-stats { display:flex; flex-wrap:wrap; gap:8px; margin: 8px 0 4px; font-size: 0.78em; }
  .mini-stats span { background: rgba(255,255,255,0.04); border: 1px solid rgba(0,255,242,0.15); border-radius: 6px; padding: 3px 8px; }
  .pos-val { color: var(--neon-green); }
  .neg-val { color: var(--neon-red); }
  canvas.chart { width: 100% !important; height: 130px !important; margin-top: 6px; }
  details { margin-top: 10px; }
  summary { cursor: pointer; color: #7fbfca; font-size: 0.8em; outline: none; }
  table { width:100%; border-collapse: collapse; margin-top: 8px; font-size:0.76em; }
  th, td { padding:5px 6px; text-align:left; border-bottom:1px solid rgba(0,255,242,0.12); white-space: nowrap; }
  th { color:#7fbfca; font-weight:normal; }
  .buy { color: var(--neon-green); } .sell { color: var(--neon-red); }
  .updated { color:#4c6b73; font-size:0.72em; margin-top:10px; text-align:right; }
  .footer-note { text-align:center; color:#3c525a; font-size:0.72em; margin-top: 34px; }
</style>
</head>
<body>
  <h1>⚡ BOT DE TRADING — MULTI-PAR ⚡</h1>
  <div class="subtitle">PAPER TRADING · SMA CROSSOVER · DATOS EN VIVO DE BINANCE</div>

  <div class="summary" id="summary"></div>
  <div class="grid" id="grid"></div>
  <div class="footer-note">Panel se actualiza solo cada 5s vía API, sin recargar la página · No es asesoría financiera</div>

<script>
const charts = {};

function fmt(n, d=2) {
  if (n === undefined || n === null || isNaN(n)) return "0.00";
  return Number(n).toFixed(d);
}

function ensureCard(sym) {
  if (document.getElementById('card-' + sym)) return;
  const grid = document.getElementById('grid');
  const card = document.createElement('div');
  card.className = 'card';
  card.id = 'card-' + sym;
  card.innerHTML = `
    <div class="card-head">
      <span class="sym">${sym}</span>
      <span class="price" id="price-${sym}">0.00</span>
    </div>
    <div class="mini-stats">
      <span id="pos-${sym}">Posición: —</span>
      <span id="equity-${sym}">Equity: —</span>
      <span id="pnl-${sym}">PnL: —</span>
      <span id="wr-${sym}">Win rate: —</span>
    </div>
    <canvas class="chart" id="chart-${sym}"></canvas>
    <details>
      <summary>Historial de operaciones</summary>
      <table>
        <thead><tr><th>Hora</th><th>Acción</th><th>Precio</th><th>Cant.</th><th>PnL</th></tr></thead>
        <tbody id="trades-${sym}"></tbody>
      </table>
    </details>
    <div class="updated" id="updated-${sym}"></div>
  `;
  grid.appendChild(card);

  const ctx = document.getElementById('chart-' + sym).getContext('2d');
  charts[sym] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: '#00fff2',
        backgroundColor: 'rgba(0,255,242,0.12)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.25,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: {
          ticks: { color: '#5c8891', font: { size: 9 } },
          grid: { color: 'rgba(0,255,242,0.06)' },
        }
      }
    }
  });
}

async function refresh() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    const symbols = Object.keys(data);

    let totalEquity = 0, totalPnl = 0, openPositions = 0, totalTrades = 0;

    symbols.forEach(sym => {
      ensureCard(sym);
      const d = data[sym];
      totalEquity += d.equity || 0;
      totalPnl += d.pnl_total || 0;
      if (d.in_position) openPositions++;
      totalTrades += d.total_trades || 0;

      document.getElementById('price-' + sym).textContent = fmt(d.current_price, 4);
      document.getElementById('pos-' + sym).textContent = 'Posición: ' + (d.in_position ? 'Abierta' : 'Cerrada');
      document.getElementById('equity-' + sym).textContent = 'Equity: ' + fmt(d.equity) + ' USDT';
      const pnlEl = document.getElementById('pnl-' + sym);
      pnlEl.textContent = 'PnL: ' + fmt(d.pnl_total) + ' (' + fmt(d.pnl_total_pct) + '%)';
      pnlEl.className = (d.pnl_total >= 0) ? 'pos-val' : 'neg-val';
      document.getElementById('wr-' + sym).textContent = 'Win rate: ' + fmt(d.win_rate_pct, 1) + '%';
      document.getElementById('updated-' + sym).textContent = 'Últ. actualización: ' + (d.last_update || '—');

      const hist = d.price_history || [];
      const chart = charts[sym];
      chart.data.labels = hist.map(p => p.t);
      chart.data.datasets[0].data = hist.map(p => p.p);
      const rising = hist.length > 1 && hist[hist.length-1].p >= hist[0].p;
      chart.data.datasets[0].borderColor = rising ? '#39ff88' : '#ff3860';
      chart.data.datasets[0].backgroundColor = rising ? 'rgba(57,255,136,0.12)' : 'rgba(255,56,96,0.12)';
      chart.update();

      const tbody = document.getElementById('trades-' + sym);
      tbody.innerHTML = (d.trade_history || []).slice(0, 15).map(t => `
        <tr>
          <td>${(t.timestamp || '').substring(11,19)}</td>
          <td class="${t.action.includes('COMPRA') ? 'buy' : 'sell'}">${t.action}</td>
          <td>${t.price}</td>
          <td>${t.qty}</td>
          <td>${t.pnl_usdt}</td>
        </tr>`).join('');
    });

    document.getElementById('summary').innerHTML = `
      <div class="stat"><div class="label">Equity total</div><div class="value">${fmt(totalEquity)} USDT</div></div>
      <div class="stat"><div class="label">PnL total</div><div class="value" style="color:${totalPnl>=0?'#39ff88':'#ff3860'}">${fmt(totalPnl)} USDT</div></div>
      <div class="stat"><div class="label">Posiciones abiertas</div><div class="value">${openPositions}/${symbols.length}</div></div>
      <div class="stat"><div class="label">Operaciones totales</div><div class="value">${totalTrades}</div></div>
    `;
  } catch (e) {
    console.error('Error actualizando panel', e);
  }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""

flask_app = Flask(__name__)
shared_state: SharedState = None  # se asigna en __main__


@flask_app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)


@flask_app.route("/api/state")
def api_state():
    return jsonify(shared_state.snapshot_all())


def run_web_dashboard(host: str, port: int):
    flask_app.run(host=host, port=port, debug=False, use_reloader=False)


# ------------------------------------------------------------------
if __name__ == "__main__":
    config = Config()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("trading_bot")

    shared_state = SharedState(config.symbols, config.price_history_points)
    telegram = TelegramNotifier(
        config.telegram_bot_token, config.telegram_chat_id, config.telegram_enabled, log
    )

    if config.web_enabled:
        web_thread = threading.Thread(
            target=run_web_dashboard, args=(config.web_host, config.web_port), daemon=True
        )
        web_thread.start()
        print(f"Panel web disponible en http://<ip-del-servidor>:{config.web_port}  "
              f"(o http://localhost:{config.web_port} si corres localmente)")

    telegram.send(
        "🤖 Bot multi-par iniciado (paper trading)\n"
        f"Pares: {', '.join(config.symbols)}\n"
        f"Balance inicial por par: {config.initial_balance_usdt:.2f} USDT"
    )

    bots = [SymbolBot(sym, config, telegram, shared_state, log) for sym in config.symbols]
    threads = [threading.Thread(target=bot.run, daemon=True) for bot in bots]
    for t in threads:
        t.start()
        time.sleep(2)  # escalona el arranque para no golpear la API de Binance de una vez

    # Mantiene vivo el proceso principal (los bots corren en hilos daemon)
    while True:
        time.sleep(3600)