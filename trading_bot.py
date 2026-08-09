"""
Bot de Trading Automático - Binance - Paper Trading
=====================================================

Estrategia: Cruce de Medias Moviles (SMA Crossover)
  - Senal de COMPRA: la media corta cruza por ENCIMA de la media larga (cruce dorado)
  - Senal de VENTA:  la media corta cruza por DEBAJO de la media larga (cruce de la muerte)

Gestion de riesgo / margen de ganancias:
  - TAKE_PROFIT_PCT: cierra la posicion automaticamente si la ganancia alcanza este %
  - STOP_LOSS_PCT:   cierra la posicion automaticamente si la perdida alcanza este %

Extras de esta version:
  - Notificaciones a Telegram en cada compra/venta y resumen periodico.
  - Panel web (Flask) con estadisticas en vivo: precio, equity, PnL, % de
    operaciones ganadoras e historial de trades. Se abre en el navegador en
    http://<ip-del-servidor>:5000

IMPORTANTE:
  - Este bot corre en modo PAPER TRADING: no envia ordenes reales, solo simula
    con un balance virtual, usando datos de mercado reales de Binance (API
    publica, no requiere API key para leer precios).
  - No es asesoria financiera. El trading conlleva riesgo real de perdida de
    capital. Ningun bot garantiza ganancias.
  - Para que corra sin depender de tu telefono, este script debe ejecutarse
    en un servidor (por ejemplo un droplet de DigitalOcean), no solo en
    Termux -- en Termux funciona igual, pero se detiene si cierras la app.

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
from flask import Flask, render_template_string


# ------------------------------------------------------------------
# CONFIGURACION -- ajusta estos parametros a tu gusto
# ------------------------------------------------------------------
@dataclass
class Config:
    symbol: str = "BTC/USDT"        # Par a operar
    timeframe: str = "15m"          # Temporalidad de las velas: 1m, 5m, 15m, 1h, 4h, 1d...
    short_window: int = 9           # Periodo de la media movil corta
    long_window: int = 21           # Periodo de la media movil larga
    take_profit_pct: float = 0.02   # Margen de ganancia objetivo: 2%
    stop_loss_pct: float = 0.01     # Perdida maxima aceptada: 1%
    trade_fraction: float = 0.95    # Fraccion del balance a usar en cada compra
    fee_pct: float = 0.001          # Comision simulada de Binance (~0.1%)
    initial_balance_usdt: float = 1000.0  # Balance virtual inicial (paper trading)
    poll_seconds: int = 60          # Cada cuanto revisa el mercado (segundos)
    candles_lookback: int = 200     # Cuantas velas historicas descarga
    log_file: str = "trades_log.csv"

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
# NOTIFICADOR DE TELEGRAM
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
# ------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "symbol": "",
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
        }

    def update(self, **kwargs):
        with self.lock:
            self.data.update(kwargs)

    def snapshot(self):
        with self.lock:
            return dict(self.data)


shared_state = SharedState()


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
            "price": round(price, 2),
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
# BOT PRINCIPAL
# ------------------------------------------------------------------
class TradingBot:
    # Se usa data-api.binance.vision en vez de api.binance.com porque Binance
    # bloquea (HTTP 451) las peticiones desde IPs de EE.UU. Este es el espejo
    # publico de solo lectura de mercado, sin esa restriccion.
    BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

    def __init__(self, config: Config):
        self.cfg = config
        self.binance_symbol = config.symbol.replace("/", "")
        self.strategy = SmaCrossStrategy(config.short_window, config.long_window)
        self.account = PaperAccount(balance_usdt=config.initial_balance_usdt)
        self._setup_logging()
        self._setup_csv()
        self.telegram = TelegramNotifier(
            config.telegram_bot_token, config.telegram_chat_id, config.telegram_enabled, self.log
        )
        self._cycle_count = 0

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        self.log = logging.getLogger("trading_bot")

    def _setup_csv(self):
        if not os.path.exists(self.cfg.log_file):
            with open(self.cfg.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "action", "price", "qty", "fee",
                                  "pnl_usdt", "pnl_pct", "balance_after"])

    def _append_csv(self, entry: dict):
        with open(self.cfg.log_file, "a", newline="") as f:
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
            msg = f"TAKE PROFIT ejecutado a {current_price:.2f} ({change_pct*100:.2f}%)"
            self.log.info(f"[TP] {msg}")
            self.telegram.send(f"🎯 {msg}")

        elif change_pct <= -self.cfg.stop_loss_pct:
            entry = self.account.sell(current_price, self.cfg.fee_pct, reason="stop-loss")
            self._append_csv(entry)
            msg = f"STOP LOSS ejecutado a {current_price:.2f} ({change_pct*100:.2f}%)"
            self.log.info(f"[SL] {msg}")
            self.telegram.send(f"🛑 {msg}")

    def step(self):
        df = self.fetch_data()
        current_price = df["close"].iloc[-1]

        self.check_risk_management(current_price)

        signal = self.strategy.compute_signal(df)

        if signal == "buy" and not self.account.in_position:
            self.account.buy(current_price, self.cfg.trade_fraction, self.cfg.fee_pct)
            self._append_csv(self.account.trade_history[-1])
            msg = f"COMPRA a {current_price:.2f}"
            self.log.info(f"[BUY] {msg}")
            self.telegram.send(f"🟢 {msg}")

        elif signal == "sell" and self.account.in_position:
            entry = self.account.sell(current_price, self.cfg.fee_pct, reason="senal-cruce")
            self._append_csv(entry)
            msg = f"VENTA (senal) a {current_price:.2f} | PnL: {entry['pnl_usdt']} USDT ({entry['pnl_pct']}%)"
            self.log.info(f"[SELL] {msg}")
            self.telegram.send(f"🔴 {msg}")

        equity = self.account.equity(current_price)
        pnl_total = equity - self.cfg.initial_balance_usdt
        pnl_total_pct = (equity / self.cfg.initial_balance_usdt - 1) * 100
        stats = self.account.stats()

        shared_state.update(
            symbol=self.cfg.symbol,
            current_price=round(float(current_price), 2),
            in_position=self.account.in_position,
            entry_price=round(self.account.entry_price, 2),
            equity=round(equity, 2),
            balance_usdt=round(self.account.balance_usdt, 2),
            pnl_total=round(pnl_total, 2),
            pnl_total_pct=round(pnl_total_pct, 2),
            last_update=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            trade_history=list(reversed(self.account.trade_history[-50:])),
            **stats,
        )

        self.log.info(
            f"Precio: {current_price:.2f} | Posicion: {'SI' if self.account.in_position else 'NO'} "
            f"| Equity: {equity:.2f} USDT | PnL total: {pnl_total:+.2f} USDT ({pnl_total_pct:+.2f}%) "
            f"| Trades: {stats['total_trades']} | Win rate: {stats['win_rate_pct']:.1f}%"
        )

        self._cycle_count += 1
        if self.cfg.telegram_summary_every_n_cycles and \
           self._cycle_count % self.cfg.telegram_summary_every_n_cycles == 0:
            self.telegram.send(
                f"📊 Resumen {self.cfg.symbol}\n"
                f"Precio: {current_price:.2f}\n"
                f"Equity: {equity:.2f} USDT\n"
                f"PnL total: {pnl_total:+.2f} USDT ({pnl_total_pct:+.2f}%)\n"
                f"Trades: {stats['total_trades']} | Win rate: {stats['win_rate_pct']:.1f}%"
            )

    def run(self):
        self.log.info(
            f"Iniciando bot PAPER TRADING | {self.cfg.symbol} | {self.cfg.timeframe} | "
            f"SMA({self.cfg.short_window}/{self.cfg.long_window}) | "
            f"TP {self.cfg.take_profit_pct*100:.1f}% | SL {self.cfg.stop_loss_pct*100:.1f}% | "
            f"Balance inicial: {self.account.balance_usdt:.2f} USDT"
        )
        self.telegram.send(
            f"🤖 Bot iniciado (paper trading)\n{self.cfg.symbol} | {self.cfg.timeframe}\n"
            f"Balance inicial: {self.account.balance_usdt:.2f} USDT"
        )
        while True:
            try:
                self.step()
            except Exception as e:
                self.log.error(f"Error en el ciclo: {e}")
            time.sleep(self.cfg.poll_seconds)


# ------------------------------------------------------------------
# PANEL WEB (Flask)
# ------------------------------------------------------------------
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="15">
<title>Bot de Trading - {{ d.symbol }}</title>
<style>
  body { font-family: -apple-system, Arial, sans-serif; background:#0f1115; color:#e6e6e6; margin:0; padding:20px; }
  h1 { font-size: 1.3em; }
  .cards { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:20px; }
  .card { background:#1a1d24; border-radius:10px; padding:14px 18px; min-width:140px; flex:1; }
  .card .label { font-size:0.75em; color:#9aa0aa; text-transform:uppercase; }
  .card .value { font-size:1.4em; font-weight:bold; margin-top:4px; }
  .pos { color:#4caf50; } .neg { color:#f44336; }
  table { width:100%; border-collapse: collapse; margin-top: 10px; font-size:0.9em; }
  th, td { padding:8px; text-align:left; border-bottom:1px solid #2a2e37; }
  th { color:#9aa0aa; font-weight:normal; }
  .buy { color:#4caf50; } .sell { color:#f44336; }
  .updated { color:#666; font-size:0.8em; margin-top:20px; }
</style>
</head>
<body>
  <h1>🤖 {{ d.symbol }} — Paper Trading</h1>
  <div class="cards">
    <div class="card"><div class="label">Precio actual</div><div class="value">{{ "%.2f"|format(d.current_price) }}</div></div>
    <div class="card"><div class="label">Posicion</div><div class="value">{{ "Abierta" if d.in_position else "Cerrada" }}</div></div>
    <div class="card"><div class="label">Equity</div><div class="value">{{ "%.2f"|format(d.equity) }} USDT</div></div>
    <div class="card"><div class="label">PnL total</div><div class="value {{ 'pos' if d.pnl_total >= 0 else 'neg' }}">{{ "%.2f"|format(d.pnl_total) }} USDT ({{ "%.2f"|format(d.pnl_total_pct) }}%)</div></div>
    <div class="card"><div class="label">Operaciones</div><div class="value">{{ d.total_trades }}</div></div>
    <div class="card"><div class="label">% Ganadoras</div><div class="value">{{ "%.1f"|format(d.win_rate_pct) }}%</div></div>
    <div class="card"><div class="label">PnL prom./op.</div><div class="value">{{ "%.2f"|format(d.avg_pnl_per_trade) }} USDT</div></div>
  </div>

  <h2>Historial de operaciones</h2>
  <table>
    <tr><th>Fecha</th><th>Accion</th><th>Precio</th><th>Cantidad</th><th>PnL</th></tr>
    {% for t in d.trade_history %}
    <tr>
      <td>{{ t.timestamp[:19] }}</td>
      <td class="{{ 'buy' if 'COMPRA' in t.action else 'sell' }}">{{ t.action }}</td>
      <td>{{ t.price }}</td>
      <td>{{ t.qty }}</td>
      <td>{{ t.pnl_usdt }}</td>
    </tr>
    {% endfor %}
  </table>

  <div class="updated">Ultima actualizacion: {{ d.last_update }} — se refresca solo cada 15s</div>
</body>
</html>
"""

flask_app = Flask(__name__)


@flask_app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE, d=shared_state.snapshot())


def run_web_dashboard(host: str, port: int):
    flask_app.run(host=host, port=port, debug=False, use_reloader=False)


# ------------------------------------------------------------------
if __name__ == "__main__":
    config = Config()
    bot = TradingBot(config)

    if config.web_enabled:
        web_thread = threading.Thread(
            target=run_web_dashboard, args=(config.web_host, config.web_port), daemon=True
        )
        web_thread.start()
        print(f"Panel web disponible en http://<ip-del-servidor>:{config.web_port}  "
              f"(o http://localhost:{config.web_port} si corres localmente)")

    bot.run()
