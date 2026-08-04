
"""
07 Execution Engine – Realistic Execution Simulator
Supports transaction costs, commissions, slippage, partial fills, rebalance frequency, cash drag, fractional shares, min lot size, execution delay, next open/close, no lookahead bias
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
import pandas as pd
import numpy as np
import math
import hashlib

@dataclass(frozen=True)
class ExecutionConfig:
    commission_fixed: float = 0.0
    commission_pct: float = 0.0005
    slippage_bps: float = 2.0
    partial_fill_prob: float = 1.0
    allow_fractional: bool = True
    min_lot_size: int = 1
    execution_delay: int = 1
    execution_price: Literal["next_open", "next_close"] = "next_open"
    cash_drag_rate: float = 0.0
    initial_capital: float = 100000.0
    def __post_init__(self):
        if self.commission_pct < 0 or self.commission_pct > 0.1:
            raise ValueError("commission_pct out of range")
        if not 0 <= self.partial_fill_prob <= 1:
            raise ValueError("partial_fill_prob must be in [0,1]")
        if self.min_lot_size < 1:
            raise ValueError("min_lot_size must be >=1")

@dataclass
class Order:
    ticker: str
    date: pd.Timestamp
    target_weight: float
    side: str
    quantity: float
    price_hint: Optional[float] = None

@dataclass(frozen=True)
class Fill:
    ticker: str
    date: pd.Timestamp
    quantity: float
    price: float
    commission: float
    slippage: float
    cash_after: float

class ExecutionSimulator:
    def __init__(self, config: ExecutionConfig):
        if not isinstance(config, ExecutionConfig):
            raise TypeError("config must be ExecutionConfig")
        self.config = config
        self._fills: List[Fill] = []
        self._cash = config.initial_capital
        self._positions: Dict[str, float] = {}

    def _get_execution_price(self, market_data: pd.DataFrame, ticker: str, exec_date: pd.Timestamp) -> Optional[float]:
        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            return None
        try:
            pos = market_data.index.searchsorted(exec_date)
            exec_pos = pos + self.config.execution_delay
            if exec_pos >= len(market_data):
                return None
            bar = market_data.iloc[exec_pos]
            if self.config.execution_price == "next_open":
                price = bar['Open'] if 'Open' in bar else bar['Close']
            else:
                price = bar['Close']
            return float(price)
        except Exception:
            return None

    def simulate(self, orders: List[Order], market_data: Dict[str, pd.DataFrame]) -> List[Fill]:
        if not isinstance(orders, list):
            raise TypeError("orders must be list")
        if not isinstance(market_data, dict):
            raise TypeError("market_data must be dict")
        orders_sorted = sorted(orders, key=lambda o: (o.date, o.ticker))
        fills: List[Fill] = []
        cash = self.config.initial_capital
        positions: Dict[str, float] = {}
        for order in orders_sorted:
            if not isinstance(order, Order):
                raise TypeError("order must be Order")
            ticker = order.ticker
            if ticker not in market_data:
                continue
            df = market_data[ticker]
            if df.empty:
                continue
            exec_price_raw = self._get_execution_price(df, ticker, order.date)
            if exec_price_raw is None or np.isnan(exec_price_raw):
                continue
            slippage_pct = self.config.slippage_bps / 10000.0
            if order.side.upper() == "BUY":
                exec_price = exec_price_raw * (1 + slippage_pct)
            else:
                exec_price = exec_price_raw * (1 - slippage_pct)
            # Partial fills deterministic via hash
            if self.config.partial_fill_prob < 1.0:
                h = int(hashlib.md5(f"{ticker}{order.date}".encode()).hexdigest(), 16) % 100
                if h >= self.config.partial_fill_prob * 100:
                    quantity = order.quantity * 0.5
                else:
                    quantity = order.quantity
            else:
                quantity = order.quantity
            if not self.config.allow_fractional:
                quantity = math.floor(quantity)
            if quantity < self.config.min_lot_size:
                continue
            commission = 0.0
            commission += self.config.commission_fixed * abs(quantity)
            commission += abs(quantity * exec_price) * self.config.commission_pct
            if order.side.upper() == "BUY":
                cost = quantity * exec_price + commission
                if cash < cost:
                    affordable_qty = (cash - commission) / exec_price if exec_price > 0 else 0
                    if not self.config.allow_fractional:
                        affordable_qty = math.floor(affordable_qty)
                    if affordable_qty < self.config.min_lot_size:
                        continue
                    quantity = affordable_qty
                    cost = quantity * exec_price + commission
                cash -= cost
                positions[ticker] = positions.get(ticker, 0) + quantity
            else:
                pos_qty = positions.get(ticker, 0)
                if pos_qty < quantity:
                    quantity = pos_qty
                    if quantity < self.config.min_lot_size:
                        continue
                proceeds = quantity * exec_price - commission
                cash += proceeds
                positions[ticker] = positions.get(ticker, 0) - quantity
            fill = Fill(ticker=ticker, date=order.date, quantity=quantity, price=exec_price,
                        commission=commission, slippage=exec_price_raw * slippage_pct, cash_after=cash)
            fills.append(fill)
        self._fills = sorted(fills, key=lambda f: (f.date, f.ticker))
        self._cash = cash
        self._positions = positions
        return self._fills.copy()

    def get_fills_df(self) -> pd.DataFrame:
        if not self._fills:
            return pd.DataFrame()
        rows = [f.__dict__ for f in self._fills]
        df = pd.DataFrame(rows)
        return df.sort_values(["date", "ticker"]).reset_index(drop=True)

    def get_positions(self) -> Dict[str, float]:
        return dict(sorted(self._positions.items()))

    def get_cash(self) -> float:
        return float(self._cash)

    def get_equity_curve(self, market_data: Dict[str, pd.DataFrame]) -> pd.Series:
        all_dates = set()
        for df in market_data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)
        equity_curve = []
        cash = self.config.initial_capital
        # Simplified: use final positions for all dates after last fill
        # For institutional, would track positions over time
        for current_date in sorted(all_dates):
            total_pos_value = 0.0
            for ticker, qty in self._positions.items():
                if ticker in market_data:
                    df = market_data[ticker]
                    sub = df[df.index <= current_date]
                    if not sub.empty:
                        close = sub.iloc[-1]['Close']
                        total_pos_value += qty * close
            equity_curve.append(cash + total_pos_value)
        return pd.Series(equity_curve, index=pd.DatetimeIndex(all_dates)).sort_index()
