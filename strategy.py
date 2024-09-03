import numpy as np
import os
import pandas as pd
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import List

import config
from demeter import (
    Broker,
    MarketInfo,
    TokenInfo,
    Actuator,
    Strategy,
    RowData,
    PeriodTrigger,
    DemeterError,
    ActionTypeEnum,
)
from demeter.uniswap import UniLpMarket, UniV3Pool, PositionInfo
from lib_market_param import get_sigma_and_fee_of
from util import value_in_error, swap_all_to_my_market, get_best_range

pd.options.display.max_columns = None
pd.set_option("display.width", 5000)


expect_time = 2 / 52



@dataclass
class StrategyPosition:
    market: MarketInfo = None
    position: PositionInfo = None
    h_price: Decimal = Decimal(0)
    l_price: Decimal = Decimal(0)
    price_when_add: Decimal = Decimal(0)
    c: float = 0
    sigma: float = 0
    v: float = 0
    c_sigma_rate: float = 0


class LvrStrategy(Strategy):

    def initialize(self):
        # check and add liq on every hour
        self.triggers.append(PeriodTrigger(timedelta(hours=1), self.check_and_quit))
        self.triggers.append(PeriodTrigger(timedelta(hours=1), self.adjust_position))
        self.last_param: StrategyPosition | None = None
        self.c_sigma_list = []

    @staticmethod
    def _check_market_stable(sigma):
        return not (sigma >= 1 or sigma < 0.005)

    def add_liq(self, row_data: RowData):
        now = row_data.timestamp
        if now == start:  # first doesn't have previous day sigma
            return
        market_params: List[StrategyPosition] = []
        # find param for each market

        for market_key, market in self.broker.markets.items():
            c, sigma = get_sigma_and_fee_of(market.config_pool, now, timedelta(days=1))

            if not LvrStrategy._check_market_stable(sigma):
                self.log(row_data.timestamp, f"Market {market_key.name} is not stable")
                continue
            h, l, v = get_best_range(sigma, expect_time)
            market_params.append(
                StrategyPosition(
                    market=market_key,
                    position=None,
                    h_price=Decimal(h) * row_data.market_status.data[market_key]["price"],
                    l_price=Decimal(l) * row_data.market_status.data[market_key]["price"],
                    price_when_add=row_data.market_status.data[market_key]["price"],
                    c=c,
                    sigma=sigma,
                    v=v,
                    c_sigma_rate=c - sigma**2 / 4,
                )
            )
        if len(market_params) < 1:
            return
        best_position = max(market_params, key=lambda x: x.c_sigma_rate)

        if best_position.c_sigma_rate > 0:
            market: UniLpMarket = self.broker.markets[best_position.market]

            if (row_data.timestamp - timedelta(hours=1) > self.prices.index[0]) and not value_in_error(
                self.prices.loc[row_data.timestamp - timedelta(hours=1)][market.base_token.name],
                self.prices.loc[row_data.timestamp][market.base_token.name],
                0.07,
            ):
                self.log(row_data.timestamp, f"price of {market.base_token.name} has change too much in last hour")
                return

            upper_tick = market.price_to_tick(best_position.h_price)
            lower_tick = market.price_to_tick(best_position.l_price)
            if market.pool_info.is_token0_quote:
                lower_tick, upper_tick = upper_tick, lower_tick
            swap_all_to_my_market(self.broker, market)
            try:
                position, _, _, _ = market.add_liquidity_by_value(lower_tick, upper_tick)
                actuator.comment_last_action(
                    f"c:{round( best_position.c,3)},sigma:{round(best_position.sigma,3)},v:{round(best_position.v,3)}, c/sigma: {round(best_position.c_sigma_rate,3)}"
                )
                best_position.position = position
                self.last_param = best_position
            except DemeterError as e:
                self.log(row_data.timestamp, f"I got some error {e.message}")

    def check_and_quit(self, row_data: RowData):
        prices = row_data.prices
        if self.last_param is None:
            return
        # out of range
        market: UniLpMarket = self.broker.markets[self.last_param.market]
        price = prices[market.base_token.name] / prices[market.quote_token.name]

        if not (self.last_param.l_price <= price <= self.last_param.h_price):
            market.remove_liquidity(self.last_param.position)
            actuator.comment_last_action("remove because out of lower range", ActionTypeEnum.uni_lp_remove_liquidity)
            self.last_param = None


    def adjust_position(self, row_data: RowData):
        if self.last_param is None:
            self.add_liq(row_data)

    def on_bar(self, row_data: RowData):
        pass



if __name__ == "__main__":
    ENABLED_POOL = [0,2,3,]

    actuator = Actuator()
    broker: Broker = actuator.broker
    price_df = pd.DataFrame()
    token_dict = {}

    start = date(2024, 1, 1) 
    end = date(2024, 8, 16)

    quote_token_num = len(
        set(
            [
                config.POOLS[x].token0.actual if config.POOLS[x].is_0_quote else config.POOLS[x].token1.actual
                for x in ENABLED_POOL
            ]
        )
    )
    if quote_token_num != 1:
        raise RuntimeError("Only one quote token is allowed")

    for pool_index in ENABLED_POOL:
        pool = config.POOLS[pool_index]
        token0 = TokenInfo(pool.token0.name, pool.token0.decimal)
        token1 = TokenInfo(pool.token1.name, pool.token1.decimal)
        token_dict[pool.token0.name] = token0
        token_dict[pool.token1.name] = token1
        uni_pool = UniV3Pool(token0, token1, pool.fee * 100, token0 if pool.is_0_quote else token1)

        market_uni = UniLpMarket(
            MarketInfo(pool.name), uni_pool, data_path=str(os.path.join(config.APP.data_path, pool.name))
        )
        market_uni.load_data(config.CHAIN.name, pool.address, start, end)
        market_uni.config_pool = pool
        market_price, global_quote_token = market_uni.get_price_from_data()
        if token0.name not in price_df.columns:
            price_df[token0.name] = market_price[token0.name]
        if token1.name not in price_df.columns:
            price_df[token1.name] = market_price[token1.name]

        broker.add_market(market_uni)  # add market


    actuator.set_price(price_df)

    broker.set_balance(token_dict[config.usdc.name], 10000)
    broker.set_balance(token_dict[config.usdt.name], 10000)
    broker.set_balance(token_dict[config.weth.name], 30)

    actuator.strategy = LvrStrategy()
    actuator.print_action = False
    actuator.run(False)
    files = actuator.save_result(
        "/result", f"3_pools_{start.strftime('%y%m%d')}-{end.strftime('%y%m%d')}"
    )
