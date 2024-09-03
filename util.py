
import numpy as np
from decimal import Decimal
from typing import List
from scipy.optimize import minimize
from typing import Dict, List

from demeter import MarketTypeEnum
from demeter.uniswap import UniLpMarket

def value_in_error(a, b, error=0.10):
    return np.abs((b - a) / a) <= error

def swap_all_to_my_market(broker, market: UniLpMarket):
    need_token = [market.base_token, market.quote_token]
    for t in broker.assets.keys():
        if t in need_token:
            continue
        if broker.assets[t].balance != Decimal(0):
            # find available market.
            for m in [x for x in broker.markets.values() if x.market_info.type == MarketTypeEnum.uniswap_v3]:
                tbd_market_tokens = [m.base_token, m.quote_token]
                if t not in tbd_market_tokens:
                    continue
                if market.base_token in tbd_market_tokens:
                    target_token = market.base_token
                elif market.quote_token in tbd_market_tokens:
                    target_token = market.quote_token
                else:
                    continue
                m.swap(broker.assets[t].balance, t, target_token)
        if broker.assets[t].balance > Decimal(0):
            raise RuntimeWarning(
                f"Can not swap token {t} to {market.market_info.name}, which means this token will not be used"
            )

def get_h(l, sigma, t) -> float:
    return np.power(np.e, -t * sigma**2 / np.log(l))


def VnoP(l, sigma, T):
    return 1 / (2 - np.sqrt(l) - 1 / (np.sqrt(np.power(np.e, -T * (sigma**2) / np.log(l)))))

     
def _maximize(narray, c, sigma):
    l = narray
    return -VnoP(l, c, sigma) 

def get_best_range(sigma, t, extra_cons: List[Dict] = []):
    initial_guess = np.array([0.99])
    cons = [
        {"type": "ineq", "fun": lambda x: x[0]},  # l > 0
        {"type": "ineq", "fun": lambda x: 0.99 - x[0]},  # l < 0.99
    ]

    if len(extra_cons) > 0:
        cons.extend(extra_cons)
    cons = tuple(cons)
    res = minimize(_maximize, initial_guess, args=(sigma, t), constraints=cons, method="SLSQP")

    l = res.x[0]

    h = get_h(l, sigma, t)

    liq = -res.fun
    return h, l, liq