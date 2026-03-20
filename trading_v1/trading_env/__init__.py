from trading_env.trading_env import TradingEnv, Actions, Positions, transform
from trading_env.stocks_env import StocksEnv
from trading_env.bitcoin_env import BitcoinEnv
from trading_env.wrappers import SharpeRewardWrapper, DrawdownPenaltyWrapper

__all__ = [
    "TradingEnv", "StocksEnv", "BitcoinEnv", "Actions", "Positions", "transform",
    "SharpeRewardWrapper", "DrawdownPenaltyWrapper",
]

# Optional: custom networks require stable-baselines3
try:
    from trading_env.custom_networks import LargeMLPExtractor, AttentionExtractor
    __all__ += ["LargeMLPExtractor", "AttentionExtractor"]
except ImportError:
    pass
