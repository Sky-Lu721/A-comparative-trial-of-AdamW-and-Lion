"""Run every configured learning-rate candidate for both optimizers."""
from _common import config_from_cli
from src.tuning import run_search

if __name__ == "__main__":
    run_search(config_from_cli(), "lr")
