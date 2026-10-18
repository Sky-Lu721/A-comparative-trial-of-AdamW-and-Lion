"""Tune weight decay with each optimizer's validation-selected LR."""
from _common import config_from_cli
from src.tuning import run_search

if __name__ == "__main__":
    run_search(config_from_cli(), "weight_decay")
