"""Run two epochs per optimizer on a small real CIFAR-10 subset."""
from _common import config_from_cli
from src.tuning import run_final

if __name__ == "__main__":
    run_final(config_from_cli(), sanity=True)
