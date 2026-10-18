"""Run six final experiments and automatically generate statistics and plots."""
from _common import config_from_cli
from src.tuning import run_final

if __name__ == "__main__":
    run_final(config_from_cli())
