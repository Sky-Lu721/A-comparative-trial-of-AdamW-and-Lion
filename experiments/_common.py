"""Direct-script bootstrap and shared command-line parsing."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config


def config_from_cli() -> dict:
    """Accept an optional experiment configuration path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    return load_config(parser.parse_args().config)
