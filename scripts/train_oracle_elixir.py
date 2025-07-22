"""
CLI entry point for training the Oracle's Elixir draft model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pbai.training.train_oracle_elixir import train_oracle_elixir

if __name__ == "__main__":
    train_oracle_elixir()
