"""One command: generate data, train, evaluate, write outputs."""
from src.generate_data import generate
from src.model import run
from pathlib import Path
import json

Path("data").mkdir(exist_ok=True)
generate().to_csv("data/customers.csv", index=False)
m = run("data/customers.csv", "outputs")
print(json.dumps({k: v for k, v in m.items() if k != "model_comparison"}, indent=2))
print("\nSee outputs/SUMMARY.md")
