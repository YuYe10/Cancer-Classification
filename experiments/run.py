import argparse
import yaml
from src.pipeline import run_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)

args = parser.parse_args()

config = yaml.safe_load(open(args.config))

acc, report = run_pipeline(config)

print("ACC:", acc)
print(report)