import os
from src.pipeline import SparkPipeline

CONFIG_PATH = os.path.join(os.getcwd(), "config.yaml")


def main():
    pipeline = SparkPipeline(CONFIG_PATH)
    pipeline.evaluate()


if __name__ == "__main__":
    main()
