"""CLI: python -m src.etmcp.evaluate -- Evaluate an ET-MCP case."""

import argparse

from src.engine import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate an ET-MCP case")
    parser.add_argument("--case-dir", required=True, help="Case directory path")
    args = parser.parse_args()

    results = evaluate(args.case_dir)
    print(f"Success rate: {results['metrics']['success_rate']:.2%}")
    print(f"Token consumption: {results['metrics']['token_consumption']:.1f}")
    print(f"Communication density: {results['metrics']['communication_density']:.3f}")


if __name__ == "__main__":
    main()
