#!/usr/bin/env python3
"""
SPX GEX Weekly Report — Main Entry Point

Usage:
  python main.py                          # Run report now
  python main.py --config config.yaml    # Specify config
  python main.py --no-email              # Skip email even if enabled in config
  python main.py --no-pdf                # HTML only
  python main.py --verbose               # Debug logging
"""
import argparse
import logging
import os
import sys

# Ensure the gex_weekly package root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from reports.builder import run_weekly_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPX GEX Weekly Report")
    parser.add_argument("--config",   default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-email", action="store_true",   help="Skip email delivery")
    parser.add_argument("--no-pdf",   action="store_true",   help="Generate HTML only (no PDF)")
    parser.add_argument("--verbose",  action="store_true",   help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    config_path = os.path.abspath(args.config)

    try:
        path = run_weekly_report(
            config_path=config_path,
            skip_email=args.no_email,
            html_only=args.no_pdf,
        )
        if path:
            print(f"\n✅ Report ready: {path}\n")
        else:
            print("\n⚠️  Report generation completed with warnings. Check logs.\n")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Report failed: {e}", exc_info=True)
        sys.exit(1)
