"""
Run the weekly GEX report on a schedule.

Usage:
  python scheduler.py              # Start scheduler (blocking)
  python scheduler.py --now        # Run once immediately, then exit
  python scheduler.py --config path/to/config.yaml
"""
import argparse
import logging
import sys
import os

# Ensure gex_weekly/ is on the path when run as a script
sys.path.insert(0, os.path.dirname(__file__))

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from reports.builder import run_weekly_report


def main():
    parser = argparse.ArgumentParser(description="SPX GEX Report Scheduler")
    parser.add_argument("--now", action="store_true", help="Run immediately and exit")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # Load config for schedule settings
    config_path = os.path.abspath(args.config)
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Cannot load config from {config_path}: {e}")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    if args.now:
        logging.info("Running report immediately (--now flag).")
        path = run_weekly_report(config_path)
        print(f"\n✅ Report ready: {path}\n")
        return

    sched_cfg = config.get("schedule", {})
    tz = sched_cfg.get("timezone", "Europe/London")
    day_of_week = sched_cfg.get("day_of_week", "mon-fri")
    hour = sched_cfg.get("hour", 7)
    minute = sched_cfg.get("minute", 0)

    scheduler = BlockingScheduler(timezone=tz)
    scheduler.add_job(
        run_weekly_report,
        CronTrigger(
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            timezone=tz,
        ),
        args=[config_path],
        id="daily_gex_report",
        name="SPX GEX Daily Report",
        misfire_grace_time=3600,   # If missed, run up to 1h late
        coalesce=True,             # Don't stack missed runs
    )

    jobs = scheduler.get_jobs()
    if jobs:
        logging.info(f"Scheduler started. Next run: {jobs[0].next_run_time}")
    logging.info(f"Schedule: {day_of_week} at {hour:02d}:{minute:02d} {tz}")
    logging.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
