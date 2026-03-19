# Setup & Operating Guide

## 1. Install
```bash
pip install -r requirements.txt
```

### macOS (WeasyPrint system dependencies)
```bash
brew install pango cairo gdk-pixbuf
```

### Ubuntu/Debian
```bash
sudo apt-get install python3-weasyprint libpango-1.0-0 libcairo2
```

## 2. Configure

Edit `config.yaml`:
- Set your email SMTP credentials (use Gmail App Passwords)
- Adjust instruments, expiries, timezone
- Set `risk_free_rate` to current 3m T-bill rate

## 3. Test run
```bash
cd gex_weekly/
python main.py --no-email --verbose
```

## 4. Start the scheduler
```bash
python scheduler.py
```

## 5. Run immediately (any time)
```bash
python scheduler.py --now
# or
python main.py
```

## 6. HTML-only mode (no PDF, faster)
```bash
python main.py --no-pdf --no-email
```

## 7. Automate with cron (alternative to Python scheduler)
```bash
# Edit crontab: crontab -e
# Add this line (runs every Monday 7:30 AM ET):
30 7 * * 1 cd /path/to/gex_weekly && /usr/bin/python3 main.py >> logs/cron.log 2>&1
```

## 8. Run as a background service (Linux systemd)
```ini
# Create /etc/systemd/system/gex-scheduler.service:
[Unit]
Description=GEX Weekly Report Scheduler

[Service]
ExecStart=/usr/bin/python3 /path/to/gex_weekly/scheduler.py
WorkingDirectory=/path/to/gex_weekly
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
# Enable:
sudo systemctl enable gex-scheduler
sudo systemctl start gex-scheduler
```

## Archive Structure

```
archive/
└── 2026-03/
    ├── gex_weekly_2026-03-02/
    │   ├── gex_weekly_2026-03-02.pdf
    │   ├── gex_weekly_2026-03-02.html
    │   ├── gex_data.csv
    │   └── charts/
    ├── gex_weekly_2026-03-09/
    └── run.log
```

## Updating risk-free rate

Edit `config.yaml` → `options.risk_free_rate`.
Current rates: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/

## Adding a new instrument

Add to `config.yaml` under `instruments` list, set `primary: false`.
No code changes needed — the system picks it up automatically.

## Email setup (Gmail)

1. Enable 2FA on your Google account
2. Go to myaccount.google.com → Security → App Passwords
3. Generate password for "Mail"
4. Paste into `config.yaml` → `email.smtp_password`
5. Set `email.enabled: true`

## Validation checks

```bash
# 1. Import test
python -c "from reports.builder import run_weekly_report; print('✅ imports OK')"

# 2. Greeks unit test
python -c "
import numpy as np
from analytics.greeks import bs_gamma, bs_delta
g = bs_gamma(5000, np.array([5000.]), np.array([0.1]), 0.05, np.array([0.15]))
d = bs_delta(5000, np.array([4900.]), np.array([0.1]), 0.05, np.array([0.15]), np.array([-1]))
print(f'✅ Gamma ATM: {g[0]:.6f}  Put Delta: {d[0]:.4f}')
"

# 3. Data fetch test (requires internet)
python -c "
from data.fetcher import get_spot, get_expirations
s = get_spot('SPY')
e = get_expirations('SPY')
print(f'✅ SPY spot={s:.2f}, {len(e)} expiries available')
"

# 4. Full run (no email)
python main.py --no-email --verbose

# 5. Scheduler test (starts, prints next run time, Ctrl+C to stop)
python scheduler.py
```

## Troubleshooting

**WeasyPrint fails on PDF generation:**
- On macOS: `brew install pango cairo gdk-pixbuf`
- On Ubuntu: `sudo apt-get install libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0`
- Use `--no-pdf` flag to generate HTML only

**yfinance returns empty data:**
- SPX (`^SPX`) has no options — use SPY as primary for testing
- Some tickers require market hours or may be rate-limited
- Cached data is stored in `data/cache/` — delete stale `.parquet` files to force refresh

**kaleido not found (chart export fails):**
- `pip install kaleido`
- Charts will fall back to 1×1 blank PNG and the report still generates

**Email fails:**
- Use Gmail App Passwords (not your account password)
- Check `smtp_port: 587` and `starttls` support
- Set `email.enabled: false` during testing
