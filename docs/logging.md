# Logging & Troubleshooting

Pysparta uses [Loguru](https://github.com/delgan/loguru) to provide informative, color-coded console output. By default, logging is **disabled** to avoid cluttering your application's output, but it can be easily activated for debugging.

## Activating Logs

To see what Pysparta is doing (API calls, cache status, interpolation details), use the `set_logger` utility:

```python
from pysparta import set_logger

# Enable standard info logs
set_logger(enable=True, level="INFO")

# Enable detailed debug logs (includes function names and fine-grained steps)
set_logger(enable=True, level="DEBUG")
```

## Log Format Breakdown

Pysparta uses a **level-aware** formatting system. Depending on the severity, the output will look different:


| Level | Icon | Description |
| :--- | :--- | :--- |
| **DEBUG** | 🐞 | Technical details and internal state. Shows the function name. |
| **INFO** | ℹ️ | General progress (e.g., "Loading data from cache"). |
| **SUCCESS** | ✅ | Critical steps completed successfully. |
| **WARNING** | ⚠️ | Non-critical issues or automatic corrections (like fuzzy matching). |
| **ERROR** | ❌ | Critical failures that might stop the execution. |

## Common Issues

### Missing SoDA Credentials
If you see a `ValueError` regarding `soda_user_email`, ensure you have configured your email:
```python
import pysparta
pysparta.set_option("soda_user_email", "your@email.com")
```

### Capturing Python Warnings
By default, `set_logger(capture_warnings=True)` intercepts standard Python warnings (like those from `pandas` or `scipy`) and formats them as Pysparta logs. This helps keep your terminal clean and consistent.

## Multiprocessing
If you are running SPARTA in a parallel environment, you can include the process name in your logs to identify which worker is emitting each message:

```python
# In your logtools configuration
set_logger(enable=True, level="DEBUG", with_mp=True)
```

!!! tip "Pro-Tip"
If you are already using Loguru in your own project, you don't need to call `set_logger`. Just use `logger.enable("pysparta")` to let Pysparta logs flow into your existing configuration.

