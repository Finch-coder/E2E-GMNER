"""Optional SwanLab tracking helpers."""

from typing import Any, Dict, Optional

try:
    import swanlab  # pip install swanlab
except Exception:
    swanlab = None

def _swan_log(run, data: Dict[str, Any], *, step: Optional[int] = None):
    if run is None:
        return
    try:
        if hasattr(run, "log"):
            if step is None:
                run.log(data)
            else:
                run.log(data, step=step)
        else:
            if step is None:
                swanlab.log(data)
            else:
                swanlab.log(data, step=step)
    except Exception as e:
        print(f"[SwanLab] log failed (step={step}): {e}")


def _swan_finish(run):
    if run is None:
        return
    try:
        if hasattr(run, "finish"):
            run.finish()
        else:
            if hasattr(swanlab, "finish"):
                swanlab.finish()
    except Exception:
        pass
