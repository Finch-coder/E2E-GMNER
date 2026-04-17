#!/usr/bin/env python3
"""Convenience wrapper so the project can be run without editable install."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent 
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e2egmner.cli.train import main

#debug用 在正式训练前请注释掉该代码
# import debugpy
# try:
#         # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#         pass
if __name__ == "__main__":
   

    main()
