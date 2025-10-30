import os
import sys

# Put <repo root> on sys.path so "import src.*" works
_THIS = os.path.dirname(os.path.abspath(__file__))  # .../repo/tests
ROOT = os.path.abspath(os.path.join(_THIS, ".."))  # .../repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
