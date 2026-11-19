"""Module execution entrypoint (`python -m azzani_enhance`)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
