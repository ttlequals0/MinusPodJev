"""
Run database migrations.
"""

import subprocess
import sys


def main():
    """Run Alembic migrations."""
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"])


if __name__ == "__main__":
    main()
