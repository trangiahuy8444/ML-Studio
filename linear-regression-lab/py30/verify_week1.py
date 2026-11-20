"""Giữ để tương thích: nghiệm thu tuần 1 bằng verify_py30.py."""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).with_name('verify_py30.py')), '--days', '1-7', *sys.argv[1:]]))
