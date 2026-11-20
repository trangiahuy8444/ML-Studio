"""Tải dữ liệu được giao vào data/kaggle bằng Kaggle CLI."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

SETS = {
    'house': ('competitions', 'house-prices-advanced-regression-techniques'),
    'titanic': ('competitions', 'titanic'),
    'digits': ('competitions', 'digit-recognizer'),
    'fraud': ('datasets', 'mlg-ulb/creditcardfraud'),
    'imdb': ('datasets', 'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=SETS)
    options = parser.parse_args()
    credentials = Path.home() / '.kaggle/kaggle.json'
    if not credentials.is_file():
        parser.exit(2, 'Thiếu ~/.kaggle/kaggle.json. Tạo token trong Kaggle, lưu tại đó và chạy chmod 600 ~/.kaggle/kaggle.json.\n')
    if credentials.stat().st_mode & 0o077:
        parser.exit(2, 'Chạy chmod 600 ~/.kaggle/kaggle.json trước khi tải.\n')
    category, slug = SETS[options.dataset]
    dest = Path(__file__).resolve().parent / 'data/kaggle' / slug.rsplit('/', 1)[-1]
    if category == 'competitions':
        print(f'Hãy chấp nhận luật cuộc thi tại https://www.kaggle.com/competitions/{slug}/rules trước khi tải.', flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    bundled_cli = Path(sys.executable).parent / 'kaggle'
    cli = str(bundled_cli) if bundled_cli.is_file() else shutil.which('kaggle')
    if not cli:
        parser.exit(2, 'Chưa cài Kaggle CLI. Cài requirements-py30.txt bằng Python của Jupyter.\n')
    command = [cli, category, 'download',
               '-c' if category == 'competitions' else '-d', slug, '-p', str(dest)]
    environment = {key: value for key, value in os.environ.items() if not key.startswith('KAGGLE_')}
    environment['KAGGLE_CONFIG_DIR'] = str(credentials.parent)
    status = subprocess.run(command, env=environment, check=False).returncode
    if status:
        parser.exit(status, 'Tải chưa thành công. Kiểm tra gói kaggle, luật cuộc thi và kết nối.\n')
    for archive in dest.glob('*.zip'):
        with zipfile.ZipFile(archive) as bundle:
            if any(not (dest / member.filename).resolve().is_relative_to(dest) for member in bundle.infolist()):
                parser.exit(2, 'Tệp nén chứa đường dẫn vượt thư mục dữ liệu.\n')
            bundle.extractall(dest)
    print('Dữ liệu đã tải vào', dest)


if __name__ == '__main__':
    main()
