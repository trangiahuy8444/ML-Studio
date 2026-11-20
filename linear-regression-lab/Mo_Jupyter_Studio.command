#!/bin/zsh
set -e
studio_dir="${0:A:h}"
studio_env="$HOME/.venvs/ml-studio-jupyter"
studio_python=""
for candidate in /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 /usr/local/bin/python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert sys.version_info >= (3,11)' >/dev/null 2>&1; then
    studio_python="$candidate"
    break
  fi
done
if [[ -z "$studio_python" ]]; then
  print 'Cần Python 3.11 trở lên. Cài từ python.org rồi mở lại tệp này.'
  read '?Nhấn Enter để đóng.'
  exit 1
fi
if [[ ! -x "$studio_env/bin/python" ]]; then
  "$studio_python" -m venv "$studio_env"
fi
if ! "$studio_env/bin/python" -c 'import nbclassic,jupyter_server,ipykernel,numpy,pandas,matplotlib,sklearn,nbclient,nbformat' >/dev/null 2>&1; then
  print 'Đang cài Jupyter thật và thư viện học máy. Lần đầu cần Internet.'
  if ! "$studio_env/bin/python" -m pip install -r "$studio_dir/requirements-jupyter.txt"; then
    print 'Cài chưa thành công. Kiểm tra Internet rồi mở lại.'
    read '?Nhấn Enter để đóng.'
    exit 1
  fi
fi
"$studio_env/bin/python" "$studio_dir/start_jupyter_studio.py"
read '?Đã dừng. Nhấn Enter để đóng.'
