"""Start ML Studio and a real, authenticated Jupyter NbClassic server."""
import argparse
import errno
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
BOOKS = {'Du_an_01_Du_doan_gia_nha': 'Du_an_01_Du_doan_gia_nha.ipynb', 'regression': 'ML_Studio_Hoi_quy_tuyen_tinh.ipynb'}
BOOKS['algorithm-observer'] = 'Quan_sat_thuat_toan.ipynb'
BOOKS['tree-pca'] = 'Cay_quyet_dinh_va_PCA.ipynb'
BOOKS['neural-network'] = 'Mang_no_ron_tung_buoc.ipynb'
BOOKS['cnn-attention'] = 'CNN_va_Attention_tung_buoc.ipynb'
BOOKS['gnn-foundations'] = 'GNN_Tung_Buoc.ipynb'
BOOKS['gnn-project'] = 'GNN_Du_An_Phan_Loai_Nut.ipynb'
BOOKS['assessment'] = 'Tu_kiem_tra_va_Rubric.ipynb'
BOOKS.update({f'AI_Ngay_{i:02}': f'Ngay_{i:02}.ipynb' for i in range(1, 31)})


def notebook_paths(root):
    return {key: filename if (root / filename).is_file() else 'course-notebooks/' + filename for key, filename in BOOKS.items()}


def py30_notebook(path):
    parts = path.split('/')
    if (len(parts) < 2 or parts[0] != 'py30' or not path.endswith('.ipynb')
            or any(part in ('', '.', '..') or any(ord(c) < 32 for c in part)
                   for part in parts) or '\\' in path):
        return None
    resolved = (ROOT / path).resolve()
    return path if resolved.is_relative_to(ROOT / 'py30') and resolved.is_file() else None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.web_directory), **kwargs)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path)
        if path.path.startswith('/jupyter/'):
            if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
                self.send_error(403)
                return
            alive = self.server.kernel_process.poll() is None
            if path.path == '/jupyter/status':
                payload = json.dumps({'ready': alive, 'engine': 'Jupyter NbClassic', 'books': list(BOOKS)}).encode()
                self.send_response(200 if alive else 503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if path.path == '/jupyter/session':
                expected = {f'http://127.0.0.1:{self.server.server_port}',
                            f'http://localhost:{self.server.server_port}'}
                if (self.headers.get('Sec-Fetch-Site', 'same-origin') != 'same-origin'
                        or self.headers.get('Origin', f'http://{self.headers["Host"]}') not in expected):
                    self.send_error(403)
                    return
                if not alive:
                    self.send_error(503, 'Jupyter stopped.')
                    return
                payload = json.dumps({'url': self.server.kernel_url, 'token': self.server.kernel_token,
                                      'classicPrefix': self.server.classic_prefix, 'root': 'py30'}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
                self.end_headers()
                self.wfile.write(payload)
                return
            if path.path == '/jupyter/open':
                query = urllib.parse.parse_qs(path.query)
                if 'path' in query:
                    book_path = py30_notebook(query['path'][0])
                else:
                    book_path = self.server.books.get(query.get('book', [''])[0])
                if not book_path:
                    self.send_error(404, 'Unknown notebook')
                    return
                if not alive:
                    self.send_error(503, 'Jupyter stopped. Restart Mo_Jupyter_Studio.command.')
                    return
                target = self.server.kernel_url + self.server.classic_prefix + '/notebooks/' + urllib.parse.quote(book_path, safe='/')
                self.send_response(302)
                self.send_header('Location', target + '?token=' + self.server.kernel_token)
                self.send_header('Referrer-Policy', 'no-referrer')
                self.end_headers()
                return
            self.send_error(404)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--web-port', type=int, default=8765)
    parser.add_argument('--jupyter-port', type=int, default=8888)
    parser.add_argument('--allow-root', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    missing = [n for n in ['nbclassic', 'jupyter_server', 'ipykernel'] if importlib.util.find_spec(n) is None]
    if missing:
        raise SystemExit('Thiếu ' + ', '.join(missing) + '. Mở Mo_Jupyter_Studio.command để cài môi trường.')
    optional = [name for name in ('torch', 'torchvision', 'kaggle')
                if importlib.util.find_spec(name) is None]
    if optional:
        print('Gói bổ sung Python → ML chưa có: ' + ', '.join(optional) +
              '. Cài requirements-py30.txt khi cần Kaggle hoặc PyTorch.', flush=True)
    directory = next((p for p in [ROOT.parent / 'ml-studio-react' / 'dist', ROOT / 'dist', ROOT / 'web'] if (p / 'index.html').is_file()), None)
    if directory is None:
        raise SystemExit('Cần giải nén toàn bộ ZIP, bao gồm thư mục web.')
    books = notebook_paths(ROOT)
    for filename in books.values():
        if not (ROOT / filename).is_file():
            raise SystemExit('Thiếu notebook: ' + filename)
    Handler.web_directory = directory
    for port in range(args.web_port, args.web_port + 25):
        try:
            web = ThreadingHTTPServer(('127.0.0.1', port), Handler)
            break
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise SystemExit('Không mở được cổng web: ' + str(exc) + '. Chạy Mo_Jupyter_Studio.command từ Terminal trên máy em.')
    else:
        raise SystemExit('Các cổng web đang bận. Đóng lần chạy cũ và thử lại.')
    runtime = Path(tempfile.mkdtemp(prefix='ml-studio-jupyter-'))
    spec = runtime / 'kernels/python3'
    spec.mkdir(parents=True)
    (spec / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'Python 3 (ML Studio)', 'language': 'python'}))
    token = secrets.token_urlsafe(32)
    env = {**os.environ, 'JUPYTER_TOKEN': token, 'JUPYTER_DATA_DIR': str(runtime / 'data'), 'JUPYTER_CONFIG_DIR': str(runtime / 'config'), 'IPYTHONDIR': str(runtime / 'ipython'), 'MPLCONFIGDIR': str(runtime / 'matplotlib'), 'XDG_CACHE_HOME': str(runtime / 'cache'), 'JUPYTER_RUNTIME_DIR': str(runtime), 'JUPYTER_PATH': str(runtime) + os.pathsep + os.environ.get('JUPYTER_PATH', '')}
    log = tempfile.TemporaryFile(mode='w+')
    origin = rf'^http://(127\.0\.0\.1|localhost):{web.server_port}$'
    process = None
    try:
        for port in range(args.jupyter_port, args.jupyter_port + 25):
            cmd = [sys.executable, '-m', 'nbclassic', '--no-browser', '--ServerApp.ip=127.0.0.1', f'--ServerApp.port={port}', '--ServerApp.port_retries=0', f'--ServerApp.root_dir={ROOT}', f'--ServerApp.allow_origin_pat={origin}']
            if args.allow_root:
                cmd.append('--allow-root')
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log)
            url = f'http://127.0.0.1:{port}'
            ready = False
            for _ in range(120):
                if process.poll() is not None:
                    break
                try:
                    req = urllib.request.Request(url + '/api/status', headers={'Authorization': 'token ' + token})
                    with urllib.request.urlopen(req, timeout=0.5) as response:
                        ready = response.status == 200
                    if ready:
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.25)
            if ready and process.poll() is None:
                break
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        else:
            raise RuntimeError('Jupyter chưa khởi động được. Kiểm tra môi trường Python và cổng 8888–8912.')
        # NbClassic serves /tree when used alone; installations with Notebook 7
        # can use /nbclassic/tree. Detect the route instead of assuming one.
        prefix = None
        for candidate in ['/nbclassic', '']:
            try:
                req = urllib.request.Request(url + candidate + '/tree', headers={'Authorization': 'token ' + token})
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        prefix = candidate
                        break
            except urllib.error.HTTPError:
                pass
        if prefix is None:
            raise RuntimeError('Jupyter đã chạy nhưng chưa tìm thấy giao diện Classic.')
        web.kernel_process, web.kernel_token, web.kernel_url = process, token, url
        web.classic_prefix, web.books = prefix, books
        homepage = f'http://127.0.0.1:{web.server_port}/'
        print(f'\nJupyter Notebook thật đã sẵn sàng.\nMở bài học: {homepage}\nBấm “Mở Jupyter Notebook” trong bài để thực hành.\nGiữ Terminal mở; Ctrl+C để dừng cả web và Jupyter.\n', flush=True)
        if not args.no_browser:
            webbrowser.open(homepage)
        web.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        web.server_close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        log.close()
        shutil.rmtree(runtime, ignore_errors=True)


if __name__ == '__main__':
    main()
