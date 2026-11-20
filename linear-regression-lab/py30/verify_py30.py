"""Nghiệm thu bài tập Python → ML bằng nbclient, mỗi trường hợp một kernel mới.

Với mỗi bài kiểm tra ba loại trường hợp:
  starter  — khung code chưa làm: phải CHƯA ĐẠT và báo NotImplementedError;
  solution — lời giải tham khảo: phải ĐẠT mọi tiêu chí;
  wrong    — từng lời giải sai cố ý trong wrong_cases.py: phải CHƯA ĐẠT.

    python py30/verify_py30.py                 # mọi tuần đã có
    python py30/verify_py30.py --days 8-14     # chỉ một khoảng ngày
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from build_notebooks import lesson_files  # noqa: E402
from wrong_cases import WRONG  # noqa: E402


def run_case(lesson, exercise, mode, source, label):
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(lesson['setup']),
                                           nbformat.v4.new_code_cell(source),
                                           nbformat.v4.new_code_cell(exercise['test'])])
    NotebookClient(book, timeout=180, kernel_name='python3', allow_errors=True,
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    setup_errors = [o for o in book.cells[0].outputs if o.get('output_type') == 'error']
    if setup_errors:
        return False, f"{exercise['id']} · {label} · Ô chuẩn bị lỗi: {setup_errors[0].get('evalue')}"
    lines = ''.join(o.get('text', '') for o in book.cells[-1].outputs).splitlines()
    reports = [json.loads(line.removeprefix('__PY30__')) for line in lines if line.startswith('__PY30__')]
    if len(reports) != 1 or reports[0]['id'] != exercise['id'] or not reports[0]['checks']:
        return False, f"{exercise['id']} · {label} · không có báo cáo hợp lệ"
    checks = reports[0]['checks']
    passed = all(c['ok'] for c in checks)
    ok = passed == (mode == 'solution')
    if mode == 'starter':
        ok = ok and any('NotImplementedError' in c['detail'] for c in checks)
    detail = '' if ok else ' · ' + '; '.join(f"{c['name']}={'đạt' if c['ok'] else 'chưa'} ({c['detail']})" for c in checks)
    return ok, f"{exercise['id']} · {label} · {'OK' if ok else 'SAI KỲ VỌNG'}{detail}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--days', default='1-30')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.days.split('-')) if '-' in args.days else (int(args.days),) * 2
    tasks = []
    for path in lesson_files():
        for lesson in json.loads(path.read_text(encoding='utf-8')):
            if not lo <= lesson['id'] <= hi:
                continue
            nbformat.validate(nbformat.read(ROOT / f"Ngay_{lesson['id']:02}.ipynb", as_version=4))
            for ex in lesson['exercises']:
                tasks.append((lesson, ex, 'starter', ex['starter'], 'starter'))
                tasks.append((lesson, ex, 'solution', ex['solution'], 'solution'))
                wrongs = WRONG.get(ex['id'])
                if not wrongs:
                    raise SystemExit(f"Thiếu lời giải sai cho {ex['id']} trong wrong_cases.py")
                for label, code in wrongs:
                    tasks.append((lesson, ex, 'wrong', code, 'sai: ' + label))
    with tempfile.TemporaryDirectory(prefix='py30-kernels-') as tmp:
        spec = Path(tmp) / 'kernels/python3'
        spec.mkdir(parents=True)
        (spec / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python 3 (ML Studio test)', 'language': 'python'}))
        os.environ['JUPYTER_PATH'] = tmp + os.pathsep + os.environ.get('JUPYTER_PATH', '')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda t: run_case(*t), tasks))
    bad = 0
    for ok, line in results:
        print(line, flush=True)
        bad += not ok
    print(f'{len(results) - bad}/{len(results)} trường hợp đúng kỳ vọng.')
    if bad:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
