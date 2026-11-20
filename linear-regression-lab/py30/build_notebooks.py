"""Sinh notebook mẫu py30/Ngay_XX.ipynb từ dữ liệu bài học của web (weekN.json).

Notebook mẫu chỉ chứa khung code (raise NotImplementedError) và ô kiểm tra, không chứa lời giải.
Bài làm của học viên nằm trong bai_lam/ và không bao giờ bị ghi đè.

    python py30/build_notebooks.py            # tìm ml-studio-react/src/track-py/week*.json
    python py30/build_notebooks.py --check    # chỉ báo notebook nào lệch với JSON
"""
import argparse
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent


def lesson_files():
    relative = Path('ml-studio-react/src/track-py')
    for base in ROOT.parents[:2]:
        folder = base / relative
        if folder.is_dir():
            return sorted(folder.glob('week*.json'))
    raise SystemExit('Không tìm thấy ml-studio-react/src/track-py/week*.json')


def build(lesson):
    nb = nbformat.v4.new_notebook()
    nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3 (ML Studio)', 'language': 'python'},
                   'language_info': {'name': 'python'}, 'ml_studio': {'track': 'py30', 'day': lesson['id']}}
    goals = '\n'.join(f'- {g}' for g in lesson['goals'])
    reads = '\n'.join(f'- {r}' for r in lesson['reads'])
    cells = [nbformat.v4.new_markdown_cell(f"# Ngày {lesson['id']} · {lesson['title']}\n\n{goals}\n\n**Đọc:**\n{reads}"),
             nbformat.v4.new_code_cell(lesson['setup'], metadata={'tags': ['setup']})]
    for ex in lesson['exercises']:
        cells.append(nbformat.v4.new_markdown_cell(f"## {ex['title']}\n\nGợi ý: {ex['hint']}"))
        cells.append(nbformat.v4.new_code_cell(ex['starter'], metadata={'tags': [ex['id']]}))
        cells.append(nbformat.v4.new_code_cell(ex['test'], metadata={'tags': [ex['id'] + '-check']}))
    cells.append(nbformat.v4.new_markdown_cell(f"## Dự án\n\n{lesson['project']}"))
    for i, cell in enumerate(cells):
        cell['id'] = f"py30-d{lesson['id']:02}-{i}"
    nb.cells = cells
    nbformat.validate(nb)
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = []
    for path in lesson_files():
        for lesson in json.loads(path.read_text(encoding='utf-8')):
            target = ROOT / f"Ngay_{lesson['id']:02}.ipynb"
            text = nbformat.writes(build(lesson)) + '\n'
            if target.is_file() and target.read_text(encoding='utf-8') == text:
                continue
            stale.append(target.name)
            if not args.check:
                target.write_text(text, encoding='utf-8')
    print(('Lệch: ' if args.check else 'Đã ghi: ') + (', '.join(stale) if stale else 'không có'))
    if args.check and stale:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
