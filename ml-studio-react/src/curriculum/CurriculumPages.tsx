import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle,
  ChevronRight,
  Circle,
  Code2,
  Download,
  FlaskConical,
  Play,
  RotateCcw,
  Sparkles,
} from 'lucide-react';
import { MathBlock, MathText } from '../components/MathText';
import { PythonEditor } from '../components/PythonEditor';
import { download } from '../lib/domain';
import { PythonKernel } from '../lib/kernel';
import type { Output, Route } from '../types';
import { GradSimulation } from './GradSimulations';
import { UnitSymbolGlossary } from './MathGlossary';
import { GRAD_MODULES, allReadyUnits, getModule, getUnit } from './index';
import type { GradExercise, GradSection, GradUnit } from './types';
import './curriculum.css';

const GRAD_PROGRESS_KEY = 'mlStudioGrad-progress-v1';
const GRAD_CODE_KEY = 'mlStudioGrad-code-v1';
const GRAD_MODE_KEY = 'mlStudioGrad-mode-v1';
const GRAD_MARKER = '__GRAD__';

export type LearnerDepthMode = 'guided' | 'graduate';

export function loadLearnerMode(): LearnerDepthMode {
  try {
    const raw = localStorage.getItem(GRAD_MODE_KEY);
    if (raw === 'graduate' || raw === 'guided') return raw;
  } catch {
    /* Ignore unavailable storage */
  }
  return 'guided';
}

function saveLearnerMode(mode: LearnerDepthMode) {
  try {
    localStorage.setItem(GRAD_MODE_KEY, mode);
  } catch {
    /* Ignore unavailable storage */
  }
}

function LearnerModeBanner({
  mode,
  onChange,
}: {
  mode: LearnerDepthMode;
  onChange: (next: LearnerDepthMode) => void;
}) {
  return (
    <div className="learner-mode-bar" role="region" aria-label="Chế độ học cá nhân hóa">
      <div className="learner-mode-info">
        <span className="eyebrow">Lộ trình cá nhân hóa · Từ Python &amp; Trực giác → Cao học CS</span>
        <p>
          {mode === 'guided'
            ? 'Đang bật Chế độ 4 Tầng: Ví dụ đời thường → Tính tay số nhỏ → Code vòng lặp sang NumPy → Mở công thức Cao học khi sẵn sàng.'
            : 'Đang bật Chế độ Chuẩn Cao học: Hiển thị đầy đủ công thức toán học và dẫn xuất ngay lập tức cùng bảng giải mã ký hiệu.'}
        </p>
      </div>
      <div className="learner-mode-toggle" role="group" aria-label="Chọn độ sâu bài học">
        <button
          type="button"
          className={`mode-btn ${mode === 'guided' ? 'active' : ''}`}
          onClick={() => onChange('guided')}
        >
          4 Tầng: Dễ hiểu → Cao học
        </button>
        <button
          type="button"
          className={`mode-btn ${mode === 'graduate' ? 'active' : ''}`}
          onClick={() => onChange('graduate')}
        >
          Chuẩn Cao học (Hiện đủ)
        </button>
      </div>
    </div>
  );
}

function SectionFormalLayer({ sec }: { sec: GradSection }) {
  return (
    <>
      <MathText text={sec.body} />
      {sec.math && <MathBlock math={sec.math} />}
      {sec.symbols && sec.symbols.length > 0 && (
        <div className="inline-symbol-box">
          <span className="callout-title">Đọc từng ký hiệu trong công thức trên</span>
          <div className="inline-symbol-list">
            {sec.symbols.map((sym) => (
              <div className="inline-symbol-row" key={sym.symbol}>
                <span className="inline-symbol-math">
                  <MathText text={sym.symbol} />
                </span>
                <span>
                  <MathText text={`**${sym.readAs}**: ${sym.meaning}`} />
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {sec.intuition && (
        <div className="callout-intuition">
          <span className="callout-title">Trực giác &amp; Ý nghĩa thuật toán</span>
          <MathText text={sec.intuition} />
        </div>
      )}
      {sec.pitfall && (
        <div className="callout-pitfall">
          <span className="callout-title">Lưu ý kỹ thuật &amp; Điểm dễ nhầm</span>
          <MathText text={sec.pitfall} />
        </div>
      )}
    </>
  );
}

export interface UnitProgress {
  theoryDone: boolean;
  seminarDone: boolean;
  checks: Record<string, boolean>;
  note: string;
}

export interface GradProgressStore {
  version: 1;
  units: Record<string, UnitProgress>;
}

const blankUnitProgress = (): UnitProgress => ({
  theoryDone: false,
  seminarDone: false,
  checks: {},
  note: '',
});

export function loadGradProgress(): GradProgressStore {
  try {
    const raw = JSON.parse(localStorage.getItem(GRAD_PROGRESS_KEY) || 'null') as Partial<GradProgressStore> | null;
    if (raw && raw.version === 1 && raw.units && typeof raw.units === 'object') {
      return { version: 1, units: raw.units };
    }
  } catch {
    /* Ignore invalid local storage */
  }
  return { version: 1, units: {} };
}

function saveGradProgress(store: GradProgressStore) {
  try {
    localStorage.setItem(GRAD_PROGRESS_KEY, JSON.stringify(store));
    if (typeof window !== 'undefined') window.dispatchEvent(new Event('mlstudio-grad-progress'));
  } catch {
    /* Ignore unavailable storage */
  }
}

function loadUnitCodes(unit: GradUnit): Record<string, string> {
  const result = Object.fromEntries(unit.exercises.map((ex) => [ex.id, ex.starter]));
  try {
    const all = JSON.parse(localStorage.getItem(GRAD_CODE_KEY) || '{}') as Record<string, Record<string, string>>;
    const saved = all[unit.id];
    if (saved && typeof saved === 'object') {
      for (const ex of unit.exercises) {
        if (typeof saved[ex.id] === 'string') result[ex.id] = saved[ex.id];
      }
    }
  } catch {
    /* Ignore unavailable storage */
  }
  return result;
}

function saveUnitCodes(unitId: string, codes: Record<string, string>) {
  try {
    const all = JSON.parse(localStorage.getItem(GRAD_CODE_KEY) || '{}') as Record<string, Record<string, string>>;
    all[unitId] = codes;
    localStorage.setItem(GRAD_CODE_KEY, JSON.stringify(all));
  } catch {
    /* Ignore unavailable storage */
  }
}

export function isUnitComplete(unit: GradUnit, up?: UnitProgress): boolean {
  if (!up) return false;
  const exercisesPassed = unit.exercises.every((ex) => up.checks?.[ex.id] === true);
  return up.theoryDone && exercisesPassed;
}

interface GradCheck {
  name: string;
  ok: boolean;
  detail: string;
}

interface GradReport {
  id: string;
  checks: GradCheck[];
}

function extractGradReports(outputs: Output[]): { cleanOutputs: Output[]; reports: GradReport[] } {
  const reports: GradReport[] = [];
  const cleanOutputs: Output[] = [];
  for (const out of outputs) {
    const rawText = Array.isArray(out.text) ? out.text.join('') : out.text ?? '';
    if (out.output_type !== 'stream' || !rawText.includes(GRAD_MARKER)) {
      cleanOutputs.push(out);
      continue;
    }
    const keptLines: string[] = [];
    for (const line of rawText.split('\n')) {
      if (line.startsWith(GRAD_MARKER)) {
        try {
          const parsed = JSON.parse(line.slice(GRAD_MARKER.length)) as GradReport;
          if (parsed && typeof parsed.id === 'string' && Array.isArray(parsed.checks)) {
            reports.push({
              id: parsed.id,
              checks: parsed.checks.map((c) => ({
                name: String(c.name),
                ok: c.ok === true,
                detail: String(c.detail ?? ''),
              })),
            });
          }
        } catch {
          keptLines.push(line);
        }
      } else {
        keptLines.push(line);
      }
    }
    const joined = keptLines.join('\n');
    if (joined.trim()) cleanOutputs.push({ ...out, text: joined });
  }
  return { cleanOutputs, reports };
}

function exportUnitNotebook(unit: GradUnit, moduleTitle: string, codes: Record<string, string>) {
  const cells: Record<string, unknown>[] = [
    {
      cell_type: 'markdown',
      id: `${unit.id}-header`,
      metadata: {},
      source: [
        `# Học phần ${unit.moduleId} · Bài ${unit.moduleId}.${unit.order}: ${unit.title}\n`,
        `**${moduleTitle}**\n\n`,
        `${unit.summary}\n\n`,
        '## Tài liệu tham khảo\n',
        ...unit.references.map((r) => `- \`${r.source}\` (${r.section}): ${r.note}\n`),
      ],
    },
  ];
  for (const ex of unit.exercises) {
    cells.push(
      {
        cell_type: 'markdown',
        id: `${ex.id}-md`,
        metadata: {},
        source: [`## Bài tập: ${ex.title}\n\n`, `${ex.prompt}\n`],
      },
      {
        cell_type: 'code',
        id: `${ex.id}-setup`,
        metadata: { tags: ['setup'] },
        execution_count: null,
        outputs: [],
        source: ex.setup.split('\n').map((l, i, arr) => (i < arr.length - 1 ? l + '\n' : l)),
      },
      {
        cell_type: 'code',
        id: `${ex.id}-work`,
        metadata: {},
        execution_count: null,
        outputs: [],
        source: (codes[ex.id] ?? ex.starter).split('\n').map((l, i, arr) => (i < arr.length - 1 ? l + '\n' : l)),
      },
      {
        cell_type: 'code',
        id: `${ex.id}-test`,
        metadata: { tags: ['test'] },
        execution_count: null,
        outputs: [],
        source: ex.test.split('\n').map((l, i, arr) => (i < arr.length - 1 ? l + '\n' : l)),
      },
    );
  }
  const nb = {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: { name: 'python3', display_name: 'Python 3', language: 'python' },
      language_info: { name: 'python' },
    },
    cells,
  };
  download(`ML_CaoHoc_${unit.id}.ipynb`, JSON.stringify(nb, null, 2), 'application/x-ipynb+json');
}

type Nav = (r: Route, anchor?: string) => void;

export function CurriculumOverviewPage({ navigate }: { navigate: Nav }) {
  const [store] = useState(loadGradProgress);
  const [mode, setMode] = useState<LearnerDepthMode>(loadLearnerMode);
  const readyUnits = allReadyUnits();
  const readyCompleted = readyUnits.filter((u) => isUnitComplete(u, store.units[u.id])).length;

  function changeMode(next: LearnerDepthMode) {
    setMode(next);
    saveLearnerMode(next);
  }

  return (
    <>
      <header className="page-heading">
        <span className="eyebrow">Chương trình Cao học Khoa học Máy tính · 10 Học phần</span>
        <h1>Lộ trình Học máy &amp; Trí tuệ Nhân tạo Chuyên sâu</h1>
        <p>
          Thiết kế trực tiếp từ kho giáo trình chuẩn quốc tế (Murphy PML, Stanford CS229, UC Berkeley CS189, ISLR,
          Sutton &amp; Barto, Liu &amp; Zhou GNN) và tuyển tập bài báo NLP/CV cốt lõi. Kết hợp chặt chẽ chứng minh toán học,
          mô phỏng tương tác và thực hành cài đặt thuật toán từ đầu trên Python/NumPy.
        </p>
        <div className="heading-actions">
          <button className="btn-primary" onClick={() => navigate({ kind: 'unit', id: 'm1-u1' })}>
            Bắt đầu Học phần 1 · Bài 1.1 <ArrowRight size={15} />
          </button>
          <span className="module-status-pill ready">
            <CheckCircle size={13} /> Đã hoàn thành {readyCompleted}/{readyUnits.length} bài sẵn sàng
          </span>
        </div>
      </header>

      <LearnerModeBanner mode={mode} onChange={changeMode} />

      <div className="curriculum-grid">
        {GRAD_MODULES.map((mod) => {
          const doneInMod = mod.units.filter((u) => isUnitComplete(u, store.units[u.id])).length;
          return (
            <article className="module-card" key={mod.id} style={{ ['--phase' as string]: mod.color }}>
              <div className="module-badge">{mod.id}</div>
              <div>
                <div className="module-card-header">
                  <span className="eyebrow">Học phần {mod.id}</span>
                  <span className={`module-status-pill ${mod.status}`}>
                    {mod.status === 'ready'
                      ? `Sẵn sàng (${doneInMod}/${mod.units.length} bài hoàn thành)`
                      : 'Lộ trình Đợt tiếp theo'}
                  </span>
                </div>
                <h2>{mod.title}</h2>
                <p className="en-sub">{mod.englishTitle}</p>
                <p style={{ margin: '0 0 10px', fontSize: '14px', color: 'var(--text-2)', lineHeight: 1.65 }}>
                  {mod.overview}
                </p>
                <div className="module-sources">
                  {mod.sources.map((src) => (
                    <span className="chip" key={src}>
                      {src}
                    </span>
                  ))}
                </div>
              </div>
              <button className="btn-outline" onClick={() => navigate({ kind: 'module', id: mod.id })}>
                {mod.status === 'ready' ? 'Mở học phần' : 'Xem đề cương'} <ChevronRight size={15} />
              </button>
            </article>
          );
        })}
      </div>
    </>
  );
}

export function ModulePage({ id, navigate }: { id: number; navigate: Nav }) {
  const mod = getModule(id);
  const [store] = useState(loadGradProgress);

  if (!mod) {
    return (
      <section className="panel">
        <h2>Không tìm thấy Học phần {id}</h2>
        <button className="btn-outline" onClick={() => navigate({ kind: 'curriculum' })}>
          Quay về Lộ trình 10 Học phần
        </button>
      </section>
    );
  }

  return (
    <div style={{ ['--phase' as string]: mod.color }}>
      <header className="page-heading">
        <button className="back-link" onClick={() => navigate({ kind: 'curriculum' })}>
          <ArrowLeft size={14} /> Toàn bộ 10 Học phần
        </button>
        <span className="eyebrow">Học phần {mod.id} · {mod.englishTitle}</span>
        <h1>{mod.title}</h1>
        <p>{mod.overview}</p>
      </header>

      <section className="panel">
        <span className="eyebrow">Chuẩn đầu ra học phần</span>
        <h2>Kiến thức &amp; Năng lực đạt được</h2>
        <ul className="outcomes big" style={{ marginTop: 12 }}>
          {mod.outcomes.map((outcome) => (
            <li key={outcome}>
              <CheckCircle size={15} style={{ color: mod.color, flexShrink: 0 }} />
              <MathText text={outcome} />
            </li>
          ))}
        </ul>
        <div className="module-sources" style={{ marginTop: 16 }}>
          {mod.sources.map((src) => (
            <span className="chip" key={src}>
              {src}
            </span>
          ))}
        </div>
      </section>

      {mod.status === 'ready' ? (
        <section style={{ marginTop: 22 }}>
          <span className="eyebrow">Danh sách bài học chuyên sâu</span>
          <h2>{mod.units.length} Bài giảng &amp; Thực hành Tích hợp</h2>
          <div className="unit-list">
            {mod.units.map((unit) => {
              const up = store.units[unit.id];
              const done = isUnitComplete(unit, up);
              const passedCount = unit.exercises.filter((ex) => up?.checks?.[ex.id]).length;
              return (
                <article className="unit-card" key={unit.id}>
                  <div style={{ minWidth: 0 }}>
                    <div className="module-card-header">
                      <span className="eyebrow">
                        Bài {mod.id}.{unit.order}
                      </span>
                      <span className={`module-status-pill ${done ? 'ready' : ''}`}>
                        {done ? '✓ Hoàn thành' : `${passedCount}/${unit.exercises.length} bài code đạt`}
                      </span>
                    </div>
                    <h3>{unit.title}</h3>
                    <MathText text={unit.summary} />
                  </div>
                  <button className="btn-primary" onClick={() => navigate({ kind: 'unit', id: unit.id })}>
                    Học bài này <ArrowRight size={15} />
                  </button>
                </article>
              );
            })}
          </div>
        </section>
      ) : (
        <section className="panel" style={{ marginTop: 22 }}>
          <span className="eyebrow">Đề cương chi tiết · Triển khai theo lộ trình cuốn chiếu</span>
          <h2>Các bài chuyên sâu trong Học phần {mod.id}</h2>
          <p style={{ color: 'var(--text-2)' }}>
            Học phần này thuộc lộ trình Đợt triển khai tiếp theo. Toàn bộ nội dung sẽ bám sát tài liệu gốc liệt kê ở trên:
          </p>
          <div className="unit-list">
            {(mod.plannedTopics ?? []).map((topic) => (
              <div className="unit-card" key={topic}>
                <MathText text={`**${topic}**`} />
                <span className="module-status-pill">Đang biên soạn</span>
              </div>
            ))}
          </div>
          <div className="heading-actions" style={{ marginTop: 18 }}>
            <button className="btn-outline" onClick={() => navigate({ kind: 'lab' })}>
              Mở Phòng lab Hồi quy tương tác
            </button>
            <button className="btn-outline" onClick={() => navigate({ kind: 'gnn' })}>
              Mở Phòng lab GNN
            </button>
            <button className="btn-outline" onClick={() => navigate({ kind: 'py' })}>
              Luyện thêm Python → ML
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

interface ExerciseRunState {
  running: boolean;
  outputs: Output[];
  report?: GradReport;
}

export function UnitPage({
  unitId,
  navigate,
  onBusy,
}: {
  unitId: string;
  navigate: Nav;
  onBusy: (v: boolean) => void;
}) {
  const found = getUnit(unitId);
  const [store, setStore] = useState(loadGradProgress);
  const [mode, setMode] = useState<LearnerDepthMode>(loadLearnerMode);
  const [codes, setCodes] = useState<Record<string, string>>(() =>
    found ? loadUnitCodes(found.unit) : {},
  );

  function changeMode(next: LearnerDepthMode) {
    setMode(next);
    saveLearnerMode(next);
  }
  const [kernelStatus, setKernelStatus] = useState('Sẵn sàng (Pyodide chạy trong trình duyệt)');
  const [runs, setRuns] = useState<Record<string, ExerciseRunState>>({});
  const [workedCodes, setWorkedCodes] = useState<Record<string, string>>({});
  const [workedRuns, setWorkedRuns] = useState<Record<string, { running: boolean; outputs: Output[] }>>({});

  const liveRef = useRef(true);
  const kernel = useMemo(() => new PythonKernel(setKernelStatus), []);

  useEffect(() => {
    liveRef.current = true;
    return () => {
      liveRef.current = false;
      kernel.dispose();
      onBusy(false);
    };
  }, [kernel, onBusy]);

  if (!found) {
    return (
      <section className="panel">
        <h2>Chưa tìm thấy bài học {unitId}</h2>
        <button className="btn-outline" onClick={() => navigate({ kind: 'curriculum' })}>
          Quay về Lộ trình 10 Học phần
        </button>
      </section>
    );
  }

  const { module: mod, unit } = found;
  const up = store.units[unit.id] ?? blankUnitProgress();
  const readyUnits = allReadyUnits();
  const currentIdx = readyUnits.findIndex((u) => u.id === unit.id);
  const prevUnit = currentIdx > 0 ? readyUnits[currentIdx - 1] : null;
  const nextUnit = currentIdx >= 0 && currentIdx < readyUnits.length - 1 ? readyUnits[currentIdx + 1] : null;

  function patchUnitProgress(patch: Partial<UnitProgress>) {
    setStore((old) => {
      const next: GradProgressStore = {
        version: 1,
        units: {
          ...old.units,
          [unit.id]: { ...(old.units[unit.id] ?? blankUnitProgress()), ...patch },
        },
      };
      saveGradProgress(next);
      return next;
    });
  }

  function updateExerciseCode(exId: string, value: string) {
    setCodes((old) => {
      const next = { ...old, [exId]: value };
      saveUnitCodes(unit.id, next);
      return next;
    });
  }

  async function runWorkedCode(cellKey: string, defaultCode: string) {
    const code = workedCodes[cellKey] ?? defaultCode;
    onBusy(true);
    setWorkedRuns((old) => ({ ...old, [cellKey]: { running: true, outputs: [] } }));
    try {
      await kernel.start();
      const res = await kernel.run(code);
      if (!liveRef.current) return;
      setWorkedRuns((old) => ({ ...old, [cellKey]: { running: false, outputs: res.outputs } }));
    } catch (err) {
      if (liveRef.current) {
        setWorkedRuns((old) => ({
          ...old,
          [cellKey]: {
            running: false,
            outputs: [
              {
                output_type: 'error',
                ename: 'KernelError',
                evalue: err instanceof Error ? err.message : String(err),
              },
            ],
          },
        }));
      }
    } finally {
      if (liveRef.current) onBusy(false);
    }
  }

  async function runExercise(ex: GradExercise) {
    const userCode = codes[ex.id] ?? ex.starter;
    onBusy(true);
    setRuns((old) => ({ ...old, [ex.id]: { running: true, outputs: [] } }));
    try {
      await kernel.start();
      const setupRes = await kernel.run(ex.setup);
      if (!liveRef.current) return;
      if (!setupRes.ok) {
        setRuns((old) => ({ ...old, [ex.id]: { running: false, outputs: setupRes.outputs } }));
        return;
      }
      const userRes = await kernel.run(userCode);
      if (!liveRef.current) return;
      if (!userRes.ok) {
        setRuns((old) => ({ ...old, [ex.id]: { running: false, outputs: userRes.outputs } }));
        return;
      }
      const testRes = await kernel.run(ex.test);
      if (!liveRef.current) return;
      const merged = [...userRes.outputs, ...testRes.outputs];
      const { cleanOutputs, reports } = extractGradReports(merged);
      const matched = reports.find((r) => r.id === ex.id);
      setRuns((old) => ({
        ...old,
        [ex.id]: { running: false, outputs: cleanOutputs, report: matched },
      }));
      if (matched && matched.checks.length > 0 && matched.checks.every((c) => c.ok)) {
        setStore((old) => {
          const cur = old.units[unit.id] ?? blankUnitProgress();
          const next: GradProgressStore = {
            version: 1,
            units: {
              ...old.units,
              [unit.id]: { ...cur, checks: { ...cur.checks, [ex.id]: true } },
            },
          };
          saveGradProgress(next);
          return next;
        });
      }
    } catch (err) {
      if (liveRef.current) {
        setRuns((old) => ({
          ...old,
          [ex.id]: {
            running: false,
            outputs: [
              {
                output_type: 'error',
                ename: 'KernelError',
                evalue: err instanceof Error ? err.message : String(err),
              },
            ],
          },
        }));
      }
    } finally {
      if (liveRef.current) onBusy(false);
    }
  }

  return (
    <div style={{ ['--phase' as string]: mod.color }}>
      <header className="page-heading">
        <button className="back-link" onClick={() => navigate({ kind: 'module', id: mod.id })}>
          <ArrowLeft size={14} /> Học phần {mod.id}: {mod.title}
        </button>
        <span className="eyebrow">
          Học phần {mod.id} · Bài {mod.id}.{unit.order}
        </span>
        <h1>{unit.title}</h1>
        {unit.bigPicture && (
          <div className="big-picture-card">
            <span className="callout-title">Học bài này để giải quyết chuyện gì ngoài đời?</span>
            <MathText text={unit.bigPicture} />
            {unit.bigPictureTable && (
              <div className="df-head-box">
                <div className="df-head-caption">
                  <code>{unit.bigPictureTable.caption}</code>
                </div>
                <div className="df-head-scroll">
                  <table className="df-head-table">
                    <thead>
                      <tr>
                        {unit.bigPictureTable.columns.map((col, ci) => (
                          <th
                            key={col}
                            className={ci === unit.bigPictureTable?.highlightTargetCol ? 'df-target-col' : ''}
                          >
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {unit.bigPictureTable.rows.map((row, ri) => (
                        <tr key={ri}>
                          {row.map((cell, ci) => (
                            <td
                              key={ci}
                              className={
                                ci === 0
                                  ? 'df-index-col'
                                  : ci === unit.bigPictureTable?.highlightTargetCol
                                    ? 'df-target-col'
                                    : ''
                              }
                            >
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {unit.bigPictureTable.footnote && (
                  <div className="df-head-footnote">
                    <MathText text={unit.bigPictureTable.footnote} />
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        <MathText text={unit.summary} />
        <div className="heading-actions">
          <button
            className="btn-outline"
            onClick={() => document.getElementById('unit-theory')?.scrollIntoView({ behavior: 'smooth' })}
          >
            <BookOpen size={14} /> Lý thuyết toán học
          </button>
          <button
            className="btn-outline"
            onClick={() => document.getElementById('unit-sim')?.scrollIntoView({ behavior: 'smooth' })}
          >
            <FlaskConical size={14} /> Mô phỏng tương tác
          </button>
          <button
            className="btn-outline"
            onClick={() => document.getElementById('unit-practice')?.scrollIntoView({ behavior: 'smooth' })}
          >
            <Code2 size={14} /> Thực hành Python ({unit.exercises.length} bài)
          </button>
          <button className="btn-outline" onClick={() => exportUnitNotebook(unit, mod.title, codes)}>
            <Download size={14} /> Tải .ipynb bài học
          </button>
        </div>
      </header>

      <LearnerModeBanner mode={mode} onChange={changeMode} />

      <div className="unit-layout">
        <div className="unit-main">
          {/* Chuẩn đầu ra */}
          <section className="panel">
            <span className="eyebrow">Chuẩn đầu ra bài học</span>
            <h2>Mục tiêu nắm vững</h2>
            <ul className="outcomes big" style={{ marginTop: 10 }}>
              {unit.outcomes.map((o) => (
                <li key={o}>
                  <CheckCircle size={15} style={{ color: mod.color, flexShrink: 0 }} />
                  <MathText text={o} />
                </li>
              ))}
            </ul>
          </section>

          {/* Phần 1: Lý thuyết 4 Tầng + Từ điển ký hiệu */}
          <section className="panel" id="unit-theory">
            <span className="eyebrow">Phần 1 · Lý thuyết 4 Tầng (Trực giác → Tính tay → Code → Toán Cao học)</span>
            <h2>Cơ sở hình thức &amp; Trực giác</h2>
            <UnitSymbolGlossary unit={unit} defaultOpen={mode === 'guided'} />
            {unit.sections.map((sec, secIdx) => {
              const hasLayers = Boolean(sec.realWorld || sec.workedExample || sec.codeBridge);
              const workedKey = `${unit.id}::sec-${secIdx}`;
              const workedCodeVal = sec.workedExampleCode ? (workedCodes[workedKey] ?? sec.workedExampleCode) : '';
              const workedRunState = workedRuns[workedKey];
              return (
                <article className="unit-section-card" key={sec.heading}>
                  <h3>{sec.heading}</h3>

                  {sec.realWorld && (
                    <div className="layer-card layer-1">
                      <span className="layer-badge layer-badge-1">Tầng 1 · Ví dụ đời thường</span>
                      <MathText text={sec.realWorld} />
                    </div>
                  )}

                  {sec.workedExample && (
                    <div className="layer-card layer-2">
                      <span className="layer-badge layer-badge-2">Tầng 2 · Tính tay từng bước bằng số nhỏ</span>
                      {sec.workedExampleCode ? (
                        <div className="worked-split-grid">
                          <div className="worked-math-col">
                            <MathText text={sec.workedExample} />
                          </div>
                          <div className="worked-code-panel">
                            <div className="worked-code-head">
                              <span>Cell Python chạy thử số liệu này</span>
                              <div className="worked-code-actions">
                                <button
                                  type="button"
                                  className="btn-primary worked-run-btn"
                                  disabled={workedRunState?.running}
                                  onClick={() => void runWorkedCode(workedKey, sec.workedExampleCode!)}
                                >
                                  <Play size={12} /> {workedRunState?.running ? 'Đang chạy...' : 'Chạy ví dụ'}
                                </button>
                                <button
                                  type="button"
                                  className="btn-outline worked-reset-btn"
                                  disabled={workedRunState?.running}
                                  onClick={() =>
                                    setWorkedCodes((old) => ({ ...old, [workedKey]: sec.workedExampleCode! }))
                                  }
                                >
                                  <RotateCcw size={12} /> Đặt lại
                                </button>
                              </div>
                            </div>
                            <PythonEditor
                              ariaLabel={`Ví dụ Python ${sec.heading}`}
                              value={workedCodeVal}
                              disabled={workedRunState?.running}
                              rows={Math.min(14, Math.max(5, workedCodeVal.split('\n').length + 1))}
                              onChange={(val) => setWorkedCodes((old) => ({ ...old, [workedKey]: val }))}
                              onRun={() => void runWorkedCode(workedKey, sec.workedExampleCode!)}
                            />
                            {workedRunState && workedRunState.outputs.length > 0 && (
                              <div className="worked-code-out">
                                {workedRunState.outputs.map((o, oi) => (
                                  <pre
                                    key={oi}
                                    className={o.output_type === 'error' ? 'python-error' : 'python-output'}
                                  >
                                    {o.output_type === 'error'
                                      ? (o.traceback || [`${o.ename}: ${o.evalue}`]).join('\n')
                                      : Array.isArray(o.text)
                                        ? o.text.join('')
                                        : o.text || o.data?.['text/plain'] || ''}
                                  </pre>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      ) : (
                        <MathText text={sec.workedExample} />
                      )}
                    </div>
                  )}

                  {sec.codeBridge && (
                    <div className="layer-card layer-3">
                      <span className="layer-badge layer-badge-3">
                        Tầng 3 · {sec.codeBridge.title || 'Từ vòng lặp Python quen thuộc → NumPy'}
                      </span>
                      <div className="code-bridge-grid">
                        <div className="code-bridge-col">
                          <span className="code-bridge-label">1. Vòng lặp Python cơ bản</span>
                          <pre>
                            <code>{sec.codeBridge.pythonLoop}</code>
                          </pre>
                        </div>
                        <div className="code-bridge-col numpy">
                          <span className="code-bridge-label">2. Viết gọn bằng NumPy</span>
                          <pre>
                            <code>{sec.codeBridge.numpyWay}</code>
                          </pre>
                        </div>
                      </div>
                      <MathText text={sec.codeBridge.explanation} />
                    </div>
                  )}

                  {hasLayers && mode === 'guided' ? (
                    <details className="layer4-details" key={`${sec.heading}-${mode}`}>
                      <summary>
                        <span>
                          Tầng 4 · Mở Công thức Toán chuẩn Cao học &amp; Giải thích ký hiệu (khi đã hiểu 3 tầng trên)
                        </span>
                      </summary>
                      <div className="layer4-body">
                        <SectionFormalLayer sec={sec} />
                      </div>
                    </details>
                  ) : (
                    <div className={hasLayers ? 'layer4-open' : ''}>
                      {hasLayers && (
                        <span className="layer-badge layer-badge-4" style={{ marginTop: 12 }}>
                          Tầng 4 · Công thức Toán chuẩn Cao học
                        </span>
                      )}
                      <SectionFormalLayer sec={sec} />
                    </div>
                  )}
                </article>
              );
            })}
          </section>

          {/* Phần 2: Mô phỏng tương tác */}
          {unit.simulationId && (
            <section className="panel" id="unit-sim">
              <span className="eyebrow">Phần 2 · Phòng thí nghiệm Tương tác</span>
              <h2>Kiểm chứng trực quan</h2>
              <div style={{ marginTop: 14 }}>
                <GradSimulation id={unit.simulationId} />
              </div>
            </section>
          )}

          {/* Phần 3: Thực hành Python & Kiểm thử Tự động */}
          <section className="panel" id="unit-practice">
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', gap: 10 }}>
              <div>
                <span className="eyebrow">Phần 3 · Thực hành Cài đặt từ đầu (Pyodide WebAssembly)</span>
                <h2>Bài tập lập trình &amp; Chấm điểm tự động</h2>
              </div>
              <span className="module-status-pill">{kernelStatus}</span>
            </div>

            {unit.exercises.map((ex, idx) => {
              const code = codes[ex.id] ?? ex.starter;
              const runState = runs[ex.id];
              const passed = up.checks?.[ex.id] === true;
              return (
                <article className="grad-exercise-card" key={ex.id}>
                  <div className="grad-exercise-head">
                    <div>
                      <span className="eyebrow">Bài tập {idx + 1}</span>
                      <h3 style={{ margin: '2px 0 0' }}>{ex.title}</h3>
                    </div>
                    <span className={`module-status-pill ${passed ? 'ready' : ''}`}>
                      {passed ? '✓ Đã vượt tất cả kiểm thử' : 'Chưa đạt'}
                    </span>
                  </div>

                  <MathText text={ex.prompt} />

                  <PythonEditor
                    ariaLabel={`Mã Python cho ${ex.title}`}
                    value={code}
                    disabled={runState?.running}
                    rows={Math.min(16, Math.max(5, code.split('\n').length + 1))}
                    onChange={(val) => updateExerciseCode(ex.id, val)}
                    onRun={() => void runExercise(ex)}
                  />

                  <div className="grad-exercise-actions">
                    <button
                      className="btn-primary"
                      disabled={runState?.running}
                      onClick={() => void runExercise(ex)}
                    >
                      <Play size={14} /> {runState?.running ? 'Đang chạy...' : 'Chạy & Kiểm tra'}
                    </button>
                    <button
                      className="btn-outline"
                      disabled={runState?.running}
                      onClick={() => updateExerciseCode(ex.id, ex.starter)}
                    >
                      <RotateCcw size={14} /> Khôi phục khung mẫu
                    </button>
                    <span className="small-note">Shift+Enter trong ô code để chạy nhanh</span>
                  </div>

                  {runState?.report && (
                    <ul className="grad-check-list" role="status">
                      {runState.report.checks.map((ch) => (
                        <li key={ch.name} className={`grad-check-item ${ch.ok ? 'ok' : 'fail'}`}>
                          {ch.ok ? <CheckCircle size={15} /> : <Circle size={15} />}
                          <div>
                            <strong>{ch.name}</strong>
                            {ch.detail && <div style={{ opacity: 0.85, fontSize: 12 }}>{ch.detail}</div>}
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}

                  {runState && runState.outputs.length > 0 && (
                    <div className="cell-outputs">
                      {runState.outputs.map((o, i) => (
                        <div key={i} className={o.output_type === 'error' ? 'python-error' : 'python-output'}>
                          <pre>
                            {o.output_type === 'error'
                              ? (o.traceback || [`${o.ename}: ${o.evalue}`]).join('\n')
                              : Array.isArray(o.text)
                              ? o.text.join('')
                              : o.text || o.data?.['text/plain'] || ''}
                          </pre>
                        </div>
                      ))}
                    </div>
                  )}

                  <details className="hint">
                    <summary>Gợi ý triển khai</summary>
                    <MathText text={ex.hint} />
                  </details>
                  <details className="hint">
                    <summary>Lời giải chuẩn tham khảo</summary>
                    <pre>{ex.solution}</pre>
                  </details>
                </article>
              );
            })}
          </section>

          {/* Phần 4: Seminar */}
          <section className="panel" id="unit-seminar">
            <span className="eyebrow">Phần 4 · Seminar Cao học &amp; Phản biện</span>
            <h2>Câu hỏi chứng minh &amp; Thiết kế thí nghiệm</h2>
            {unit.seminar.map((item, idx) => (
              <article className="seminar-item" key={idx}>
                <MathText text={`**Câu hỏi ${idx + 1}.** ${item.question}`} />
                <details className="hint" style={{ marginTop: 10 }}>
                  <summary>
                    <Sparkles size={13} style={{ display: 'inline', marginRight: 6 }} />
                    Hướng dẫn lập luận &amp; Đáp án chuẩn
                  </summary>
                  <MathText text={item.guidance} />
                </details>
              </article>
            ))}
          </section>

          {/* Phần 5: Tài liệu tham khảo gốc */}
          <section className="panel" id="unit-refs">
            <span className="eyebrow">Phần 5 · Tài liệu gốc trong thư mục</span>
            <h2>Đọc sâu từ Giáo trình &amp; Bài báo</h2>
            <div className="ref-list">
              {unit.references.map((ref) => (
                <div className="ref-item" key={ref.source + ref.section}>
                  <strong>{ref.section}</strong> — {ref.note}
                  <code>{ref.source}</code>
                </div>
              ))}
            </div>
          </section>

          <div className="previous-next">
            {prevUnit ? (
              <button className="btn-outline" onClick={() => navigate({ kind: 'unit', id: prevUnit.id })}>
                <ArrowLeft size={15} /> Bài {prevUnit.moduleId}.{prevUnit.order}: {prevUnit.title}
              </button>
            ) : (
              <span />
            )}
            {nextUnit && (
              <button className="btn-primary" onClick={() => navigate({ kind: 'unit', id: nextUnit.id })}>
                Bài {nextUnit.moduleId}.{nextUnit.order}: {nextUnit.title} <ArrowRight size={15} />
              </button>
            )}
          </div>
        </div>

        {/* Sidebar tự theo dõi tiến độ & ghi chú */}
        <aside className="unit-aside">
          <div className="panel">
            <h3>Tiến độ bài {mod.id}.{unit.order}</h3>
            <label className="check-line">
              <input
                type="checkbox"
                checked={up.theoryDone}
                onChange={(e) => patchUnitProgress({ theoryDone: e.target.checked })}
              />
              Đã nắm lý thuyết &amp; dẫn xuất toán học
            </label>
            {unit.exercises.map((ex, i) => (
              <label className="check-line" key={ex.id}>
                <input
                  type="checkbox"
                  checked={up.checks?.[ex.id] === true}
                  onChange={(e) =>
                    patchUnitProgress({ checks: { ...up.checks, [ex.id]: e.target.checked } })
                  }
                />
                Bài tập code {i + 1} đạt kiểm thử
              </label>
            ))}
            <label className="check-line">
              <input
                type="checkbox"
                checked={up.seminarDone}
                onChange={(e) => patchUnitProgress({ seminarDone: e.target.checked })}
              />
              Đã giải câu hỏi Seminar
            </label>
            {isUnitComplete(unit, up) && (
              <span className="complete-label">
                <CheckCircle size={16} /> Đã hoàn thành bài học
              </span>
            )}
          </div>

          <div className="panel">
            <h3>Ghi chú nghiên cứu</h3>
            <p className="small-note">Tóm tắt chứng minh, giả định mô hình hoặc câu hỏi mở để tra cứu lại.</p>
            <textarea
              aria-label={`Ghi chú bài ${unit.id}`}
              rows={8}
              value={up.note}
              onChange={(e) => patchUnitProgress({ note: e.target.value })}
              placeholder="Ghi chú toán học, nhận xét từ mô phỏng hoặc thắc mắc nghiên cứu..."
            />
          </div>
        </aside>
      </div>
    </div>
  );
}
