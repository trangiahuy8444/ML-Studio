import { PythonTrack } from './track-py/PythonTrack';
import { GraphLearning } from './components/GraphLearning';
import { SelfStudyGuide } from './components/SelfStudyGuide';
import { Assessments, CheckpointQuiz, ProjectRubric } from './components/Assessments';
import { blankAssessment, checkpoints, mergeAssessment } from './lib/assessment';
import type { AssessmentState } from './lib/assessment';
import { RegressionLab } from './components/RegressionLab';
import { Component, useCallback, useEffect, useRef, useState } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import {
  Map,
  BookOpen,
  ChartLine,
  Library,
  ArrowRight,
  ArrowLeft,
  Download,
  Upload,
  Menu,
  X,
  CheckCircle,
  Search,
  ChevronRight,
  Sun,
  SquareTerminal,
  FlaskConical,
  Network,
} from 'lucide-react';
import { Home, PHASE_COLORS, ProgressRing } from './components/Home';
import { ChaptersPage, ChapterPage, ProjectsHub, ProjectPage } from './projects/pages';
import type { Day, DayProgress, Progress, Route } from './types';
import { course } from './data';
import { blankDay, complete, download, parseRoute, routeHash, validateProgress } from './lib/domain';
import { HousingProject } from './components/HousingProject';
import { Notebook } from './components/Notebook';
import { Simulation } from './components/Simulation';
import {
  CurriculumOverviewPage,
  ModulePage,
  UnitPage,
  isUnitComplete,
  loadGradProgress,
} from './curriculum/CurriculumPages';
import { GRAD_MODULES, GRAD_RESOURCES, allReadyUnits } from './curriculum';

const PROGRESS_KEY = 'mlStudioAI30-progress-v1';
const normalize = (s: string) =>
  s.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replaceAll('đ', 'd');

function loadProgress(): Progress {
  try {
    return validateProgress(JSON.parse(localStorage.getItem(PROGRESS_KEY) || '{"version":1,"days":{}}'));
  } catch {
    return { version: 1, days: {} };
  }
}

class LessonBoundary extends Component<{ children: ReactNode }, { error: string }> {
  state = { error: '' };
  static getDerivedStateFromError(e: Error) {
    return { error: e.message };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ML Studio:', error, info.componentStack);
  }
  render() {
    return this.state.error ? (
      <section className="panel error-boundary" role="alert">
        <h2>Phần học này chưa tải được.</h2>
        <p>{this.state.error}</p>
        <p>Bạn có thể quay về Lộ trình 10 Học phần từ menu. Tiến độ đã lưu vẫn được giữ.</p>
        <button className="btn-outline" onClick={() => window.location.reload()}>
          Tải lại trang
        </button>
      </section>
    ) : (
      this.props.children
    );
  }
}

function DayPage({
  day,
  progress,
  onProgress,
  navigate,
  onBusy,
  assessment,
  onAssessment,
}: {
  assessment: AssessmentState;
  onAssessment: (v: AssessmentState) => void;
  day: Day;
  progress: DayProgress;
  onProgress: (p: DayProgress) => void;
  navigate: (r: Route, anchor?: string) => void;
  onBusy: (v: boolean) => void;
}) {
  const phase = course.phases.find((p) => p.id === day.phase)!;
  return (
    <>
      <header className="page-heading">
        <button className="back-link" onClick={() => navigate({ kind: 'phase', id: day.phase })}>
          <ArrowLeft size={14} /> {phase.title}
        </button>
        <span className="eyebrow">Ngày {day.id} · Lưu trữ tương thích</span>
        <h1>{day.title}</h1>
        <p>{day.bullets[0]?.text}</p>
        <div className="heading-actions">
          {[
            ['Mô phỏng', 'simulation'],
            ['Thực hành', 'practice'],
            ['Notebook', 'notebook'],
          ].map(([label, id]) => (
            <button key={id} className="btn-outline" onClick={() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })}>
              {label}
            </button>
          ))}
          <button className="btn-outline" onClick={() => navigate({ kind: 'guide', id: day.id }, 'guide-day')}>
            Cẩm nang bài {day.id}
          </button>
          {day.id === 30 && (
            <button className="btn-outline" onClick={() => navigate({ kind: 'gnn' })}>
              Học tiếp GNN
            </button>
          )}
          {day.id >= 28 && (
            <button className="btn-outline" onClick={() => navigate({ kind: 'assessments' }, 'project-rubric')}>
              Rubric dự án
            </button>
          )}
        </div>
      </header>
      <div className="day-layout">
        <div>
          <div id="simulation">
            <Simulation key={day.id} day={day.id} />
          </div>
          <section className="panel reading">
            <span className="eyebrow">Tài liệu lý thuyết</span>
            <h2>Câu hỏi và học liệu</h2>
            {day.bullets
              .filter((b) => b.label !== 'Mục tiêu')
              .map((b, i) => (
                <div key={i}>
                  <h3>{b.label}</h3>
                  <p>{b.text.replaceAll('**', '').replaceAll('`', '')}</p>
                </div>
              ))}
          </section>
          <section className="panel" id="practice">
            <div className="panel-heading">
              <span className="eyebrow">Ba mức thực hành</span>
              <h2>Dự đoán trước. Chạy để kiểm chứng.</h2>
            </div>
            {(['A', 'B', 'C'] as const).map((level) => (
              <article key={level} className="practice-task">
                <span className="task-letter">{level}</span>
                <div>
                  <h3>{level === 'A' ? 'Quan sát và tính tay' : level === 'B' ? 'Tự làm trong notebook' : 'Thử thách mở rộng'}</h3>
                  <p className="small-note">{level === 'A' ? 'Cốt lõi' : level === 'B' ? 'Cốt lõi' : 'Tùy chọn'}</p>
                  <p>{day.practice[level]}</p>
                  {level === 'B' && (
                    <>
                      <button className="btn-quiet" onClick={() => document.getElementById('notebook')?.scrollIntoView({ behavior: 'smooth' })}>
                        Làm trong notebook <ArrowRight size={14} />
                      </button>
                      <details className="hint">
                        <summary>Gợi ý</summary>
                        <p>{day.practice.hint}</p>
                      </details>
                      <details className="hint">
                        <summary>Đối chiếu lời giải</summary>
                        <pre>{day.practice.solution}</pre>
                      </details>
                    </>
                  )}
                </div>
              </article>
            ))}
          </section>
        </div>
        <aside className="day-aside">
          <div className="panel">
            <h3>Tiến độ tự đánh dấu</h3>
            {(
              [
                ['a', 'Đã làm và giải thích bài A'],
                ['b', 'Đã làm bài B, đọc kiểm tra'],
                ['review', 'Đã tự kiểm tra và viết nhật ký'],
                ['c', 'Đã thử bài C (tùy chọn)'],
              ] as const
            ).map(([key, label]) => (
              <label className="check-line" key={key}>
                <input type="checkbox" checked={progress[key]} onChange={(e) => onProgress({ ...progress, [key]: e.target.checked })} />
                {label}
              </label>
            ))}
            {complete(progress) && (
              <span className="complete-label">
                <CheckCircle size={16} /> Đã hoàn thành
              </span>
            )}
          </div>
          <div className="panel">
            <h3>Nhật ký nghiên cứu</h3>
            <textarea
              aria-label={`Nhật ký bài ${day.id}`}
              rows={8}
              value={progress.note}
              onChange={(e) => onProgress({ ...progress, note: e.target.value })}
              placeholder="Ghi chú toán học và thực nghiệm..."
            />
          </div>
        </aside>
      </div>
      {checkpoints.find((c) => c.day === day.id) && (
        <CheckpointQuiz
          checkpoint={checkpoints.find((c) => c.day === day.id)!}
          state={assessment}
          change={onAssessment}
          openDay={(id) => navigate({ kind: 'day', id })}
        />
      )}
      {day.id === 30 && <ProjectRubric state={assessment} change={onAssessment} openDay={(id) => navigate({ kind: 'day', id })} />}
      <Notebook key={'day-' + day.id} book={day.notebook} bookId={`AI_Ngay_${String(day.id).padStart(2, '0')}`} title={`Notebook · Ngày ${day.id}`} onBusy={onBusy} />
      <div className="previous-next">
        {day.id > 1 ? (
          <button className="btn-outline" onClick={() => navigate({ kind: 'day', id: day.id - 1 })}>
            <ArrowLeft size={15} /> Ngày {day.id - 1}
          </button>
        ) : (
          <span />
        )}
        {day.id < 30 && (
          <button className="btn-primary" onClick={() => navigate({ kind: 'day', id: day.id + 1 })}>
            Ngày {day.id + 1}
            <ArrowRight size={15} />
          </button>
        )}
      </div>
    </>
  );
}

function Exercises({ navigate }: { navigate: (r: Route, anchor?: string) => void }) {
  const [query, setQuery] = useState('');
  const [phase, setPhase] = useState('all');
  const [level, setLevel] = useState('all');
  const [limit, setLimit] = useState(24);
  const tasks = course.days
    .flatMap((d) => (['A', 'B', 'C'] as const).map((l) => ({ day: d, level: l, text: d.practice[l] })))
    .filter((t) => (phase === 'all' || t.day.phase === +phase) && (level === 'all' || level === t.level) && normalize(`${t.day.title} ${t.text}`).includes(normalize(query)));
  return (
    <>
      <header className="page-heading">
        <span className="eyebrow">Thư viện bài tập thực hành</span>
        <h1>Tìm bài tập theo chủ đề thuật toán</h1>
      </header>
      <div className="exercise-filters">
        <label className="search-box">
          <Search size={18} />
          <input
            type="search"
            aria-label="Tìm bài thực hành"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setLimit(24);
            }}
            placeholder="Thử: gradient, attention, dữ liệu…"
          />
        </label>
        <select
          aria-label="Lọc giai đoạn"
          value={phase}
          onChange={(e) => {
            setPhase(e.target.value);
            setLimit(24);
          }}
        >
          <option value="all">Mọi phần</option>
          {course.phases.map((p) => (
            <option key={p.id} value={p.id}>
              {p.id}. {p.title}
            </option>
          ))}
        </select>
        <select
          aria-label="Lọc mức thực hành"
          value={level}
          onChange={(e) => {
            setLevel(e.target.value);
            setLimit(24);
          }}
        >
          <option value="all">Mọi mức A/B/C</option>
          <option>A</option>
          <option>B</option>
          <option>C</option>
        </select>
      </div>
      <div className="exercise-grid">
        {tasks.slice(0, limit).map((t) => (
          <article className="exercise-card" key={t.day.id + t.level}>
            <span className="eyebrow">Bài {t.day.id}</span>
            <h3>
              {t.level} · {t.day.title}
            </h3>
            <p>{t.text}</p>
            <button className="btn-quiet" onClick={() => navigate({ kind: 'day', id: t.day.id }, t.level === 'B' ? 'notebook' : 'practice')}>
              Mở bài thực hành <ArrowRight size={14} />
            </button>
          </article>
        ))}
      </div>
      {tasks.length > limit && (
        <button className="btn-outline mt-5" onClick={() => setLimit((v) => v + 24)}>
          Xem thêm bài thực hành
        </button>
      )}
    </>
  );
}

function topbarLabel(route: Route): string {
  if (route.kind === 'home') return 'Trang chủ Cao học KHMT';
  if (route.kind === 'curriculum') return 'Lộ trình 10 Học phần';
  if (route.kind === 'module') return `Học phần ${route.id}`;
  if (route.kind === 'unit') return `Bài học ${route.id.toUpperCase()}`;
  if (route.kind === 'py') return 'Kho bài tập Python';
  if (route.kind === 'project') return 'Dự án giá nhà';
  if (route.kind === 'projects' || route.kind === 'proj') return 'Dự án thực nghiệm';
  if (route.kind === 'day') return `Ngày ${route.id}`;
  if (route.kind === 'phase') return `Chương ${route.id}`;
  if (route.kind === 'overview') return '6 chương (Bản trước)';
  if (route.kind === 'exercises') return 'Thực hành';
  if (route.kind === 'lab') return 'Phòng hồi quy';
  if (route.kind === 'gnn') return 'Mạng nơ-ron đồ thị';
  if (route.kind === 'assessments') return 'Kiểm tra và dự án';
  if (route.kind === 'guide') return 'Cẩm nang tự học';
  return 'Học liệu gốc';
}

export default function App() {
  const [route, setRoute] = useState<Route>(() => parseRoute(location.hash));
  const [progress, setProgress] = useState(loadProgress);
  const [kernelBusy, setKernelBusy] = useState(false);
  const [toast, setToast] = useState('');
  const [storageError, setStorageError] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [, setGradTick] = useState(0);

  useEffect(() => {
    const sync = () => setGradTick((t) => t + 1);
    window.addEventListener('mlstudio-grad-progress', sync);
    return () => window.removeEventListener('mlstudio-grad-progress', sync);
  }, []);

  const busyRef = useRef(kernelBusy);
  const currentRef = useRef(route);
  const anchor = useRef<string | null>(null);
  const importRef = useRef<HTMLInputElement>(null);
  busyRef.current = kernelBusy;
  currentRef.current = route;

  const notify = useCallback((text: string) => setToast(text), []);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(''), 6000);
    return () => clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    const handler = () => {
      const next = parseRoute(location.hash);
      if (busyRef.current && routeHash(next) !== routeHash(currentRef.current)) {
        history.replaceState(null, '', '#' + routeHash(currentRef.current));
        notify('Kernel Python đang chạy. Chờ hoàn tất hoặc bấm dừng trước khi đổi bài.');
        return;
      }
      setRoute(next);
      setMobileMenu(false);
    };
    window.addEventListener('hashchange', handler);
    return () => window.removeEventListener('hashchange', handler);
  }, [notify]);

  useEffect(() => {
    if (anchor.current) {
      const id = anchor.current;
      anchor.current = null;
      requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: 'start' }));
    } else {
      window.scrollTo({ top: 0, behavior: 'instant' });
    }
  }, [route]);

  useEffect(() => {
    try {
      localStorage.setItem(PROGRESS_KEY, JSON.stringify(progress));
      setStorageError(false);
    } catch {
      setStorageError(true);
    }
  }, [progress]);

  const navigate = useCallback(
    (next: Route, target?: string) => {
      if (busyRef.current) {
        notify('Kernel Python đang chạy. Chờ hoàn tất hoặc bấm dừng trước khi đổi bài.');
        return;
      }
      anchor.current = target || null;
      const hash = routeHash(next);
      if (location.hash.slice(1) === hash) {
        setRoute({ ...next });
        setMobileMenu(false);
      } else {
        location.hash = hash;
      }
    },
    [notify],
  );

  const patchDay = (id: number, value: DayProgress) =>
    setProgress((old) => ({ ...old, days: { ...old.days, [id]: value } }));

  const readyModulesCount = GRAD_MODULES.filter((m) => m.status === 'ready').length;
  const readyUnits = allReadyUnits();
  const gradStore = loadGradProgress();
  const gradDone = readyUnits.filter((u) => isUnitComplete(u, gradStore.units[u.id])).length;
  const phaseId = route.kind === 'phase' ? route.id : route.kind === 'day' ? course.days[route.id - 1].phase : 0;

  function exportProgress() {
    download('Tien_do_ML_CaoHoc.json', JSON.stringify({ grad: gradStore, legacy: progress }, null, 2));
    notify('Đã xuất tiến độ JSON.');
  }

  const navGroups = [
    [
      'Chương trình Cao học KHMT',
      [
        [Sun, 'home', 'Trang chủ'],
        [Map, 'curriculum', 'Lộ trình 10 Học phần'],
      ],
    ],
    [
      'Phòng lab tương tác',
      [
        [ChartLine, 'lab', 'Phòng hồi quy'],
        [Network, 'gnn', 'Mạng nơ-ron đồ thị (GNN)'],
        [FlaskConical, 'projects', 'Dự án thực nghiệm'],
      ],
    ],
    [
      'Tài liệu & Luyện tập',
      [
        [SquareTerminal, 'py', 'Kho bài tập Python'],
        [Library, 'resources', 'Học liệu gốc'],
      ],
    ],
  ] as const;

  const activeKind =
    route.kind === 'curriculum' || route.kind === 'module' || route.kind === 'unit' || route.kind === 'day' || route.kind === 'phase' || route.kind === 'overview'
      ? 'curriculum'
      : route.kind === 'proj' || route.kind === 'project'
      ? 'projects'
      : route.kind;

  const sidebar = (
    <>
      <button className="brand" onClick={() => navigate({ kind: 'home' })}>
        <span className="brand-mark">m/l</span>
        <span>
          ML Studio<small>Cao học Khoa học Máy tính</small>
        </span>
      </button>
      {navGroups.map(([group, items]) => (
        <nav className="nav-group" key={group} aria-label={group}>
          <span className="nav-label">{group}</span>
          {items.map(([Icon, kind, label]) => (
            <button
              key={kind}
              className={'nav-item ' + (activeKind === kind ? 'active' : '')}
              aria-current={activeKind === kind ? 'page' : undefined}
              onClick={() => navigate({ kind })}
            >
              <Icon size={17} />
              {label}
            </button>
          ))}
        </nav>
      ))}
      {phaseId > 0 && (
        <div className="nav-phase-strip" aria-label="Giai đoạn">
          {course.phases.map((p) => (
            <button
              key={p.id}
              title={p.title}
              className={phaseId === p.id ? 'active' : ''}
              style={{ ['--phase' as string]: PHASE_COLORS[p.id - 1] }}
              onClick={() => navigate({ kind: 'phase', id: p.id })}
            >
              {p.id}
            </button>
          ))}
        </div>
      )}
      <div className="sidebar-progress">
        <ProgressRing value={gradDone} total={readyUnits.length} size={54} stroke={6} />
        <div>
          <strong>
            {gradDone}/{readyUnits.length} bài sẵn sàng
          </strong>
          <p>
            {readyModulesCount}/{GRAD_MODULES.length} Học phần đã mở
          </p>
          <button onClick={exportProgress}>
            <Download size={13} /> Xuất tiến độ
          </button>
        </div>
      </div>
    </>
  );

  let page: ReactNode;
  if (route.kind === 'home') page = <Home progress={progress} navigate={navigate} />;
  else if (route.kind === 'curriculum') page = <CurriculumOverviewPage navigate={navigate} />;
  else if (route.kind === 'module') page = <ModulePage id={route.id} navigate={navigate} />;
  else if (route.kind === 'unit') page = <UnitPage key={route.id} unitId={route.id} navigate={navigate} onBusy={setKernelBusy} />;
  else if (route.kind === 'project') page = <HousingProject onBusy={setKernelBusy} />;
  else if (route.kind === 'day') {
    const day = course.days[route.id - 1];
    page = (
      <DayPage
        key={day.id}
        day={day}
        progress={progress.days[day.id] || blankDay()}
        onProgress={(p) => patchDay(day.id, p)}
        assessment={progress.assessment ?? blankAssessment()}
        onAssessment={(value) => setProgress((old) => ({ ...old, assessment: value }))}
        navigate={navigate}
        onBusy={setKernelBusy}
      />
    );
  } else if (route.kind === 'overview') page = <ChaptersPage progress={progress} navigate={navigate} />;
  else if (route.kind === 'phase') page = <ChapterPage key={route.id} id={route.id} progress={progress} navigate={navigate} />;
  else if (route.kind === 'projects') page = <ProjectsHub navigate={navigate} />;
  else if (route.kind === 'proj') page = <ProjectPage key={route.id} id={route.id} navigate={navigate} />;
  else if (route.kind === 'guide') page = <SelfStudyGuide key={route.id ?? 1} initialDay={route.id ?? 1} openDay={(id) => navigate({ kind: 'day', id })} />;
  else if (route.kind === 'assessments')
    page = <Assessments state={progress.assessment ?? blankAssessment()} change={(value) => setProgress((old) => ({ ...old, assessment: value }))} openDay={(id) => navigate({ kind: 'day', id })} />;
  else if (route.kind === 'exercises') page = <Exercises navigate={navigate} />;
  else if (route.kind === 'resources')
    page = (
      <>
        <header className="page-heading">
          <span className="eyebrow">Danh mục Giáo trình &amp; Bài báo Gốc</span>
          <h1>Kho Học liệu Cao học Khoa học Máy tính</h1>
          <p>Toàn bộ 10 Học phần được biên soạn trực tiếp từ các giáo trình kinh điển và bài báo nghiên cứu gốc trong thư mục lưu trữ cục bộ dưới đây.</p>
        </header>
        <div className="resource-list">
          {GRAD_RESOURCES.map((r) => (
            <article key={r.id}>
              <span>{r.id}</span>
              <div>
                <h2>{r.title}</h2>
                <p>
                  <strong>{r.authors}</strong> · <em>Áp dụng: {r.modules}</em>
                </p>
                <p>{r.note}</p>
                <code>{r.path}</code>
              </div>
            </article>
          ))}
        </div>
        <div className="heading-actions mt-6">
          <button className="btn-outline" onClick={exportProgress}>
            <Download size={14} /> Xuất tiến độ JSON
          </button>
          <button className="btn-outline" onClick={() => importRef.current?.click()}>
            <Upload size={14} /> Nhập tiến độ JSON
          </button>
          <button className="btn-primary" onClick={() => navigate({ kind: 'curriculum' })}>
            <BookOpen size={14} /> Mở Lộ trình 10 Học phần
          </button>
        </div>
      </>
    );
  else if (route.kind === 'py') page = <PythonTrack key={route.id ?? 0} day={route.id} openDay={(id) => navigate({ kind: 'py', id })} onBusy={setKernelBusy} />;
  else if (route.kind === 'gnn') page = <GraphLearning onBusy={setKernelBusy} openDay={(id) => navigate({ kind: 'day', id })} />;
  else page = <RegressionLab onBusy={setKernelBusy} />;

  return (
    <div className="app-shell">
      <aside className="sidebar">{sidebar}</aside>
      <header className="mobile-header">
        <button onClick={() => setMobileMenu((v) => !v)} aria-label={mobileMenu ? 'Đóng menu' : 'Mở menu'}>
          {mobileMenu ? <X size={21} /> : <Menu size={21} />}
        </button>
        <strong>ML Studio</strong>
        <span>
          {gradDone}/{readyUnits.length} bài
        </span>
      </header>
      {mobileMenu && <aside className="mobile-menu">{sidebar}</aside>}
      <main className="app-main">
        <div className="topbar">
          <span>
            ML Studio · Cao học KHMT <ChevronRight size={13} />
            <strong>{topbarLabel(route)}</strong>
          </span>
          <button onClick={exportProgress}>
            <Download size={14} /> Xuất tiến độ
          </button>
        </div>
        {storageError && (
          <p role="alert" className="storage-notice">
            Bộ nhớ trình duyệt bị giới hạn. Sử dụng nút Xuất tiến độ JSON để sao lưu.
          </p>
        )}
        <LessonBoundary key={routeHash(route)}>{page}</LessonBoundary>
        <footer className="app-footer">
          <span>ML Studio · Chương trình Cao học Khoa học Máy tính · Murphy PML · CS229 · CS189.</span>
          <span>Môi trường chạy Python/NumPy tích hợp trực tiếp qua Pyodide WebAssembly.</span>
        </footer>
      </main>
      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
      <input
        hidden
        ref={importRef}
        type="file"
        accept=".json"
        onChange={async (e) => {
          const f = e.target.files?.[0];
          if (!f) return;
          try {
            if (f.size > 1024 * 1024) throw Error('Tệp tiến độ tối đa 1 MB.');
            const parsed = JSON.parse(await f.text()) as { grad?: { version?: number; units?: unknown }; legacy?: unknown; version?: unknown };
            if (parsed.grad && parsed.grad.version === 1 && parsed.grad.units && typeof parsed.grad.units === 'object') {
              localStorage.setItem('mlStudioGrad-progress-v1', JSON.stringify(parsed.grad));
              window.dispatchEvent(new Event('mlstudio-grad-progress'));
            }
            const legacyData = parsed.legacy ?? (parsed.version === 1 ? parsed : null);
            if (legacyData) {
              const data = validateProgress(legacyData);
              setProgress((old) => ({
                ...old,
                days: { ...old.days, ...data.days },
                ...(data.assessment ? { assessment: mergeAssessment(old.assessment ?? blankAssessment(), data.assessment) } : {}),
              }));
            }
            notify('Đã nhập tiến độ.');
          } catch (error) {
            notify(error instanceof Error ? error.message : String(error));
          } finally {
            e.target.value = '';
          }
        }}
      />
    </div>
  );
}
