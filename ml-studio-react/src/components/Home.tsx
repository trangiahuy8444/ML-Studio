import { ArrowRight, BookOpen, ChartLine, CheckCircle, FlaskConical, Library, Map, Network, SquareTerminal, Target } from 'lucide-react';
import { useState } from 'react';
import { GRAD_MODULES, allReadyUnits } from '../curriculum';
import { isUnitComplete, loadGradProgress } from '../curriculum/CurriculumPages';
import type { Progress, Route } from '../types';

/** Màu sắc đại diện cho các học phần trên nền tối. */
export const PHASE_COLORS = ['#42d3e5', '#84e0a8', '#ffd173', '#f28db9', '#a8a7ff', '#ff9b6a'];

export function ProgressRing({
  value,
  total,
  size = 132,
  stroke = 10,
  label,
}: {
  value: number;
  total: number;
  size?: number;
  stroke?: number;
  label?: string;
}) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const frac = total ? value / total : 0;
  return (
    <svg className="progress-ring" width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`${value}/${total} ${label ?? ''}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--surface-3)" strokeWidth={stroke} />
      {frac > 0 && (
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${c * frac} ${c}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      )}
      <text x="50%" y="47%" textAnchor="middle" dominantBaseline="middle" className="ring-value" style={{ fontSize: size * 0.24 }}>
        {value}
      </text>
      {label && (
        <text x="50%" y="66%" textAnchor="middle" dominantBaseline="middle" className="ring-label" style={{ fontSize: Math.max(10, size * 0.085) }}>
          /{total} {label}
        </text>
      )}
    </svg>
  );
}

export function Home({ navigate }: { progress?: Progress; navigate: (r: Route, anchor?: string) => void }) {
  const [gradStore] = useState(loadGradProgress);
  const readyUnits = allReadyUnits();
  const doneUnits = readyUnits.filter((u) => isUnitComplete(u, gradStore.units[u.id]));
  const nextUnit = readyUnits.find((u) => !isUnitComplete(u, gradStore.units[u.id])) ?? readyUnits[0];
  const passedExercises = readyUnits.reduce(
    (sum, u) => sum + u.exercises.filter((ex) => gradStore.units[u.id]?.checks?.[ex.id]).length,
    0,
  );
  const totalExercises = readyUnits.reduce((sum, u) => sum + u.exercises.length, 0);
  const notesCount = Object.values(gradStore.units).filter((u) => u.note?.trim()).length;

  return (
    <div className="home">
      <section className="home-hero">
        <div className="home-hero-copy">
          <span className="eyebrow">Chương trình Cao học Khoa học Máy tính · 10 Học phần</span>
          <h1>
            <span className="hero-day" style={{ color: '#42d3e5' }}>
              Bài {nextUnit.moduleId}.{nextUnit.order}
            </span>
            {nextUnit.title}
          </h1>
          <p>
            Học toàn diện Học máy từ nền tảng toán giải tích ma trận, suy luận xác suất Bayes (Murphy PML, CS229, CS189)
            đến Deep Learning, NLP/Transformer, Graph Neural Networks và Reinforcement Learning (Sutton &amp; Barto).
          </p>
          <div className="heading-actions">
            <button className="btn-primary" onClick={() => navigate({ kind: 'unit', id: nextUnit.id })}>
              Vào bài học chuyên sâu <ArrowRight size={16} />
            </button>
            <button className="btn-outline" onClick={() => navigate({ kind: 'curriculum' })}>
              <Map size={15} /> Xem Lộ trình 10 Học phần
            </button>
          </div>
        </div>
        <div className="home-hero-ring">
          <ProgressRing value={doneUnits.length} total={readyUnits.length} label="bài Đợt 1" />
        </div>
      </section>

      <section className="home-stats" aria-label="Thống kê tiến độ cao học">
        <div>
          <Target size={18} />
          <strong>10</strong>
          <span>học phần cao học</span>
        </div>
        <div>
          <CheckCircle size={18} />
          <strong>
            {doneUnits.length}/{readyUnits.length}
          </strong>
          <span>bài Học phần 1 đạt chuẩn</span>
        </div>
        <div>
          <SquareTerminal size={18} />
          <strong>
            {passedExercises}/{totalExercises}
          </strong>
          <span>bài code NumPy vượt kiểm thử</span>
        </div>
        <div>
          <BookOpen size={18} />
          <strong>{notesCount}</strong>
          <span>ghi chú chứng minh</span>
        </div>
      </section>

      <section className="home-map">
        <div className="section-head">
          <div>
            <span className="eyebrow">Cấu trúc toàn khóa · 10 Học phần</span>
            <h2>Học phần 1 đã sẵn sàng; Học phần 2–10 triển khai theo lộ trình cuốn chiếu</h2>
          </div>
          <button className="btn-outline" onClick={() => navigate({ kind: 'curriculum' })}>
            Mở toàn bộ lộ trình <ArrowRight size={14} />
          </button>
        </div>

        <div className="curriculum-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
          {GRAD_MODULES.map((mod) => (
            <article
              key={mod.id}
              className="lab-card"
              style={{ borderColor: mod.status === 'ready' ? mod.color : 'var(--line)', padding: 18 }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                <span className="eyebrow" style={{ color: mod.color }}>
                  Học phần {mod.id}
                </span>
                <span className={`module-status-pill ${mod.status}`}>
                  {mod.status === 'ready' ? `Sẵn sàng (${mod.units.length} bài)` : 'Đợt tiếp theo'}
                </span>
              </div>
              <strong style={{ fontSize: '1.02rem' }}>{mod.title}</strong>
              <span style={{ fontSize: '12.5px', lineHeight: 1.55 }}>{mod.sources.slice(0, 2).join(' · ')}</span>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
                <button className="btn-quiet" onClick={() => navigate({ kind: 'module', id: mod.id })}>
                  {mod.status === 'ready' ? 'Mở học phần' : 'Xem đề cương'} <ArrowRight size={13} />
                </button>
                {mod.status === 'ready' &&
                  mod.units.map((u) => (
                    <button
                      key={u.id}
                      className="chip"
                      style={{ cursor: 'pointer' }}
                      onClick={() => navigate({ kind: 'unit', id: u.id })}
                    >
                      Bài 1.{u.order}
                    </button>
                  ))}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="home-labs">
        <div className="section-head">
          <div>
            <span className="eyebrow">Phòng thí nghiệm &amp; Thực hành tích hợp</span>
            <h2>Công cụ thực nghiệm trực tiếp trong trình duyệt</h2>
          </div>
        </div>
        <div className="lab-cards">
          <button className="lab-card" onClick={() => navigate({ kind: 'lab' })}>
            <ChartLine size={22} />
            <strong>Phòng lab Hồi quy nâng cao</strong>
            <span>Thử nghiệm nhiều đặc trưng, Iris, hoặc nạp CSV cục bộ trực tiếp lên Pyodide.</span>
          </button>
          <button className="lab-card" onClick={() => navigate({ kind: 'gnn' })}>
            <Network size={22} />
            <strong>Phòng lab Graph Neural Networks</strong>
            <span>Truyền thông điệp, GCN, GAT và phân loại nút trên đồ thị theo Liu &amp; Zhou.</span>
          </button>
          <button className="lab-card" onClick={() => navigate({ kind: 'projects' })}>
            <FlaskConical size={22} />
            <strong>9 Dự án thực nghiệm ML</strong>
            <span>Từ dữ liệu y tế, NLP lọc thư rác, chữ số viết tay đến bài toán Bandit/RL.</span>
          </button>
          <button className="lab-card" onClick={() => navigate({ kind: 'py' })}>
            <SquareTerminal size={22} />
            <strong>Kho bài tập cài đặt Python → ML</strong>
            <span>Hệ thống 30 bài thực hành lập trình từ NumPy vectorization tới Scikit-learn &amp; PyTorch.</span>
          </button>
          <button className="lab-card" onClick={() => navigate({ kind: 'resources' })}>
            <Library size={22} />
            <strong>Kho Học liệu Cao học gốc</strong>
            <span>Danh mục giáo trình (Murphy PML, CS229, CS189, ISLR, Sutton &amp; Barto) &amp; bài báo nền tảng.</span>
          </button>
        </div>
      </section>
    </div>
  );
}
