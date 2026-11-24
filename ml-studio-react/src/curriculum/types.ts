export interface GradReference {
  source: string;
  section: string;
  note: string;
}

export interface GradSymbolDef {
  symbol: string;
  readAs: string;
  meaning: string;
}

export interface GradCodeBridge {
  title?: string;
  pythonLoop: string;
  numpyWay: string;
  explanation: string;
}

export interface GradSection {
  heading: string;
  body: string;
  math?: string;
  intuition?: string;
  pitfall?: string;
  /** Tầng 1: Ví dụ đời thường & lý do thực tế */
  realWorld?: string;
  /** Tầng 2: Ví dụ tính tay từng bước bằng số nhỏ cụ thể */
  workedExample?: string;
  /** Code Python minh họa chạy ngay kế bên Tầng 2 */
  workedExampleCode?: string;
  /** Tầng 3: Cầu nối từ vòng lặp Python quen thuộc sang NumPy */
  codeBridge?: GradCodeBridge;
  /** Tầng 4: Giải nghĩa riêng từng ký hiệu xuất hiện trong mục này */
  symbols?: GradSymbolDef[];
}

export interface GradExercise {
  id: string;
  title: string;
  prompt: string;
  hint: string;
  setup: string;
  starter: string;
  solution: string;
  test: string;
}

export interface GradSeminar {
  question: string;
  guidance: string;
}

export type GradSimulationId =
  | 'm1-u1'
  | 'm1-u2'
  | 'm1-u3'
  | 'm1-u4'
  | 'm2-u1'
  | 'm2-u2'
  | 'm2-u3'
  | 'm2-u4'
  | 'm3-u1'
  | 'm3-u2'
  | 'm3-u3'
  | 'm3-u4'
  | 'm4-u1'
  | 'm4-u2'
  | 'm4-u3'
  | 'm5-u1'
  | 'm5-u2'
  | 'm5-u3'
  | 'm5-u4'
  | 'm6-u1'
  | 'm6-u2'
  | 'm6-u3'
  | 'm6-u4'
  | 'm7-u1'
  | 'm7-u2'
  | 'm7-u3'
  | 'm7-u4'
  | 'm8-u1'
  | 'm8-u2'
  | 'm8-u3'
  | 'm8-u4'
  | 'm8-u5'
  | 'm9-u1'
  | 'm9-u2'
  | 'm9-u3'
  | 'm9-u4'
  | 'm10-u1'
  | 'm10-u2'
  | 'm10-u3'
  | 'm10-u4'
  | 'm10-u5';

export interface GradSampleTable {
  caption: string;
  columns: string[];
  rows: (string | number)[][];
  highlightTargetCol?: number;
  footnote?: string;
}

export interface GradUnit {
  id: string;
  moduleId: number;
  order: number;
  title: string;
  summary: string;
  /** Bức tranh toàn cảnh bằng tiếng Việt đời thường: Bài này giải quyết chuyện gì ngoài đời? */
  bigPicture?: string;
  bigPictureTable?: GradSampleTable;
  outcomes: string[];
  references: GradReference[];
  sections: GradSection[];
  simulationId?: GradSimulationId;
  exercises: GradExercise[];
  seminar: GradSeminar[];
}

export interface GradModule {
  id: number;
  title: string;
  englishTitle: string;
  color: string;
  status: 'ready' | 'planned';
  overview: string;
  outcomes: string[];
  sources: string[];
  plannedTopics?: string[];
  units: GradUnit[];
}

export interface GradResourceItem {
  id: string;
  title: string;
  authors: string;
  path: string;
  modules: string;
  note: string;
}
