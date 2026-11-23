export interface GradReference {
  source: string;
  section: string;
  note: string;
}

export interface GradSection {
  heading: string;
  body: string;
  math?: string;
  intuition?: string;
  pitfall?: string;
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
  | 'm3-u4';

export interface GradUnit {
  id: string;
  moduleId: number;
  order: number;
  title: string;
  summary: string;
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
