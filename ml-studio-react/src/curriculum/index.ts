import { MODULE_01 } from './module01.ts';
import { MODULE_02 } from './module02.ts';
import { MODULE_03 } from './module03.ts';
import { MODULE_04 } from './module04.ts';
import { MODULE_05 } from './module05.ts';
import { MODULE_06 } from './module06.ts';
import { MODULE_07 } from './module07.ts';
import { MODULE_08 } from './module08.ts';
import { MODULE_09 } from './module09.ts';
import { MODULE_10 } from './module10.ts';
import type { GradModule, GradResourceItem, GradUnit } from './types.ts';

export type {
  GradExercise,
  GradModule,
  GradReference,
  GradResourceItem,
  GradSection,
  GradSeminar,
  GradSimulationId,
  GradUnit,
} from './types';

export const GRAD_MODULES: GradModule[] = [
  MODULE_01,
  MODULE_02,
  MODULE_03,
  MODULE_04,
  MODULE_05,
  MODULE_06,
  MODULE_07,
  MODULE_08,
  MODULE_09,
  MODULE_10,
];

export function getModule(id: number): GradModule | undefined {
  return GRAD_MODULES.find((m) => m.id === id);
}

export function getUnit(unitId: string): { module: GradModule; unit: GradUnit } | undefined {
  for (const mod of GRAD_MODULES) {
    const unit = mod.units.find((u) => u.id === unitId);
    if (unit) return { module: mod, unit };
  }
  return undefined;
}

export function allReadyUnits(): GradUnit[] {
  return GRAD_MODULES.flatMap((m) => m.units);
}

export const GRAD_RESOURCES: GradResourceItem[] = [
  {
    id: 'R01',
    title: 'Probabilistic Machine Learning: An Introduction',
    authors: 'Kevin P. Murphy (MIT Press)',
    path: '01_Sach_va_Giao_trinh_Kinh_dien/PML_An_Introduction_Kevin_Murphy.pdf',
    modules: 'Module 1, 2, 3, 4, 5, 6, 7',
    note: 'Giáo trình trụ cột về suy luận Bayes, tối ưu hóa, GLMs, mô hình sinh và đồ thị xác suất.',
  },
  {
    id: 'R02',
    title: 'Stanford CS229: Machine Learning Complete Lecture Notes',
    authors: 'Andrew Ng & Tengyu Ma',
    path: '03_Khoa_hoc_Stanford_CS229_CS224N/Stanford_CS229_Machine_Learning_Complete_Notes.pdf',
    modules: 'Module 1, 2, 4, 6, 7, 10',
    note: 'Tài liệu nền tảng về GLMs, SVM/Kernel, Learning Theory (Hoeffding/VC), EM và Reinforcement Learning.',
  },
  {
    id: 'R03',
    title: 'UC Berkeley CS189/289A: Introduction to Machine Learning (Lectures 01–13)',
    authors: 'UC Berkeley',
    path: '02_Bai_giang_Hoc_May_Nang_Cao_UC_Berkeley_CS189/',
    modules: 'Module 1, 2, 3, 6',
    note: 'Bài giảng chuẩn cao học về Data Tools, K-means, GMM, Linear/Logistic Regression, Bias–Variance và SGD.',
  },
  {
    id: 'R04',
    title: 'An Introduction to Statistical Learning (7th Printing)',
    authors: 'James, Witten, Hastie & Tibshirani',
    path: '01_Sach_va_Giao_trinh_Kinh_dien/ISLR_Seventh_Printing.pdf',
    modules: 'Module 1, 2, 5',
    note: 'Tài liệu chuẩn mực về Resampling, Cross-Validation, Regularization và Tree-Based Ensembles.',
  },
  {
    id: 'R05',
    title: 'Machine Learning cơ bản (Bản tiêu chuẩn & Bản in màu)',
    authors: 'Vũ Hữu Tiệp',
    path: '01_Sach_va_Giao_trinh_Kinh_dien/book_ML_color.pdf',
    modules: 'Module 1, 3, 4',
    note: 'Trình bày trực quan tiếng Việt về đại số tuyến tính, tối ưu lồi và SVM.',
  },
  {
    id: 'R06',
    title: 'Core NLP & Transformer Research Papers (2013–2020)',
    authors: 'Mikolov, Hochreiter, Bahdanau, Vaswani, Goldberg, Jurafsky & Martin, Nguyen (PhoBERT)',
    path: '04_Deep_Learning_va_NLP_Nghien_cuu_Cot_loi/',
    modules: 'Module 8',
    note: 'Tuyển tập các bài báo gốc định hình Word2Vec, LSTM, Attention, Transformer và PhoBERT tiếng Việt.',
  },
  {
    id: 'R07',
    title: 'Computer Vision Lectures & OpenCV Practice',
    authors: 'Bộ bài giảng Thị giác Máy tính',
    path: '05_Thi_giac_May_tinh_Computer_Vision/',
    modules: 'Module 7',
    note: 'Cơ sở biểu diễn ảnh, bộ lọc không gian, tăng cường ảnh và thực hành Python/OpenCV.',
  },
  {
    id: 'R08',
    title: 'Introduction to Graph Neural Networks',
    authors: 'Zhiyuan Liu & Jie Zhou',
    path: '01_Sach_va_Giao_trinh_Kinh_dien/Introduction_to_Graph_Neural_Networks.pdf',
    modules: 'Module 9',
    note: 'Giáo trình chuyên sâu về Graph Convolutions, Message Passing, GAT và đồ thị động.',
  },
  {
    id: 'R09',
    title: 'Reinforcement Learning: An Introduction (2nd Edition)',
    authors: 'Richard S. Sutton & Andrew G. Barto',
    path: '01_Sach_va_Giao_trinh_Kinh_dien/Reinforcement_Learning_Sutton_Barto.pdf',
    modules: 'Module 10',
    note: 'Kinh điển số 1 về Bandits, MDPs, Dynamic Programming, TD Learning và Policy Gradient.',
  },
];
