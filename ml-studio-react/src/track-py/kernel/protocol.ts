// Phần thuần (không I/O) của giao thức nhắn tin Jupyter v5: dựng thông điệp và
// gộp các thông điệp iopub thành danh sách output kiểu nbformat. Tách riêng để kiểm thử bằng node --test.

export interface CellOutput {
  output_type: 'stream' | 'execute_result' | 'display_data' | 'error';
  name?: string; text?: string; data?: Record<string, unknown>; execution_count?: number | null;
  ename?: string; evalue?: string; traceback?: string[];
}

export interface JupyterMessage {
  header: {msg_id: string; msg_type: string; session: string; username: string; date: string; version: string};
  parent_header: Partial<JupyterMessage['header']>;
  metadata: Record<string, unknown>;
  content: Record<string, unknown>;
  channel?: string;
  buffers?: unknown[];
}

export const MARKER = '__PY30__';

let counter = 0;
export function newId(): string {
  counter += 1;
  const rnd = typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : Math.random().toString(16).slice(2);
  return `${rnd}-${counter}`;
}

export function executeRequest(code: string, session: string, opts: {silent?: boolean; storeHistory?: boolean} = {}): JupyterMessage {
  return {
    header: {msg_id: newId(), msg_type: 'execute_request', session, username: 'ml-studio', date: new Date().toISOString(), version: '5.3'},
    parent_header: {}, metadata: {}, channel: 'shell', buffers: [],
    content: {code, silent: opts.silent ?? false, store_history: opts.storeHistory ?? true, user_expressions: {}, allow_stdin: false, stop_on_error: true},
  };
}

export interface ExecState {outputs: CellOutput[]; clearPending: boolean; executionCount: number | null; idle: boolean; replied: boolean; status: 'ok' | 'error' | 'aborted' | null}
export const initialExec = (): ExecState => ({outputs: [], clearPending: false, executionCount: null, idle: false, replied: false, status: null});

/** Gộp một thông điệp (iopub hoặc shell) thuộc cùng lệnh chạy vào trạng thái. Trả về trạng thái mới (bất biến). */
export function reduceMessage(state: ExecState, msg: JupyterMessage): ExecState {
  const t = msg.header.msg_type, c = msg.content;
  const push = (o: CellOutput): ExecState => {
    const base = state.clearPending ? [] : state.outputs;
    const last = base[base.length - 1];
    if (o.output_type === 'stream' && last?.output_type === 'stream' && last.name === o.name)
      return {...state, clearPending: false, outputs: [...base.slice(0, -1), {...last, text: (last.text ?? '') + (o.text ?? '')}]};
    return {...state, clearPending: false, outputs: [...base, o]};
  };
  switch (t) {
    case 'stream': return push({output_type: 'stream', name: String(c.name ?? 'stdout'), text: String(c.text ?? '')});
    case 'execute_result': return push({output_type: 'execute_result', data: c.data as Record<string, unknown>, execution_count: (c.execution_count as number) ?? null});
    case 'display_data': return push({output_type: 'display_data', data: c.data as Record<string, unknown>});
    case 'error': return push({output_type: 'error', ename: String(c.ename ?? ''), evalue: String(c.evalue ?? ''), traceback: (c.traceback as string[]) ?? []});
    case 'clear_output': return c.wait ? {...state, clearPending: true} : {...state, outputs: [], clearPending: false};
    case 'execute_input': return {...state, executionCount: (c.execution_count as number) ?? state.executionCount};
    case 'status': return c.execution_state === 'idle' ? {...state, idle: true} : state;
    case 'execute_reply': return {...state, replied: true, status: (c.status as ExecState['status']) ?? null, executionCount: (c.execution_count as number) ?? state.executionCount};
    default: return state;
  }
}

export const finished = (s: ExecState) => s.idle && s.replied;

const ANSI = /\u001b\[[0-9;]*[A-Za-z]/g;
export const stripAnsi = (s: string) => s.replace(ANSI, '');

export interface CheckResult {name: string; ok: boolean; detail: string}
export interface Report {id: string; checks: CheckResult[]}

/** Tách các dòng báo cáo kiểm tra (bắt đầu bằng MARKER) khỏi output stream; trả về output đã lọc và các báo cáo. */
export function extractReports(outputs: CellOutput[]): {outputs: CellOutput[]; reports: Report[]} {
  const reports: Report[] = [];
  const cleaned: CellOutput[] = [];
  for (const o of outputs) {
    if (o.output_type !== 'stream' || !o.text?.includes(MARKER)) {cleaned.push(o); continue}
    const keep: string[] = [];
    for (const line of o.text.split('\n')) {
      if (line.startsWith(MARKER)) {
        try {
          const r = JSON.parse(line.slice(MARKER.length)) as Report;
          if (r && typeof r.id === 'string' && Array.isArray(r.checks)) reports.push({id: r.id, checks: r.checks.map(ch => ({name: String(ch.name), ok: ch.ok === true, detail: String(ch.detail ?? '')}))});
        } catch {keep.push(line)}
      } else keep.push(line);
    }
    const text = keep.join('\n');
    if (text.trim()) cleaned.push({...o, text});
  }
  return {outputs: cleaned, reports};
}

function nbOutput(o: CellOutput): Record<string, unknown> {
  if (o.output_type === 'stream') return {output_type: 'stream', name: o.name ?? 'stdout', text: o.text ?? ''};
  if (o.output_type === 'error') return {output_type: 'error', ename: o.ename ?? '', evalue: o.evalue ?? '', traceback: o.traceback ?? []};
  if (o.output_type === 'execute_result') return {output_type: 'execute_result', data: o.data ?? {}, metadata: {}, execution_count: o.execution_count ?? null};
  return {output_type: 'display_data', data: o.data ?? {}, metadata: {}};
}

/** Chuyển ô của trang thành notebook nbformat 4 để lưu bằng Contents API. */
export interface PageCell {kind: 'markdown' | 'code'; source: string; outputs?: CellOutput[]; tag?: string}
export function toNotebook(cells: PageCell[], title: string) {
  return {
    nbformat: 4, nbformat_minor: 5,
    metadata: {kernelspec: {name: 'python3', display_name: 'Python 3 (ML Studio)', language: 'python'}, language_info: {name: 'python'}, ml_studio: {track: 'py30', title}},
    cells: cells.map((c, i) => c.kind === 'markdown'
      ? {cell_type: 'markdown', id: `py30-${i}`, metadata: {tags: c.tag ? [c.tag] : []}, source: c.source}
      : {cell_type: 'code', id: `py30-${i}`, metadata: {tags: c.tag ? [c.tag] : []}, source: c.source, execution_count: null, outputs: (c.outputs ?? []).map(nbOutput)}),
  };
}
