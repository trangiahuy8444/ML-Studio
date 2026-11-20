// Kết nối tới Jupyter Server cục bộ do launcher khởi động.
// Launcher cấp địa chỉ + token qua /jupyter/session (chỉ cùng origin, Host 127.0.0.1/localhost),
// và cấu hình Jupyter chỉ nhận origin của trang web (allow_origin_pat). REST dùng header Authorization,
// WebSocket dùng ?token=. Không có token nào được lưu vào localStorage.
import {executeRequest, finished, initialExec, reduceMessage, newId} from './protocol';
import type {CellOutput, ExecState, JupyterMessage} from './protocol';

export interface ServerInfo {url: string; token: string; classicPrefix: string; root: string}
export type KernelStatus = 'offline' | 'connecting' | 'starting' | 'idle' | 'busy' | 'restarting' | 'dead';

export class KernelUnavailable extends Error {}

export async function discoverServer(): Promise<ServerInfo> {
  if (location.protocol === 'file:') throw new KernelUnavailable('Trang đang mở từ tệp HTML. Hãy mở bằng Mo_Jupyter_Studio.command để có kernel Python trên máy.');
  let res: Response;
  try {res = await fetch('/jupyter/session', {cache: 'no-store', credentials: 'same-origin'})}
  catch {throw new KernelUnavailable('Không gọi được launcher. Hãy chạy Mo_Jupyter_Studio.command và mở trang từ địa chỉ nó in ra.')}
  if (res.status === 404) throw new KernelUnavailable('Trang này không được phục vụ bởi launcher ML Studio (thiếu /jupyter/session).');
  if (!res.ok) throw new KernelUnavailable(`Launcher trả lỗi ${res.status}. Jupyter có thể đã dừng; khởi động lại launcher.`);
  const info = await res.json() as Partial<ServerInfo>;
  if (typeof info.url !== 'string' || typeof info.token !== 'string') throw new KernelUnavailable('Phản hồi của launcher không đúng định dạng.');
  const endpoint = new URL(info.url);
  if (endpoint.protocol !== 'http:' || !['127.0.0.1', 'localhost'].includes(endpoint.hostname)
      || endpoint.username || endpoint.password || endpoint.pathname !== '/' || endpoint.search || endpoint.hash
      || info.root !== 'py30' || !info.token) throw new KernelUnavailable('Launcher trả một địa chỉ kernel không hợp lệ.');
  return {url: info.url.replace(/\/$/, ''), token: info.token, classicPrefix: info.classicPrefix ?? '', root: info.root ?? 'py30'};
}

interface Pending {state: ExecState; onUpdate: (s: ExecState) => void; resolve: (s: ExecState) => void; reject: (e: Error) => void}

export class LocalKernel {
  private ws: WebSocket | null = null;
  private sessionId = '';
  private clientSession = newId();
  private closed = false;
  private kernelId = '';
  private pending = new Map<string, Pending>();
  private queue: Promise<unknown> = Promise.resolve();
  status: KernelStatus = 'offline';
  onStatus: (s: KernelStatus) => void = () => {};

  constructor(private server: ServerInfo, private notebookPath: string) {}

  private api(path: string, init: RequestInit = {}) {
    return fetch(this.server.url + path, {...init, headers: {Authorization: 'token ' + this.server.token, 'Content-Type': 'application/json', ...(init.headers ?? {})}});
  }
  private setStatus(s: KernelStatus) {this.status = s; this.onStatus(s)}

  async start(): Promise<void> {
    if (this.closed) throw new Error('Phiên đã đóng.');
    this.setStatus('starting');
    const res = await this.api('/api/sessions', {method: 'POST', body: JSON.stringify({path: this.notebookPath, name: this.notebookPath.split('/').pop(), type: 'notebook', kernel: {name: 'python3'}})});
    if (!res.ok) {this.setStatus('dead'); throw new Error(`Không tạo được phiên kernel (${res.status}).`)}
    const body = await res.json() as {id: string; kernel: {id: string}};
    this.sessionId = body.id; this.kernelId = body.kernel.id;
    if (this.closed) {await this.shutdown(); throw new Error('Phiên đã đóng.');}
    try {await this.openSocket();}
    catch (error) {await this.shutdown(); throw error;}
  }

  private openSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = this.server.url.replace(/^http/, 'ws') + `/api/kernels/${this.kernelId}/channels?session_id=${this.clientSession}&token=${encodeURIComponent(this.server.token)}`;
      const ws = new WebSocket(wsUrl);
      this.ws = ws;
      let opened = false;
      const timer = setTimeout(() => {ws.close(); reject(new Error('Quá thời gian kết nối kernel.'));},15000);
      ws.onopen = () => {clearTimeout(timer); opened = true; this.setStatus('idle'); resolve()};
      ws.onerror = () => {if (!opened) {this.setStatus('dead'); reject(new Error('Không mở được WebSocket tới kernel. Kiểm tra launcher đã cấu hình origin.'))}};
      ws.onclose = () => {
        clearTimeout(timer);
        if (!opened) reject(new Error('WebSocket đã đóng trước khi kết nối.'));
        if (this.ws !== ws) return;
        this.setStatus('dead');
        for (const p of this.pending.values()) p.reject(new Error('Mất kết nối tới kernel.'));
        this.pending.clear();
      };
      ws.onmessage = ev => {
        if (typeof ev.data !== 'string') return; // giao thức JSON cũ: mọi thông điệp là văn bản
        let msg: JupyterMessage;
        try {msg = JSON.parse(ev.data)} catch {return}
        if (msg.header.msg_type === 'status' && msg.channel === 'iopub') {
          const st = msg.content.execution_state as string;
          if (st === 'busy' || st === 'idle') this.setStatus(st);
          if (st === 'restarting') this.setStatus('restarting');
          if (st === 'dead') this.setStatus('dead');
        }
        const parent = msg.parent_header?.msg_id;
        const p = parent ? this.pending.get(parent) : undefined;
        if (!p) return;
        p.state = reduceMessage(p.state, msg);
        p.onUpdate(p.state);
        if (finished(p.state)) {this.pending.delete(parent!); p.resolve(p.state)}
      };
    });
  }

  /** Chạy mã theo hàng đợi (một lệnh một lúc), gọi onUpdate mỗi khi có output mới. */
  execute(code: string, onUpdate: (s: ExecState) => void = () => {}, opts: {storeHistory?: boolean} = {}): Promise<ExecState> {
    const run = () => new Promise<ExecState>((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {reject(new Error('Kernel chưa kết nối.')); return}
      const msg = executeRequest(code, this.clientSession, opts);
      this.pending.set(msg.header.msg_id, {state: initialExec(), onUpdate, resolve, reject});
      try {this.ws.send(JSON.stringify(msg));}
      catch (error) {this.pending.delete(msg.header.msg_id); reject(error);}
    });
    const next = this.queue.then(run, run);
    this.queue = next.catch(() => undefined);
    return next;
  }

  async interrupt() {
    const res = await this.api(`/api/kernels/${this.kernelId}/interrupt`, {method: 'POST'});
    if (!res.ok) throw new Error(`Không ngắt được kernel (${res.status}).`);
  }

  async restart() {
    this.setStatus('restarting');
    const socket = this.ws; this.ws = null; socket?.close();
    for (const p of this.pending.values()) p.reject(new Error('Kernel đã khởi động lại.'));
    this.pending.clear();
    const res = await this.api(`/api/kernels/${this.kernelId}/restart`, {method: 'POST'});
    if (!res.ok) throw new Error(`Khởi động lại thất bại (${res.status}).`);
    if (!this.closed) await this.openSocket();
  }

  async shutdown() {
    this.closed = true;
    const ws = this.ws; this.ws = null; ws?.close();
    for (const item of this.pending.values()) item.reject(new Error('Phiên kernel đã đóng.'));
    this.pending.clear();
    const id = this.sessionId; this.sessionId = '';
    if (id) await this.api(`/api/sessions/${id}`, {method: 'DELETE'}).catch(() => undefined);
    this.setStatus('offline');
  }

  async saveNotebook(path: string, notebook: unknown) {
    const dir = path.split('/').slice(0, -1).join('/');
    if (dir) {
      const exists = await this.api(`/api/contents/${encodePath(dir)}?content=0`);
      if (exists.status === 404) await this.api(`/api/contents/${encodePath(dir)}`, {method: 'PUT', body: JSON.stringify({type: 'directory'})});
    }
    const res = await this.api(`/api/contents/${encodePath(path)}`, {method: 'PUT', body: JSON.stringify({type: 'notebook', format: 'json', content: notebook})});
    if (!res.ok) throw new Error(`Không lưu được notebook (${res.status}).`);
  }

  async loadNotebook(path: string): Promise<{cells: {cell_type: string; source: string | string[]; metadata?: {tags?: string[]}}[]} | null> {
    const res = await this.api(`/api/contents/${encodePath(path)}?type=notebook&content=1`);
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`Không đọc được notebook (${res.status}).`);
    return (await res.json() as {content: {cells: {cell_type: string; source: string | string[]}[]}}).content;
  }

}

/** Mở notebook trong giao diện Jupyter Classic: launcher chuyển hướng kèm token, trang web không giữ token trong liên kết. */
export const openInJupyterHref = (path: string) => `/jupyter/open?path=${encodeURIComponent(path)}`;

const encodePath = (p: string) => p.split('/').map(encodeURIComponent).join('/');
export type {CellOutput, ExecState};
