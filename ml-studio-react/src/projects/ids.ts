/** Danh sách mã dự án; tách riêng để lib/domain.ts kiểm tra route mà không tải giao diện. */
export const PROJECT_IDS=['tumor','spam','iris','customers','digits','sentiment','rl','gnn'] as const;
export type ProjectId=typeof PROJECT_IDS[number];
export const isProjectId=(v:string):v is ProjectId=>(PROJECT_IDS as readonly string[]).includes(v);
