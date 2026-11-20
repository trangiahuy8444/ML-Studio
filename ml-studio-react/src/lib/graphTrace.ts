export type Edge = readonly [number, number];
export type Aggregator = 'sum' | 'mean' | 'gcn' | 'gat';

export function adjacency(size: number, edges: readonly Edge[], loops: boolean) {
  const matrix = Array.from({length: size}, (_, i) =>
    Array.from({length: size}, (_, j) => Number(loops && i === j)));
  for (const [i, j] of edges) {
    if (i === j || !matrix[i] || !matrix[j]) throw new RangeError('Cạnh phải nối hai nút hợp lệ.');
    matrix[i][j] = matrix[j][i] = 1;
  }
  return matrix;
}

export function graphTrace(matrix: number[][], features: number[], mode: Aggregator, weight = 1) {
  const degrees = matrix.map(row => row.reduce((sum, value) => sum + value, 0));
  // Illustrated GAT with scalar projection and fixed attention parameters [0.6, -0.3].
  const projected = features.map(value => value * weight);
  const rows = matrix.map((row, target) => {
    const edges = row.flatMap((present, source) => {
      if (!present) return [];
      const raw = .6 * projected[target] - .3 * projected[source];
      return [{source, score: raw < 0 ? .2 * raw : raw}];
    });
    if (!edges.length) return {messages: [], output: 0};
    const maxScore = Math.max(...edges.map(edge => edge.score));
    const normalizer = edges.reduce((sum, edge) => sum + Math.exp(edge.score - maxScore), 0);
    const messages = edges.map(edge => {
      const {source, score} = edge;
      const coefficient = mode === 'gat' ? Math.exp(score - maxScore) / normalizer
        : mode === 'gcn' ? 1 / Math.sqrt(degrees[target] * degrees[source])
        : mode === 'mean' ? 1 / degrees[target] : 1;
      return {source, score, coefficient, contribution: projected[source] * coefficient};
    });
    return {messages, output: messages.reduce((sum, message) => sum + message.contribution, 0)};
  });
  return {rows, degrees, output: rows.map(row => row.output)};
}
