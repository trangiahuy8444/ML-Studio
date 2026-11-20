import {test} from 'node:test';
import assert from 'node:assert/strict';
import {adjacency, graphTrace} from '../src/lib/graphTrace.ts';

const close = (a,b) => assert(Math.abs(a-b) < 1e-10);
const chain = [[0,1],[1,2]];
const x = [2,4,8];

test('undirected adjacency is binary, deduplicates and handles self-loop policy', () => {
  assert.deepEqual(adjacency(4,[[0,1],[1,0]],false), [[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]);
  assert.throws(() => adjacency(3,[[0,3]],true), RangeError);
  assert.throws(() => adjacency(3,[[1,1]],true), RangeError);
});

test('one-hop mean and two-hop propagation use synchronous states', () => {
  const graph=adjacency(3,chain,true);
  const {output}=graphTrace(graph,x,'mean');
  assert.deepEqual(output.map(v=>v.toFixed(6)), [3,14/3,6].map(v=>v.toFixed(6)));
  close(graphTrace(graph,output,'mean').output[0],(3+14/3)/2);
  assert.equal(graphTrace(graph,x,'sum').output[1],14);
});

test('GCN uses both endpoint degrees and isolated rows stay finite', () => {
  const graph=adjacency(4,chain,true);
  const trace=graphTrace(graph,[...x,11],'gcn',2);
  close(trace.output[0],2+8/Math.sqrt(6));
  assert.equal(trace.output[3],22);
  assert.equal(graphTrace(adjacency(4,chain,false),[...x,11],'gcn').output[3],0);
});

test('GAT softmax is masked and renumbering nodes is equivariant', () => {
  const graph=adjacency(3,chain,true);
  const trace=graphTrace(graph,x,'gat');
  assert.deepEqual(trace.rows[0].messages.map(m=>m.source),[0,1]);
  close(trace.rows[0].messages.reduce((s,m)=>s+m.coefficient,0),1);
  const coefficients=[Math.exp(.6),1];
  close(trace.output[0],(2*coefficients[0]+4)/coefficients.reduce((a,b)=>a+b));
  const perm=[2,0,1];
  const reversed=graphTrace(perm.map(i=>perm.map(j=>graph[i][j])),perm.map(i=>x[i]),'gat');
  reversed.output.forEach((v,i)=>close(v,trace.output[perm[i]]));
});
