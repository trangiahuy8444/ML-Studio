import {test} from 'node:test';
import assert from 'node:assert/strict';
import {executeRequest,reduceMessage,initialExec,finished,extractReports,toNotebook} from '../src/track-py/kernel/protocol.ts';
import {parseRoute,routeHash} from '../src/lib/domain.ts';

const msg=(msg_type,content)=>({header:{msg_type},content});
test('execute_request uses Jupyter 5.3 and disallows stdin',()=>{
  const m=executeRequest('print(42)','test-session',{storeHistory:false});
  assert.equal(m.header.version,'5.3');
  assert.equal(m.channel,'shell');
  assert.equal(m.content.allow_stdin,false);
  assert.equal(m.content.store_history,false);
});
test('kernel waits for shell and idle, merges stream and clears on next output',()=>{
  let s=initialExec();
  for(const text of ['a','b'])s=reduceMessage(s,msg('stream',{name:'stdout',text}));
  assert.equal(s.outputs[0].text,'ab');
  s=reduceMessage(s,msg('clear_output',{wait:true}));
  assert.equal(s.outputs.length,1);
  s=reduceMessage(s,msg('display_data',{data:{'text/html':'<b>42</b>'}}));
  assert.equal(s.outputs[0].output_type,'display_data');
  s=reduceMessage(s,msg('execute_reply',{status:'ok',execution_count:2}));
  assert.equal(finished(s),false);
  assert.equal(finished(reduceMessage(s,msg('status',{execution_state:'idle'}))),true);
  const nb=toNotebook([{kind:'code',source:'print(42)',outputs:s.outputs}],'Example');
  assert.deepEqual(nb.cells[0].outputs[0].metadata,{});
});
test('grade markers separated from stdout; malformed marker is preserved',()=>{
  const parsed=extractReports([{output_type:'stream',name:'stdout',text:'before\n__PY30__{"id":"a","checks":[{"name":"test","ok":true,"detail":"pass"}]}\nafter\n__PY30__?'}]);
  assert.equal(parsed.reports[0].checks[0].ok,true);
  assert.equal(parsed.outputs[0].text,'before\nafter\n__PY30__?');
});
test('new routes round-trip and old routes retain their form',()=>{
  for(const route of ['py','py-1','py-7','py-30','day-30','gnn','project-house']){
    assert.equal(routeHash(parseRoute('#'+route)),route);
  }
  assert.equal(parseRoute('#py-31').kind,'home');
});
