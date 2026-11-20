import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const load=f=>JSON.parse(readFileSync(new URL('../src/track-py/'+f,import.meta.url),'utf8'));
const lessons=[...load('week1.json'),...load('week2.json'),...load('week3.json'),...load('week4.json')];
test('lộ trình liên tục từ ngày 1 và mỗi ngày có đủ phần',()=>{
  lessons.forEach((l,i)=>{
    assert.equal(l.id,i+1);
    for(const key of ['title','goals','reads','project','questions','answers','steps','setup','exercises'])assert.ok(l[key]?.length,`ngày ${l.id} thiếu ${key}`);
    assert.equal(l.questions.length,l.answers.length);
  });
});
test('mỗi bài tập: khung chưa làm, ô kiểm tra báo đúng id, lời giải không lộ trong khung',()=>{
  const ids=new Set();
  for(const l of lessons)for(const e of l.exercises){
    assert.ok(!ids.has(e.id),'trùng id '+e.id);ids.add(e.id);
    assert.match(e.starter,/raise NotImplementedError/);
    assert.ok(e.test.includes(`_report('${e.id}'`),'ô kiểm tra của '+e.id);
    assert.notEqual(e.starter.trim(),e.solution.trim());
    assert.ok([...e.starter.matchAll(/^(?:def|class)\s+(\w+)/gm)].length>0,'khung phải khai báo hàm hoặc lớp: '+e.id);
  }
});
test('dữ liệu breast2d khớp kích thước bộ Breast Cancer Wisconsin',()=>{
  const d=load('data/breast2d.json');
  assert.equal(d.points.length,569);
  assert.equal(d.points.filter(p=>p[2]===1).length,212);
  assert.equal(d.validation.y.length,d.validation.score30.length);
});
test('dữ liệu tuần 3: Iris 150 bông, digits đủ cấu trúc và bảng CV hợp lệ',()=>{
  const i=load('data/iris2d.json');
  assert.equal(i.points.length,150);
  assert.deepEqual([0,1,2].map(c=>i.points.filter(p=>p[2]===c).length),[50,50,50]);
  const d=load('data/digits.json');
  assert.equal(d.mean.length,64);assert.equal(d.components.length,30);assert.equal(d.ratio.length,64);
  assert.ok(Math.abs(d.ratio.reduce((s,v)=>s+v,0)-1)<1e-4);
  assert.equal(d.samples.z.length,d.samples.pixels.length);
  for(const r of d.cv)assert.ok(r[2]>0&&r[2]<=1);
});
test('tuần 4: bài dùng PyTorch báo rõ khi thiếu thư viện, Ngày 22–23 chỉ cần NumPy',()=>{
  for(const l of load('week4.json')){
    if(l.id<=23)assert.ok(!/import torch/.test(l.setup),'ngày '+l.id+' không được cần PyTorch');
    else assert.match(l.setup,/requirements-py30\.txt/);
  }
});
