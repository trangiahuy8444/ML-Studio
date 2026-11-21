import test from 'node:test';
import assert from 'node:assert/strict';
import {parseRoute,routeHash} from '../src/lib/domain.ts';
import {PROJECT_IDS} from '../src/projects/ids.ts';
import {CHAPTERS,PROJECTS} from '../src/projects/chapters.ts';

test('project routes round-trip; unknown project ids fall back to home',()=>{
  assert.deepEqual(parseRoute('#projects'),{kind:'projects'});
  for(const id of PROJECT_IDS){assert.equal(routeHash({kind:'proj',id}),'proj-'+id);assert.deepEqual(parseRoute('#proj-'+id),{kind:'proj',id})}
  for(const h of ['#proj-','#proj-abc','#proj-housing'])assert.deepEqual(parseRoute(h),{kind:'home'});
});

test('six chapters cover days 1–30 in order and every outcome list is concrete',()=>{
  assert.equal(CHAPTERS.length,6);
  let next=1;
  for(const ch of CHAPTERS){assert.equal(ch.days[0],next);next=ch.days[1]+1;assert.ok(ch.outcomes.length>=3);assert.ok(ch.product.length>10)}
  assert.equal(next,31);
});

test('every project belongs to exactly one chapter that matches its metadata',()=>{
  const listed=CHAPTERS.flatMap(c=>c.projects.map(p=>[p,c.id]));
  assert.equal(new Set(listed.map(x=>x[0])).size,listed.length);
  for(const id of [...PROJECT_IDS,'housing']){const hit=listed.find(x=>x[0]===id);assert.ok(hit,id+' missing');assert.equal(PROJECTS[id].chapter,hit[1]);
    for(const d of PROJECTS[id].days)assert.ok(d>=1&&d<=30)}
});
