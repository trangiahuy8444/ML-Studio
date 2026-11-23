import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseRoute, routeHash } from '../src/lib/domain.ts';
import { GRAD_MODULES, GRAD_RESOURCES, allReadyUnits, getModule, getUnit } from '../src/curriculum/index.ts';

test('graduate curriculum defines 10 contiguous modules within graduate CS scope', () => {
  assert.equal(GRAD_MODULES.length, 10);
  assert.deepEqual(
    GRAD_MODULES.map((m) => m.id),
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  );
  for (const mod of GRAD_MODULES) {
    assert(mod.title.length > 5);
    assert(mod.englishTitle.length > 5);
    assert(mod.overview.length > 30);
    assert(mod.outcomes.length >= 3);
    assert(mod.sources.length >= 1);
  }
  assert(GRAD_RESOURCES.length >= 8);
});

test('Modules 1 to 7 have 27 complete graduate units with theory, references, exercises and seminar', () => {
  const expectedCounts = { 1: 4, 2: 4, 3: 4, 4: 3, 5: 4, 6: 4, 7: 4 };
  for (const [modIdStr, count] of Object.entries(expectedCounts)) {
    const modId = Number(modIdStr);
    const mod = getModule(modId);
    assert(mod);
    assert.equal(mod.status, 'ready');
    assert.equal(mod.units.length, count);
    assert.deepEqual(
      mod.units.map((u) => u.id),
      Array.from({ length: count }, (_, idx) => `m${modId}-u${idx + 1}`),
    );
  }

  const ready = allReadyUnits();
  assert.equal(ready.length, 27);
  assert.equal(ready.reduce((sum, u) => sum + u.exercises.length, 0), 40);

  for (const u of ready) {
    assert(u.outcomes.length >= 3, `${u.id} outcomes`);
    assert(u.references.length >= 2, `${u.id} references`);
    assert(u.sections.length >= 2, `${u.id} sections`);
    assert(u.simulationId, `${u.id} simulationId`);
    assert(u.exercises.length >= 1, `${u.id} exercises`);
    assert(u.seminar.length >= 1, `${u.id} seminar`);

    for (const ex of u.exercises) {
      assert(ex.starter.includes('raise NotImplementedError'), `${ex.id} starter placeholder`);
      assert(ex.solution.trim().length > 20 && !ex.solution.includes('NotImplementedError'), `${ex.id} solution`);
      assert(ex.setup.includes('__GRAD__'), `${ex.id} __GRAD__ reporting marker in setup`);
      assert(ex.test.includes(`_report('${ex.id}'`), `${ex.id} _report call with report id`);
    }
  }
  assert(getUnit('m1-u2')?.unit.title.includes('MLE'));
  assert(getUnit('m2-u4')?.unit.title.includes('Exponential Family'));
  assert(getUnit('m3-u4')?.unit.title.includes('Newton'));
  assert(getUnit('m4-u3')?.unit.title.includes('Kernel'));
  assert(getUnit('m5-u4')?.unit.title.includes('XGBoost'));
  assert(getUnit('m6-u3')?.unit.title.includes('EM'));
  assert(getUnit('m7-u4')?.unit.title.includes('CNN'));
});

test('graduate curriculum routes round-trip and invalid module ids fall back to home', () => {
  assert.deepEqual(parseRoute('#' + routeHash({ kind: 'curriculum' })), { kind: 'curriculum' });
  for (let id = 1; id <= 10; id++) {
    assert.deepEqual(parseRoute('#' + routeHash({ kind: 'module', id })), { kind: 'module', id });
  }
  for (const u of allReadyUnits()) {
    assert.deepEqual(parseRoute('#' + routeHash({ kind: 'unit', id: u.id })), { kind: 'unit', id: u.id });
  }
  for (const bad of ['#module-0', '#module-11', '#unit-bad']) {
    assert.deepEqual(parseRoute(bad), { kind: 'home' });
  }
});
