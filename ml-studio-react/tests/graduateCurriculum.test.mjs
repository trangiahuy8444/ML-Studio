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

test('Module 1 has 4 complete graduate units with theory, references, exercises and seminar', () => {
  const m1 = getModule(1);
  assert(m1);
  assert.equal(m1.status, 'ready');
  assert.equal(m1.units.length, 4);
  assert.deepEqual(
    m1.units.map((u) => u.id),
    ['m1-u1', 'm1-u2', 'm1-u3', 'm1-u4'],
  );

  for (const u of allReadyUnits()) {
    assert(u.outcomes.length >= 3, `${u.id} outcomes`);
    assert(u.references.length >= 3, `${u.id} references`);
    assert(u.sections.length >= 3, `${u.id} sections`);
    assert(u.simulationId, `${u.id} simulationId`);
    assert(u.exercises.length >= 1, `${u.id} exercises`);
    assert(u.seminar.length >= 2, `${u.id} seminar`);

    for (const ex of u.exercises) {
      assert(ex.starter.includes('raise NotImplementedError'), `${ex.id} starter placeholder`);
      assert(ex.solution.trim().length > 20 && !ex.solution.includes('NotImplementedError'), `${ex.id} solution`);
      assert(ex.setup.includes('__GRAD__'), `${ex.id} __GRAD__ reporting marker in setup`);
      assert(ex.test.includes(`_report('${ex.id}'`), `${ex.id} _report call with report id`);
    }
  }
  assert(getUnit('m1-u2')?.unit.title.includes('MLE'));
});

test('graduate curriculum routes round-trip and invalid module ids fall back to home', () => {
  assert.deepEqual(parseRoute('#' + routeHash({ kind: 'curriculum' })), { kind: 'curriculum' });
  for (let id = 1; id <= 10; id++) {
    assert.deepEqual(parseRoute('#' + routeHash({ kind: 'module', id })), { kind: 'module', id });
  }
  for (const id of ['m1-u1', 'm1-u2', 'm1-u3', 'm1-u4']) {
    assert.deepEqual(parseRoute('#' + routeHash({ kind: 'unit', id })), { kind: 'unit', id });
  }
  for (const bad of ['#module-0', '#module-11', '#unit-bad']) {
    assert.deepEqual(parseRoute(bad), { kind: 'home' });
  }
});
