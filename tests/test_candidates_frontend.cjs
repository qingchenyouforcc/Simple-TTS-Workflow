// Run with: node --test tests/test_candidates_frontend.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/app.js', 'utf8');

function harness() {
  const context = {
    regenerationCount: 1,
    FormData, generationInProgress: false, resultGroups: [], lastGenerationData: new FormData(),
    setResultState() {}, showToast() {}, renderResultGroups() {}, renderCandidateSummary() {},
    setLoading(value) { context.generationInProgress = value; },
    parseResponse: async response => response.payload,
    regenerateButton: {}, clearResults: {}, message: {}, emptyResults: {}, showView() {},
  };
  vm.createContext(context);
  for (const [start, end] of [
    ['async function generateAudio(', 'function getSingleInput('],
    ['function getSingleInput(', 'function renderResultGroups('],
    ['async function regenerateOne(', 'async function parseResponse('],
  ]) vm.runInContext(source.slice(source.indexOf(start), source.indexOf(end)), context);
  return context;
}

test('single-item input preserves an entire assisted block and skips blank normal lines', () => {
  const c = harness();
  const data = new FormData();
  data.set('texts', ' first\r\n\r\n second ');
  assert.equal(c.getSingleInput(data, 1), 'second');
  data.set('mode', 'scene_dubbing');
  data.set('scene_dubbing_mode', 'assisted');
  data.set('texts', 'One\r\ncalm\r\nkeyword: calm\r\n \r\nTwo\r\nangry\r\nkeyword: angry');
  assert.equal(c.getSingleInput(data, 1), 'Two\nangry\nkeyword: angry');
});

test('whole-batch regeneration also appends versions and failed new batches keep existing results', async () => {
  const c = harness();
  const data = new FormData();
  data.set('texts', 'first\nsecond');
  c.fetch = async () => ({ok: true, payload: {items: [{index: 1}, {index: 2}], output_dir: 'out'}});
  await c.generateAudio(data);
  assert.equal(c.resultGroups.length, 2);
  assert.equal(c.resultGroups[1].input, 'second');
  await c.generateAudio(c.lastGenerationData);
  assert.equal(c.resultGroups[0].versions.length, 2);
  assert.equal(c.resultGroups[1].versions.length, 2);
  const previousGroups = c.resultGroups;
  c.fetch = async () => ({ok: false, payload: {detail: 'failed'}});
  await c.generateAudio(new FormData());
  assert.equal(c.resultGroups, previousGroups);
  assert.equal(c.lastGenerationData, data);
  assert.equal(c.emptyResults.hidden, true);
});

test('repeated single retries keep earlier candidates and other lines; failure is non-destructive', async () => {
  const c = harness();
  c.lastGenerationData.set('texts', 'first\nsecond');
  c.lastGenerationData.set('voice_preset', 'narrator');
  const group = {index: 2, input: 'second', selected: 0, versions: [{number: 1, item: {candidate_id: 'original'}}]};
  const other = {index: 1, versions: [{item: {candidate_id: 'other'}}]};
  c.resultGroups = [other, group];
  let calls = 0;
  c.fetch = async (_, options) => {
    calls++;
    assert.equal(options.body.get('texts'), 'second');
    assert.equal(options.body.get('voice_preset'), 'narrator');
    assert.equal(c.generationInProgress, true);
    await c.regenerateOne(group); // duplicate attempts must not create a request
    return {ok: true, payload: {items: [{candidate_id: 'v' + calls}], emotion_analyses: [], output_dir: 'out'}};
  };
  await c.regenerateOne(group);
  await c.regenerateOne(group);
  assert.equal(calls, 2);
  assert.equal(group.versions.length, 3);
  assert.equal(group.selected, 0);
  assert.equal(other.versions.length, 1);
  assert.equal(c.lastGenerationData.get('texts'), 'first\nsecond');
  c.fetch = async () => { throw new Error('offline'); };
  await c.regenerateOne(group);
  assert.equal(group.versions.length, 3);
  assert.equal(c.generationInProgress, false);
});

test('configured retries run sequentially and preserve partial success', async () => {
  const c = harness();
  c.regenerationCount = 3;
  const group = {index: 1, input: 'hello', selected: 0, versions: [{number: 1, item: {}}]};
  let calls = 0;
  c.fetch = async () => {
    calls++;
    return {ok: true, payload: {items: [{index: 1}], output_dir: 'out'}};
  };
  await c.regenerateOne(group);
  assert.equal(calls, 3);
  assert.equal(group.versions.length, 4);
  assert.equal(group.selected, 0);
  calls = 0;
  c.fetch = async () => {
    if (++calls === 2) throw new Error('offline');
    return {ok: true, payload: {items: [{index: 1}], output_dir: 'out'}};
  };
  await c.regenerateOne(group);
  assert.equal(calls, 2);
  assert.equal(group.versions.length, 5);
  assert.equal(c.generationInProgress, false);
});

test('settings default to three and validate stored counts', () => {
  const c = harness();
  vm.runInContext(source.slice(source.indexOf('function loadRegenerationCount('), source.indexOf('const modeNotes')), c);
  for (const [stored, expected] of [[null, 3], ['5', 5], ['0', 3], ['21', 3], ['1.5', 3], ['bad', 3]]) {
    c.localStorage = {getItem: () => stored};
    assert.equal(c.loadRegenerationCount(), expected);
  }
  c.localStorage = {getItem() { throw new Error('denied'); }};
  assert.equal(c.loadRegenerationCount(), 3);
});

test('retaining choices submits exact discard IDs and only prunes after success', async () => {
  const c = harness();
  const group = {selected: 1, versions: ['a', 'b', 'c'].map(candidate_id => ({item: {candidate_id}}))};
  c.resultGroups = [group];
  c.fetch = async (url, options) => {
    assert.equal(url, '/api/results/retain');
    assert.deepEqual(JSON.parse(options.body), {groups: [{selected: 'b', discard: ['a', 'c']}]});
    return {ok: false, payload: {detail: 'failed'}};
  };
  await c.retainSelectedCandidates();
  assert.equal(group.versions.length, 3);
  c.fetch = async () => ({ok: true, payload: {removed: 2}});
  await c.retainSelectedCandidates();
  assert.equal(group.versions.length, 1);
  assert.equal(group.versions[0].item.candidate_id, 'b');
  assert.equal(group.selected, 0);
  assert.equal(c.generationInProgress, false);
});
