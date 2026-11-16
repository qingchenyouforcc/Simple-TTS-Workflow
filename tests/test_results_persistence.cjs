const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const source = fs.readFileSync('static/app.js', 'utf8');
function setup(snapshot) {
  const c = {FormData, structuredClone, restoringResults: false, resultGroups: [], lastGenerationData: null,
    submitButton: {}, clearResults: {}, regenerateButton: {}, emptyResults: {}, currentView: 'studio',
    renderResultGroups() {}, renderCandidateSummary() {}, setResultState() {}, showToast() {},
    showView(view) {c.currentView = view;},
    resultStorage: {load: async () => snapshot, save: async value => {c.saved = value;}},
  };
  vm.createContext(c);
  vm.runInContext(source.slice(source.indexOf('function persistResults('), source.indexOf('function bindEvents(')), c);
  return c;
}
test('round trip restores all candidates, selection, text and reference file', async () => {
  const c = setup();
  c.resultGroups = [{index: 1, selected: 1, versions: [{number: 1}, {number: 2}]}];
  c.lastGenerationData = new FormData();
  c.lastGenerationData.set('texts', 'hello');
  c.lastGenerationData.set('ref_audio', new Blob(['reference']), 'voice.wav');
  c.persistResults();
  c.resultGroups[0].selected = 0;
  assert.equal(c.saved.groups[0].selected, 1);
  const reopened = setup(c.saved);
  await reopened.restoreSavedResults();
  assert.equal(reopened.resultGroups[0].versions.length, 2);
  assert.equal(reopened.resultGroups[0].selected, 1);
  assert.equal(reopened.lastGenerationData.get('texts'), 'hello');
  assert.equal(await reopened.lastGenerationData.get('ref_audio').text(), 'reference');
  assert.equal(reopened.currentView, 'results');
  assert.equal(reopened.submitButton.disabled, false);
});
test('cleared results stay empty and storage failures release controls', async () => {
  const c = setup({version: 1, groups: [], generationEntries: null});
  await c.restoreSavedResults();
  assert.equal(c.currentView, 'studio');
  c.resultStorage.load = async () => {throw new Error('blocked');};
  await c.restoreSavedResults();
  assert.equal(c.restoringResults, false);
  assert.equal(c.submitButton.disabled, false);
  c.persistResults();
  assert.equal(c.saved.groups.length, 0);
});
