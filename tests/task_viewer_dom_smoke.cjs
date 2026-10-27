// Offline DOM smoke test of the shipped page script; no browser/network/device.
// Usage: node tests/task_viewer_dom_smoke.cjs <exported-demo.html>
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.textContent = '';
    this.attributes = {};
  }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  add(child) { this.children.push(child); }
  querySelector(tag) { return this.children.find(child => child.tag === tag) || null; }
  addEventListener() {}
  removeAttribute(name) { delete this.attributes[name]; }
}

const html = fs.readFileSync(process.argv[2], 'utf8');
const initial = html.match(/<script type="application\/json" id="initial-state">([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const elements = new Map();
for (const match of html.matchAll(/\bid="([^"]+)"/g)) elements.set(match[1], new Element());
elements.get('initial-state').textContent = initial;
const document = {
  getElementById(id) {
    assert.ok(elements.has(id), `Missing element ${id}`);
    return elements.get(id);
  },
  createElement(tag) { return new Element(tag); },
};
const context = vm.createContext({
  document, URLSearchParams, location: {search: ''},
  Option: class extends Element { constructor(label, value) { super('option'); this.textContent = label; this.value = value; } },
  setTimeout() { throw new Error('Offline export must not poll'); },
  clearTimeout() {},
  fetch() { throw new Error('Offline export must not access network'); },
});
vm.runInContext(script, context);
assert.equal(elements.get('task').textContent, '把藍色方塊放進盒子');
assert.equal(elements.get('motion-count').textContent, '0');
assert.equal(elements.get('success').textContent, '任務成功：未確認');
assert.equal(elements.get('coordinates').children[0].children[3].textContent, '+5.00');
assert.ok(elements.get('caption-front_rgb').textContent.includes('離線'));

// Exercise preview rendering with fabricated health; no image URL is requested.
vm.runInContext(`
initial.preview.streams = {front_rgb:'mock:front', right_rgb:'mock:right', wrist_rgb:'mock:wrist'};
const health = {available:true, cameras:{}};
roles.forEach(role=>health.cameras[role]={ok:true,mapping_verified:true,fps:29.7,age_s:0.02});
showPreview(health);
`, context);
assert.equal(elements.get('image-front_rgb').querySelector('img').src, 'mock:front');
assert.ok(elements.get('caption-front_rgb').textContent.includes('29.7 FPS'));
vm.runInContext('draw({...snapshot, current:null})', context);
assert.equal(elements.get('content').hidden, true);
assert.ok(elements.get('image-front_rgb').querySelector('img'), 'Cameras independent of task');
vm.runInContext('showPreview({available:false,cameras:{}})', context);
assert.equal(elements.get('image-front_rgb').querySelector('img'), null);
assert.ok(elements.get('caption-front_rgb').textContent.includes('沒有可用'));
console.log('PASS: HTML script, task rendering, preview mapping, disconnect, and zero network');
