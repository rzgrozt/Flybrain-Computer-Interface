// Optional dependency-free Chrome DevTools smoke test for an already running local panel.
// Run: node tests/js/browser_panel_v2_smoke.mjs http://127.0.0.1:8765/
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {setTimeout as sleep} from 'node:timers/promises';

const pageUrl = process.argv[2] || 'http://127.0.0.1:8765/';
if (!/^http:\/\/127\.0\.0\.1:\d+\/$/.test(pageUrl)) {
  throw Error('Browser smoke requires an explicit loopback panel URL.');
}
const directory = await mkdtemp(join(tmpdir(), 'flybrain-browser-v2-'));
const chrome = spawn('/usr/bin/google-chrome', [
  '--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
  '--remote-debugging-port=0', '--user-data-dir=' + directory,
  '--window-size=1600,1000', '--enable-unsafe-swiftshader', 'about:blank',
], {stdio: 'ignore'});
let socket;
let nextId = 0;
const pending = new Map();
const exceptions = [];
try {
  let port;
  for (let attempt = 0; attempt < 80; attempt++) {
    try {port = Number((await readFile(join(directory, 'DevToolsActivePort'), 'utf8')).split('\n')[0]); break;}
    catch {await sleep(100);}
  }
  assert.ok(port, 'Chrome DevTools never started');
  const tabs = await (await fetch('http://127.0.0.1:' + port + '/json')).json();
  const tab = tabs.find(item => item.type === 'page');
  assert.ok(tab, 'Chrome page is missing');
  socket = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, {once: true});
    socket.addEventListener('error', reject, {once: true});
  });
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails.text);
    if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') {
      exceptions.push(message.params.entry.text);
    }
    const entry = pending.get(message.id);
    if (!entry) return;
    pending.delete(message.id);
    if (message.error) entry.reject(Error(message.error.message));
    else entry.resolve(message.result);
  });
  function send(method, params = {}) {
    const id = ++nextId;
    return new Promise((resolve, reject) => {
      pending.set(id, {resolve, reject});
      socket.send(JSON.stringify({id, method, params}));
    });
  }
  async function evaluate(expression) {
    const result = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    if (result.exceptionDetails) throw Error(result.exceptionDetails.text);
    return result.result.value;
  }
  await send('Runtime.enable');
  await send('Log.enable');
  await send('Page.enable');
  await send('Page.navigate', {url: pageUrl});
  let loaded = false;
  for (let i = 0; i < 100; i++) {
    const value = await evaluate("document.getElementById('sandboxSource')?.textContent");
    if (value === 'RECORDED CURSOR · NOT A VM') {loaded = true; break;}
    await sleep(100);
  }
  assert.ok(loaded, 'Recorded virtual cursor did not initialize');
  const initial = await evaluate("({source:document.getElementById('sandboxSource').textContent,vm:document.getElementById('sandboxConnection').textContent,motor:document.getElementById('motorSource').textContent,timeline:document.getElementById('timelineSource').textContent,episode:document.getElementById('timelineEpisode').textContent,step:document.getElementById('replayStepText').textContent,cursor:document.getElementById('virtualCursor').style.left})");
  assert.equal(initial.vm, 'Not connected');
  assert.equal(initial.motor, 'RECORDED READOUT');
  assert.equal(initial.timeline, 'RECORDED STEPS');
  assert.ok(initial.cursor.endsWith('%'));
  await evaluate("document.getElementById('replayNext').click()");
  const next = await evaluate("document.getElementById('replayStepText').textContent");
  assert.notEqual(next, initial.step, 'Next replay step did not advance');
  await evaluate("document.getElementById('modeLive').click()");
  const guest = await evaluate("({source:document.getElementById('sandboxSource').textContent,motor:document.getElementById('motorSource').textContent,timeline:document.getElementById('timelineSource').textContent,empty:document.getElementById('sandboxLiveEmpty').hidden})");
  assert.equal(guest.source, 'VM NOT CONNECTED');
  assert.equal(guest.motor, 'NO LIVE MOTOR FEED');
  assert.equal(guest.timeline, 'NO LIVE EPISODE FEED');
  assert.equal(guest.empty, false);
  await evaluate("document.getElementById('modeReplay').click()");
  const replay = await evaluate("document.getElementById('sandboxSource').textContent");
  assert.equal(replay, 'RECORDED CURSOR · NOT A VM');
  const screenshot = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: false});
  await writeFile('/tmp/flybrain-panel-v2-inspection.png', Buffer.from(screenshot.data, 'base64'));
  // WebGL warnings from unsupported headless GPU are not a JavaScript crash.
  assert.deepEqual(exceptions, [], 'Browser application errors: ' + exceptions.join('; '));
  console.log(JSON.stringify({status: 'PASS', initial, nextStep: next, guest, screenshot: '/tmp/flybrain-panel-v2-inspection.png'}));
} finally {
  if (socket) socket.close();
  chrome.kill('SIGTERM');
  await rm(directory, {recursive:true, force:true});
}
