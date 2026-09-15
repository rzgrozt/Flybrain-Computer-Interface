import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';

const atlas = new URL(
  '../../../src/flybrain_interface/panel/static/brain-atlas/',
  import.meta.url,
);
const manifest = JSON.parse(await readFile(new URL('manifest.json', atlas), 'utf8'));

for (const [name, expected] of Object.entries(manifest.exportSha256)) {
  const bytes = await readFile(new URL(name, atlas));
  const actual = createHash('sha256').update(bytes).digest('hex');
  if (actual !== expected) throw Error(`Atlas checksum mismatch: ${name}`);
}

console.log('MaleCNS anatomical atlas hashes verified.');
