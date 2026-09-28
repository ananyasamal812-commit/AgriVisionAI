import { mkdir, copyFile, readdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = fileURLToPath(new URL('../', import.meta.url))
const runtime = path.join(root, 'node_modules/onnxruntime-web/dist')
const destination = path.join(root, 'public/ort')
await mkdir(destination, { recursive: true })
// Ship the non-JSEP WASM backend only; all files must match the installed JS version.
for (const name of await readdir(runtime)) {
  if (/^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(name)) await copyFile(path.join(runtime, name), path.join(destination, name))
}
