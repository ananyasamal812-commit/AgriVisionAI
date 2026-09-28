import { readFile, writeFile } from 'node:fs/promises'
import assert from 'node:assert/strict'
import * as ort from 'onnxruntime-web/wasm'
import { preprocessRGBA, rankLogits } from '../src/inference-core.js'

const fixtures = new URL('../../outputs/browser-validation/', import.meta.url)
const metadata = JSON.parse(await readFile(new URL('../public/model/metadata.json', import.meta.url), 'utf8'))
const references = JSON.parse(await readFile(new URL('reference.json', fixtures), 'utf8'))
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = new URL('../node_modules/onnxruntime-web/dist/', import.meta.url).href
const model = await readFile(new URL('../public/model/agrivision.onnx', import.meta.url))
const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] })
let maximumLogitError = 0, maximumCamError = 0, maximumPreprocessConfidenceDifference = 0
for (const reference of references) {
  const bytes = await readFile(new URL(`input-${reference.index}.bin`, fixtures))
  const input = new ort.Tensor('float32', new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4), [1, 3, metadata.image_size, metadata.image_size])
  const outputs = await session.run({ image: input })
  const ranked = rankLogits(outputs.logits.data, metadata.classes)
  assert.equal(ranked[0].class_index, reference.class_index)
  reference.logits.forEach((expected, index) => { maximumLogitError = Math.max(maximumLogitError, Math.abs(expected - outputs.logits.data[index])) })
  reference.heatmap.forEach((expected, index) => { maximumCamError = Math.max(maximumCamError, Math.abs(expected - outputs.heatmap.data[index])) })
  const rgba = await readFile(new URL(`rgba-${reference.index}.bin`, fixtures))
  const preprocessed = preprocessRGBA(rgba, reference.width, reference.height, metadata.image_size, metadata.mean, metadata.std)
  const photoInput = new ort.Tensor('float32', preprocessed, [1, 3, metadata.image_size, metadata.image_size])
  const photoOutputs = await session.run({ image: photoInput })
  const photoRanked = rankLogits(photoOutputs.logits.data, metadata.classes)
  assert.equal(photoRanked[0].class_index, reference.class_index, `Photo pipeline prediction changed for image ${reference.index}`)
  maximumPreprocessConfidenceDifference = Math.max(maximumPreprocessConfidenceDifference, Math.abs(photoRanked[0].confidence - ranked[0].confidence))
  input.dispose(); photoInput.dispose()
  Object.values(outputs).forEach(t => t.dispose()); Object.values(photoOutputs).forEach(t => t.dispose())
  console.log('WASM prediction and heatmap verified:', reference.index + 1)
}
assert.ok(maximumLogitError < .001, maximumLogitError)
assert.ok(maximumCamError < .001, maximumCamError)
assert.ok(maximumPreprocessConfidenceDifference < 2, maximumPreprocessConfidenceDifference)
const report = { images: references.length, top1Agreement: '100%', maximumLogitError, maximumCamError, maximumPreprocessConfidenceDifferencePercentagePoints: maximumPreprocessConfidenceDifference }
await writeFile(new URL('wasm-validation.json', fixtures), JSON.stringify(report, null, 2))
await session.release()
console.log(JSON.stringify(report))
