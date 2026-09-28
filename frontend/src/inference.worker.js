import * as ort from 'onnxruntime-web/wasm'
import { preprocessRGBA, rankLogits, qualityWarnings } from './inference-core'

ort.env.wasm.numThreads = 1
ort.env.wasm.proxy = false
ort.env.wasm.wasmPaths = new URL('/ort/', globalThis.location.href).href
let modelPromise

async function loadModel(progress) {
  const metadataResponse = await fetch('/model/metadata.json', { cache: 'no-cache' })
  if (!metadataResponse.ok) throw new Error('The model could not be downloaded. Check your connection and try again.')
  const metadata = await metadataResponse.json()
  const response = await fetch('/model/' + metadata.model_file + '?v=' + metadata.sha256)
  if (!response.ok) throw new Error('The model could not be downloaded. Check your connection and try again.')
  let bytes
  if (response.body?.getReader) {
    const reader = response.body.getReader(), chunks = []
    let received = 0, previous = -1
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value); received += value.length
      const percent = Math.min(100, Math.round(received / metadata.model_bytes * 100))
      if (percent !== previous) { progress('Downloading the model', percent); previous = percent }
    }
    bytes = new Uint8Array(received)
    let position = 0
    for (const chunk of chunks) { bytes.set(chunk, position); position += chunk.length }
  } else bytes = new Uint8Array(await response.arrayBuffer())
  const checksum = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('')
  if (checksum !== metadata.sha256) throw new Error('The downloaded model is incomplete. Refresh the page and try again.')
  progress('Preparing analysis on your device')
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
  return { session, metadata }
}

globalThis.onmessage = async ({ data }) => {
  const { id, buffer, width, height } = data
  const progress = (key, percent) => globalThis.postMessage({ id, type: 'progress', key, percent })
  let input, outputs
  try {
    modelPromise ??= loadModel(progress).catch(error => { modelPromise = null; throw error })
    const { session, metadata } = await modelPromise
    progress('Analyzing on your device')
    const rgba = new Uint8ClampedArray(buffer)
    input = new ort.Tensor('float32', preprocessRGBA(rgba, width, height, metadata.image_size, metadata.mean, metadata.std), [1, 3, metadata.image_size, metadata.image_size])
    outputs = await session.run({ image: input })
    const ranked = rankLogits(outputs.logits.data, metadata.classes)
    const uncertain = ranked[0].probability < .7 || ranked[0].probability - ranked[1].probability < .15
    const warnings = qualityWarnings(rgba, width, height)
    if (uncertain) warnings.push('The result is uncertain. Compare the possible matches and try another clear photo before taking action.')
    globalThis.postMessage({ id, type: 'result', result: {
      ...metadata.disease_info[ranked[0].disease], disease: ranked[0].disease, confidence: ranked[0].confidence,
      class_index: ranked[0].class_index, top_predictions: ranked.map(({ class_index, disease, confidence }) => ({ class_index, disease, confidence })),
      uncertain, warnings, heatmap: Array.from(outputs.heatmap.data), heatmap_shape: outputs.heatmap.dims,
      model_version: metadata.sha256.slice(0, 12), generated_at: new Date().toISOString(),
    } })
  } catch (error) {
    globalThis.postMessage({ id, type: 'error', message: /download|connection|incomplete/i.test(error.message) ? error.message : 'Analysis could not run on this device. Close other tabs and retry, or use a recent Chrome, Edge, Firefox, or Safari browser.' })
  } finally {
    input?.dispose()
    if (outputs) Object.values(outputs).forEach(tensor => tensor.dispose())
  }
}
