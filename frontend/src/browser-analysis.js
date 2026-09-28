let worker
let busy = false

export async function decodePhoto(file) {
  if (!file.size || file.size > 10 * 1024 * 1024) throw new Error('Choose a non-empty image smaller than 10 MB.')
  const header = new Uint8Array(await file.slice(0, 16).arrayBuffer())
  const jpeg = header[0] === 255 && header[1] === 216 && header[2] === 255
  const png = header[0] === 137 && header[1] === 80 && header[2] === 78 && header[3] === 71
  const webp = String.fromCharCode(...header.slice(0, 4)) === 'RIFF' && String.fromCharCode(...header.slice(8, 12)) === 'WEBP'
  if (!jpeg && !png && !webp) throw new Error('This file is not a valid image.')
  let bitmap
  try { bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' }) }
  catch { throw new Error('This file is not a valid image.') }
  try {
    if (bitmap.width * bitmap.height > 20_000_000) throw new Error('Image must contain no more than 20 million pixels.')
    const scale = Math.min(1, 2048 / Math.max(bitmap.width, bitmap.height))
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(bitmap.width * scale)); canvas.height = Math.max(1, Math.round(bitmap.height * scale))
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    return canvas
  } finally { bitmap.close() }
}

export function renderHeatmap(original, values, shape, canvasFactory = () => document.createElement('canvas')) {
  const rows = shape.at(-2), columns = shape.at(-1)
  const small = canvasFactory(); small.width = columns; small.height = rows
  const smallContext = small.getContext('2d'), pixels = smallContext.createImageData(columns, rows)
  for (let i = 0; i < values.length; i++) {
    const value = Math.max(0, Math.min(255, Math.round(values[i] * 255)))
    pixels.data.set([value, value, value, 255], i * 4)
  }
  smallContext.putImageData(pixels, 0, 0)
  const scale = Math.min(1, 1000 / Math.max(original.width, original.height))
  const canvas = canvasFactory(); canvas.width = Math.max(1, Math.round(original.width * scale)); canvas.height = Math.max(1, Math.round(original.height * scale))
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height)
  const mask = ctx.getImageData(0, 0, canvas.width, canvas.height)
  ctx.drawImage(original, 0, 0, canvas.width, canvas.height)
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height)
  for (let i = 0; i < image.data.length; i += 4) {
    const v = mask.data[i] / 255
    const jet = [3, 2, 1].map(offset => 255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - offset))))
    for (let c = 0; c < 3; c++) image.data[i + c] = Math.round(.5 * image.data[i + c] + .5 * jet[c])
  }
  ctx.putImageData(image, 0, 0)
  return canvas.toDataURL('image/png')
}

export async function analyzeLeaf(file, onProgress) {
  if (busy) throw new Error('An analysis is already running.')
  busy = true
  try {
    onProgress({ key: 'Checking your photo' })
    const original = await decodePhoto(file)
    const image = original.getContext('2d').getImageData(0, 0, original.width, original.height)
    worker ??= new Worker(new URL('./inference.worker.js', import.meta.url), { type: 'module' })
    const id = crypto.randomUUID()
    const result = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => { worker?.terminate(); worker = null; reject(new Error('Analysis took too long on this device. Close other tabs and try again.')) }, 240_000)
      const finish = (callback, value) => { clearTimeout(timeout); if (worker) { worker.onmessage = null; worker.onerror = null } callback(value) }
      worker.onmessage = ({ data }) => {
        if (data.id !== id) return
        if (data.type === 'progress') onProgress({ key: data.key, percent: data.percent })
        else if (data.type === 'result') finish(resolve, data.result)
        else finish(reject, new Error(data.message))
      }
      worker.onerror = () => { const current = worker; worker = null; current.terminate(); finish(reject, new Error('Analysis could not run on this device. Close other tabs and retry, or use a recent Chrome, Edge, Firefox, or Safari browser.')) }
      worker.postMessage({ id, buffer: image.data.buffer, width: image.width, height: image.height }, [image.data.buffer])
    })
    return { ...result, report_id: id, original: original.toDataURL('image/jpeg', .9),
      gradcam: renderHeatmap(original, result.heatmap, result.heatmap_shape) }
  } finally { busy = false }
}
