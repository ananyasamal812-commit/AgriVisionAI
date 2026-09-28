// Pure functions shared by the browser worker and parity tests.
export function preprocessRGBA(rgba, width, height, size, mean, std) {
  if (rgba.length !== width * height * 4) throw new Error('Invalid image dimensions')
  const result = new Float32Array(3 * size * size)
  for (let y = 0; y < size; y++) {
    const sourceY = Math.max(0, Math.min(height - 1, (y + 0.5) * height / size - 0.5))
    const y0 = Math.floor(sourceY), y1 = Math.min(y0 + 1, height - 1), dy = sourceY - y0
    for (let x = 0; x < size; x++) {
      const sourceX = Math.max(0, Math.min(width - 1, (x + 0.5) * width / size - 0.5))
      const x0 = Math.floor(sourceX), x1 = Math.min(x0 + 1, width - 1), dx = sourceX - x0
      for (let channel = 0; channel < 3; channel++) {
        const top = rgba[(y0 * width + x0) * 4 + channel] * (1 - dx) + rgba[(y0 * width + x1) * 4 + channel] * dx
        const bottom = rgba[(y1 * width + x0) * 4 + channel] * (1 - dx) + rgba[(y1 * width + x1) * 4 + channel] * dx
        const pixel = Math.round(top * (1 - dy) + bottom * dy)
        result[channel * size * size + y * size + x] = (pixel / 255 - mean[channel]) / std[channel]
      }
    }
  }
  return result
}

export function rankLogits(logits, classes) {
  if (logits.length !== classes.length || !Array.from(logits).every(Number.isFinite)) throw new Error('Invalid model output')
  const maximum = Math.max(...logits)
  const scores = Array.from(logits, value => Math.exp(value - maximum))
  const total = scores.reduce((sum, value) => sum + value, 0)
  return scores.map((value, class_index) => ({ class_index, disease: classes[class_index],
    probability: value / total, confidence: Math.round(value / total * 10000) / 100 }))
    .sort((a, b) => b.probability - a.probability).slice(0, 3)
}

export function qualityWarnings(rgba, width, height) {
  const warnings = []
  if (Math.min(width, height) < 128) warnings.push('This photo is small. A higher-resolution close-up may give a better result.')
  let sum = 0, count = 0, lap = 0, lapSquared = 0
  const gray = (x, y) => { const i = (y * width + x) * 4; return .299 * rgba[i] + .587 * rgba[i + 1] + .114 * rgba[i + 2] }
  const stride = Math.max(1, Math.floor(Math.max(width, height) / 512))
  for (let y = 1; y < height - 1; y += stride) for (let x = 1; x < width - 1; x += stride) {
    const center = gray(x, y)
    const value = gray(x - 1, y) + gray(x + 1, y) + gray(x, y - 1) + gray(x, y + 1) - 4 * center
    sum += center; count++; lap += value; lapSquared += value * value
  }
  if (count && (sum / count < 35 || sum / count > 230)) warnings.push('Lighting may hide leaf details. Try an evenly lit photo without glare.')
  if (count && lapSquared / count - (lap / count) ** 2 < 15) warnings.push('This photo may lack sharp detail. Try a focused close-up of one leaf.')
  return warnings
}
