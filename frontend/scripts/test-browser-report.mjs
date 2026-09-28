import { createCanvas, loadImage, GlobalFonts } from '@napi-rs/canvas'
import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import assert from 'node:assert/strict'
import { createReportPdf } from '../src/browser-report.js'
import { renderHeatmap } from '../src/browser-analysis.js'

const fixtures = new URL('../../outputs/browser-validation/', import.meta.url)
const metadata = JSON.parse(await readFile(new URL('../public/model/metadata.json', import.meta.url), 'utf8'))
const reference = JSON.parse(await readFile(new URL('reference.json', fixtures), 'utf8'))[0]
assert.ok(GlobalFonts.registerFromPath(fileURLToPath(new URL('../public/fonts/NotoSansDevanagari-Full.ttf', import.meta.url)), 'AgriReport'))
const canvas = createCanvas(reference.width, reference.height), ctx = canvas.getContext('2d')
const image = ctx.createImageData(reference.width, reference.height)
image.data.set(await readFile(new URL('rgba-0.bin', fixtures))); ctx.putImageData(image, 0, 0)
const disease = metadata.classes[reference.class_index]
const result = { ...metadata.disease_info[disease], disease, report_id: 'browser-validation-0001', generated_at: new Date().toISOString(),
  original: canvas.toDataURL('image/png'), gradcam: renderHeatmap(canvas, reference.heatmap, reference.heatmap_shape, () => createCanvas(1, 1)),
  confidence: 99.97, warnings: [], uncertain: false,
  top_predictions: [{ disease, confidence: 99.97 }, { disease: metadata.classes[1], confidence: .02 }, { disease: metadata.classes[2], confidence: .01 }] }
for (const language of ['en', 'hi']) for (const uncertain of [false, true]) {
  const data = uncertain ? { ...result, uncertain: true, confidence: 52.5,
    top_predictions: result.top_predictions.map((p, i) => ({ ...p, confidence: [52.5, 30, 12][i] })),
    warnings: ['This photo is small. A higher-resolution close-up may give a better result.', 'Lighting may hide leaf details. Try an evenly lit photo without glare.', 'This photo may lack sharp detail. Try a focused close-up of one leaf.', 'The result is uncertain. Compare the possible matches and try another clear photo before taking action.'] } : result
  const doc = await createReportPdf(data, language, { canvasFactory: createCanvas, loadImage, fontReady: async () => {} })
  const bytes = new Uint8Array(doc.output('arraybuffer'))
  assert.ok(bytes.length > 5000)
  assert.ok(doc.getNumberOfPages() >= 1 && doc.getNumberOfPages() <= 3)
  const filename = `browser-report-${language}${uncertain ? '-uncertain' : ''}.pdf`
  await writeFile(new URL(filename, fixtures), bytes)
  console.log(filename, doc.getNumberOfPages(), 'pages', bytes.length, 'bytes')
}
