import { jsPDF } from 'jspdf'
import { translate, classLabel } from './i18n.js'

let fontPromise
async function ensureFont() {
  fontPromise ??= (async () => {
    const font = new FontFace('AgriReport', 'url(/fonts/NotoSansDevanagari-Full.ttf)')
    await font.load(); document.fonts.add(font)
  })().catch(error => { fontPromise = null; throw error })
  return fontPromise
}

const browserCanvas = (width, height) => { const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height; return canvas }
const browserImage = url => new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = url })

// Draw text with the browser's native shaping engine before embedding the page.
// This preserves Hindi conjuncts and vowel marks in the downloadable PDF.
export async function createReportPdf(result, language, options = {}) {
  const canvasFactory = options.canvasFactory || browserCanvas
  const loadImage = options.loadImage || browserImage
  await (options.fontReady || ensureFont)()
  const t = value => translate(value, language)
  const doc = new jsPDF({ unit: 'mm', format: 'a4', compress: true })
  const width = 1240, height = 1754, margin = 70, contentWidth = width - 2 * margin
  let canvas, ctx, y, page = 0
  function startPage() {
    canvas = canvasFactory(width, height); ctx = canvas.getContext('2d')
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height)
    ctx.textBaseline = 'top'; y = margin
  }
  function font(size = 23, color = '#233b31') { ctx.font = `${size}px AgriReport`; ctx.fillStyle = color }
  function lines(value, maxWidth, size = 23) {
    font(size)
    const words = String(value).split(/\s+/), wrapped = []
    let current = ''
    for (const word of words) {
      const next = current ? current + ' ' + word : word
      if (current && ctx.measureText(next).width > maxWidth) { wrapped.push(current); current = word }
      else current = next
    }
    if (current) wrapped.push(current)
    return wrapped
  }
  function finishPage() {
    font(18, '#63776d')
    ctx.fillText('AgriVisionAI | ' + t('Model-generated analysis'), margin, height - 55)
    ctx.textAlign = 'right'; ctx.fillText(t('Page') + ' ' + (page + 1), width - margin, height - 55); ctx.textAlign = 'left'
    if (page++) doc.addPage()
    doc.addImage(canvas.toDataURL('image/png'), 'PNG', 0, 0, 210, 297, undefined, 'FAST')
  }
  function room(amount) { if (y + amount > height - 95) { finishPage(); startPage() } }
  function text(value, size = 23, color = '#233b31', gap = 10) {
    const wrapped = lines(value, contentWidth, size), lineHeight = size * 1.5
    room(wrapped.length * lineHeight + gap); font(size, color)
    for (const line of wrapped) { ctx.fillText(line, margin, y); y += lineHeight }
    y += gap
  }
  function photo(image, x, top, boxWidth, boxHeight) {
    const scale = Math.min(boxWidth / image.width, boxHeight / image.height)
    ctx.drawImage(image, x + (boxWidth - image.width * scale) / 2, top + (boxHeight - image.height * scale) / 2, image.width * scale, image.height * scale)
  }
  startPage()
  text('AgriVisionAI', 46, '#006c43', 0)
  text(t('LEAF ANALYSIS REPORT'), 29, '#233b31', 12)
  text(t('Generated') + ': ' + result.generated_at.replace('T', ' ').slice(0, 19) + ' UTC', 19, '#63776d', 0)
  text(t('Report ID') + ': ' + result.report_id, 19, '#63776d', 18)
  text(classLabel(result.disease, language), 32, '#006c43', 3)
  text(t('Leading prediction confidence') + ': ' + result.confidence.toFixed(2) + '%', 23, '#233b31', 18)
  const [original, heatmap] = await Promise.all([loadImage(result.original), loadImage(result.gradcam)])
  const boxWidth = (contentWidth - 30) / 2
  room(310)
  photo(original, margin, y, boxWidth, 260); photo(heatmap, margin + boxWidth + 30, y, boxWidth, 260)
  y += 270; font(20)
  ctx.fillText(t('Uploaded leaf'), margin, y); ctx.fillText(t('Model attention heatmap'), margin + boxWidth + 30, y); y += 44
  if (result.uncertain) text(t('Uncertain result - compare alternatives and seek confirmation.'), 22, '#8a5714', 6)
  for (const warning of result.warnings || []) text(t(warning), 20, '#8a5714', 5)
  room(280); text(t('Top 3 predictions'), 28, '#006c43', 5)
  for (const prediction of result.top_predictions) {
    const wrapped = lines(classLabel(prediction.disease, language), contentWidth - 155, 23)
    room(Math.max(58, wrapped.length * 33 + 18)); font(23)
    for (let i = 0; i < wrapped.length; i++) ctx.fillText(wrapped[i], margin, y + i * 33)
    ctx.textAlign = 'right'; ctx.fillText(prediction.confidence.toFixed(2) + '%', width - margin, y); ctx.textAlign = 'left'
    y += wrapped.length * 33 + 8
    ctx.fillStyle = '#e4eee6'; ctx.fillRect(margin, y, contentWidth, 7)
    ctx.fillStyle = '#168153'; ctx.fillRect(margin, y, contentWidth * prediction.confidence / 100, 7); y += 20
  }
  text(t('Scores are estimates across all supported classes; the top three need not total 100%.'), 19, '#63776d', 16)
  for (const pair of [['symptoms', 'cause'], ['treatment', 'prevention']]) {
    const entries = pair.map(key => ({ key, body: lines(t(result[key] || 'Not available'), boxWidth - 15, 22) }))
    const rowHeight = Math.max(...entries.map(entry => entry.body.length)) * 33 + 65
    room(rowHeight)
    entries.forEach((entry, index) => {
      const x = margin + index * (boxWidth + 30)
      font(24, '#006c43'); ctx.fillText(t(entry.key[0].toUpperCase() + entry.key.slice(1)), x, y)
      font(22); entry.body.forEach((line, i) => ctx.fillText(line, x, y + 39 + i * 33))
    })
    y += rowHeight
  }
  text(t('This is not a confirmed diagnosis. Unsupported photos can still receive confident predictions. Confirm with a local agricultural expert before treatment. Heatmaps show model attention, not a measured diseased area.'), 19, '#63776d', 12)
  finishPage()
  doc.setProperties({ title: language === 'hi' ? 'AgriVisionAI Hindi leaf report' : 'AgriVisionAI leaf report', subject: 'Model estimates for supported crops', creator: 'AgriVisionAI' })
  return doc
}

export async function downloadReport(result, language) {
  const doc = await createReportPdf(result, language)
  doc.save(`AgriVisionAI-${result.report_id.slice(0, 8)}-${language}.pdf`)
}
