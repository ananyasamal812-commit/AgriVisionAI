import { useEffect, useRef, useState } from 'react'
import { analyzeLeaf } from './browser-analysis'
import { translate, classLabel } from './i18n'

const crops = ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 'Pepper', 'Potato', 'Raspberry', 'Soybean', 'Squash', 'Strawberry', 'Tomato']
function Icon({ name = 'leaf', ...props }) {
  const paths = {
    leaf: <><path d="M20 4c-9-1-16 2-16 9a7 7 0 0 0 7 7c7 0 10-7 9-16Z" /><path d="m4 20 11-11M9 15v-5m0 5h5" /></>,
    upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5" /></>,
    arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
    download: <><path d="M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4" /></>,
    camera: <><path d="m4 6 3 0 2-3h6l2 3h3v15H4Z" /><circle cx="12" cy="13" r="4" /></>,
    check: <path d="m5 12 4 4L19 6" />,
  }
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>
}

export default function App() {
  const [language, setLanguage] = useState(() => {
    try { return localStorage.getItem('agrivision-language') === 'hi' ? 'hi' : 'en' }
    catch { return 'en' }
  })
  const t = value => translate(value, language)
  useEffect(() => {
    document.documentElement.lang = language
    try { localStorage.setItem('agrivision-language', language) } catch { /* Storage may be unavailable. */ }
  }, [language])
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [downloading, setDownloading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [view, setView] = useState('heatmap')
  const [elapsed, setElapsed] = useState(0)
  const [progress, setProgress] = useState(null)
  const [imageError, setImageError] = useState(false)
  const fileInput = useRef(null)
  const cameraInput = useRef(null)

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])
  useEffect(() => {
    if (!loading) return
    const timer = setInterval(() => setElapsed(value => value + 1), 1000)
    return () => clearInterval(timer)
  }, [loading])

  function choose(file) {
    if (!file || loading) return
    setError('')
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) &&
      !/\.(jpe?g|jfif|png|webp)$/i.test(file.name)) {
      setError(t("Choose a JPEG, PNG, or WebP photo. Convert HEIC photos to JPEG first."))
      return
    }
    if (!file.size || file.size > 10 * 1024 * 1024) {
      setError(t("Choose a non-empty image smaller than 10 MB."))
      return
    }
    setImage(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setImageError(false)
    setView('heatmap')
  }

  async function analyze() {
    if (!image || loading) return
    setLoading(true); setElapsed(0); setError(''); setResult(null); setImageError(false)
    setProgress(null)
    try {
      const data = await analyzeLeaf(image, setProgress)
      setResult(data)
    } catch (err) {
      setError(err.message || 'Analysis could not run on this device. Close other tabs and retry, or use a recent Chrome, Edge, Firefox, or Safari browser.')
    } finally { setLoading(false) }
  }

  async function download() {
    if (!result || downloading) return
    setDownloading(true); setError('')
    try {
      const { downloadReport } = await import('./browser-report')
      await downloadReport(result, language)
    } catch { setError('The report could not be created. Check your connection and try downloading again.') }
    finally { setDownloading(false) }
  }

  return <div className="app-shell">
    <header className="site-header">
      <a href="#" className="brand" aria-label="AgriVisionAI home"><span className="brand-mark"><Icon /></span>AgriVision<span>AI</span></a>
      <nav aria-label="Main navigation"><div className="language-switch" aria-label="Language / भाषा"><button onClick={() => setLanguage('en')} aria-pressed={language === 'en'} lang="en">EN</button><button onClick={() => setLanguage('hi')} aria-pressed={language === 'hi'} lang="hi">हिन्दी</button></div><a href="#how-it-works">{t("How it works")}</a><a href="#supported-crops">{t("Supported crops")}</a><a className="nav-action" href="#analyze">{t("Analyze a leaf")} <Icon name="arrow" /></a></nav>
    </header>
    <main>
      <section className="hero">
        <div><p className="eyebrow"><span /> {t("A clearer picture of plant health")}</p>
          <h1>{t("Healthy plants start")}<br />{t("with a closer")} <em>{t("look.")}</em></h1>
          <p className="hero-copy">{t("Turn a leaf photo into useful insights. Explore possible conditions, understand the result, and take the next step with confidence.")}</p>
          <div className="hero-tags"><span><Icon name="check" /> {t("No sign-up")}</span><span><Icon name="check" /> {t("Photo to PDF")}</span><span><Icon name="check" /> {t("Photo guidance")}</span></div>
        </div>
        <aside className="field-note"><span className="note-icon"><Icon width="42" height="42" /></span><p className="eyebrow">{t("A better photo. A better starting point.")}</p><h2>{t("One leaf.")}<br />{t("Natural light.")}<br />{t("A clear view.")}</h2><p>{t("Fill the frame with a single leaf and keep the background simple.")}</p><span className="note-line" /></aside>
      </section>

      <section id="analyze" className="workspace">
        <div className="upload-column">
          <article className="panel upload-panel">
            <div className="panel-heading"><span className="step-number">01</span><div><p className="eyebrow">{t("Start here")}</p><h2>{t("Your leaf photo")}</h2></div></div>
            <input ref={fileInput} id="leaf-file" className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" disabled={loading} onChange={event => { choose(event.target.files?.[0]); event.target.value = '' }} />
            <input ref={cameraInput} className="visually-hidden" aria-label="Take a leaf photo" type="file" accept="image/*" capture="environment" disabled={loading} onChange={event => { choose(event.target.files?.[0]); event.target.value = '' }} />
            <div className={'drop-zone ' + (dragging ? 'is-dragging ' : '') + (preview ? 'has-photo' : '')}
              onDragOver={event => { event.preventDefault(); if (!loading) setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={event => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files?.[0]) }}>
              {preview ? <img src={preview} alt="Leaf selected for analysis" /> : <>
                <span className="upload-icon"><Icon name="upload" width="28" height="28" /></span>
                <h3>{t("Drop your leaf photo here")}</h3><p>{t("or choose an image from your device")}</p>
                <button className="button button-secondary" disabled={loading} onClick={() => fileInput.current?.click()}>{t("Choose photo")}</button>
                <small>{t("JPG, PNG, WebP · Up to 10 MB")}</small>
              </>}
            </div>
            <div className="photo-actions">
              {image ? <><span className="file-name" title={image.name}>{image.name}</span><button className="text-button" disabled={loading} onClick={() => fileInput.current?.click()}>{t("Change photo")}</button></> :
                <button className="text-button camera-button" disabled={loading} onClick={() => cameraInput.current?.click()}><Icon name="camera" /> {t("Take a photo")}</button>}
            </div>
            <button className="button button-primary analyze-button" disabled={!image || loading} onClick={analyze}>
              {loading ? <><span className="spinner" /> {t("Analyzing your leaf...")}</> : <>{t("Analyze leaf")} <Icon name="arrow" /></>}
            </button>
            {loading && <p className="loading-note" role="status">{t(progress?.key || 'Checking your photo')}{progress?.percent !== undefined ? ` (${progress.percent}%)` : ''}{elapsed > 15 ? ' — ' + t('The first analysis can take longer while your device loads the model.') : ''}</p>}
            {error && <p role="alert" className="error-message">{t(error)}</p>}
            <p className="privacy-note">{t('Analysis runs in your browser. Your photo is not uploaded. Save your PDF before closing or refreshing this page.')}</p>
          </article>
          <aside className="photo-tip"><Icon /><p><strong>{t("Keep it close and clear.")}</strong> {t("Avoid screenshots, collages, and several leaves in one photo.")}</p></aside>
        </div>

        <article className="panel result-panel" aria-live="polite" aria-busy={loading}>
          <div className="panel-heading"><span className="step-number">02</span><div><p className="eyebrow">{t("Understand your plant")}</p><h2>{t("Leaf insights")}</h2></div>{result && <span className="complete-badge"><Icon name="check" /> {t("Complete")}</span>}</div>
          {!result ? <div className="empty-result"><div className={'leaf-orbit ' + (loading ? 'is-loading' : '')}><Icon width="50" height="50" /></div><h3>{loading ? t("Taking a closer look") : t("A little insight goes a long way.")}</h3><p>{loading ? t("We are comparing the photo with supported conditions and creating your report.") : t("Upload a photo to see possible matches, a visual heatmap, and helpful care information.")}</p><div className="result-preview-tags"><span>{t("Possible matches")}</span><span>{t("Visual explanation")}</span><span>{t("PDF report")}</span></div></div> : <>
            <div className={'diagnosis ' + (result.uncertain ? 'uncertain' : '')}><p className="eyebrow">{result.uncertain ? t("Needs a closer look") : t("Leading possible match")}</p><h3>{classLabel(result.disease, language)}</h3><div className="confidence-row"><span>{t("Model confidence")}</span><strong>{result.confidence.toFixed(2)}%</strong></div><div className="score-track"><span style={{ width: result.confidence + '%' }} /></div></div>
            {result.warnings?.length > 0 && <div className="result-notice">{result.warnings.map(warning => <p key={t(warning)}>{t(warning)}</p>)}</div>}
            <div className="matches"><h4>{t("Top 3 possible matches")}</h4>{result.top_predictions.map((item, index) => <div className="match-row" key={item.class_index}><span className="match-rank">0{index + 1}</span><div><span>{classLabel(item.disease, language)}</span><div className="score-track small"><span style={{ width: item.confidence + '%' }} /></div></div><strong>{item.confidence.toFixed(2)}%</strong></div>)}<p className="fine-print">{t("Model scores are estimates, not a confirmed diagnosis.")}</p></div>
            <div className="care-grid">{['symptoms', 'cause', 'treatment', 'prevention'].map(key => <div key={key}><h4>{key === 'treatment' ? t("Care guidance") : t(key)}</h4><p>{t(result[key]) || t("Information is not available for this condition.")}</p></div>)}</div>
            <div className="visual-heading"><h4>{t("A closer look")}</h4><div className="segmented-control" aria-label="Image view">{['original', 'heatmap'].map(option => <button key={option} aria-pressed={view === option} className={view === option ? 'selected' : ''} onClick={() => setView(option)}>{option === 'original' ? t("Original") : t("Heatmap")}</button>)}</div></div>
            <figure className="result-visual">{imageError && view === 'heatmap' ? <p>{t("Heatmap unavailable. Analyze the photo again to refresh it.")}</p> : <img src={view === 'original' ? result.original : result.gradcam} onError={() => setImageError(true)} alt={view === 'original' ? 'Original leaf photograph' : 'Regions that influenced the prediction'} />}<figcaption>{t("Heatmap colors show model attention, not the severity of disease.")}</figcaption></figure>
            <button className="button button-secondary report-button" disabled={downloading} onClick={download}><Icon name="download" />{downloading ? t("Preparing download...") : t("Download PDF report")}</button>
            <p className="result-disclaimer">{t("For supported crops only. Ask a local agricultural expert to confirm the condition before treatment.")}</p>
          </>}
        </article>
      </section>

      <section id="how-it-works" className="how-section"><p className="eyebrow">{t("Simple from the first photo")}</p><h2>{t("From leaf to insight, in three steps.")}</h2><div className="how-grid">{[['01', t("Choose a clear photo"), t("Photograph one leaf in natural light, or upload a saved photo.")], ['02', t("Explore possible matches"), t("Review the leading results and the visual explanation together.")], ['03', t("Keep your findings"), t("Download your report to revisit it or discuss it with an expert.")]].map(([n, title, copy]) => <article key={n}><span>{n}</span><h3>{title}</h3><p>{copy}</p></article>)}</div></section>
      <section id="supported-crops" className="crops-section"><div><p className="eyebrow">{t("Before you upload")}</p><h2>{t("Made for these crops.")}</h2><p>{t("38 supported crop and condition combinations. Photos of other plants may produce an incorrect match.")}</p></div><div className="crop-pills">{crops.map(crop => <span key={t(crop)}>{t(crop)}</span>)}</div></section>
    </main>
    <footer className="site-footer"><span className="brand"><Icon /> AgriVisionAI</span><p>{t("Helpful insights. Thoughtful plant care.")}</p><a href="#analyze">{t("Back to analysis ↑")}</a></footer>
  </div>
}

