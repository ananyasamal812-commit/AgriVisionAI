import { useEffect, useState } from 'react'
import axios from 'axios'

export default function Evaluation() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    axios.get('/api/evaluation', { signal: controller.signal, timeout: 15000 })
      .then(response => { setData(response.data); setError('') })
      .catch(err => { if (!axios.isCancel(err)) setError('Evaluation information could not be loaded. Check that the updated app is running.') })
    return () => controller.abort()
  }, [attempt])
  return <section className="mx-auto max-w-6xl rounded-3xl bg-white p-6 shadow-lg sm:p-8">
    <h2 className="text-3xl font-bold text-green-800">Model evaluation</h2>
    <p className="mt-3 text-gray-600">Recorded performance, supported classes, and what still needs testing.</p>
    {error ? <div role="alert" className="mt-6 text-red-800">{error}<button className="ml-3 underline" onClick={() => setAttempt(value => value + 1)}>Retry</button></div> : !data ?
      <p className="mt-6" role="status">Loading evaluation...</p> : <>
        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          <div className="rounded-2xl bg-green-50 p-5"><p className="text-sm text-gray-600">Recorded validation accuracy</p><p className="mt-2 text-3xl font-bold text-green-800">{data.validation_accuracy === null ? 'Not recorded' : data.validation_accuracy + '%'}</p><p className="mt-2 text-sm text-gray-600">Used for checkpoint selection</p></div>
          <div className="rounded-2xl bg-gray-50 p-5"><p className="text-sm text-gray-600">Independent test accuracy</p><p className="mt-2 text-2xl font-bold">Not measured</p><p className="mt-2 text-sm text-gray-600">No verified held-out evaluation available</p></div>
          <div className="rounded-2xl bg-gray-50 p-5"><p className="text-sm text-gray-600">Supported classes</p><p className="mt-2 text-3xl font-bold">{data.classes.length}</p><p className="mt-2 text-sm text-gray-600">Crop and disease combinations</p></div>
        </div>
        <h3 className="mt-8 text-xl font-bold">Where these numbers come from</h3>
        <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
          {[['Model', data.model], ['Input size', data.image_size + ' × ' + data.image_size], ['Recorded epoch', data.recorded_epoch ?? 'Not recorded'], ['Evaluation split', data.split], ['Source', data.source]].map(([label, value]) => <div key={label}><dt className="font-semibold text-gray-800">{label}</dt><dd className="mt-1 break-words text-gray-600">{value}</dd></div>)}
        </dl>
        <div className="mt-8 rounded-2xl border border-amber-200 bg-amber-50 p-5">
          <h3 className="text-xl font-bold">Limitations</h3>
          <ul className="mt-3 list-disc space-y-2 pl-5 text-gray-700">{data.limitations.map(item => <li key={item}>{item}</li>)}</ul>
        </div>
        <details className="mt-8 rounded-2xl border border-green-100 p-5"><summary className="cursor-pointer font-semibold">View all supported classes</summary>
          <ul className="mt-4 grid gap-2 text-sm text-gray-700 sm:grid-cols-2">{data.classes.map(item => <li key={item}>{item.replaceAll('___', ' · ').replaceAll('_', ' ')}</li>)}</ul>
        </details>
      </>}
  </section>
}
