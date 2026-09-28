import hindi from './locales/hi.json' with { type: 'json' }
export function translate(text, language) {
 if (language !== 'hi' || typeof text !== 'string') return text
 if (hindi[text]) return hindi[text]
 if (text.endsWith(' fungus.')) return text.replace(' fungus.', ' फफूंद।')
 if (text.endsWith(' bacteria.')) return text.replace(' bacteria.', ' जीवाणु।')
 return text
}
export function classLabel(value, language) {
 const [crop, condition] = value.split('___')
 const cropName = crop.replace('_(including_sour)', '').replace('_(maize)', '').replace('Pepper,_bell', 'Pepper').replaceAll('_', ' ').trim()
 const disease = (condition || '').replaceAll('_', ' ').trim()
 return translate(cropName, language) + (disease ? ' · ' + translate(disease, language) : '')
}

