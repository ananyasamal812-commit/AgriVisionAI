import axios from 'axios'

const origin = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
export const api = axios.create({ baseURL: origin || '/api', timeout: 180000 })
export const assetUrl = path => origin ? origin + '/' + path.replace(/^\//, '') : '/' + path.replace(/^\//, '')
export const displayClass = value => value.replaceAll('___', ' · ').replaceAll('_', ' ')

