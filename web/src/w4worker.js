import { makeHandler } from './w4worker_core.js';
const h = makeHandler((m, tr) => postMessage(m, tr || []));
onmessage = (e) => h(e.data);
