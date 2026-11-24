import { parentPort } from 'node:worker_threads';
import { makeHandler } from '../src/w4worker_core.js';
const h = makeHandler((m, tr) => parentPort.postMessage(m, tr || []));
parentPort.on('message', h);
