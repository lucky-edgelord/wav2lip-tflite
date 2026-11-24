import { Core } from './w4core.js';
export function makeHandler(post) {
  let core = null;
  return (msg) => {
    try {
      if (msg.t === 'init') { core = new Core(msg.module, msg.memMB); post({ t: 'ready' }); }
      else if (msg.t === 'load') { core.load(msg.slot, msg.bytes, msg.maxJobs); post({ t: 'loaded' }); }
      else if (msg.t === 'run') {
        core.prepare(msg.slot, msg.inputs, msg.n);
        for (let j = 0; j < msg.n; j++) core.job(msg.slot, j);
        const outputs = core.collect(msg.slot, msg.n); post({ t: 'done', outputs }, outputs.map((o) => o.buffer));
      }
    } catch (e) { post({ t: 'error', error: String((e && e.stack) || e) }); }
  };
}
