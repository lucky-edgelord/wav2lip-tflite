// GPU correctness check: runs one real Wav2Lip sample (speech and silence) on CPU and on
// WebGPU at different precisions / mel layouts, and reports whether the GPU output follows
// the audio. Open gpu-check.html on the device to test.
import { loadLiteRt, loadAndCompile, Tensor } from '@litertjs/core';

const out = document.getElementById('out');
const say = (s) => { out.textContent += `${s}\n`; };
const base = (p) => new URL(p, location.href).href;
const f32 = async (p) => new Float32Array(await (await fetch(base(p), { cache: 'no-store' })).arrayBuffer());

function stats(a, b) { // mean |a-b| over all pixels and over the mouth region
  let all = 0, mouth = 0, n = 0;
  for (let y = 0; y < 96; y++) for (let x = 0; x < 96; x++) for (let c = 0; c < 3; c++) {
    const i = (y * 96 + x) * 3 + c, d = Math.abs(a[i] - b[i]);
    all += d; if (y >= 60 && y < 85 && x >= 30 && x < 66) { mouth += d; n++; }
  }
  return { all: all / (96 * 96 * 3), mouth: mouth / n };
}

async function run(model, accel, mel, face) {
  const ins = model.getInputDetails();
  const t = [];
  t[ins.find((d) => d.shape[3] === 1 || d.shape[1] === 80).index] = new Tensor(mel, [1, 80, 16, 1]);
  t[ins.find((d) => d.shape[3] === 6).index] = new Tensor(face, [1, 96, 96, 6]);
  const moved = accel === 'webgpu' ? await Promise.all(t.map((x) => x.moveTo('webgpu'))) : t;
  const s = performance.now();
  const outs = await model.run(moved);
  const host = await outs[0].moveTo('wasm');
  const r = host.toTypedArray().slice();
  const ms = performance.now() - s;
  host.delete(); moved.forEach((x) => x.delete());
  return { r, ms };
}

(async () => {
  try {
    say(`UA: ${navigator.userAgent}`);
    const adapter = await navigator.gpu?.requestAdapter();
    say(`WebGPU: ${adapter ? `${adapter.info?.vendor} ${adapter.info?.architecture}` : 'none'}`);
    await loadLiteRt(base('./litert/'));
    const bytes = new Uint8Array(await (await fetch(base('./assets/wav2lip_gan_b1_int8w.tflite'))).arrayBuffer());
    const mel = await f32('./test/mel.f32'), face = await f32('./test/face.f32');
    const expect = await f32('./test/expect.f32'), expectSilent = await f32('./test/expect_silent.f32');
    const silent = new Float32Array(mel.length).fill(-4);
    const ref = stats(expect, expectSilent);
    say(`Reference (Python CPU): speech vs silence differ by ${ref.all.toFixed(4)} overall, ${ref.mouth.toFixed(4)} in the mouth\n`);

    const configs = [
      ['CPU (wasm)', { accelerator: 'wasm' }, 'wasm'],
      ['WebGPU default precision', { accelerator: 'webgpu' }, 'webgpu'],
      ['WebGPU fp32', { accelerator: 'webgpu', gpuOptions: { precision: 'fp32' } }, 'webgpu'],
      ['WebGPU fp16', { accelerator: 'webgpu', gpuOptions: { precision: 'fp16' } }, 'webgpu'],
    ];
    for (const [name, opts, acc] of configs) {
      if (acc === 'webgpu' && !adapter) { say(`${name}: skipped (no WebGPU)`); continue; }
      try {
        const m = await loadAndCompile(bytes, opts);
        const a = await run(m, acc, mel, face);
        const b = await run(m, acc, silent, face);
        const vsExpect = stats(a.r, expect), follows = stats(a.r, b.r);
        const verdict = follows.mouth > ref.mouth * 0.3 && vsExpect.all < 0.02 ? 'OK' : follows.mouth < ref.mouth * 0.1 ? 'BROKEN: ignores audio' : 'SUSPECT';
        say(`${name}: ${verdict}\n  error vs reference ${vsExpect.all.toFixed(4)} (mouth ${vsExpect.mouth.toFixed(4)})\n  speech vs silence ${follows.all.toFixed(4)} (mouth ${follows.mouth.toFixed(4)})\n  ${a.ms.toFixed(0)} ms/run`);
        m.delete?.();
      } catch (e) { say(`${name}: ERROR ${e.message}`); }
    }
    say('\nDone.');
    document.title = 'done';
  } catch (e) { say(`FAILED: ${e.stack || e}`); document.title = 'done'; }
})();
