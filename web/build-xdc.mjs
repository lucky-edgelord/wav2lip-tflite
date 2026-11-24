// Builds single/wav2lip-int4.xdc: the single-file page packed as a webxdc app (zip with index.html, manifest.toml, icon.png).
import { readFileSync, writeFileSync, mkdirSync, rmSync } from 'fs';
import { execSync } from 'child_process';
const dir = 'single/xdc'; rmSync(dir, { recursive: true, force: true }); mkdirSync(dir, { recursive: true });
let html = readFileSync('single/w2l_int4.html', 'utf8');
html = html.replace(/<script>\n\/\/ Enable cross-origin isolation[\s\S]*?<\/script>/, '')            // no service worker in webxdc
  .replace(/<link rel="preconnect"[^>]*>\n?/g, '').replace(/<link href="https:\/\/fonts[^>]*>\n?/g, '') // no network in webxdc
  .replace('<link rel="icon" href="data:,">', '<link rel="icon" href="data:,">\n<script src="webxdc.js"></script>');
writeFileSync(`${dir}/index.html`, html);
writeFileSync(`${dir}/manifest.toml`, 'name = "Wav2Lip INT4"\nsource_code_url = "https://github.com/lucky-edgelord/wav2lip-tflite"\n');
execSync(`python3 -c "
from PIL import Image, ImageDraw
S=256; im=Image.new('RGB',(S,S),(7,6,13)); d=ImageDraw.Draw(im)
for i in range(0,S,16): d.line([(i,0),(i,S)],fill=(20,24,40)); d.line([(0,i),(S,i)],fill=(20,24,40))
d.ellipse([68,40,188,190],outline=(5,217,232),width=6)
d.ellipse([100,95,114,109],fill=(5,217,232)); d.ellipse([142,95,156,109],fill=(5,217,232))
d.arc([96,120,160,170],20,160,fill=(249,240,2),width=8)
d.rectangle([30,206,226,216],fill=(255,42,109)); d.rectangle([30,222,150,228],fill=(5,217,232))
im.save('${dir}/icon.png')"`);
execSync(`cd ${dir} && rm -f ../wav2lip-int4.xdc && zip -q -9 ../wav2lip-int4.xdc index.html manifest.toml icon.png`);
console.log('built single/wav2lip-int4.xdc');
