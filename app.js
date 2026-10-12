// CutoutBG — on-device background removal (BiRefNet_lite, MIT licence) via Transformers.js
import { AutoModel, AutoProcessor, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5';

env.allowLocalModels = false;
const MODEL_ID = 'onnx-community/BiRefNet_lite-ONNX';
const FREE_MAX = 1500; // free tier long-side cap

const $ = (s) => document.querySelector(s);
const drop = $('#drop'), fileIn = $('#file'), status = $('#status'), barI = $('#bar i'), stTxt = $('#stTxt');
const upload = $('#upload'), result = $('#result');
const beforeImg = $('#beforeImg'), outCanvas = $('#out'), beforeWrap = $('#beforeWrap'), handle = $('#handle'), cmp = $('#cmp');

let model = null, processor = null, loading = null;
let cutout = null; // canvas with transparent cutout at original size
let bg = 'transparent';

async function loadModel() {
  if (model) return;
  if (loading) return loading;
  const hasGPU = !!navigator.gpu;
  const files = {};
  loading = (async () => {
    const progress_callback = (p) => {
      if (p.status === 'progress' && p.total) {
        files[p.file] = [p.loaded, p.total];
        let l = 0, t = 0; Object.values(files).forEach(([a, b]) => { l += a; t += b; });
        setProg(Math.round((l / t) * 90), `Loading AI model… ${Math.round(l / 1e6)} / ${Math.round(t / 1e6)} MB (first time only, then cached)`);
      }
    };
    try {
      model = await AutoModel.from_pretrained(MODEL_ID, { dtype: 'fp16', device: hasGPU ? 'webgpu' : 'wasm', progress_callback });
    } catch (e) {
      console.warn('GPU/fp16 failed, falling back to wasm fp32', e);
      model = await AutoModel.from_pretrained(MODEL_ID, { dtype: 'fp32', device: 'wasm', progress_callback });
    }
    processor = await AutoProcessor.from_pretrained(MODEL_ID);
  })();
  return loading;
}

function setProg(pct, txt) { status.style.display = 'block'; barI.style.width = pct + '%'; stTxt.textContent = txt; }

async function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) { alert('Please choose a JPG, PNG or WEBP image.'); return; }
  const url = URL.createObjectURL(file);
  await run(url);
}

async function run(url) {
  try {
    setProg(3, 'Starting…');
    await loadModel();
    setProg(92, navigator.gpu ? 'Removing background…' : 'Removing background… (10–30 s on phones)');
    const image = await RawImage.fromURL(url);
    const { pixel_values } = await processor(image);
    const out = await model({ input_image: pixel_values });
    const t = Object.values(out)[0];
    const mask = await RawImage.fromTensor(t[0].sigmoid().mul(255).to('uint8')).resize(image.width, image.height);

    const c = document.createElement('canvas'); c.width = image.width; c.height = image.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(image.toCanvas(), 0, 0);
    const px = ctx.getImageData(0, 0, c.width, c.height);
    for (let i = 0; i < mask.data.length; i++) px.data[i * 4 + 3] = mask.data[i];
    ctx.putImageData(px, 0, 0);
    cutout = c;

    beforeImg.src = url;
    render();
    setProg(100, 'Done');
    upload.style.display = 'none'; result.style.display = 'block'; status.style.display = 'none';
    setSplit(50);
    track('cutout_done');
  } catch (e) {
    console.error(e);
    setProg(0, 'Sorry, something went wrong: ' + (e.message || e) + '. Try a smaller image or a desktop browser.');
  }
}

function composite(maxSide) {
  let w = cutout.width, h = cutout.height;
  if (maxSide && Math.max(w, h) > maxSide) { const s = maxSide / Math.max(w, h); w = Math.round(w * s); h = Math.round(h * s); }
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const x = c.getContext('2d');
  if (bg !== 'transparent') { x.fillStyle = bg; x.fillRect(0, 0, w, h); }
  x.imageSmoothingQuality = 'high';
  x.drawImage(cutout, 0, 0, w, h);
  return c;
}

function render() {
  const c = composite(1400);
  outCanvas.width = c.width; outCanvas.height = c.height;
  outCanvas.getContext('2d').drawImage(c, 0, 0);
  outCanvas.parentElement.classList.toggle('checker', bg === 'transparent');
  beforeImg.style.width = outCanvas.getBoundingClientRect().width + 'px';
}

// before/after slider
function setSplit(p) {
  p = Math.max(0, Math.min(100, p));
  beforeWrap.style.clipPath = `inset(0 ${100 - p}% 0 0)`;
  handle.style.left = `calc(${p}% - 1.5px)`;
}
function onMove(e) {
  const r = cmp.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
  setSplit((x / r.width) * 100);
}
let dragging = false;
cmp.addEventListener('pointerdown', (e) => { dragging = true; onMove(e); });
window.addEventListener('pointermove', (e) => dragging && onMove(e));
window.addEventListener('pointerup', () => (dragging = false));
window.addEventListener('resize', () => cutout && render());

// background swatches
document.querySelectorAll('.sw').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.sw').forEach((x) => x.classList.remove('on')); b.classList.add('on');
  bg = b.dataset.bg; render();
}));
$('#customColor').addEventListener('input', (e) => { document.querySelectorAll('.sw').forEach((x) => x.classList.remove('on')); bg = e.target.value; render(); });

// downloads
$('#dlFree').addEventListener('click', () => {
  const c = composite(FREE_MAX);
  c.toBlob((b) => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'cutoutbg.png'; a.click(); }, 'image/png');
  track('download_free');
});
$('#dlHD').addEventListener('click', () => { openModal('HD download is part of CutoutBG Pro'); track('click_hd'); });
$('#again').addEventListener('click', () => { result.style.display = 'none'; upload.style.display = 'block'; fileIn.value = ''; });

// input methods
drop.addEventListener('click', (e) => { if (!e.target.closest('.samples')) fileIn.click(); });
fileIn.addEventListener('change', () => handleFile(fileIn.files[0]));
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
window.addEventListener('paste', (e) => { const f = [...(e.clipboardData?.files || [])][0]; if (f) handleFile(f); });
document.querySelectorAll('.samples button').forEach((b) => b.addEventListener('click', (e) => { e.stopPropagation(); run(b.dataset.src); }));

// warm the model when the user shows intent
drop.addEventListener('pointerenter', () => { if (navigator.connection?.saveData) return; }, { once: true });

// Pro modal (payments not live yet)
const modal = $('#modal');
function openModal(title) { $('#modalTitle').textContent = title || 'CutoutBG Pro is launching soon'; modal.classList.add('on'); }
window.openModal = openModal;
document.querySelectorAll('[data-pro]').forEach((b) => b.addEventListener('click', () => { openModal('CutoutBG Pro is launching soon'); track('click_pro_' + b.dataset.pro); }));
modal.addEventListener('click', (e) => { if (e.target === modal || e.target.closest('.close')) modal.classList.remove('on'); });
$('#waitForm').addEventListener('submit', (e) => {
  e.preventDefault();
  $('#waitForm').innerHTML = '<p><b>You\'re on the list.</b> We\'ll email you the launch discount.</p>';
  track('waitlist');
});

function track(name) { try { window.gtag && window.gtag('event', name); } catch (_) {} }
