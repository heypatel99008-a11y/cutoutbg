// CutoutBG — on-device background removal (BiRefNet_lite, MIT licence) via Transformers.js
const FREE_MAX = 1500; // free tier long-side cap

const $ = (s) => document.querySelector(s);
const drop = $('#drop'), fileIn = $('#file'), status = $('#status'), barI = $('#bar i'), stTxt = $('#stTxt');
const upload = $('#upload'), result = $('#result');
const beforeImg = $('#beforeImg'), outCanvas = $('#out'), beforeWrap = $('#beforeWrap'), handle = $('#handle'), cmp = $('#cmp');

let cutout = null; // canvas with transparent cutout at original size
let bg = 'transparent';
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let pending = null, usingGPU = false;
worker.onmessage = ({ data }) => {
  if (data.type === 'progress') setProg(Math.round((data.loaded / data.total) * 90), `Loading AI model… ${Math.round(data.loaded / 1e6)} / ${Math.round(data.total / 1e6)} MB (first time only)`);
  else if (data.type === 'ready') usingGPU = data.gpu;
  else if (data.type === 'working') setProg(93, usingGPU ? 'Removing background…' : 'Removing background…');
  else if (data.type === 'done' && pending) { pending.resolve(data); pending = null; }
  else if (data.type === 'error' && pending) { pending.reject(new Error(data.message)); pending = null; }
};
function segment(blob) { return new Promise((resolve, reject) => { pending = { resolve, reject }; worker.postMessage({ type: 'run', blob }); }); }

function setProg(pct, txt) { status.style.display = 'block'; barI.style.width = pct + '%'; stTxt.textContent = txt; }

async function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) { alert('Please choose a JPG, PNG or WEBP image.'); return; }
  await run(file);
}

async function run(src) {
  try {
    setProg(3, 'Starting…');
    const blob = typeof src === 'string' ? await (await fetch(src)).blob() : src;
    const url = URL.createObjectURL(blob);
    const img = new Image(); img.src = url; await img.decode();
    const { mask, width, height } = await segment(blob);
    const c = document.createElement('canvas'); c.width = width; c.height = height;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, width, height);
    const px = ctx.getImageData(0, 0, width, height);
    for (let i = 0; i < mask.length; i++) px.data[i * 4 + 3] = mask[i];
    ctx.putImageData(px, 0, 0);
    cutout = c;
    beforeImg.src = url;
    upload.style.display = 'none'; result.style.display = 'block'; status.style.display = 'none';
    render(); setSplit(50);
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
