import { AutoModel, AutoProcessor, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5';
env.allowLocalModels = false;
// HD model (MIT) on WebGPU; light model (Apache-2.0) on CPU-only devices
const HD = { id: 'onnx-community/BiRefNet_lite-ONNX', dtype: 'fp16', device: 'webgpu', sigmoid: true };
const LITE = { id: 'Xenova/modnet', dtype: 'q8', device: 'wasm', sigmoid: false };
let model, processor, cfg, loading;
const files = {};
const progress_callback = (p) => {
  if (p.status === 'progress' && p.total) {
    files[p.file] = [p.loaded, p.total];
    let l = 0, t = 0; Object.values(files).forEach(([a, b]) => { l += a; t += b; });
    postMessage({ type: 'progress', loaded: l, total: t });
  }
};
async function loadCfg(c) {
  model = await AutoModel.from_pretrained(c.id, { dtype: c.dtype, device: c.device, progress_callback });
  processor = await AutoProcessor.from_pretrained(c.id);
  cfg = c;
}
async function load() {
  if (model) return;
  if (!loading) loading = (async () => {
    let gpu = false;
    try { const a = navigator.gpu && await navigator.gpu.requestAdapter(); gpu = !!a && !a.isFallbackAdapter && !(a.info && a.info.isFallbackAdapter) && !/swiftshader|llvmpipe/i.test(((a.info||{}).description||'')+((a.info||{}).vendor||'')); } catch (_) {}
    if (gpu) { try { await loadCfg(HD); } catch (e) { console.warn(e); gpu = false; } }
    if (!gpu) await loadCfg(LITE);
    postMessage({ type: 'ready', gpu });
  })();
  return loading;
}
onmessage = async ({ data }) => {
  try {
    await load();
    if (data.type !== 'run') return;
    postMessage({ type: 'working' });
    const image = await RawImage.fromBlob(data.blob);
    const { pixel_values } = await processor(image);
    const out = await model(cfg.sigmoid ? { input_image: pixel_values } : { input: pixel_values });
    let t = Object.values(out)[0][0];
    t = cfg.sigmoid ? t.sigmoid() : t;
    const mask = await RawImage.fromTensor(t.mul(255).to('uint8')).resize(image.width, image.height);
    postMessage({ type: 'done', mask: mask.data, width: image.width, height: image.height }, [mask.data.buffer]);
  } catch (e) { postMessage({ type: 'error', message: String(e && e.message || e) }); }
};
