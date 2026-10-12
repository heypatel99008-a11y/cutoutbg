import { AutoModel, AutoProcessor, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5';
env.allowLocalModels = false;
const MODEL_ID = 'onnx-community/BiRefNet_lite-ONNX';
let model, processor, loading;
const files = {};
const progress_callback = (p) => {
  if (p.status === 'progress' && p.total) {
    files[p.file] = [p.loaded, p.total];
    let l = 0, t = 0; Object.values(files).forEach(([a, b]) => { l += a; t += b; });
    postMessage({ type: 'progress', loaded: l, total: t });
  }
};
async function load() {
  if (model) return;
  if (!loading) loading = (async () => {
    const gpu = !!(self.navigator && navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null));
    try { model = await AutoModel.from_pretrained(MODEL_ID, { dtype: 'fp16', device: gpu ? 'webgpu' : 'wasm', progress_callback }); }
    catch (e) { model = await AutoModel.from_pretrained(MODEL_ID, { dtype: 'fp32', device: 'wasm', progress_callback }); }
    processor = await AutoProcessor.from_pretrained(MODEL_ID);
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
    const out = await model({ input_image: pixel_values });
    const t = Object.values(out)[0];
    const mask = await RawImage.fromTensor(t[0].sigmoid().mul(255).to('uint8')).resize(image.width, image.height);
    postMessage({ type: 'done', mask: mask.data, width: image.width, height: image.height }, [mask.data.buffer]);
  } catch (e) { postMessage({ type: 'error', message: String(e && e.message || e) }); }
};
