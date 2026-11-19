// Runs the carve off the page's main thread, so the page stays usable while Graph Cut works.
// seamcarve.js / seamcarve.wasm are the C++ code of this repository, built by web/build.sh.
importScripts('seamcarve.js');
const ready = createSeamcarve();

onmessage = async (event) => {
  try {
    const reply = await carve(event.data);
    postMessage(reply, reply.error ? [] : [reply.image.buffer, reply.seams.buffer]);
  } catch (e) {  // the module did not load, or the C++ aborted: tell the page instead of staying silent
    postMessage({ id: event.data.id, error: 'The carving code failed: ' + (e.message || e) });
  }
};

async function carve({ id, rgba, width, height, targetWidth, targetHeight, method, energy }) {
  const M = await ready;
  const input = M._malloc(rgba.length);
  M.HEAPU8.set(rgba, input);
  const failed = M.ccall('sc_carve', 'number',
    ['number', 'number', 'number', 'number', 'number', 'string', 'string'],
    [input, width, height, targetWidth, targetHeight, method, energy]);
  M._free(input);
  if (failed) return { id, error: M.UTF8ToString(M._sc_error()) };
  const w = M._sc_result_width(), h = M._sc_result_height();
  const image = M.HEAPU8.slice(M._sc_result(), M._sc_result() + w * h * 4);
  const first = M._sc_seams() >> 2;
  const seams = M.HEAP32.slice(first, first + M._sc_seams_length());
  return { id, width: w, height: h, image, seams,
           cost: M._sc_total_cost(), findTime: M._sc_find_time(), totalTime: M._sc_total_time() };
}
