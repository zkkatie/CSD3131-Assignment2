// WebAssembly half of the browser-build check (see web/build.sh).
//
//   node compare.js <seamcarve.js> <pixels_in> <pixels_out> <target_width> <target_height> <method>
//
// Carves <pixels_in> with the WebAssembly build and compares the result with <pixels_out>,
// which the native build made from the same pixels. Exits with 1 if any pixel differs.
const fs = require('fs');
const [modulePath, pixelsIn, pixelsOut, targetWidth, targetHeight, method] = process.argv.slice(2);

// File format: see write_pixels() in native_carve.cpp.
function readPixels(path) {
  const b = fs.readFileSync(path);
  return { w: b.readInt32LE(0), h: b.readInt32LE(4), bgr: b.subarray(8) };
}

require(modulePath)().then((M) => {
  const src = readPixels(pixelsIn), want = readPixels(pixelsOut);
  const n = src.w * src.h;
  const input = M._malloc(n * 4);
  const rgba = M.HEAPU8.subarray(input, input + n * 4);
  for (let i = 0; i < n; i++) {  // BGR -> RGBA, the format the page gets from a canvas
    rgba[4 * i] = src.bgr[3 * i + 2];
    rgba[4 * i + 1] = src.bgr[3 * i + 1];
    rgba[4 * i + 2] = src.bgr[3 * i];
    rgba[4 * i + 3] = 255;
  }
  const failed = M.ccall('sc_carve', 'number',
    ['number', 'number', 'number', 'number', 'number', 'string', 'string'],
    [input, src.w, src.h, Number(targetWidth), Number(targetHeight), method, 'sobel']);
  if (failed) {
    console.log('browser  ERROR: ' + M.UTF8ToString(M._sc_error()));
    process.exit(1);
  }
  const w = M._sc_result_width(), h = M._sc_result_height();
  const out = M.HEAPU8.subarray(M._sc_result(), M._sc_result() + w * h * 4);
  let different = (w !== want.w || h !== want.h) ? w * h : 0;
  for (let i = 0; different === 0 && i < w * h; i++) {
    if (out[4 * i] !== want.bgr[3 * i + 2] || out[4 * i + 1] !== want.bgr[3 * i + 1] ||
        out[4 * i + 2] !== want.bgr[3 * i]) different++;
  }
  console.log(`browser  ${method.padEnd(8)} ${src.w}x${src.h} -> ${w}x${h}  seams=${M._sc_seam_count()}` +
    `  cost=${M._sc_total_cost().toFixed(0)}  time=${M._sc_total_time().toFixed(3)}s  ` +
    (different ? 'DIFFERENT from native' : 'identical to native'));
  process.exit(different ? 1 : 0);
});
