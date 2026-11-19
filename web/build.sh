#!/bin/bash
# Builds the web page into build/site and checks it (run from anywhere: web/build.sh).
# Needs the same packages as the normal build (g++, cmake, libopencv-dev) plus git and python3.
#
#   1. Tools, downloaded and built once into $WASM_TOOLS (default ~/.cache/seamcarve-wasm):
#      the Emscripten SDK, and OpenCV's core and imgproc modules compiled to WebAssembly.
#   2. The project's C++ (src/ and web/seamcarve_web.cpp) -> seamcarve.js + seamcarve.wasm.
#   3. The page, and the test photos as PNG, so a browser gets exactly the pixels OpenCV reads.
#   4. Check: the native build and the WebAssembly build carve the same photos, and the
#      results must be identical, pixel for pixel. The script fails if they are not.
#
# To look at the page afterwards:  python3 -m http.server -d build/site  ->  http://localhost:8000
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="${WASM_TOOLS:-$HOME/.cache/seamcarve-wasm}"
SITE="$ROOT/build/site"
EMSDK_VERSION=7.0.0
OPENCV_VERSION=4.10.0
mkdir -p "$TOOLS" "$SITE/images"

echo "== 1. tools in $TOOLS"
if [ ! -d "$TOOLS/emsdk" ]; then
  git clone --quiet --depth 1 https://github.com/emscripten-core/emsdk.git "$TOOLS/emsdk"
fi
quiet() {  # runs a long command with its output in a log file; shows the end of the log if it fails
  local log="$1"; shift
  "$@" > "$log" 2>&1 || { tail -n 40 "$log"; echo "FAILED: $* (full log: $log)"; exit 1; }
}
quiet "$TOOLS/emsdk-install.log" "$TOOLS/emsdk/emsdk" install $EMSDK_VERSION
quiet "$TOOLS/emsdk-activate.log" "$TOOLS/emsdk/emsdk" activate $EMSDK_VERSION
source "$TOOLS/emsdk/emsdk_env.sh" > /dev/null 2>&1
em++ --version | head -1

if [ ! -f "$TOOLS/opencv/lib/libopencv_imgproc.a" ]; then
  echo "building OpenCV $OPENCV_VERSION (core, imgproc) for WebAssembly, a few minutes"
  rm -rf "$TOOLS/opencv-src" "$TOOLS/opencv-build"
  git clone --quiet --depth 1 -b $OPENCV_VERSION https://github.com/opencv/opencv.git "$TOOLS/opencv-src"
  # Only what the seam carving code uses: no image files, no windows, no threads, no SIMD.
  quiet "$TOOLS/opencv-configure.log" emcmake cmake -S "$TOOLS/opencv-src" -B "$TOOLS/opencv-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TOOLS/opencv" \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_LIST=core,imgproc -DBUILD_ZLIB=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
    -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_js=OFF \
    -DWITH_PTHREADS_PF=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_TBB=OFF \
    -DWITH_OPENMP=OFF -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF -DWITH_PROTOBUF=OFF -DWITH_QUIRC=OFF \
    -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF \
    -DWITH_JASPER=OFF -DWITH_OPENJPEG=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF \
    -DWITH_V4L=OFF -DWITH_1394=OFF -DWITH_ADE=OFF -DWITH_VTK=OFF \
    -DCPU_BASELINE="" -DCPU_DISPATCH="" -DCV_ENABLE_INTRINSICS=OFF -DCV_TRACE=OFF -DENABLE_PIC=FALSE
  quiet "$TOOLS/opencv-build.log" cmake --build "$TOOLS/opencv-build" -j "$(nproc)" --target install
fi

echo "== 2. C++ -> WebAssembly"
SOURCES=("$ROOT"/src/energy.cpp "$ROOT"/src/seam.cpp "$ROOT"/src/carve.cpp "$ROOT"/src/methods/*.cpp)
em++ -O3 -std=c++17 -fexceptions -I"$ROOT/include" -I"$TOOLS/opencv/include/opencv4" \
  "$ROOT/web/seamcarve_web.cpp" "${SOURCES[@]}" \
  "$TOOLS/opencv/lib/libopencv_imgproc.a" "$TOOLS/opencv/lib/libopencv_core.a" \
  "$TOOLS/opencv/lib/opencv4/3rdparty/libzlib.a" \
  -sMODULARIZE=1 -sEXPORT_NAME=createSeamcarve -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=web,worker,node \
  -sEXPORTED_FUNCTIONS=_malloc,_free -sEXPORTED_RUNTIME_METHODS=ccall,UTF8ToString,HEAPU8,HEAP32 \
  -o "$SITE/seamcarve.js"
ls -l "$SITE/seamcarve.wasm" | awk '{print "seamcarve.wasm: " $5 " bytes"}'

echo "== 3. page and photos"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$ROOT/build" -j "$(nproc)" --target seamcarve > /dev/null
cp "$ROOT/web/index.html" "$ROOT/web/worker.js" "$SITE/"
for photo in "$ROOT"/images/*.jpg; do
  # No target size = nothing removed: this only turns the JPEG into a PNG with the same pixels.
  "$ROOT/build/seamcarve" "$photo" --out "$SITE/images/$(basename "$photo" .jpg).png" > /dev/null
done
ls "$SITE" "$SITE/images" | tr '\n' ' '; echo

echo "== 4. check: native build against WebAssembly build"
CHECK="$ROOT/build/web-check"
mkdir -p "$CHECK"
g++ -O3 -std=c++17 -I"$ROOT/include" $(pkg-config --cflags opencv4) "$ROOT/web/check/native_carve.cpp" \
  "${SOURCES[@]}" -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lpthread -o "$CHECK/native_carve"
# $EMSDK_NODE is the Node.js that comes with Emscripten, so the check does not depend on the machine's own.
check() {  # photo, target width, target height, method
  "$CHECK/native_carve" "$ROOT/images/$1" "$2" "$3" "$4" "$CHECK/in.pixels" "$CHECK/out.pixels"
  "$EMSDK_NODE" "$ROOT/web/check/compare.js" "$SITE/seamcarve.js" "$CHECK/in.pixels" "$CHECK/out.pixels" "$2" "$3" "$4"
}
check broadway.jpg 600 450 dp
check windmill.jpg 640 426 dp
check broadway.jpg 600 450 greedy
check sunset.jpg 640 533 baseline
check broadway_small.jpg 100 70 graphcut
echo "== done: $SITE"
