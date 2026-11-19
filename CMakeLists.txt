cmake_minimum_required(VERSION 3.16)
project(seamcarve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # optimised by default: timings in the report should use Release
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)  # std::thread, used by the multithreaded DP variant

# Every seam finder in src/methods/ is picked up automatically, including ones
# teammates add later (CONFIGURE_DEPENDS re-checks the folder on each build).
file(GLOB METHOD_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/methods/*.cpp)

# OBJECT library (not static): keeps every method's REGISTER_SEAM_FINDER in the
# final programs. A static library would let the linker drop them.
add_library(seamcarve_core OBJECT
  src/energy.cpp
  src/seam.cpp
  src/carve.cpp
  src/cli.cpp
  ${METHOD_SOURCES}
)
target_include_directories(seamcarve_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(seamcarve_core PUBLIC ${OpenCV_LIBS} Threads::Threads)

if(MSVC)
  target_compile_options(seamcarve_core PUBLIC /W4)
else()
  target_compile_options(seamcarve_core PUBLIC -Wall -Wextra -Wpedantic)
endif()

# The program
add_executable(seamcarve src/main.cpp)
target_link_libraries(seamcarve PRIVATE seamcarve_core)

# Tests:  ctest --test-dir build --output-on-failure   (or ./build/tests)
# Every tests/test_*.cpp is compiled into the single test program (test_core.cpp provides main()).
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp)
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE seamcarve_core)
enable_testing()
add_test(NAME tests COMMAND tests)

# Benchmark: full vs incremental energy update
add_executable(benchmark_energy bench/benchmark_energy.cpp)
target_link_libraries(benchmark_energy PRIVATE seamcarve_core)

# Benchmark: DP variants (time vs memory) and multithreading
add_executable(benchmark_dp bench/benchmark_dp.cpp)
target_link_libraries(benchmark_dp PRIVATE seamcarve_core)

# Evaluation: compares every seam finder (quality, optimality, scaling, time breakdown)
add_executable(evaluate bench/evaluate.cpp)
target_link_libraries(evaluate PRIVATE seamcarve_core)
