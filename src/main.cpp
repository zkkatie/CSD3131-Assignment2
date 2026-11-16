// Entry point. Examples:
//   ./build/seamcarve images/broadway.jpg --width 600 --method dp
//   ./build/seamcarve images/windmill.jpg --scale 0.75 --save-seams --save-energy
//   ./build/seamcarve images/sunset.jpg --width 600 --energy gradient --energy-mode full
#include "seamcarve/cli.hpp"

int main(int argc, char** argv) {
    return seamcarve::run_cli(std::vector<std::string>(argv + 1, argv + argc));
}
