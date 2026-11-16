#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace seamcarve {

// Runs the command-line tool with the given arguments (not including the program
// name). Returns the process exit code: 0 on success, 1 on any error.
int run_cli(const std::vector<std::string>& args, std::ostream& out = std::cout, std::ostream& err = std::cerr);

}  // namespace seamcarve
