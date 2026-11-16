#include <stdexcept>

#include "seamcarve/methods.hpp"

namespace seamcarve {

// A function-local static is created on first use, so it already exists when
// another file's REGISTER_SEAM_FINDER runs during program start-up.
static std::map<std::string, SeamFinder>& registry() {
    static std::map<std::string, SeamFinder> r;
    return r;
}

const std::map<std::string, SeamFinder>& methods() { return registry(); }

bool has_method(const std::string& name) { return registry().count(name) > 0; }

SeamFinder get_method(const std::string& name) {
    auto it = registry().find(name);
    if (it == registry().end()) {
        std::string known;
        for (const auto& [n, f] : registry()) known += (known.empty() ? "" : ", ") + n;
        throw std::invalid_argument("unknown method '" + name + "' (available: " + known + ")");
    }
    return it->second;
}

bool register_seam_finder(const std::string& name, SeamFinder finder) {
    if (!registry().emplace(name, std::move(finder)).second)
        throw std::logic_error("seam finder '" + name + "' registered twice");
    return true;
}

}  // namespace seamcarve
