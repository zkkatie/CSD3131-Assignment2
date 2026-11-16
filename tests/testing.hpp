// A tiny test framework, so the project needs nothing beyond OpenCV.
//
//   TEST(name) { CHECK(1 + 1 == 2); }
//   TEST_EACH_METHOD(name) { ... uses `method`, a std::string ... }   one test per planned method
//
//   CHECK(cond)              fail if cond is false
//   CHECK_NEAR(a, b, tol)    fail if |a - b| > tol
//   CHECK_THROWS(expr)       fail if expr does not throw
//   SKIP(reason)             mark the test as skipped
//
// Run all tests:              ./build/tests
// Run tests whose name has X: ./build/tests X
#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "seamcarve/methods.hpp"

namespace testing {

struct Failure {
    std::string message;
};
struct Skipped {
    std::string reason;
};

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Register {
    Register(const std::string& name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

struct RegisterEachMethod {
    RegisterEachMethod(const std::string& name, void (*fn)(const std::string&)) {
        for (const std::string& m : seamcarve::PLANNED_METHODS)
            registry().push_back({name + "[" + m + "]", [fn, m] { fn(m); }});
    }
};

inline std::string where(const char* file, int line) {
    std::ostringstream os;
    os << file << ":" << line;
    return os.str();
}

inline int run_all(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    int passed = 0, failed = 0, skipped = 0;
    for (const TestCase& t : registry()) {
        if (!filter.empty() && t.name.find(filter) == std::string::npos) continue;
        try {
            t.fn();
            ++passed;
            std::cout << "PASS  " << t.name << "\n";
        } catch (const Skipped& s) {
            ++skipped;
            std::cout << "SKIP  " << t.name << "  (" << s.reason << ")\n";
        } catch (const Failure& f) {
            ++failed;
            std::cout << "FAIL  " << t.name << "\n      " << f.message << "\n";
        } catch (const std::exception& e) {
            ++failed;
            std::cout << "FAIL  " << t.name << "\n      unexpected exception: " << e.what() << "\n";
        }
    }
    std::cout << "\n" << passed << " passed, " << skipped << " skipped, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}

}  // namespace testing

#define TESTING_CAT_INNER(a, b) a##b
#define TESTING_CAT(a, b) TESTING_CAT_INNER(a, b)

#define TEST(name)                                                         \
    static void name();                                                    \
    static testing::Register TESTING_CAT(register_, name)(#name, name);    \
    static void name()

#define TEST_EACH_METHOD(name)                                                       \
    static void name(const std::string& method);                                     \
    static testing::RegisterEachMethod TESTING_CAT(register_, name)(#name, name);    \
    static void name([[maybe_unused]] const std::string& method)

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) throw testing::Failure{testing::where(__FILE__, __LINE__) + ": CHECK(" #cond ") failed"}; \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                    \
    do {                                                                                         \
        const double check_a_ = (a), check_b_ = (b);                                             \
        if (!(std::abs(check_a_ - check_b_) <= (tol))) {                                         \
            std::ostringstream os_;                                                              \
            os_ << testing::where(__FILE__, __LINE__) << ": CHECK_NEAR(" #a ", " #b ") failed: " \
                << check_a_ << " vs " << check_b_;                                               \
            throw testing::Failure{os_.str()};                                                   \
        }                                                                                        \
    } while (0)

#define CHECK_THROWS(expr)                                                                               \
    do {                                                                                                 \
        bool threw_ = false;                                                                             \
        try {                                                                                            \
            (void)(expr);                                                                                \
        } catch (...) {                                                                                  \
            threw_ = true;                                                                               \
        }                                                                                                \
        if (!threw_) throw testing::Failure{testing::where(__FILE__, __LINE__) + ": expected " #expr " to throw"}; \
    } while (0)

#define SKIP(reason) throw testing::Skipped{reason}
