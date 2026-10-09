// Local portability shim - NOT part of the upstream DNA Storage Toolkit.
//
// The upstream 1-encoding-decoding sources include <bits/stdc++.h>, which
// is provided by GCC/libstdc++ but not by clang/libc++ (default on macOS
// and in zig's C++ target). This header supplies the same aggregate
// include set using standard library headers, so the sources compile
// unchanged with `python -m ziglang c++` (or clang++) on macOS/Linux.
//
// Copy (or add this folder as an include path) before building:
//   cp -r 02_Software/DNA-Storage-Toolkit/bits \
//         02_Software/DNA-Storage-Toolkit/DNAStorageToolkit-main/1-encoding-decoding/bits
#pragma once

#if __has_include(<bits/stdc++.h>)
#  include <bits/stdc++.h>
#else

// C
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cfloat>
#include <cinttypes>
#include <climits>
#include <clocale>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// C++
#include <algorithm>
#include <bitset>
#include <complex>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ostream>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <typeinfo>
#include <utility>
#include <valarray>
#include <vector>

// C++11
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <forward_list>
#include <future>
#include <initializer_list>
#include <mutex>
#include <random>
#include <ratio>
#include <regex>
#include <scoped_allocator>
#include <system_error>
#include <thread>
#include <tuple>
#include <typeindex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#endif
