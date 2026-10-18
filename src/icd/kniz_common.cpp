#include "kniz_common.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace kniz {
namespace {

struct LevelName {
  LogLevel level;
  const char* name;
};

constexpr LevelName kNames[] = {
    {LogLevel::Error, "error"},
    {LogLevel::Warn, "warn"},
    {LogLevel::Info, "info"},
    {LogLevel::Debug, "debug"},
    {LogLevel::Trace, "trace"},
};

}  // namespace

int log_level() {
  // Read once; the log level is not expected to change while the process runs.
  static const int level = [] {
    const char* env = std::getenv("KNIZ_LOG");
    if (env == nullptr || *env == '\0')
      return static_cast<int>(LogLevel::Info);
    char* end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end == env || value < 0 || value > static_cast<long>(LogLevel::Trace))
      return static_cast<int>(LogLevel::Info);
    return static_cast<int>(value);
  }();
  return level;
}

void log_write(LogLevel level, const char* fmt, ...) {
  const char* name = "?";
  for (const auto& entry : kNames) {
    if (entry.level == level) {
      name = entry.name;
      break;
    }
  }

  std::fprintf(stderr, "kniz: [%s] ", name);

  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);

  std::fputc('\n', stderr);
}

}  // namespace kniz
