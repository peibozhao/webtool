#include "utils.h"
#include "gflags/gflags.h"
#include <fstream>
#include <limits.h>
#include <unistd.h>

DEFINE_string(logdir, ".", "Log file storage directory");
DEFINE_string(datadir, ".", "Database file storage directory");

std::string GetProcessName() {
  std::string name("unkown");
  std::getline(std::ifstream("/proc/self/comm"), name);
  return name;
}

std::filesystem::path GetExecFileDirectory() {
  char exec_fpath_str[PATH_MAX];
  ssize_t exec_fpath_len =
      readlink("/proc/self/exe", exec_fpath_str, sizeof(exec_fpath_str) - 1);
  if (exec_fpath_len == -1) {
    throw std::system_error(errno, std::generic_category(),
                            "Failed to get executable path");
  }
  exec_fpath_str[exec_fpath_len] = '\0';
  return std::filesystem::path(exec_fpath_str);
}

std::filesystem::path GetModelFileDirectory() {
  std::filesystem::path exec_file_path = GetExecFileDirectory();
  return exec_file_path.parent_path() / "models";
}

std::filesystem::path GetLogDirectory() { return FLAGS_logdir; }

std::filesystem::path GetDataDirectory() { return FLAGS_datadir; }
