#pragma once

#include <string>
#include <filesystem>

std::string GetProcessName();

std::filesystem::path GetExecFileDirectory();

std::filesystem::path GetModelFileDirectory();

std::filesystem::path GetLogDirectory();

std::filesystem::path GetDataDirectory();
