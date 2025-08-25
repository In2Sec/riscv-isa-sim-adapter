#include "spike_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <cstring>
#include <sys/wait.h>
#include <fcntl.h>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <fstream>

// Global error information storage
static std::string g_last_error;

// Run Spike debugging from a file path 
// – use --debug-cmd-from-string to avoid creating temporary files for debug commands.
static std::string run_spike_debug_cmd_str_elf_file(const std::string& elf_file_path,
                                           const std::string& debug_cmds_string,
                                           const std::string& isa_string) {
    try {
        // Build the Spike command – use the --debug-cmd-from-string option to pass debug commands directly.
        char spike_cmd[4096];
        snprintf(spike_cmd, sizeof(spike_cmd),
                "spike -d --isa=%s --debug-cmd-from-string='%s' %s 2>&1",
                isa_string.c_str(), debug_cmds_string.c_str(), elf_file_path.c_str());

        // Execute the Spike command and capture its output
        FILE* pipe = popen(spike_cmd, "r");
        if (!pipe) {
            throw std::runtime_error("Failed to execute spike command");
        }

        // Fetch output
        std::string result;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }

        pclose(pipe);
        return result;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Spike execution failed: ") + e.what());
    }
}

static std::string run_spike_debug_cmd_file_elf_file(const std::string& elf_file_path,
                                           const std::string& debug_cmds_path,
                                           const std::string& isa_string) {
    try {
        // Build the Spike command – use the --debug-cmd option to pass debug commands directly.
        char spike_cmd[4096];
        snprintf(spike_cmd, sizeof(spike_cmd),
                "spike -d --isa=%s --debug-cmd='%s' %s 2>&1",
                isa_string.c_str(), debug_cmds_path.c_str(), elf_file_path.c_str());

        // Execute the Spike command and capture its output
        FILE* pipe = popen(spike_cmd, "r");
        if (!pipe) {
            throw std::runtime_error("Failed to execute spike command");
        }

        // Fetch output
        std::string result;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }

        pclose(pipe);
        return result;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Spike execution failed: ") + e.what());
    }
}



// C API implementation
extern "C" {

int spike_debug_cmd_str_elf_file(const char* elf_file_path,
                         const char* debug_cmds_string,
                         const char* isa_string,
                         char* output_buffer,
                         size_t buffer_size) {
    try {
        if (!elf_file_path || !debug_cmds_string || !isa_string || !output_buffer) {
            g_last_error = "Invalid parameters";
            return -1;
        }

        std::string result = run_spike_debug_cmd_str_elf_file(
            std::string(elf_file_path),
            std::string(debug_cmds_string),
            std::string(isa_string)
        );

        size_t copy_size = std::min(result.length(), buffer_size - 1);
        std::memcpy(output_buffer, result.c_str(), copy_size);
        output_buffer[copy_size] = '\0';

        return static_cast<int>(copy_size);

    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -1;
    }
}

int spike_debug_cmd_file_elf_file(const char* elf_file_path,
                         const char* debug_cmds_path,
                         const char* isa_string,
                         char* output_buffer,
                         size_t buffer_size) {
    try {
        if (!elf_file_path || !debug_cmds_path || !isa_string || !output_buffer) {
            g_last_error = "Invalid parameters";
            return -1;
        }

        std::string result = run_spike_debug_cmd_file_elf_file(
            std::string(elf_file_path),
            std::string(debug_cmds_path),
            std::string(isa_string)
        );

        size_t copy_size = std::min(result.length(), buffer_size - 1);
        std::memcpy(output_buffer, result.c_str(), copy_size);
        output_buffer[copy_size] = '\0';

        return static_cast<int>(copy_size);

    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -1;
    }
}



const char* spike_get_last_error(void) {
    return g_last_error.c_str();
}

}
