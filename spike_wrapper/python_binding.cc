#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "spike_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(spike_wrapper, m) {
    m.doc() = "Spike RISC-V simulator Python wrapper";
    
    // Export the main function — accepts an ELF file path.
    m.def("debug_cmd_str_elf_file", [](const std::string& elf_file_path, const std::string& debug_cmds,
                               const std::string& isa_string) -> std::string {
        char output_buffer[65536];  // 64KB buffer
        int result = spike_debug_cmd_str_elf_file(
            elf_file_path.c_str(),
            debug_cmds.c_str(),
            isa_string.c_str(),
            output_buffer,
            sizeof(output_buffer)
        );

        if (result < 0) {
            throw std::runtime_error(spike_get_last_error());
        }

        return std::string(output_buffer);
    }, "Run spike debug commands with ELF file path",
       py::arg("elf_file_path"), py::arg("debug_cmds"), py::arg("isa_string"));

    m.def("debug_cmd_file_elf_file", [](const std::string& elf_file_path, const std::string& debug_cmds,
                               const std::string& isa_string) -> std::string {
        char output_buffer[65536];  // 64KB buffer
        int result = spike_debug_cmd_file_elf_file(
            elf_file_path.c_str(),
            debug_cmds.c_str(),
            isa_string.c_str(),
            output_buffer,
            sizeof(output_buffer)
        );

        if (result < 0) {
            throw std::runtime_error(spike_get_last_error());
        }

        return std::string(output_buffer);
    }, "Run spike debug commands with ELF file path",
       py::arg("elf_file_path"), py::arg("debug_cmds"), py::arg("isa_string"));
    
    m.def("get_last_error", &spike_get_last_error, "Get the last error message");
}
