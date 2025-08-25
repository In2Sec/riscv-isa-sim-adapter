#ifndef SPIKE_WRAPPER_H
#define SPIKE_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run Spike debug commands from an ELF file path
 * @param elf_file_path Path to the ELF file
 * @param debug_cmds_string Debug command string
 * @param isa_string ISA string
 * @param output_buffer Output buffer (allocated by the caller)
 * @param buffer_size Size of the output buffer
 * @return Number of characters actually written, or -1 on error
 */
int spike_debug_cmd_str_elf_file(const char* elf_file_path,
                         const char* debug_cmds_string,
                         const char* isa_string,
                         char* output_buffer,
                         size_t buffer_size);

/**
 * Run Spike debug commands from an ELF file path
 * @param elf_file_path Path to the ELF file
 * @param debug_cmds_path Path to the debug command file
 * @param isa_string ISA string
 * @param output_buffer Output buffer (allocated by the caller)
 * @param buffer_size Size of the output buffer
 * @return Number of characters actually written, or -1 on error
 */
int spike_debug_cmd_file_elf_file(const char* elf_file_path,
                         const char* debug_cmds_path,
                         const char* isa_string,
                         char* output_buffer,
                         size_t buffer_size);

/**
 * Get the description of the last error
 * @return Error description string
 */
const char* spike_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // SPIKE_WRAPPER_H
