// Copyright (c) 2024-2025 DiveFuzz Project
// SPDX-License-Identifier: Mulan PSL v2

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "spike_engine.h"

namespace py = pybind11;
using namespace spike_engine;

PYBIND11_MODULE(spike_engine, m) {
    m.doc() = "Efficient Spike execution engine with checkpointing for DiveFuzz";

    // Constants
    m.attr("IMMEDIATE_NOT_PRESENT") = IMMEDIATE_NOT_PRESENT;

    // Floating-point register index offset
    // Register index convention:
    // - 0-31: Integer registers (x0-x31)
    // - 32-63: Floating-point registers (f0-f31, use FPR_OFFSET + reg_num)
    m.attr("FPR_OFFSET") = 32;

    // ExecutionResult class
    py::class_<ExecutionResult>(m, "ExecutionResult")
        .def(py::init<>())
        .def(py::init<const std::vector<uint64_t>&, const std::vector<uint64_t>&>(),
             py::arg("source_values_before"),
             py::arg("dest_values_after"))
        .def_readwrite("source_values_before", &ExecutionResult::source_values_before,
             "Source register values captured BEFORE execution (for XOR computation)")
        .def_readwrite("dest_values_after", &ExecutionResult::dest_values_after,
             "Destination register values captured AFTER execution (for bug filtering)");

    // Checkpoint class
    py::class_<Checkpoint>(m, "Checkpoint")
        .def(py::init<>())
        .def_readwrite("xpr", &Checkpoint::xpr, "General-purpose registers (x0-x31)")
        .def_readwrite("fpr", &Checkpoint::fpr, "Floating-point registers (f0-f31)")
        .def_readwrite("pc", &Checkpoint::pc, "Program counter")
        .def_readwrite("instr_index", &Checkpoint::instr_index, "Current instruction index");

    // SpikeEngine class
    py::class_<SpikeEngine>(m, "SpikeEngine")
        .def(py::init<const std::string&, const std::string&, size_t, bool>(),
             py::arg("elf_path"),
             py::arg("isa") = "rv64gc",
             py::arg("num_instrs") = 200,
             py::arg("verbose") = false,
             R"pbdoc(
             Create a SpikeEngine instance

             Args:
                 elf_path: Path to pre-compiled ELF file with nops
                 isa: ISA string (default: "rv64gc")
                 num_instrs: Number of instructions to generate (default: 200)
                 verbose: Enable verbose output (default: false)
             )pbdoc")

        .def_static("get_instruction_size", &SpikeEngine::get_instruction_size,
             py::arg("machine_code"),
             R"pbdoc(
             Detect instruction size from machine code

             Args:
                 machine_code: 32-bit machine code

             Returns:
                 Instruction size in bytes (2 for compressed, 4 for standard)
             )pbdoc")

        .def("initialize", &SpikeEngine::initialize,
             R"pbdoc(
             Initialize Spike and execute template initialization code
             Returns True on success, False on error (check get_last_error())
             )pbdoc")

        .def("set_checkpoint", &SpikeEngine::set_checkpoint,
             "Save current processor state as checkpoint")

        .def("restore_checkpoint", &SpikeEngine::restore_checkpoint,
             "Restore processor state from last checkpoint")

        .def("execute_instruction", &SpikeEngine::execute_instruction,
             py::arg("machine_code"),
             py::arg("source_regs"),
             py::arg("dest_regs"),
             py::arg("immediate") = 0,
             R"pbdoc(
             Execute one instruction and return register values

             Args:
                 machine_code: 32-bit machine code
                 source_regs: List of source register indices (read before execution)
                 dest_regs: List of destination register indices (read after execution)
                 immediate: Immediate value (default: 0)

             Returns:
                 ExecutionResult with:
                 - source_values_before: Source register values before execution (for XOR)
                 - dest_values_after: Destination register values after execution (for bug filtering)
             )pbdoc")

        .def("execute_instruction_sequence", &SpikeEngine::execute_instruction_sequence,
             py::arg("machine_codes"),
             py::arg("sizes"),
             R"pbdoc(
             Execute a sequence of instructions (for jump sequences)

             Writes all instructions to memory first, then executes them sequentially.
             Used for forward jumps where we need to execute jump + middle instructions.

             Args:
                 machine_codes: List of machine codes to execute
                 sizes: List of instruction sizes (2 or 4 bytes each)

             Returns:
                 Number of instructions successfully executed
             )pbdoc")

        .def("execute_loop_sequence", &SpikeEngine::execute_loop_sequence,
             py::arg("init_code"),
             py::arg("init_size"),
             py::arg("loop_body_codes"),
             py::arg("loop_body_sizes"),
             py::arg("decr_code"),
             py::arg("decr_size"),
             py::arg("branch_code"),
             py::arg("branch_size"),
             py::arg("max_iterations") = 100,
             R"pbdoc(
             Execute a loop sequence until branch condition fails

             Structure: init + (loop_body + decr + branch)*
             Executes init once, then loops body+decr+branch until branch doesn't jump back.

             Args:
                 init_code: Initialization instruction machine code
                 init_size: Size of init instruction (2 or 4)
                 loop_body_codes: List of loop body instruction codes
                 loop_body_sizes: List of loop body instruction sizes
                 decr_code: Decrement instruction code
                 decr_size: Size of decrement instruction
                 branch_code: Branch instruction code
                 branch_size: Size of branch instruction
                 max_iterations: Maximum iterations (default: 100)

             Returns:
                 Actual number of iterations executed
             )pbdoc")

        .def("get_xpr", &SpikeEngine::get_xpr,
             py::arg("reg_index"),
             "Get general-purpose register value (x0-x31)")

        .def("get_fpr", &SpikeEngine::get_fpr,
             py::arg("reg_index"),
             "Get floating-point register value (f0-f31)")

        .def("get_pc", &SpikeEngine::get_pc,
             "Get program counter value")

        .def("get_all_xpr", &SpikeEngine::get_all_xpr,
             "Get all general-purpose register values (x0-x31)")

        .def("get_all_fpr", &SpikeEngine::get_all_fpr,
             "Get all floating-point register values (f0-f31)")

        .def("get_csr", &SpikeEngine::get_csr,
             py::arg("csr_addr"),
             "Get CSR value by address (e.g., 0x300 for mstatus)")

        .def("get_all_csrs", &SpikeEngine::get_all_csrs,
             "Get all accessible CSR values as dict {addr: value}")

        .def("get_mem_region_start", &SpikeEngine::get_mem_region_start,
             "Get mem_region start address (for testing memory operations)")

        .def("get_mem_region_size", &SpikeEngine::get_mem_region_size,
             "Get mem_region size in bytes")

        .def("read_mem", &SpikeEngine::read_mem,
             py::arg("addr"),
             py::arg("size"),
             "Read memory at specified address, returns list of bytes")

        .def("get_current_index", &SpikeEngine::get_current_index,
             "Get current instruction index")

        .def("get_num_instrs", &SpikeEngine::get_num_instrs,
             "Get total number of instructions")

        .def("get_last_error", &SpikeEngine::get_last_error,
             "Get last error message");

    // Version info
    m.attr("__version__") = "2.0.0";
}
