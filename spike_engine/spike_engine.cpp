#include "spike_engine.h"
#include "../riscv/sim.h"
#include "../riscv/processor.h"
#include "../riscv/mmu.h"
#include "../riscv/cfg.h"
#include "../riscv/decode.h"
#include "../riscv/trap.h"
#include "../riscv/encoding.h"

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <algorithm>

namespace spike_engine {

//==============================================================================
// Checkpoint Implementation
//==============================================================================

Checkpoint::Checkpoint()
    : xpr(32, 0)
    , fpr(32, 0)
    , pc(0)
    , instr_index(0)
    , next_instruction_addr(0)
    , prv(3)  // Default to M-mode
    , v(false)
    , debug_mode(false)
{
}

//==============================================================================
// SpikeEngine Implementation
//==============================================================================

SpikeEngine::SpikeEngine(const std::string& elf_path,
                         const std::string& isa,
                         size_t num_instrs,
                         bool verbose)
    : elf_path_(elf_path)
    , isa_(isa)
    , num_instrs_(num_instrs)
    , verbose_(verbose)
    , proc_(nullptr)
    , instruction_region_start_(0)
    , instruction_region_end_(0)
    , next_instruction_addr_(0)
    , mem_region_start_(0)
    , mem_region_size_(0)
    , current_instr_index_(0)
    , checkpoint_valid_(false)
    , initialized_(false)
{
}

size_t SpikeEngine::get_instruction_size(uint32_t machine_code) {
    // RISC-V instruction length encoding (from RISC-V spec):
    // - If bits[1:0] != 0b11, it's a 16-bit compressed instruction
    // - If bits[1:0] == 0b11 and bits[4:2] != 0b111, it's a 32-bit instruction
    // - Otherwise, it's a longer instruction (48/64-bit, not commonly used)

    uint8_t opcode_low = machine_code & 0x3;

    if (opcode_low != 0x3) {
        // Compressed instruction (16-bit)
        return 2;
    }

    uint8_t opcode_mid = (machine_code >> 2) & 0x7;
    if (opcode_mid != 0x7) {
        // Standard 32-bit instruction
        return 4;
    }

    // Instructions longer than 32 bits are not supported
    throw std::runtime_error("Unsupported instruction length (>32 bits)");
}

SpikeEngine::~SpikeEngine() {
    // sim_ will be automatically destroyed
}

bool SpikeEngine::initialize() {
    try {
        // Create configuration
        cfg_.reset(new cfg_t());
        cfg_->isa = isa_.c_str();
        cfg_->priv = "MSU";
        cfg_->misaligned = false;
        cfg_->endianness = endianness_little;
        cfg_->pmpregions = 16;
        cfg_->hartids = std::vector<size_t>{0};
        cfg_->real_time_clint = false;
        cfg_->trigger_count = 4;

        // Set memory layout
        cfg_->mem_layout.push_back(mem_cfg_t(0x80000000, 0x10000000));

        // Setup memory regions
        // We need two memory regions:
        // 1. Low memory for HTIF boot code (avoid [0, 0x1000) which Spike reserves)
        // 2. Main memory for our ELF (0x80000000 - 0x90000000)
        std::vector<std::pair<reg_t, abstract_mem_t*>> mems;

        // Low memory for boot code (starting after Spike's reserved region)
        mems.push_back(std::make_pair(0x1000, new mem_t(0xF000)));  // 60KB from 0x1000
        if (verbose_) {
            std::cout << "[SpikeEngine] Created boot memory: 0x1000 - 0x10000" << std::endl;
        }

        // Main memory for ELF
        reg_t mem_base = 0x80000000;
        size_t mem_size = 0x10000000; // 256MB
        mems.push_back(std::make_pair(mem_base, new mem_t(mem_size)));
        if (verbose_) {
            std::cout << "[SpikeEngine] Created main memory: 0x" << std::hex << mem_base
                      << " - 0x" << (mem_base + mem_size) << std::dec << std::endl;
        }

        // Create simulator
        std::vector<std::string> htif_args = {elf_path_};
        std::vector<device_factory_sargs_t> plugin_devices;
        debug_module_config_t dm_config;

        sim_.reset(new sim_t(
            cfg_.get(),
            /*halted=*/false,
            mems,  // Empty - let HTIF/ELF determine memory layout
            plugin_devices,
            htif_args,
            dm_config,
            /*log_path=*/nullptr,
            /*dtb_enabled=*/false,  // Disable DTB for simple testing
            /*dtb_file=*/nullptr,
            /*socket_enabled=*/false,
            /*cmd_file=*/nullptr,
            /*instruction_limit=*/std::nullopt
        ));

        // CRITICAL: Start HTIF to load ELF program
        // htif_t::start() will:
        // 1. Call load_program() to parse ELF and load into memory
        // 2. Call reset() to initialize processor
        if (verbose_) {
            std::cout << "[SpikeEngine] Starting HTIF to load ELF..." << std::endl;
        }
        sim_->start();
        if (verbose_) {
            std::cout << "[SpikeEngine] HTIF start() completed" << std::endl;
        }

        // Get processor 0 (after HTIF start)
        proc_ = sim_->get_core(0);
        if (!proc_) {
            last_error_ = "Failed to get processor 0";
            return false;
        }

        // Check PC after HTIF initialization
        uint64_t boot_pc = proc_->get_state()->pc;
        if (verbose_) {
            std::cout << "[SpikeEngine] PC after HTIF start (boot address): 0x" << std::hex << boot_pc << std::dec << std::endl;
        }

        // Check if boot address has valid instructions (HTIF boot code)
        if (verbose_) {
            std::cout << "[SpikeEngine] Reading memory at boot address..." << std::endl;
        }
        uint32_t boot_instr = 0;
        try {
            boot_instr = read_memory(boot_pc);
            if (verbose_) {
                std::cout << "[SpikeEngine] Instruction at boot address: 0x" << std::hex << boot_instr << std::dec << std::endl;
            }
        } catch (...) {
            if (verbose_) {
                std::cout << "[SpikeEngine] Failed to read memory at boot address" << std::endl;
            }
            throw;
        }

        // Find main symbol address
        if (!find_nop_region()) {
            last_error_ = "Failed to find main symbol in ELF";
            return false;
        }

        // Find _start symbol
        uint64_t start_addr = read_symbol_address("_start");
        if (verbose_) {
            std::cout << "[SpikeEngine] _start address: 0x" << std::hex << start_addr << std::dec << std::endl;
        }

        // Verify _start has valid instructions
        uint32_t start_instr = read_memory(start_addr);
        if (verbose_) {
            std::cout << "[SpikeEngine] Instruction at _start: 0x" << std::hex << start_instr << std::dec << std::endl;
        }

        if (start_instr == 0) {
            last_error_ = "Memory at _start is empty - ELF not loaded properly";
            return false;
        }

        // If boot PC has valid code, execute from boot to _start
        // Otherwise, directly set PC to _start
        uint64_t entry_pc;
        if (boot_instr != 0 && boot_pc != start_addr) {
            if (verbose_) {
                std::cout << "[SpikeEngine] Boot code found, executing from boot to _start..." << std::endl;
            }
            entry_pc = boot_pc;
        } else {
            if (verbose_) {
                std::cout << "[SpikeEngine] No boot code, setting PC directly to _start" << std::endl;
            }
            proc_->get_state()->pc = start_addr;
            entry_pc = start_addr;
        }

        if (verbose_) {
            std::cout << "[SpikeEngine] main address: 0x" << std::hex << instruction_region_start_ << std::dec << std::endl;
        }

        // Execute from entry to main
        if (entry_pc != instruction_region_start_) {
            const size_t max_steps = 100000;
            size_t steps = 0;

            if (verbose_) {
                std::cout << "[SpikeEngine] Executing from entry (0x" << std::hex << entry_pc
                          << ") to main (0x" << instruction_region_start_ << ")" << std::dec << std::endl;
            }

            while (proc_->get_state()->pc != instruction_region_start_ && steps < max_steps) {
                uint64_t current_pc = proc_->get_state()->pc;

                if (!step_processor()) {
                    std::ostringstream oss;
                    oss << "Failed at PC 0x" << std::hex << current_pc;
                    last_error_ = oss.str();
                    return false;
                }

                steps++;
            }

            if (steps >= max_steps) {
                std::ostringstream oss;
                oss << "Initialization timeout: PC did not reach main"
                    << " (stopped at 0x" << std::hex << proc_->get_state()->pc
                    << ", target was 0x" << instruction_region_start_ << ")";
                last_error_ = oss.str();
                return false;
            }

            if (verbose_) {
                std::cout << "[SpikeEngine] Reached main after " << steps << " steps" << std::endl;
            }
        } else {
            if (verbose_) {
                std::cout << "[SpikeEngine] Entry point is already at main" << std::endl;
            }
        }

        initialized_ = true;
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Initialization exception: ") + e.what();
        std::cerr << "[SpikeEngine] Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        last_error_ = "Unknown exception during initialization";
        std::cerr << "[SpikeEngine] Unknown exception!" << std::endl;
        return false;
    }
}

uint64_t SpikeEngine::read_symbol_address(const std::string& symbol_name) {
    // Use objdump to read symbol table from ELF
    // Format: objdump -t <elf_file> | grep <symbol>
    // Output: "address  flags section size name"

    std::string cmd = "riscv64-unknown-elf-objdump -t " + elf_path_ + " 2>/dev/null | grep ' " + symbol_name + "$'";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return 0;
    }

    char buffer[256];
    uint64_t address = 0;

    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        // Parse first field (address in hex)
        char* endptr;
        address = strtoull(buffer, &endptr, 16);
    }

    pclose(pipe);
    return address;
}

bool SpikeEngine::find_nop_region() {
    // Find instruction region using ELF symbol table
    // Read the 'main' symbol address to locate the instruction region

    // Try to read 'main' symbol address from ELF
    uint64_t main_addr = read_symbol_address("main");

    if (main_addr == 0) {
        // Fallback: try common alternatives
        main_addr = read_symbol_address("_start");
        if (main_addr == 0) {
            last_error_ = "Failed to find 'main' or '_start' symbol in ELF";
            return false;
        }
    }

    // Set instruction region bounds
    instruction_region_start_ = main_addr;
    next_instruction_addr_ = main_addr;

    // Calculate available space (assuming worst case: all 4-byte instructions)
    // This provides enough space even if all instructions are standard 32-bit
    instruction_region_end_ = main_addr + num_instrs_ * 4;

    // Find mem_region for checkpoint memory backup
    // This is the data region that can be modified by instructions
    uint64_t mem_region_addr = read_symbol_address("mem_region");
    uint64_t mem_region_end_addr = read_symbol_address("mem_region_end");

    if (mem_region_addr != 0 && mem_region_end_addr != 0) {
        mem_region_start_ = mem_region_addr;
        mem_region_size_ = mem_region_end_addr - mem_region_addr;
        if (verbose_) {
            std::cout << "[SpikeEngine] mem_region: 0x" << std::hex << mem_region_start_
                      << " - 0x" << (mem_region_start_ + mem_region_size_)
                      << " (size: " << std::dec << mem_region_size_ << " bytes)" << std::endl;
        }
    } else {
        // Fallback: use a default size based on template (8KB = 8192 bytes)
        // The mem_region typically starts right after .data section
        mem_region_addr = read_symbol_address("region_0");
        if (mem_region_addr != 0) {
            // region_0 is 32 bytes, mem_region follows
            mem_region_start_ = mem_region_addr + 32;
            mem_region_size_ = 8192;  // Default .mem_region size
        }
        if (verbose_) {
            std::cout << "[SpikeEngine] mem_region symbols not found, using fallback" << std::endl;
        }
    }

    return true;
}

void SpikeEngine::set_checkpoint() {
    save_state(checkpoint_);
    checkpoint_valid_ = true;
}

void SpikeEngine::restore_checkpoint() {
    if (!checkpoint_valid_) {
        throw std::runtime_error("No valid checkpoint to restore");
    }
    restore_state(checkpoint_);
}

size_t SpikeEngine::execute_instruction_sequence(
    const std::vector<uint32_t>& machine_codes,
    const std::vector<size_t>& sizes) {

    if (!initialized_) {
        throw std::runtime_error("SpikeEngine not initialized");
    }

    if (machine_codes.size() != sizes.size()) {
        throw std::runtime_error("machine_codes and sizes must have the same length");
    }

    if (machine_codes.empty()) {
        return 0;
    }

    // Calculate total size needed
    size_t total_size = 0;
    for (size_t s : sizes) {
        total_size += s;
    }

    // Check if we have enough space
    if (next_instruction_addr_ + total_size > instruction_region_end_) {
        throw std::runtime_error("Out of instruction region space for sequence");
    }

    // Verify PC matches next instruction address
    uint64_t current_pc = get_pc();
    if (current_pc != next_instruction_addr_) {
        std::ostringstream oss;
        oss << "PC mismatch in sequence: expected 0x" << std::hex << next_instruction_addr_
            << " but got 0x" << current_pc << std::dec;
        throw std::runtime_error(oss.str());
    }

    // Step 1: Write all instructions to memory
    uint64_t write_addr = next_instruction_addr_;
    for (size_t i = 0; i < machine_codes.size(); ++i) {
        if (!write_memory(write_addr, machine_codes[i], sizes[i])) {
            throw std::runtime_error("Failed to write instruction to memory in sequence");
        }
        write_addr += sizes[i];
    }

    // Step 2: Execute all instructions with trap handling
    size_t executed = 0;
    for (size_t i = 0; i < machine_codes.size(); ++i) {
        if (!step_until_in_region()) {
            std::ostringstream oss;
            oss << "Failed to execute instruction " << i << " in sequence: " << last_error_;
            throw std::runtime_error(oss.str());
        }
        executed++;
        current_instr_index_++;
    }

    // Update next instruction address
    next_instruction_addr_ = get_pc();

    return executed;
}

size_t SpikeEngine::execute_loop_sequence(
    uint32_t init_code,
    size_t init_size,
    const std::vector<uint32_t>& loop_body_codes,
    const std::vector<size_t>& loop_body_sizes,
    uint32_t decr_code,
    size_t decr_size,
    uint32_t branch_code,
    size_t branch_size,
    size_t max_iterations) {

    if (!initialized_) {
        throw std::runtime_error("SpikeEngine not initialized");
    }

    if (loop_body_codes.size() != loop_body_sizes.size()) {
        throw std::runtime_error("loop_body_codes and loop_body_sizes must have the same length");
    }

    // Calculate total size for one iteration
    size_t body_size = 0;
    for (size_t s : loop_body_sizes) {
        body_size += s;
    }
    size_t loop_iteration_size = body_size + decr_size + branch_size;
    size_t total_size = init_size + loop_iteration_size;

    // Check space (we only need space for one iteration since we reuse it)
    if (next_instruction_addr_ + total_size > instruction_region_end_) {
        throw std::runtime_error("Out of instruction region space for loop");
    }

    // Verify PC
    uint64_t current_pc = get_pc();
    if (current_pc != next_instruction_addr_) {
        std::ostringstream oss;
        oss << "PC mismatch in loop: expected 0x" << std::hex << next_instruction_addr_
            << " but got 0x" << current_pc << std::dec;
        throw std::runtime_error(oss.str());
    }

    // Step 1: Write init instruction
    uint64_t init_addr = next_instruction_addr_;
    if (!write_memory(init_addr, init_code, init_size)) {
        throw std::runtime_error("Failed to write init instruction");
    }

    // Step 2: Write loop body
    uint64_t loop_start_addr = init_addr + init_size;
    uint64_t write_addr = loop_start_addr;
    for (size_t i = 0; i < loop_body_codes.size(); ++i) {
        if (!write_memory(write_addr, loop_body_codes[i], loop_body_sizes[i])) {
            throw std::runtime_error("Failed to write loop body instruction");
        }
        write_addr += loop_body_sizes[i];
    }

    // Step 3: Write decrement instruction
    uint64_t decr_addr = write_addr;
    if (!write_memory(decr_addr, decr_code, decr_size)) {
        throw std::runtime_error("Failed to write decrement instruction");
    }
    write_addr += decr_size;

    // Step 4: Write branch instruction
    uint64_t branch_addr = write_addr;
    if (!write_memory(branch_addr, branch_code, branch_size)) {
        throw std::runtime_error("Failed to write branch instruction");
    }
    uint64_t loop_end_addr = write_addr + branch_size;

    // Step 5: Execute init with trap handling
    if (!step_until_in_region()) {
        throw std::runtime_error("Failed to execute init instruction: " + last_error_);
    }
    current_instr_index_++;

    // Step 6: Execute loop iterations with trap handling
    size_t iterations = 0;
    while (iterations < max_iterations) {
        // Execute loop body
        for (size_t i = 0; i < loop_body_codes.size(); ++i) {
            if (!step_until_in_region()) {
                throw std::runtime_error("Failed to execute loop body: " + last_error_);
            }
            current_instr_index_++;
        }

        // Execute decrement
        if (!step_until_in_region()) {
            throw std::runtime_error("Failed to execute decrement: " + last_error_);
        }
        current_instr_index_++;

        // Execute branch
        if (!step_until_in_region()) {
            throw std::runtime_error("Failed to execute branch: " + last_error_);
        }
        current_instr_index_++;

        iterations++;

        // Check if branch jumped back or fell through
        current_pc = get_pc();
        if (current_pc >= loop_end_addr) {
            // Branch fell through (condition false), exit loop
            break;
        }
        // Otherwise, branch jumped back, continue loop
    }

    // Update next instruction address
    next_instruction_addr_ = get_pc();

    return iterations;
}

ExecutionResult SpikeEngine::execute_instruction(uint32_t machine_code,
                                                  const std::vector<int>& source_regs,
                                                  const std::vector<int>& dest_regs,
                                                  int64_t immediate) {
    if (!initialized_) {
        throw std::runtime_error("SpikeEngine not initialized");
    }

    // Detect instruction size (2 for compressed, 4 for standard)
    size_t instr_size = get_instruction_size(machine_code);

    // Check if we have enough space
    if (next_instruction_addr_ + instr_size > instruction_region_end_) {
        throw std::runtime_error("Out of instruction region space");
    }

    // Verify PC matches next instruction address
    uint64_t current_pc = get_pc();
    if (current_pc != next_instruction_addr_) {
        std::ostringstream oss;
        oss << "PC mismatch: expected 0x" << std::hex << next_instruction_addr_
            << " but got 0x" << current_pc << std::dec;
        throw std::runtime_error(oss.str());
    }

    // Get instruction address
    uint64_t instr_addr = next_instruction_addr_;

    // Write machine code with correct size
    if (!write_memory(instr_addr, machine_code, instr_size)) {
        throw std::runtime_error("Failed to write machine code to memory");
    }

    // STEP 1: Read source register values BEFORE execution
    // This ensures we capture the values that will be used by the instruction,
    // even if the destination register overlaps with source registers
    // (e.g., add x10, x10, x11 - we want the OLD value of x10, not the result)
    //
    // Register index convention:
    // - 0-31: Integer registers (x0-x31)
    // - 32-63: Floating-point registers (f0-f31, mapped as 32+reg_num)
    std::vector<uint64_t> source_values;
    for (int reg_idx : source_regs) {
        if (reg_idx >= 0 && reg_idx < 32) {
            // Integer register (x0-x31)
            source_values.push_back(get_xpr(reg_idx));
        } else if (reg_idx >= 32 && reg_idx < 64) {
            // Floating-point register (f0-f31, mapped as 32+reg_num)
            source_values.push_back(get_fpr(reg_idx - 32));
        }
    }

    // Add immediate if instruction has one (including immediate=0)
    if (immediate != IMMEDIATE_NOT_PRESENT) {
        source_values.push_back(static_cast<uint64_t>(immediate));
    }

    // STEP 2: Execute instruction with trap handling
    // Use step_until_in_region() instead of step_processor() to handle traps.
    // If the instruction triggers a trap, Spike's internal take_trap() will
    // jump to the trap handler. We need to continue executing until the
    // trap handler completes (via mret/sret) and PC returns to instruction region.
    if (!step_until_in_region()) {
        std::ostringstream oss;
        oss << "Failed to execute instruction at 0x" << std::hex << instr_addr
            << std::dec << " - " << last_error_;
        throw std::runtime_error(oss.str());
    }


    // STEP 3: Read destination register values AFTER execution (and after trap handling)
    // These values are used for bug filtering in Python (e.g., checking if sc.w returned 1)
    //
    // Register index convention (same as source registers):
    // - 0-31: Integer registers (x0-x31)
    // - 32-63: Floating-point registers (f0-f31, mapped as 32+reg_num)
    std::vector<uint64_t> dest_values;
    for (int reg_idx : dest_regs) {
        if (reg_idx >= 0 && reg_idx < 32) {
            // Integer register (x0-x31)
            dest_values.push_back(get_xpr(reg_idx));
        } else if (reg_idx >= 32 && reg_idx < 64) {
            // Floating-point register (f0-f31, mapped as 32+reg_num)
            dest_values.push_back(get_fpr(reg_idx - 32));
        }
    }

    // Update next instruction address to current PC
    // Now PC should be at the instruction after the one we executed (post trap handling)
    next_instruction_addr_ = get_pc();

    // Increment instruction index
    current_instr_index_++;

    // Return both source values (for XOR) and dest values (for bug filtering)
    return ExecutionResult(source_values, dest_values);
}

uint64_t SpikeEngine::get_xpr(int reg_index) const {
    if (!proc_ || reg_index < 0 || reg_index >= 32) {
        return 0;
    }
    return proc_->get_state()->XPR[reg_index];
}

uint64_t SpikeEngine::get_fpr(int reg_index) const {
    if (!proc_ || reg_index < 0 || reg_index >= 32) {
        return 0;
    }
    return proc_->get_state()->FPR[reg_index].v[0];
}

uint64_t SpikeEngine::get_pc() const {
    if (!proc_) {
        return 0;
    }
    return proc_->get_state()->pc;
}

std::vector<uint64_t> SpikeEngine::get_all_xpr() const {
    std::vector<uint64_t> result(32, 0);
    if (!proc_) {
        return result;
    }
    for (int i = 0; i < 32; ++i) {
        result[i] = proc_->get_state()->XPR[i];
    }
    return result;
}

std::vector<uint64_t> SpikeEngine::get_all_fpr() const {
    std::vector<uint64_t> result(32, 0);
    if (!proc_) {
        return result;
    }
    for (int i = 0; i < 32; ++i) {
        result[i] = proc_->get_state()->FPR[i].v[0];
    }
    return result;
}

uint64_t SpikeEngine::get_csr(uint64_t csr_addr) const {
    if (!proc_) {
        return 0;
    }
    try {
        auto state = proc_->get_state();
        auto it = state->csrmap.find(csr_addr);
        if (it != state->csrmap.end() && it->second) {
            return it->second->read();
        }
    } catch (...) {
        // CSR read may throw exception, return 0
    }
    return 0;
}

std::map<uint64_t, uint64_t> SpikeEngine::get_all_csrs() const {
    std::map<uint64_t, uint64_t> result;
    if (!proc_) {
        return result;
    }
    try {
        auto state = proc_->get_state();
        for (const auto& [addr, csr] : state->csrmap) {
            if (csr) {
                try {
                    result[addr] = csr->read();
                } catch (...) {
                    // Skip CSRs that throw on read
                }
            }
        }
    } catch (...) {
        // Return partial result on error
    }
    return result;
}

std::vector<uint8_t> SpikeEngine::read_mem(uint64_t addr, size_t size) const {
    std::vector<uint8_t> result(size, 0);
    if (!sim_ || !sim_->debug_mmu) {
        return result;
    }
    try {
        for (size_t i = 0; i < size; ++i) {
            result[i] = sim_->debug_mmu->load<uint8_t>(addr + i);
        }
    } catch (...) {
        // Return partial result on error
    }
    return result;
}

//==============================================================================
// Private Helper Methods
//==============================================================================

bool SpikeEngine::write_memory(uint64_t addr, uint32_t code, size_t size) {
    try {
        mmu_t* mmu = proc_->get_mmu();

        if (size == 2) {
            // Compressed instruction: write only 16 bits
            uint16_t code_16 = static_cast<uint16_t>(code & 0xFFFF);
            mmu->store<uint16_t>(addr, code_16);
        } else if (size == 4) {
            // Standard instruction: write full 32 bits
            mmu->store<uint32_t>(addr, code);
        } else {
            last_error_ = "Invalid instruction size: " + std::to_string(size);
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        last_error_ = std::string("Memory write failed: ") + e.what();
        return false;
    }
}

uint32_t SpikeEngine::read_memory(uint64_t addr) {
    try {
        mmu_t* mmu = proc_->get_mmu();
        return mmu->load<uint32_t>(addr);
    } catch (const std::exception&) {
        return 0;
    }
}

bool SpikeEngine::is_in_instruction_region(uint64_t pc) const {
    return pc >= instruction_region_start_ && pc < instruction_region_end_;
}

bool SpikeEngine::step_processor() {
    try {
        // Execute one instruction
        proc_->step(1);
        return true;
    } catch (trap_t& t) {
        // Catch RISC-V traps (illegal instruction, access faults, etc.)
        std::ostringstream oss;
        oss << "Processor trap: " << t.name()
            << " (cause=" << t.cause() << ")";

        if (t.has_tval()) {
            oss << ", tval=0x" << std::hex << t.get_tval() << std::dec;
        }

        last_error_ = oss.str();
        return false;
    } catch (trap_debug_mode&) {
        // Debug mode entry (not an error in fuzzing context, just skip)
        last_error_ = "Processor entered debug mode";
        return false;
    } catch (const std::exception& e) {
        last_error_ = std::string("Processor step failed: ") + e.what();
        return false;
    } catch (...) {
        // Catch all other exceptions
        last_error_ = "Processor step failed: Caught an unknown exception!";
        return false;
    }
}

bool SpikeEngine::step_until_in_region() {
    // Maximum steps to allow for trap handler execution
    // Trap handlers are typically short (< 20 instructions), but we allow
    // more steps for safety (e.g., nested handlers, complex dispatch)
    const size_t MAX_TRAP_HANDLER_STEPS = 1000;
    size_t trap_steps = 0;

    // Step 1: Execute the target instruction
    try {
        proc_->step(1);
    } catch (trap_t& t) {
        // Note: Spike normally handles traps internally via take_trap() in execute.cc
        // and does NOT throw here. This catch is a fallback for unusual situations.
        std::ostringstream oss;
        oss << "Unexpected trap exception: " << t.name()
            << " (cause=" << t.cause() << ")";
        if (t.has_tval()) {
            oss << ", tval=0x" << std::hex << t.get_tval() << std::dec;
        }
        last_error_ = oss.str();
        return false;
    } catch (trap_debug_mode&) {
        last_error_ = "Processor entered debug mode";
        return false;
    } catch (const std::exception& e) {
        last_error_ = std::string("Initial step failed: ") + e.what();
        return false;
    } catch (...) {
        last_error_ = "Initial step failed: Unknown exception";
        return false;
    }

    // Step 2: Check if PC is in instruction region
    // If not, we're likely in a trap handler and need to continue executing
    uint64_t current_pc = get_pc();

    while (!is_in_instruction_region(current_pc) && trap_steps < MAX_TRAP_HANDLER_STEPS) {
        try {
            proc_->step(1);
        } catch (trap_t& t) {
            std::ostringstream oss;
            oss << "Trap in handler: " << t.name()
                << " (cause=" << t.cause() << ")";
            last_error_ = oss.str();
            return false;
        } catch (trap_debug_mode&) {
            last_error_ = "Debug mode in trap handler";
            return false;
        } catch (const std::exception& e) {
            last_error_ = std::string("Exception in trap handler: ") + e.what();
            return false;
        } catch (...) {
            last_error_ = "Unknown exception in trap handler";
            return false;
        }

        current_pc = get_pc();
        trap_steps++;
    }

    // Step 3: Check if we exited the loop successfully
    if (trap_steps >= MAX_TRAP_HANDLER_STEPS) {
        std::ostringstream oss;
        oss << "Trap handler did not return within " << MAX_TRAP_HANDLER_STEPS
            << " steps. Last PC: 0x" << std::hex << current_pc << std::dec;
        last_error_ = oss.str();
        return false;
    }

    // Success: PC is back in instruction region
    if (verbose_ && trap_steps > 0) {
        std::cout << "[SpikeEngine] Trap handler completed in " << trap_steps
                  << " steps, PC now at 0x" << std::hex << current_pc << std::dec << std::endl;
    }

    return true;
}

uint64_t SpikeEngine::compute_xor(const std::vector<uint64_t>& values) const {
    uint64_t result = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        result ^= (values[i] << i);
    }
    return result;
}

void SpikeEngine::save_state(Checkpoint& checkpoint) {
    if (!proc_) {
        throw std::runtime_error("No processor to save state from");
    }

    try {
        auto state = proc_->get_state();

        // Save general-purpose registers
        try {
            for (int i = 0; i < 32; ++i) {
                checkpoint.xpr[i] = state->XPR[i];
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to save XPR registers: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Failed to save XPR registers: Unknown exception");
        }

        // Save floating-point registers
        try {
            for (int i = 0; i < 32; ++i) {
                checkpoint.fpr[i] = state->FPR[i].v[0];
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to save FPR registers: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Failed to save FPR registers: Unknown exception");
        }

        // Save program counter
        checkpoint.pc = state->pc;

        // Save instruction index
        // This marks the execution position at checkpoint time
        checkpoint.instr_index = current_instr_index_;

        // Save next instruction address
        // This ensures PC and instruction placement remain synchronized after restore
        checkpoint.next_instruction_addr = next_instruction_addr_;

        // Save privilege and mode state
        // These are essential for correct trap handling after restore
        checkpoint.prv = state->prv;
        checkpoint.v = state->v;
        checkpoint.debug_mode = state->debug_mode;

        // Save ALL CSRs by iterating through csrmap
        // This ensures complete state restoration including all trap handling,
        // interrupt, PMP, floating-point, and custom CSRs
        checkpoint.csr_values.clear();

        // Iterate through all CSRs in the csrmap and save their values
        // Use try-catch for each CSR since some may throw exceptions on read
        for (const auto& [addr, csr] : state->csrmap) {
            if (csr) {
                try {
                    checkpoint.csr_values[addr] = csr->read();
                } catch (...) {
                    // Skip CSRs that throw exceptions on read
                }
            }
        }

        // Save memory region (essential for correct rollback of AMO/store instructions)
        if (mem_region_size_ > 0 && mem_region_start_ != 0) {
            checkpoint.mem_region_backup.resize(mem_region_size_);
            for (size_t i = 0; i < mem_region_size_; i += 8) {
                uint64_t addr = mem_region_start_ + i;
                uint64_t value = 0;
                // Read 8 bytes at a time
                for (size_t j = 0; j < 8 && (i + j) < mem_region_size_; ++j) {
                    uint8_t byte = sim_->debug_mmu->load<uint8_t>(addr + j);
                    value |= (static_cast<uint64_t>(byte) << (j * 8));
                }
                // Store in backup
                for (size_t j = 0; j < 8 && (i + j) < mem_region_size_; ++j) {
                    checkpoint.mem_region_backup[i + j] = (value >> (j * 8)) & 0xFF;
                }
            }
        }

    } catch (const std::runtime_error&) {
        // Re-throw RuntimeError as-is (already has detailed message)
        throw;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to save checkpoint: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Failed to save checkpoint: Unknown exception in outer block");
    }
}

void SpikeEngine::restore_state(const Checkpoint& checkpoint) {
    if (!proc_) {
        throw std::runtime_error("No processor to restore state to");
    }

    try {
        auto state = proc_->get_state();

        // Restore privilege and mode state FIRST
        // This is critical because CSR access permissions depend on privilege level
        // If we restore CSRs before privilege level, some writes may fail
        state->prv = checkpoint.prv;
        state->v = checkpoint.v;
        state->debug_mode = checkpoint.debug_mode;

        // Restore CSRs from the saved map using direct pointer access
        // We use the direct CSR pointers from state_t to avoid issues with
        // some CSRs that may cause crashes when written via csrmap iteration
        auto restore_csr = [&checkpoint](reg_t addr, const csr_t_p& csr) {
            if (csr && checkpoint.csr_values.count(addr)) {
                try {
                    csr->write(checkpoint.csr_values.at(addr));
                } catch (...) {
                    // Silently skip CSRs that fail to write
                }
            }
        };

        // Machine mode trap handling CSRs
        restore_csr(0x300, state->mstatus);
        restore_csr(0x301, state->misa);
        restore_csr(0x302, state->medeleg);
        restore_csr(0x303, state->mideleg);
        restore_csr(0x304, state->mie);
        restore_csr(0x305, state->mtvec);
        restore_csr(0x306, state->mcounteren);
        restore_csr(0x310, state->mstatush);
        restore_csr(0x320, state->mcountinhibit);
        restore_csr(0x340, state->csrmap.count(0x340) ? state->csrmap.at(0x340) : nullptr);
        restore_csr(0x341, state->mepc);
        restore_csr(0x342, state->mcause);
        restore_csr(0x343, state->mtval);
        restore_csr(0x344, state->mip);

        // Supervisor mode CSRs
        restore_csr(0x100, state->sstatus);
        restore_csr(0x104, state->nonvirtual_sie);
        restore_csr(0x105, state->stvec);
        restore_csr(0x106, state->scounteren);
        restore_csr(0x140, state->csrmap.count(0x140) ? state->csrmap.at(0x140) : nullptr);
        restore_csr(0x141, state->sepc);
        restore_csr(0x142, state->scause);
        restore_csr(0x143, state->stval);
        restore_csr(0x144, state->nonvirtual_sip);
        restore_csr(0x180, state->satp);

        // Floating-point CSRs
        // CRITICAL: Before restoring fflags/frm, we must temporarily enable mstatus.FS
        // This is because float_csr_t::unlogged_write() calls dirty_fp_state which calls
        // sstatus->dirty(). The dirty() function calls abort() if FS is disabled (FS==0).
        //
        // Strategy:
        // 1. Temporarily set mstatus.FS to Dirty (0b11) to pass the enabled() check
        // 2. Restore fflags and frm (this will trigger dirty_fp_state, but won't abort)
        // 3. Restore the correct mstatus value at the end (see FINAL STATUS RESTORE section)
        {
            // Save current mstatus and set FS to Dirty (bits 13-14 = 0b11)
            reg_t current_mstatus = state->mstatus->read();
            reg_t temp_mstatus = (current_mstatus & ~MSTATUS_FS) | MSTATUS_FS;  // FS = Dirty (0b11)
            state->mstatus->write(temp_mstatus);

            // Now safe to restore floating-point CSRs
            restore_csr(0x001, state->fflags);
            restore_csr(0x002, state->frm);
        }

        // Environment configuration CSRs
        restore_csr(0x30A, state->menvcfg);
        restore_csr(0x10A, state->senvcfg);

        // Additional machine CSRs
        restore_csr(0x323, state->csrmap.count(0x323) ? state->csrmap.at(0x323) : nullptr);
        restore_csr(0x7A0, state->tselect);
        restore_csr(0x7A2, state->tdata2);
        restore_csr(0x7A5, state->tcontrol);

        // Hypervisor CSRs (if H extension enabled)
        restore_csr(0x600, state->hstatus);
        restore_csr(0x602, state->hedeleg);
        restore_csr(0x603, state->hideleg);
        restore_csr(0x604, state->hvip);
        restore_csr(0x605, state->htimedelta);  // htimedelta
        restore_csr(0x606, state->hcounteren);
        restore_csr(0x60A, state->henvcfg);     // henvcfg
        restore_csr(0x643, state->htval);
        restore_csr(0x64A, state->htinst);      // htinst
        restore_csr(0x680, state->hgatp);

        // Additional machine mode CSRs for H extension
        restore_csr(0x34A, state->mtinst);      // mtinst
        restore_csr(0x34B, state->mtval2);      // mtval2

        // VS mode CSRs
        restore_csr(0x200, state->vsstatus);
        restore_csr(0x205, state->vstvec);
        restore_csr(0x240, state->csrmap.count(0x240) ? state->csrmap.at(0x240) : nullptr);  // vsscratch
        restore_csr(0x241, state->vsepc);
        restore_csr(0x242, state->vscause);
        restore_csr(0x243, state->vstval);
        restore_csr(0x280, state->vsatp);

        // Debug CSRs
        restore_csr(0x7B0, state->dcsr);
        restore_csr(0x7B1, state->dpc);
        restore_csr(0x7B2, state->csrmap.count(0x7B2) ? state->csrmap.at(0x7B2) : nullptr);  // dscratch0
        restore_csr(0x7B3, state->csrmap.count(0x7B3) ? state->csrmap.at(0x7B3) : nullptr);  // dscratch1

        // Debug context CSRs
        restore_csr(0x5A8, state->scontext);    // scontext
        restore_csr(0x7A8, state->mcontext);    // mcontext

        // Zcmt extension
        restore_csr(0x017, state->jvt);         // jvt

        // Zicfiss extension
        restore_csr(0x011, state->ssp);         // ssp

        // Security configuration
        restore_csr(0x747, state->mseccfg);     // mseccfg

        // AIA extension CSRs
        restore_csr(0x308, state->mvien);       // mvien
        restore_csr(0x309, state->mvip);        // mvip
        restore_csr(0x609, state->hvictl);      // hvictl
        restore_csr(0xEB0, state->vstopi);      // vstopi

        // Sstc extension CSRs
        restore_csr(0x14D, state->stimecmp);    // stimecmp
        restore_csr(0x24D, state->vstimecmp);   // vstimecmp

        // Smstateen extension CSRs
        for (int i = 0; i < 4; ++i) {
            restore_csr(0x30C + i, state->mstateen[i]);  // mstateen0-3
            restore_csr(0x10C + i, state->sstateen[i]);  // sstateen0-3
            restore_csr(0x60C + i, state->hstateen[i]);  // hstateen0-3
        }

        // CLIC/RNMI extension CSRs
        restore_csr(0x741, state->mnepc);       // mnepc
        restore_csr(0x744, state->mnstatus);    // mnstatus

        // Supervisor count inhibit (Smcdeleg extension)
        restore_csr(0x120, state->scountinhibit);  // scountinhibit

        // Counter CSRs
        // mcycle and minstret are wide_counter_csr_t with a 'written' flag
        // that asserts on consecutive writes without an intervening bump().
        // We call bump(0) first to reset the 'written' flag without incrementing.
        if (state->mcycle) {
            state->mcycle->bump(0);  // Reset 'written' flag
            restore_csr(0xB00, state->mcycle);
        }
        if (state->minstret) {
            state->minstret->bump(0);  // Reset 'written' flag
            restore_csr(0xB02, state->minstret);
        }

        // PMP CSRs - restore pmpaddr registers
        for (int i = 0; i < 64 && state->pmpaddr[i]; ++i) {
            restore_csr(0x3B0 + i, state->pmpaddr[i]);  // pmpaddr0-63
        }

        // pmpcfg CSRs (0x3A0-0x3AF)
        for (reg_t addr = 0x3A0; addr <= 0x3AF; ++addr) {
            if (state->csrmap.count(addr)) {
                restore_csr(addr, state->csrmap.at(addr));
            }
        }

        // ========== SKIPPED CSRs ==========
        // The following CSRs are intentionally NOT restored:
        //
        // 1. Read-only CSRs (cannot be written):
        //    - mvendorid (0xF11): Vendor ID
        //    - marchid (0xF12): Architecture ID
        //    - mimpid (0xF13): Implementation ID
        //    - mhartid (0xF14): Hardware thread ID
        //    - mconfigptr (0xF15): Configuration pointer
        //
        // 2. Time-related CSRs (managed by simulator):
        //    - time (0xC01): Timer register (read-only, updated by CLINT)
        //    - timeh (0xC81): Upper 32 bits of time (RV32 only)
        //
        // 3. Performance counter read proxies (derived from mcycle/minstret):
        //    - cycle (0xC00): Cycle counter (user-mode proxy for mcycle)
        //    - instret (0xC02): Instructions-retired counter (proxy for minstret)
        //    - cycleh/instreth (0xC80/0xC82): Upper 32 bits (RV32 only)
        //    - hpmcounter3-31 (0xC03-0xC1F): Performance counters (read-only)
        //
        // 4. Vector extension CSRs (if V extension used, need special handling):
        //    - vstart, vxsat, vxrm, vcsr, vl, vtype, vlenb
        //    Note: Vector register file would also need separate handling.
        //
        // 5. Crypto extension CSRs:
        //    - seed (0x015): Entropy source (read has side effects)
        // ===================================

        // ========== FINAL STATUS RESTORE ==========
        // Re-restore mstatus/sstatus AFTER floating-point CSRs to override dirty bits
        // This is necessary because restoring fflags/frm triggers dirty_fp_state
        // which sets mstatus.FS to dirty (0b11)
        restore_csr(0x300, state->mstatus);
        restore_csr(0x100, state->sstatus);

        // Restore general-purpose registers
        for (int i = 0; i < 32; ++i) {
            state->XPR.write(i, checkpoint.xpr[i]);
        }

        // Restore floating-point registers
        for (int i = 0; i < 32; ++i) {
            freg_t freg_val;
            freg_val.v[0] = checkpoint.fpr[i];
            freg_val.v[1] = 0;
            state->FPR.write(i, freg_val);
        }

        // Restore program counter
        state->pc = checkpoint.pc;

        // Restore instruction index
        current_instr_index_ = checkpoint.instr_index;

        // Restore next instruction address
        next_instruction_addr_ = checkpoint.next_instruction_addr;

        // Restore memory region (essential for correct rollback of AMO/store instructions)
        // Without this, rejected instructions' memory modifications would persist
        if (!checkpoint.mem_region_backup.empty() && mem_region_size_ > 0) {
            for (size_t i = 0; i < checkpoint.mem_region_backup.size(); ++i) {
                sim_->debug_mmu->store<uint8_t>(mem_region_start_ + i, checkpoint.mem_region_backup[i]);
            }
        }

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to restore checkpoint: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Failed to restore checkpoint: Unknown exception");
    }
}

} // namespace spike_engine
