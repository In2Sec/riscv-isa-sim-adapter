// Copyright (c) 2024-2025 DiveFuzz Project
// SPDX-License-Identifier: Mulan PSL v2

#ifndef SPIKE_ENGINE_H
#define SPIKE_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <map>

// Forward declarations
class sim_t;
class processor_t;
class cfg_t;

namespace spike_engine {

// Special value to indicate instruction without immediate operand
constexpr int64_t IMMEDIATE_NOT_PRESENT = INT64_MIN;

/**
 * Execution result containing register values before and after instruction execution
 *
 * This structure is returned by execute_instruction() and contains:
 * - source_values_before: Source register values BEFORE execution (for XOR computation)
 * - dest_values_after: Destination register values AFTER execution (for bug filtering)
 *
 * Design Rationale:
 * - Python layer computes XOR from source_values_before for deduplication
 * - Python layer uses dest_values_after for bug pattern matching
 * - Separating these concerns keeps spike_engine simple and focused on execution
 */
struct ExecutionResult {
    // Source register values captured BEFORE instruction execution
    // Used for XOR-based deduplication in Python layer
    std::vector<uint64_t> source_values_before;

    // Destination register values captured AFTER instruction execution
    // Used for bug filtering in Python layer (e.g., checking if sc.w returned 1)
    std::vector<uint64_t> dest_values_after;

    ExecutionResult() = default;
    ExecutionResult(const std::vector<uint64_t>& src, const std::vector<uint64_t>& dst)
        : source_values_before(src), dest_values_after(dst) {}
};

/**
 * Checkpoint state for processor
 *
 * Represents a lightweight snapshot of processor state at a specific point.
 * Designed for efficient rollback during instruction validation (e.g., when
 * XOR value collisions occur and we need to retry with different instructions).
 *
 * Design Philosophy:
 *   - Minimal footprint: Only stores essential state (registers, PC, index)
 *   - Fast restore: Uses vector truncation instead of memory writes
 *   - Single checkpoint: Optimized for the common pattern of one active checkpoint
 *
 * Memory Consistency:
 *   After restore, memory may contain instructions that were executed after
 *   the checkpoint, but this is safe because:
 *   1. PC and next_instruction_addr point to the checkpoint position
 *   2. Subsequent execute_instruction() calls will overwrite old instructions
 *   3. Instructions past the checkpoint position will never be executed
 */
struct Checkpoint {
    // General purpose registers (x0-x31)
    std::vector<uint64_t> xpr;

    // Floating point registers (f0-f31)
    std::vector<uint64_t> fpr;

    // Program counter
    uint64_t pc;

    // Index into instruction sequence
    // Marks how many instructions have been executed at checkpoint time
    size_t instr_index;

    // Next available memory address for instruction placement
    uint64_t next_instruction_addr;

    // Memory region backup for checkpoint/restore
    // This is essential for correct rollback when instructions modify memory
    // (e.g., AMO instructions, store instructions)
    std::vector<uint8_t> mem_region_backup;

    // ========== Privilege and Mode State ==========
    // Privilege level (0=U, 1=S, 3=M)
    // Essential for correct trap handling after restore
    uint64_t prv;

    // Virtualization mode (for H extension)
    bool v;

    // Debug mode flag
    bool debug_mode;

    // ========== All CSRs ==========
    // Complete CSR state captured as address->value map
    // This ensures ALL CSRs are properly restored, including:
    // - Trap handling CSRs (mtvec, mepc, mcause, mtval, etc.)
    // - Status registers (mstatus, sstatus, etc.)
    // - Interrupt/exception delegation registers
    // - PMP configuration
    // - Floating-point CSRs (fflags, frm, fcsr)
    // - Custom CSRs
    std::map<uint64_t, uint64_t> csr_values;

    Checkpoint();
};

/**
 * SpikeEngine: Efficient Spike execution engine with checkpointing
 *
 * A lightweight wrapper around the Spike RISC-V ISA simulator, designed for
 * instruction-by-instruction execution with support for mixed-length instructions
 * (16-bit compressed and 32-bit standard) and efficient checkpoint/restore.
 *
 * Key Features:
 *   - Mixed-length instruction support (RV*C compressed + standard)
 *   - Dynamic address allocation (no pre-allocated slots)
 *   - Lightweight checkpointing (registers + index only)
 *   - XOR-based instruction validation for fuzzing
 *
 * Typical Workflow:
 *   1. Initialize with pre-compiled ELF template (containing nop placeholder region)
 *   2. Engine executes template initialization code until reaching main()
 *   3. For each instruction to test:
 *      a. set_checkpoint() - save current processor state
 *      b. execute_instruction() - write instruction to memory and execute
 *      c. Check if XOR value is unique (collision detection)
 *      d. If collision: restore_checkpoint() and retry with different instruction
 *         If unique: proceed to next instruction (checkpoint will be updated)
 *
 * Memory Management:
 *   Instructions are written sequentially starting from main symbol address.
 *   2-byte compressed instructions are detected automatically and written with
 *   correct size. PC tracking ensures proper alignment for mixed-length sequences.
 *
 * Thread Safety:
 *   Not thread-safe. Each thread should use its own SpikeEngine instance.
 *
 * Example:
 *   SpikeEngine engine("template.elf", "rv64gc", 1000);
 *   engine.initialize();
 *
 *   engine.set_checkpoint();
 *   uint64_t xor_val = engine.execute_instruction(0x003100b3, {2, 3}, 0);
 *   if (is_duplicate(xor_val)) {
 *       engine.restore_checkpoint();  // Try different instruction
 *   }
 */
class SpikeEngine {
public:
    /**
     * Constructor
     * @param elf_path Path to pre-compiled ELF file
     * @param isa ISA string (e.g., "rv64gc")
     * @param num_instrs Number of instructions to generate (nops in ELF)
     * @param verbose Enable verbose output (default: false)
     */
    SpikeEngine(const std::string& elf_path,
                const std::string& isa,
                size_t num_instrs,
                bool verbose = false);

    /**
     * Destructor
     */
    ~SpikeEngine();

    /**
     * Initialize Spike and run until main function entry
     * This executes template initialization code
     * @return true on success, false on error
     */
    bool initialize();

    /**
     * Create a checkpoint of current processor state
     * Saves PC, registers, and modified memory
     */
    void set_checkpoint();

    /**
     * Restore processor state from last checkpoint
     */
    void restore_checkpoint();

    /**
     * Execute a sequence of instructions (for jump sequences)
     *
     * Writes all instructions to memory first, then executes them sequentially.
     * Used for forward jumps where we need to execute jump + middle instructions.
     *
     * @param machine_codes List of machine codes to execute
     * @param sizes List of instruction sizes (2 or 4 bytes each)
     * @return Number of instructions successfully executed
     *
     * @throws std::runtime_error if execution fails
     */
    size_t execute_instruction_sequence(
        const std::vector<uint32_t>& machine_codes,
        const std::vector<size_t>& sizes);

    /**
     * Execute a loop sequence until branch condition fails
     *
     * Structure: init + (loop_body + decr + branch)*
     * Executes init once, then loops body+decr+branch until branch doesn't jump back.
     *
     * @param init_code Initialization instruction (e.g., li s11, 5)
     * @param init_size Size of init instruction
     * @param loop_body_codes Instructions in loop body
     * @param loop_body_sizes Sizes of loop body instructions
     * @param decr_code Decrement instruction (e.g., addi s11, s11, -1)
     * @param decr_size Size of decrement instruction
     * @param branch_code Branch instruction (e.g., bne s11, zero, offset)
     * @param branch_size Size of branch instruction
     * @param max_iterations Maximum iterations (safety limit)
     * @return Actual number of iterations executed
     *
     * @throws std::runtime_error if execution fails
     */
    size_t execute_loop_sequence(
        uint32_t init_code,
        size_t init_size,
        const std::vector<uint32_t>& loop_body_codes,
        const std::vector<size_t>& loop_body_sizes,
        uint32_t decr_code,
        size_t decr_size,
        uint32_t branch_code,
        size_t branch_size,
        size_t max_iterations = 100);

    /**
     * Execute one instruction and return register values for XOR and bug filtering
     *
     * Automatically detects instruction size (2 or 4 bytes), writes it to the
     * next available memory address, executes it, and captures register values
     * before and after execution.
     *
     * Process:
     *   1. Detect instruction size from machine_code bits[1:0]
     *   2. Verify sufficient space in instruction region
     *   3. Verify PC matches expected address (consistency check)
     *   4. Write instruction to memory (2 or 4 bytes as appropriate)
     *   5. Read source register values (BEFORE execution) -> source_values_before
     *   6. Execute instruction (single step)
     *   7. Read destination register values (AFTER execution) -> dest_values_after
     *   8. Update next_instruction_addr to current PC
     *   9. Increment instruction index counter
     *  10. Return ExecutionResult with both sets of values
     *
     * @param machine_code 32-bit instruction encoding (may be 16-bit compressed
     *                     instruction zero-padded to 32 bits)
     * @param source_regs List of source register indices (0-31) to read before execution
     * @param dest_regs List of destination register indices (0-31) to read after execution
     * @param immediate Optional immediate value to include in source_values (default: 0)
     *
     * @return ExecutionResult containing:
     *   - source_values_before: Source register values + immediate (for XOR in Python)
     *   - dest_values_after: Destination register values (for bug filtering in Python)
     *
     * @throws std::runtime_error if:
     *   - Engine not initialized
     *   - Out of instruction region space
     *   - PC mismatch (internal consistency error)
     *   - Memory write failure
     *   - Instruction execution failure
     *
     * @note For compressed instructions (bits[1:0] != 0b11), only the low 16 bits
     *       are written to memory. Spike automatically updates PC by 2 or 4 bytes
     *       based on the instruction encoding.
     */
    ExecutionResult execute_instruction(uint32_t machine_code,
                                       const std::vector<int>& source_regs,
                                       const std::vector<int>& dest_regs,
                                       int64_t immediate = 0);

    /**
     * Get value of a general-purpose register
     * @param reg_index Register index (0-31)
     * @return Register value
     */
    uint64_t get_xpr(int reg_index) const;

    /**
     * Get value of a floating-point register
     * @param reg_index Register index (0-31)
     * @return Register value (as uint64_t)
     */
    uint64_t get_fpr(int reg_index) const;

    /**
     * Get program counter value
     * @return PC value
     */
    uint64_t get_pc() const;

    /**
     * Get all general-purpose register values
     * @return Vector of 32 register values (x0-x31)
     */
    std::vector<uint64_t> get_all_xpr() const;

    /**
     * Get all floating-point register values
     * @return Vector of 32 register values (f0-f31)
     */
    std::vector<uint64_t> get_all_fpr() const;

    /**
     * Get a CSR value by address
     * @param csr_addr CSR address (e.g., 0x300 for mstatus)
     * @return CSR value, or 0 if not found/accessible
     */
    uint64_t get_csr(uint64_t csr_addr) const;

    /**
     * Get all accessible CSR values
     * @return Map of CSR address -> value
     */
    std::map<uint64_t, uint64_t> get_all_csrs() const;

    /**
     * Get mem_region start address
     * @return Start address of mem_region (for testing memory operations)
     */
    uint64_t get_mem_region_start() const { return mem_region_start_; }

    /**
     * Get mem_region size
     * @return Size of mem_region in bytes
     */
    size_t get_mem_region_size() const { return mem_region_size_; }

    /**
     * Read memory at specified address
     * @param addr Memory address to read from
     * @param size Number of bytes to read
     * @return Vector of bytes read from memory
     */
    std::vector<uint8_t> read_mem(uint64_t addr, size_t size) const;

    /**
     * Get current instruction index
     * @return Index of next instruction to replace
     */
    size_t get_current_index() const { return current_instr_index_; }

    /**
     * Get total number of instructions
     */
    size_t get_num_instrs() const { return num_instrs_; }

    /**
     * Get last error message
     */
    std::string get_last_error() const { return last_error_; }

    /**
     * Detect instruction size from machine code (static utility)
     *
     * Determines whether an instruction is 16-bit compressed or 32-bit standard
     * based on RISC-V encoding rules specified in the ISA manual.
     *
     * RISC-V Instruction Length Encoding (bits[1:0]):
     *   - bits[1:0] != 0b11  →  16-bit compressed instruction (RV*C)
     *   - bits[1:0] == 0b11  →  32-bit or longer instruction
     *     - If bits[4:2] != 0b111  →  32-bit standard instruction
     *     - If bits[4:2] == 0b111  →  48/64-bit instruction (not supported)
     *
     * @param machine_code 32-bit instruction encoding (may contain zero-padded
     *                     16-bit compressed instruction in lower half)
     *
     * @return Instruction size in bytes: 2 for compressed, 4 for standard
     *
     * @throws std::runtime_error if instruction length > 32 bits (unsupported)
     *
     * @note This is a static method and can be called without an instance:
     *       size_t sz = SpikeEngine::get_instruction_size(0x00008522);
     */
    static size_t get_instruction_size(uint32_t machine_code);

private:
    // Configuration
    std::string elf_path_;
    std::string isa_;
    size_t num_instrs_;
    bool verbose_;

    // Spike simulator
    std::unique_ptr<cfg_t> cfg_;
    std::unique_ptr<sim_t> sim_;
    processor_t* proc_;  // Main processor (core 0)

    // Instruction region
    uint64_t instruction_region_start_;
    uint64_t instruction_region_end_;
    uint64_t next_instruction_addr_;  // Next available address for instruction

    // Memory region for checkpoint/restore (e.g., .mem_region section)
    uint64_t mem_region_start_;
    size_t mem_region_size_;

    // Execution state
    size_t current_instr_index_;

    Checkpoint checkpoint_;
    bool checkpoint_valid_;
    bool initialized_;

    // Error handling
    std::string last_error_;

    // Internal helper methods

    /**
     * Find instruction region using ELF symbols
     * Reads 'main' symbol from ELF to locate instruction region
     */
    bool find_nop_region();

    /**
     * Read symbol address from ELF file
     * Uses objdump to extract symbol address
     * @param symbol_name Symbol name to look up (e.g., "main")
     * @return Symbol address, or 0 if not found
     */
    uint64_t read_symbol_address(const std::string& symbol_name);

    /**
     * Write machine code to memory address
     * @param addr Physical address
     * @param code Machine code
     * @param size Instruction size (2 or 4 bytes)
     */
    bool write_memory(uint64_t addr, uint32_t code, size_t size);

    /**
     * Read memory from physical address
     * @param addr Physical address
     * @return 32-bit value
     */
    uint32_t read_memory(uint64_t addr);

    /**
     * Check if PC is within the instruction region
     * @param pc Program counter value to check
     * @return true if PC is within [instruction_region_start_, instruction_region_end_)
     */
    bool is_in_instruction_region(uint64_t pc) const;

    /**
     * Single step execution
     * Executes one instruction on processor
     */
    bool step_processor();

    /**
     * Step execution with trap handling
     *
     * Executes one instruction and continues stepping if the PC ends up
     * outside the instruction region (e.g., in a trap handler). This ensures
     * that trap handlers are fully executed before returning control.
     *
     * @return true if execution completed successfully with PC back in instruction region
     * @return false if:
     *   - Initial step failed
     *   - Trap handler execution exceeded MAX_STEPS limit
     *   - Exception occurred during trap handler execution
     */
    bool step_until_in_region();

    /**
     * Compute XOR value from register values
     * XOR = (v[0] << 0) ^ (v[1] << 1) ^ (v[2] << 2) ^ ...
     */
    uint64_t compute_xor(const std::vector<uint64_t>& values) const;

    /**
     * Save processor state to checkpoint
     */
    void save_state(Checkpoint& checkpoint);

    /**
     * Restore processor state from checkpoint
     */
    void restore_state(const Checkpoint& checkpoint);
};

} // namespace spike_engine

#endif // SPIKE_ENGINE_H
