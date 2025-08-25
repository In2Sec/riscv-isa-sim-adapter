# Makefile for Spike Python Wrapper
# 
# Usage:
#   make          - Build the Python wrapper
#   make clean    - Clean build artifacts
#   make test     - Run tests
#   make install  - Install the module system-wide
#   make help     - Show help information

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2 -g -Wall -Wextra

# Directories
SPIKE_ROOT = ..
SPIKE_BUILD_DIR = $(SPIKE_ROOT)/build
CURRENT_DIR = $(shell pwd)

# Python configuration
PYTHON = python3
PYTHON_VERSION = $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Try to find the correct python config
PYTHON_CONFIG = $(shell \
	if command -v python$(PYTHON_VERSION)-config >/dev/null 2>&1; then \
		echo "python$(PYTHON_VERSION)-config"; \
	else \
		echo "python3-config"; \
	fi)

# Python build parameters
PYBIND11_INCLUDE = $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_include())" 2>/dev/null)
PYTHON_INCLUDES = $(shell $(PYTHON_CONFIG) --includes 2>/dev/null)
PYTHON_LDFLAGS = $(shell $(PYTHON_CONFIG) --ldflags 2>/dev/null)
PYTHON_EXT_SUFFIX = $(shell $(PYTHON_CONFIG) --extension-suffix 2>/dev/null)

# Include directories
INCLUDES = -I$(SPIKE_ROOT) \
           -I$(SPIKE_BUILD_DIR) \
           -I$(SPIKE_ROOT)/riscv \
           -I$(SPIKE_ROOT)/fesvr \
           -I$(SPIKE_ROOT)/softfloat \
           -I$(SPIKE_ROOT)/disasm \
           -I$(SPIKE_ROOT)/fdt

# Source files
WRAPPER_SOURCES = spike_wrapper.cc
BINDING_SOURCES = python_binding.cc

# Object files
WRAPPER_OBJECTS = $(WRAPPER_SOURCES:.cc=.o)
BINDING_OBJECTS = $(BINDING_SOURCES:.cc=.o)

# Target files
PYTHON_MODULE = spike_wrapper$(PYTHON_EXT_SUFFIX)

# Default target
.PHONY: all
all: check-deps $(PYTHON_MODULE)

# Check dependencies
.PHONY: check-deps
check-deps:
	@echo "Checking dependencies..."
	@if [ ! -f "$(SPIKE_BUILD_DIR)/spike" ]; then \
		echo "Error: Spike not built. Please run 'make' in $(SPIKE_ROOT) first."; \
		exit 1; \
	fi
	@if ! $(PYTHON) -c "import pybind11" >/dev/null 2>&1; then \
		echo "Error: pybind11 not found. Please install it:"; \
		echo "  pip install pybind11"; \
		exit 1; \
	fi
	@if [ -z "$(PYBIND11_INCLUDE)" ]; then \
		echo "Error: Failed to get pybind11 include path"; \
		exit 1; \
	fi
	@if [ -z "$(PYTHON_EXT_SUFFIX)" ]; then \
		echo "Error: Failed to get Python extension suffix"; \
		exit 1; \
	fi
	@echo "Dependencies check passed."
	@echo "Python version: $(PYTHON_VERSION)"
	@echo "Python config: $(PYTHON_CONFIG)"
	@echo "Extension suffix: $(PYTHON_EXT_SUFFIX)"

# Build wrapper object
$(WRAPPER_OBJECTS): $(WRAPPER_SOURCES) spike_wrapper.h
	@echo "Compiling wrapper..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYTHON_INCLUDES) -c $< -o $@

# Build Python binding object
$(BINDING_OBJECTS): $(BINDING_SOURCES) spike_wrapper.h
	@echo "Compiling Python binding..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(PYBIND11_INCLUDE) $(PYTHON_INCLUDES) -c $< -o $@

# Build Python module
$(PYTHON_MODULE): $(WRAPPER_OBJECTS) $(BINDING_OBJECTS)
	@echo "Linking Python module..."
	$(CXX) -shared $(PYTHON_LDFLAGS) -o $@ $^
	@echo "Build completed successfully!"
	@echo "Python module: $(CURRENT_DIR)/$(PYTHON_MODULE)"

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f *.o
	rm -f spike_wrapper*.so
	rm -f spike_wrapper*.dylib
	rm -f *.pyc
	rm -rf __pycache__
	@echo "Clean completed."

# Run tests
.PHONY: test
test: $(PYTHON_MODULE)
	@echo "Running tests..."
	$(PYTHON) test_wrapper.py

# Run example
.PHONY: example
example: $(PYTHON_MODULE)
	@echo "Running example..."
	$(PYTHON) example_usage.py



# Help information
.PHONY: help
help:
	@echo "Spike Python Wrapper Makefile"
	@echo "============================="
	@echo ""
	@echo "Available targets:"
	@echo "  all (default) - Build the Python wrapper"
	@echo "  clean         - Clean build artifacts"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Spike must be built first (run 'make' in parent directory)"
	@echo "  - Python 3.11 with development headers"
	@echo "  - pybind11 (pip install pybind11)"
