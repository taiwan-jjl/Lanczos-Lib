# Makefile Template

#------------------------------------------------------------------------------
# Makefile for pure-C cuBLAS app (static link) using NVIDIA HPC SDK 25.3
# Module loaded: nvhpc
#------------------------------------------------------------------------------

# Compute capability for A3000 (Ampere, CC 8.6)
SM       := 86

# Compiler and flags
CC := nvc
CFLAGS := -O2 -std=c23 -cuda -gpu=cc$(SM),sm_$(SM) -Minfo=all -tp=host	# std is "C23", be careful for -O3
LDFLAGS := -lm -cuda -lcublas_static -lculibos

# Project structure
SRC_DIR := .#src
OBJ_DIR := .#obj
BIN_DIR := .#bin
TARGET := $(BIN_DIR)/basic_lanczos

# Find all source files and corresponding object files
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# Default rule
all: $(TARGET)

# Link the target
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Build complete."

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
# clean:
# 	rm -rf $(OBJ_DIR) $(BIN_DIR)
# 	@echo "Cleaned."
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)
	@echo "Cleaned object files and executable."


# Phony targets
.PHONY: all clean
