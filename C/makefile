# Makefile Template

# Intel MKL configuration
#MKLROOT ?= /opt/intel/oneapi/mkl/latest
MKL_INCLUDE := -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS := -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

# Compiler and flags
CC := icx
CFLAGS := -Wall -Wextra -O2 $(MKL_INCLUDE)
LDFLAGS := $(MKL_LIBS)

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
