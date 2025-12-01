# Makefile for INTER
# Detect operating system
UNAME_S := $(shell uname -s)

# Fortran compiler and flags
FC = gfortran
FFLAGS = -O3 -c -fcheck=all -ffree-form -fdefault-double-8 -fdefault-real-8 -Wall

# Directories
SRCDIR = ../src
MODDIR = ../mod
OBJDIR = ../obj
BINDIR = .

# Executable name
EXECUTABLE = INTER

# Source files in dependency order
SOURCES = Lagrange_Interpolation_Method.f90 \
          Piecewise_Polynomial_Interpolation.f90 \
          INTERP.f90 \
          mainI.f90

# Object files
OBJECTS = $(SOURCES:.f90=.o)
OBJ_FILES = $(addprefix $(OBJDIR)/, $(OBJECTS))

# Platform-specific BLAS/LAPACK libraries
ifeq ($(UNAME_S),Darwin)
    # macOS: use Accelerate framework
    LDFLAGS = -framework Accelerate
else ifeq ($(UNAME_S),Linux)
    # Linux: try MKL first, fall back to OpenBLAS
    MKL_ROOT ?= /opt/intel/oneapi/mkl/latest
    ifneq ($(wildcard $(MKL_ROOT)),)
        # Intel MKL found
        LDFLAGS = -L$(MKL_ROOT)/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
    else
        # Fall back to OpenBLAS
        LDFLAGS = -lopenblas -llapack
    endif
else
    # Other platforms: assume standard BLAS/LAPACK
    LDFLAGS = -lblas -llapack
endif

# Default target
all: $(BINDIR)/$(EXECUTABLE)

# Link the executable
$(BINDIR)/$(EXECUTABLE): $(OBJ_FILES)
	$(FC) -o $@ $^ $(LDFLAGS)

# Compile rules for each source file (in dependency order)
$(OBJDIR)/Lagrange_Interpolation_Method.o: $(SRCDIR)/Lagrange_Interpolation_Method.f90
	$(FC) $(FFLAGS) -J$(MODDIR) -o $@ $<

$(OBJDIR)/Piecewise_Polynomial_Interpolation.o: $(SRCDIR)/Piecewise_Polynomial_Interpolation.f90 $(OBJDIR)/Lagrange_Interpolation_Method.o
	$(FC) $(FFLAGS) -J$(MODDIR) -I$(MODDIR) -o $@ $<

$(OBJDIR)/INTERP.o: $(SRCDIR)/INTERP.f90 $(OBJDIR)/Piecewise_Polynomial_Interpolation.o
	$(FC) $(FFLAGS) -J$(MODDIR) -I$(MODDIR) -o $@ $<

$(OBJDIR)/mainI.o: $(SRCDIR)/mainI.f90 $(OBJDIR)/INTERP.o
	$(FC) $(FFLAGS) -J$(MODDIR) -I$(MODDIR) -o $@ $<

# Clean target
clean:
	rm -f $(OBJDIR)/*.o $(MODDIR)/*.mod $(BINDIR)/$(EXECUTABLE)

.PHONY: all clean
