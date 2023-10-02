PLATFORM?=win64

ifeq ($(PLATFORM), linux)
	EXECOUT = -shared -o
	BINEXT=.so
	PYTHON_DIR = lnx_x64
	PYTHON_INC_PATH = /python3.8
	DIR = lnx_x64
	PYTHON_LIB = -L$(AME)/libcosim/lib/$(DIR) -lgeneric_cosim -lm -lrt -L$(AME)/sys/python/$(PYTHON_DIR)/lib -lpython3.8
	CC = gcc
	CFLAGS = -fPIC -m64
else ifeq ($(PLATFORM), win64)
	EXECOUT = -LD -Fe
	BINEXT = .dll
	PYTHON_DIR = win64
	PYTHON_INC_PATH =
	DIR = win64_vc140
	PYTHON_LIB = $(AME)/libcosim/lib/$(DIR)/generic_cosim.lib ws2_32.lib $(AME)/sys/python/$(PYTHON_DIR)/libs/python38.lib
	CC = cl
	CFLAGS = -DWIN32 -DWIN64
else
$(error $(PLATFORM) not supported.)
endif

CFLAGS := $(CFLAGS) -I$(AME)/libcosim/include -I$(AME)/sys/python/$(PYTHON_DIR)/include$(PYTHON_INC_PATH)

PYEXT=.pyd

all: $(DIR) $(DIR)/binding_amecommunication$(PYEXT) 

$(DIR):
	mkdir -p $@

$(DIR)/binding_amecommunication$(BINEXT): binding_amecommunication.c
	$(CC) $(CFLAGS) $< $(PYTHON_LIB) $(EXECOUT)$@

$(DIR)/binding_amecommunication$(PYEXT): $(DIR)/binding_amecommunication$(BINEXT)
	cp $< $@

clean:
	rm -f *$(PYEXT) *$(BINEXT) *.o *.obj
