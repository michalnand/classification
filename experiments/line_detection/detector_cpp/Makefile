ELF = bin/embedded_neural_nework_test.elf

MODEL_PATH = ../models/net_0/export/



EMBEDDED_NEURAL_NETWORK_PATH = ../../../embedded_neural_nework/

CXX_SRCS+= $(shell find  . -name '*.cpp')
CXX_SRCS+= $(shell find  $(EMBEDDED_NEURAL_NETWORK_PATH) -name '*.cpp')
CXX_SRCS+= $(shell find  $(MODEL_PATH) -name '*.cpp')

INC = -I .
INC+= -I $(EMBEDDED_NEURAL_NETWORK_PATH)
INC+= -I $(MODEL_PATH)

CXX = g++
LD  = g++

OBJDUMP = objdump
SIZE = size

OPTIM_FLAGS = -march=native -O3 -s -funroll-loops
CXXFLAGS = -c $(OPTIM_FLAGS) -Wall -Wextra -pedantic -std=c++17 -Wno-unused-command-line-argument
CXX_OBJS = $(CXX_SRCS:.cpp=.o)

LDFLAGS = $(OPTIM_FLAGS) -Wall -Wextra -pedantic -std=c++17 -Wno-unused-command-line-argument

CXXFLAGS+= `pkg-config --cflags --libs /usr/local/Cellar/opencv/4.4.0_1/lib/pkgconfig/opencv4.pc`
LDFLAGS+=  `pkg-config --cflags --libs /usr/local/Cellar/opencv/4.4.0_1/lib/pkgconfig/opencv4.pc`

all: $(ELF)

$(ELF): $(CXX_OBJS)
	$(LD) $(LDFLAGS) -o $@ $(CXX_OBJS)
	$(OBJDUMP) --disassemble $(ELF) > $(ELF).asm
	$(SIZE) $(ELF)

%.o: %.cpp
	$(CXX) $(INC) $(CXXFLAGS) $< -o $@

clean:
	rm $(CXX_OBJS) -rf
