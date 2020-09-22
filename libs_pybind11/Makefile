LIB_NAME = build/network_python

EMBEDDED_NEURAL_NETWORK_LIB_PATH = ../embedded_neural_nework/

CC 	= g++



CFLAGS = -O3 -funroll-loops -Wall -Wextra -pedantic -std=c++17 -fPIC -shared -undefined dynamic_lookup

INC = -I.
INC+= -I$(EMBEDDED_NEURAL_NETWORK_LIB_PATH)
INC+= `python3-config --include`

SRCS = $(shell find . -name '*.cpp')
SRCS+= $(shell find $(EMBEDDED_NEURAL_NETWORK_LIB_PATH) -name '*.cpp')

OBJS = $(SRCS:.cpp=.o)

all: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(LIB_NAME)`python3-config --extension-suffix`

%.o: %.cpp
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

clean:
	rm -f $(LIB_NAME)*
	rm -f $(OBJS)
	
