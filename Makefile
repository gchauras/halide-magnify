#
# HALIDE_DIR must be set to root directory of Halide
#

MKDIR := mkdir -p
RM	  := rm -f
CXX	  := g++ -std=c++11 -rdynamic

BUILD_DIR  := build
SRC_DIR	   := HalideApps

HALIDE_DIR ?= $(HOME)/Projects/halide/
HALIDE_LIB := $(HALIDE_DIR)/bin/libHalide.a

SRC := $(SRC_DIR)/EulerianMagnifier.cpp \
	   $(SRC_DIR)/HalideApps.cpp \
	   $(SRC_DIR)/NamedWindow.cpp \
	   $(SRC_DIR)/RieszMagnifier.cpp \
	   $(SRC_DIR)/Util.cpp \
	   $(SRC_DIR)/VideoApp.cpp \
	   $(SRC_DIR)/filter_util.cpp \
	   $(SRC_DIR)/stdafx.cpp

OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cpp=.o)))
INC := $(wildcard *.h)

APP := phase_magnifier

all: $(APP)

CXXFLAGS += -I$(OPENCV_DIR)/include/ -I$(HALIDE_DIR)/include/ -g -Wall
LDFLAGS  += -lz -lpthread -ldl -lncurses -lopencv_core -lopencv_highgui

$(APP): $(OBJ)
	$(CXX) $(OBJ) $(HALIDE_LIB) $(LDFLAGS) -o $(APP)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(MKDIR) $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ) $(APP)
