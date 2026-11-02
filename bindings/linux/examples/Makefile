# Build the C examples against a LinnetTTS distribution directory.
#   make DIST=../dist/linnet-tts-0.0.1-linux-x86_64          # shared library (default)
#   make DIST=... STATIC=1                                     # static library
# Inside an unpacked distribution archive `make` works as is (DIST defaults to ..).
DIST ?= ..
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra
CPPFLAGS += -I$(DIST)/include

ifeq ($(STATIC),1)
  LIBS = $(DIST)/lib/liblinnet_tts.a -lpthread -ldl -lm
else
  LIBS = -L$(DIST)/lib -llinnet_tts -Wl,-rpath,$(abspath $(DIST)/lib) -lm
endif

all: synth stream

synth: synth.c wav.h
	$(CC) $(CFLAGS) $(CPPFLAGS) synth.c $(LIBS) -o $@

stream: stream.c wav.h
	$(CC) $(CFLAGS) $(CPPFLAGS) stream.c $(LIBS) -o $@

clean:
	rm -f synth stream

.PHONY: all clean
