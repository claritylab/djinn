# This file was generated using the following command:
# ./configure 

# Rules that enable valgrind debugging ("make valgrind")

valgrind: .valgrind

.valgrind:
	echo -n > valgrind.out
	for x in $(TESTFILES); do echo $$x>>valgrind.out; valgrind ./$$x >/dev/null 2>> valgrind.out; done
	! ( grep 'ERROR SUMMARY' valgrind.out | grep -v '0 errors' )
	! ( grep 'definitely lost' valgrind.out | grep -v -w 0 )
	rm valgrind.out
	touch .valgrind


CONFIGURE_VERSION := 2
OPENFSTLIBS = -L/home/ypkang/git/brainiac/tonic-suite/asr/tools/openfst/lib -lfst
OPENFSTLDFLAGS = -Wl,-rpath=/home/ypkang/git/brainiac/tonic-suite/asr/tools/openfst/lib
FSTROOT = /home/ypkang/git/brainiac/tonic-suite/asr/tools/openfst
ATLASINC = /home/ypkang/kaldi-trunk/tools/ATLAS/include
ATLASLIBS = /usr/lib/atlas-base/libatlas.so.3.0 /usr/lib/atlas-base/libf77blas.so.3.0 /usr/lib/atlas-base/libcblas.so.3 /usr/lib/atlas-base/liblapack_atlas.so.3 -Wl,-rpath=/usr/lib/atlas-base
# You have to make sure ATLASLIBS is set...

ifndef FSTROOT
$(error FSTROOT not defined.)
endif

ifndef ATLASINC
$(error ATLASINC not defined.)
endif

ifndef ATLASLIBS
$(error ATLASLIBS not defined.)
endif


CXXFLAGS = -msse -msse2 -Wall -I.. \
	   -pthread \
      -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN \
      -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self \
      -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
      -DHAVE_ATLAS -I$(ATLASINC) \
      -I$(FSTROOT)/include \
      $(EXTRA_CXXFLAGS) \
      -g # -O0 -DKALDI_PARANOID 

ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = -rdynamic $(OPENFSTLDFLAGS)
LDLIBS = $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(ATLASLIBS) -lm -lpthread -ldl -lglog
CC = g++
CXX = g++
AR = ar
AS = as
RANLIB = ranlib
