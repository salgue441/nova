# Intel MKL detection
set(MKL_ROOT $ENV{MKLROOT} CACHE PATH "MKL root directory")

find_path(MKL_INCLUDE_DIR
  NAMES mkl.h
  PATHS ${MKL_ROOT}/include
)

find_library(MKL_CORE_LIB
  NAMES mkl_core
  PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64
)

find_library(MKL_THREAD_LIB
  NAMES mkl_gnu_thread mkl_intel_thread
  PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64
)

find_library(MKL_INTERFACE_LIB
  NAMES mkl_intel_lp64
  PATHS ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL
  REQUIRED_VARS
  MKL_INCLUDE_DIR
  MKL_CORE_LIB
  MKL_THREAD_LIB
  MKL_INTERFACE_LIB
)

if(MKL_FOUND AND NOT TARGET MKL::MKL)
  add_library(MKL::MKL INTERFACE IMPORTED)
  target_include_directories(MKL::MKL
    INTERFACE ${MKL_INCLUDE_DIR}
  )
  target_link_libraries(MKL::MKL
    INTERFACE
    ${MKL_INTERFACE_LIB}
    ${MKL_THREAD_LIB}
    ${MKL_CORE_LIB}
    Threads::Threads
  )
endif()