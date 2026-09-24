# Tier-1 unit tests: no solver run, no cluster, no SWMM call.
#
# They include swmm_triton.h directly, so they are built only in a coupled
# (TRITON_ENABLE_SWMM=ON) configuration -- the header is entirely inside
# #ifdef TRITON_SWMM and compiles to nothing otherwise.

add_executable(test_exchange_log_header test_exchange_log_header.cpp)

target_include_directories(test_exchange_log_header PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)

target_link_libraries(test_exchange_log_header PRIVATE kokkos swmm5)

# One CTest node id per property, so a result is accounted by name rather than
# by a pass/fail total that cannot say which property it contains.
set(TRITON_EXCHANGE_LOG_HEADER_TESTS
  T9_magic_v2_differs_from_retained_v1
  T10_header_write_layout)

foreach(_t IN LISTS TRITON_EXCHANGE_LOG_HEADER_TESTS)
  add_test(NAME UNIT-EXCHANGE-LOG-${_t}
           COMMAND test_exchange_log_header ${_t})
endforeach()
