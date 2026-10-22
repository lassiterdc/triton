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
  T10_header_write_layout
  T11_header_roundtrip_ok
  T1_stored_width_disagreement_is_refused
  T12_node_count_mismatch_refused
  T13_short_header_is_bad_magic
  T14_validation_order_magic_then_nodes_then_width
  T2_v1_legacy_file_lands_on_bad_magic
  T15_v1_third_field_is_record_bytes_not_a_width
  T16_v2_file_is_not_mistaken_for_legacy)

foreach(_t IN LISTS TRITON_EXCHANGE_LOG_HEADER_TESTS)
  add_test(NAME UNIT-EXCHANGE-LOG-${_t}
           COMMAND test_exchange_log_header ${_t})
endforeach()

# The coupled-node list and rim rule (item 11): parse, startup comparison with the
# engine, and the token-keep rim rule. Same build arrangement as above.
add_executable(test_coupled_list test_coupled_list.cpp)

target_include_directories(test_coupled_list PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)

target_link_libraries(test_coupled_list PRIVATE kokkos swmm5)

set(TRITON_COUPLED_LIST_TESTS
  P1_standard_input_gives_todays_list
  P2_header_layouts
  P3_order_follows_junctions
  P4_flow_test_is_swmms
  P5_names_case_insensitive
  P6_tokenizer
  C1_valid_list_passes
  C2_out_of_order_refused
  C3_entry_without_flow_refused
  C4_compensating_pair_refused
  C5_unequal_counts_refused
  C6_unknown_name_refused
  E1_error_line_matcher
  E2_echo_over_a_report
  R1_rim_rule
  U1_us_flow_units_mean_feet
  U2_engine_length_in_metres
  U3_us_rim_never_keeps_the_token)

foreach(_t IN LISTS TRITON_COUPLED_LIST_TESTS)
  add_test(NAME UNIT-COUPLED-LIST-${_t}
           COMMAND test_coupled_list ${_t})
endforeach()

# The reviewer's section-header test (one property, one node). Same build arrangement.
add_executable(test_coupled_list_review test_coupled_list_review.cpp)

target_include_directories(test_coupled_list_review PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)

target_link_libraries(test_coupled_list_review PRIVATE kokkos swmm5)

add_test(NAME UNIT-COUPLED-LIST-RV1_section_headers_by_swmm_keyword_prefix
         COMMAND test_coupled_list_review)

# How the coupling nodes are declared (item 16): the declaration resolver, the list
# file's format and the repeat refusal. Same build arrangement as above.
add_executable(test_coupling_declaration test_coupling_declaration.cpp)

target_include_directories(test_coupling_declaration PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)

target_link_libraries(test_coupling_declaration PRIVATE kokkos swmm5)

set(TRITON_COUPLING_DECLARATION_TESTS
  D1_list_is_exactly_the_list
  D2_all_junctions_is_exactly_the_junctions
  D3_legacy_is_exactly_the_flow_set
  D4_list_file_format
  D5_repeated_name_after_case_fold)

foreach(_t IN LISTS TRITON_COUPLING_DECLARATION_TESTS)
  add_test(NAME UNIT-COUPLING-DECLARATION-${_t}
           COMMAND test_coupling_declaration ${_t})
endforeach()
