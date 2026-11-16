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

# The coupled-node list and rim rule: parse, the startup comparison with the
# engine, the token-keep rim rule, and the stop's report echo. Same build
# arrangement as above.
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

# Section headers matched by SWMM's keyword prefix (one property, one node).
# Same build arrangement.
add_executable(test_coupled_list_review test_coupled_list_review.cpp)

target_include_directories(test_coupled_list_review PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)

target_link_libraries(test_coupled_list_review PRIVATE kokkos swmm5)

add_test(NAME UNIT-COUPLED-LIST-RV1_section_headers_by_swmm_keyword_prefix
         COMMAND test_coupled_list_review)

# How the coupling nodes are declared: the declaration resolver, the list file's
# format and the repeat refusal. Same build arrangement as above.
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

# Coupling nodes that share a TRITON cell: the per-cell exchange, its one-node
# identity with the single-node kernel, the limiter, and the cell index.
add_executable(test_coupled_cells test_coupled_cells.cpp)

target_include_directories(test_coupled_cells PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)

target_link_libraries(test_coupled_cells PRIVATE kokkos swmm5)

set(TRITON_COUPLED_CELLS_TESTS
  K1_one_node_cell_is_the_single_node_kernel
  K2_nodes_share_one_surface
  K3_limiter_drains_to_exactly_zero
  K4_filling_cell_sums_inflows
  K5_cell_index_groups_by_cell)

foreach(_t IN LISTS TRITON_COUPLED_CELLS_TESTS)
  add_test(NAME UNIT-COUPLED-CELLS-${_t}
           COMMAND test_coupled_cells ${_t})
endforeach()

# A second test file for the same per-cell exchange: a cell holding several
# coupling nodes that drains to a depth in [0, EPS12) must still conserve water.
add_executable(test_coupled_cells_review test_coupled_cells_review.cpp)

target_include_directories(test_coupled_cells_review PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)

target_link_libraries(test_coupled_cells_review PRIVATE kokkos swmm5)

add_test(NAME UNIT-COUPLED-CELLS-REVIEW-RV2_several_node_limiter_conserves_at_the_band_edge
         COMMAND test_coupled_cells_review)

# The coupling interval and lag (swmm_coupling_interval, swmm_coupling_lag): the
# bookkeeping that decides when SWMM steps and when its depths reach the surface.
# Same build arrangement as above.
add_executable(test_coupling_interval test_coupling_interval.cpp)

target_include_directories(test_coupling_interval PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver/include
  ${CMAKE_SOURCE_DIR}/external/swmm/src/solver)

target_link_libraries(test_coupling_interval PRIVATE kokkos swmm5)

# I10 reads the solver source to check the test-only switch T-D2 is wired.
target_compile_definitions(test_coupling_interval PRIVATE
  TRITON_H_PATH="${CMAKE_SOURCE_DIR}/src/triton.h")

set(TRITON_COUPLING_INTERVAL_TESTS
  I1_defaults_are_inactive
  I2_interval_mean_is_dt_weighted
  I3_one_step_interval_passes_q_unchanged
  I4_step_due_every_n_or_on_drain
  I5_lag_applies_depths_from_the_second_step_after
  I6_drain_applies_freshest_depths_under_lag
  I7_drain_leaves_nothing_carried
  I8_held_depths_between_steps
  I9_cfg_range
  I10_t_d2_switch_is_wired)

foreach(_t IN LISTS TRITON_COUPLING_INTERVAL_TESTS)
  add_test(NAME UNIT-COUPLING-INTERVAL-${_t}
           COMMAND test_coupling_interval ${_t})
endforeach()

# The T3 drain, read from src/triton.h: a source-structure test with no solver link.
add_executable(test_coupling_interval_review test_coupling_interval_review.cpp)

set(TRITON_COUPLING_INTERVAL_REVIEW_TESTS
  R1_default_drain_depends_on_run_end
  R2_default_drain_depends_on_print_interval
  R3_no_drain_build_keeps_only_the_run_end
  R4_one_drain_boolean_feeds_step_and_settle)

foreach(_t IN LISTS TRITON_COUPLING_INTERVAL_REVIEW_TESTS)
  add_test(NAME UNIT-COUPLING-INTERVAL-REVIEW-${_t}
           COMMAND test_coupling_interval_review ${CMAKE_SOURCE_DIR}/src/triton.h ${_t})
endforeach()

# A coupled cfg whose SWMM model couples no node must stop at startup with
# "couples no node" rather than finish as a surface-only run. One triton.exe launch
# under the simulation tests' run command, held to a hard 5 s budget. Registered
# only for the SERIAL backend, the one its runtime is measured on.
if("${BACKEND}" STREQUAL "SERIAL")
  add_test(NAME UNIT-ZERO-NODE-REFUSAL
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_zero_node_refusal.py
                   --run-command "${RUN_COMMAND}"
                   --exe $<TARGET_FILE:${TRITON_EXECUTABLE}>)
  set_tests_properties(UNIT-ZERO-NODE-REFUSAL PROPERTIES TIMEOUT 5 RUN_SERIAL TRUE)
endif()

# A snapshot resume of a coupled model that routes water quality must equal the
# uninterrupted run in every quality output: one clean run and two resumes from
# its checkpoint 2 (the snapshot route, and the exchange replay as the control),
# compared byte for byte in swmm/*.out and by section in the .rpt. Registered
# only for the SERIAL backend, like UNIT-ZERO-NODE-REFUSAL.
if("${BACKEND}" STREQUAL "SERIAL")
  add_test(NAME UNIT-QUALITY-RESUME
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_quality_resume.py
                   --run-command "${RUN_COMMAND}"
                   --exe $<TARGET_FILE:${TRITON_EXECUTABLE}>)
  set_tests_properties(UNIT-QUALITY-RESUME PROPERTIES TIMEOUT 120)
endif()

# Chunk 2b.1: the T6 tie-break of diagnostics::min_dt_loc (interior wins over a ghost twin).
add_executable(test_dt_tiebreak test_dt_tiebreak.cpp)
target_include_directories(test_dt_tiebreak PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_dt_tiebreak PRIVATE kokkos)
foreach(_t IN ITEMS T1_interior_wins_over_ghost_twin T2_west_ghost_twin T3_smaller_index_among_interior
                    T4_strict_ghost_minimum_reports_the_ghost)
  add_test(NAME UNIT-DT-TIEBREAK-${_t} COMMAND test_dt_tiebreak ${_t})
# Chunk 2b.1 repair: the timer-call order of super_timer start() and stop() (Other residual).
add_executable(test_supertimer_order test_supertimer_order.cpp)
foreach(_t IN ITEMS T1_start_reads_the_clock_before_the_lookup T2_stop_looks_up_once_before_the_clock
                    T3_control_old_start_is_flagged T4_control_old_stop_is_flagged)
  add_test(NAME UNIT-SUPERTIMER-ORDER-${_t}
           COMMAND test_supertimer_order ${CMAKE_SOURCE_DIR}/src/supertimer.h ${_t})
endforeach()

# Chunk 2b.1 reviewer test: source-structure properties of the surface captures (rule 6).
# Needs no link line; it reads src/diagnostics.h and src/kernels.h beside the triton.h path.
add_executable(test_surface_diag_review test_surface_diag_review.cpp)
set(TRITON_SURFACE_DIAG_REVIEW_TESTS
  S1_capture_code_never_writes_through_a_solver_pointer
  S2_collectives_are_not_under_a_rank_dependent_condition
  S3_capture_calls_sit_only_at_rule_6_positions
  S4_every_diag_bracket_opens_with_no_other_category_open
  S5_print_instant_row_precedes_the_flush_and_the_cfg
  S6_t6_collective_follows_the_min_allreduce
  S7_close_row_writes_pending_rows_before_closing_the_files)
foreach(_t IN LISTS TRITON_SURFACE_DIAG_REVIEW_TESTS)
  add_test(NAME UNIT-SURFACE-DIAG-REVIEW-${_t}
           COMMAND test_surface_diag_review ${CMAKE_SOURCE_DIR}/src/triton.h ${_t})
endforeach()
