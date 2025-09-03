function(add_triton_test test_name cfg_name)
  file(COPY ${TRITON_SOURCE_DIR}/test/${cfg_name}
       DESTINATION ${TRITON_BUILD_DIR}/test)
  set(_TEST_CMD "${RUN_COMMAND} ../${TRITON_EXECUTABLE} ./${cfg_name}")
  string (STRIP "${_TEST_CMD}" _TEST_CMD_STRIPPED)
  string (REPLACE " " ";" _CMD_LIST "${_TEST_CMD_STRIPPED}")
  add_test(
    NAME ${test_name}
    COMMAND ${_CMD_LIST}
  )
endfunction()

add_triton_test(
    PARABOLOID
    paraboloid.cfg
)

add_test(
    NAME BIT4BIT-PARABOLOID
    COMMAND ${Python3_EXECUTABLE} 
                ${CMAKE_BINARY_DIR}/compare_runs_simple.py
                --ref ${CMAKE_SOURCE_DIR}/test/reference/paraboloid
                --test ${CMAKE_BINARY_DIR}/test/output_paraboloid/asc
)
add_triton_test(
    CIRCULAR-DAMBREAK
    circular_dambreak.cfg
)

add_test(
    NAME BIT4BIT-CIRCULAR-DAMBREAK
    COMMAND ${Python3_EXECUTABLE} 
                ${CMAKE_BINARY_DIR}/compare_runs_simple.py
                --ref ${CMAKE_SOURCE_DIR}/test/reference/circular
                --test ${CMAKE_BINARY_DIR}/test/output_circular_dambreak/asc
)

add_triton_test(
    ALLATOONA
    allatoona.cfg
)

add_test(
    NAME BIT4BIT-ALLATOONA
    COMMAND ${Python3_EXECUTABLE} 
                ${CMAKE_BINARY_DIR}/compare_runs_simple.py
                --ref ${CMAKE_SOURCE_DIR}/test/reference/allatoona
                --test ${CMAKE_BINARY_DIR}/test/output_allatoona/asc
)
