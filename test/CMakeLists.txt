function(add_triton_test test_name cfg_name)
  file(COPY ${TRITON_SOURCE_DIR}/test/${cfg_name}
       DESTINATION ${TRITON_BUILD_DIR}/test)
  set(_TEST_CMD "${RUN_COMMAND} ../triton.exe ./${cfg_name}")
  string (STRIP "${_TEST_CMD}" _TEST_CMD_STRIPPED)
  string (REPLACE " " ";" _CMD_LIST "${_TEST_CMD_STRIPPED}")
  add_test(
    NAME ${test_name}
    COMMAND ${_CMD_LIST}
  )
endfunction()

add_triton_test(
    CASE01
    case01.cfg
)

add_triton_test(
    CASE02-A
    case02a.cfg
)

add_triton_test(
    CASE02-B
    case02b.cfg
)

add_triton_test(
    CASE02-C
    case02c.cfg
)

add_triton_test(
    CASE02-D
    case02d.cfg
)

add_triton_test(
    CASE03
    case03.cfg
)
