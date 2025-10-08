
macro(process_target tname src)
  add_executable(${tname} ${src})
  target_compile_options(${tname} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${TRITON_CXX_FLAGS}>)
  target_link_libraries(${tname} PUBLIC ${TRITON_LINK_FLAGS})
  target_link_libraries(${tname} PUBLIC kokkos)
endmacro()

macro(run_bash_command command outvar)
  execute_process(
    COMMAND bash -c "${command}"
    OUTPUT_VARIABLE ${outvar}     
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endmacro()

macro(run_win_command command outvar)
  execute_process(
    COMMAND cmd.exe /c "${command}"
    OUTPUT_VARIABLE ${outvar}     
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endmacro()

macro(add_build_and_run_scripts)

  cmake_host_system_information(RESULT N_PHYSICAL_CORES QUERY NUMBER_OF_PHYSICAL_CORES)

  # create a build script         
  set(_BuildScript ${CMAKE_BINARY_DIR}/triton_build.sh)
  file(WRITE ${_BuildScript}  "#!/usr/bin/env bash\n\n")
  file(APPEND ${_BuildScript} "source ./${ENVFILE}\n\n")
  # On Windows, you can omit -j for Visual Studio, or use -j<num_cores> for Ninja/Mak
  file(APPEND ${_BuildScript} "cmake --build . -j ${N_PHYSICAL_CORES}\n\n")
  execute_process(COMMAND chmod +x ${_BuildScript})

  # create a run script           
  set(_RunScript ${CMAKE_BINARY_DIR}/triton_run.sh)
  file(WRITE ${_RunScript}  "#!/usr/bin/env bash\n\n")
  file(APPEND ${_RunScript} "source ./${ENVFILE}\n\n")
  file(APPEND ${_RunScript} "CFG_FILE=\${1:-./input/paraboloid/paraboloid.cfg}\n")
  file(APPEND ${_RunScript} "MPI_CMD=\${2:-${RUN_COMMAND}}\n")
  file(APPEND ${_RunScript} "\${MPI_CMD} ./${TRITON_EXECUTABLE} \${CFG_FILE}\n\n")
  execute_process(COMMAND chmod +x ${_RunScript})

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_SOURCE_DIR}/input"
            "${CMAKE_BINARY_DIR}/input"
  )

endmacro()

macro(add_test_script)
  # create a ctest script         
  set(_CtestScript ${CMAKE_BINARY_DIR}/triton_ctest.sh)
  file(WRITE ${_CtestScript}  "#!/usr/bin/env bash\n\n")
  file(APPEND ${_CtestScript} "source ./${ENVFILE}\n\n")
  file(APPEND ${_CtestScript} "ctest $*\n\n")
  execute_process(COMMAND chmod +x ${_CtestScript})

  configure_file(${CMAKE_SOURCE_DIR}/test/reference/compare_runs_simple.py
      ${CMAKE_BINARY_DIR}/compare_runs_simple.py COPYONLY)

endmacro()

macro(add_clean_script)
  set(_CleanScript ${CMAKE_BINARY_DIR}/triton_clean.sh)
  file(WRITE ${_CleanScript}  "#!/usr/bin/env bash\n\n")
  file(APPEND ${_CleanScript} "rm -rf \\\n")
  file(APPEND ${_CleanScript} "    cid \\\n")
  file(APPEND ${_CleanScript} "    CMakeFiles \\\n")
  file(APPEND ${_CleanScript} "    cmake_packages \\\n")
  file(APPEND ${_CleanScript} "    git-state.txt \\\n")
  file(APPEND ${_CleanScript} "    Makefile \\\n")
  file(APPEND ${_CleanScript} "    CTestTestfile.cmake \\\n")
  file(APPEND ${_CleanScript} "    test \\\n")
  file(APPEND ${_CleanScript} "    tools \\\n")
  file(APPEND ${_CleanScript} "    Testing \\\n")
  file(APPEND ${_CleanScript} "    triton_* \\\n")
  file(APPEND ${_CleanScript} "    ${TRITON_EXECUTABLE} \\\n")
  file(APPEND ${_CleanScript} "    CMakeCache.txt \\\n")
  file(APPEND ${_CleanScript} "    cmake_install.cmake \\\n")
  file(APPEND ${_CleanScript} "    compare_runs_simple.py \\\n")
  file(APPEND ${_CleanScript} "    output_allatoona \\\n")
  file(APPEND ${_CleanScript} "    output_circular_dambreak \\\n")
  file(APPEND ${_CleanScript} "    output_paraboloid \\\n")
  file(APPEND ${_CleanScript} "    external \\\n")
  file(APPEND ${_CleanScript} "    input \\\n")
  file(APPEND ${_CleanScript} "    output")
  execute_process(COMMAND chmod +x ${_CleanScript})

endmacro()
