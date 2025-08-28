add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kokkos)

# Find GDAL
find_package(GDAL)
if(GDAL_FOUND)
    message(STATUS "Found GDAL: ${GDAL_INCLUDE_DIR}")
    set(TRITON_ENABLE_GDAL ON)
else()
    message(WARNING "GDAL is not found. GDAL is disabled.")
    set(TRITON_ENABLE_GDAL OFF)
endif()


