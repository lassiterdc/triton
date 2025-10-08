add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kokkos)

# Find GDAL
find_package(GDAL)
if(GDAL_FOUND)
    set(TRITON_ENABLE_GDAL ON CACHE INTERNAL "TRITON_ENABLE_GDAL variable")
else()
    message(WARNING "GDAL is not found. GDAL is disabled.")
    set(TRITON_ENABLE_GDAL OFF CACHE INTERNAL "TRITON_ENABLE_GDAL variable")
endif()


