add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kokkos)

if(${ENSEMBLE_BUILD})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp)
endif()

# Add bundled SWMM if coupling is enabled
if(TRITON_ENABLE_SWMM)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/swmm/CMakeLists.txt)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/swmm)
    else()
        message(FATAL_ERROR "SWMM coupling requested but bundled SWMM source not found in external/swmm")
    endif()
endif()

# Find GDAL
find_package(GDAL)
if(GDAL_FOUND)
    set(ENABLE_GDAL ON CACHE INTERNAL "ENABLE_GDAL variable")
else()
    message(WARNING "GDAL is not found. GDAL is disabled.")
    set(ENABLE_GDAL OFF CACHE INTERNAL "ENABLE_GDAL variable")
endif()


