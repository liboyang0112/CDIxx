cmake_minimum_required(VERSION 3.22)
project(CDIxx LANGUAGES CXX C)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")
set(CMAKE_C_COMPILER "/usr/bin/gcc-10")
set(CMAKE_LINK_DEPENDS_NO_SHARED TRUE)
find_package( OpenCV REQUIRED )
find_package(VTK COMPONENTS 
  jsoncpp
  CommonColor
  CommonCore
  CommonDataModel
  FiltersGeometry
  IOXML
  InteractionStyle
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
  RenderingVolumeOpenGL2
  )
find_package(CUDA)
find_package(MPI REQUIRED)
find_package(TBB REQUIRED)
find_package(HDF5 REQUIRED)
set(TBB_LIBRARIES TBB::tbb TBB::tbbmalloc TBB::tbbmalloc_proxy)
set(Alllibs fftw3 config++ fftw3_mpi fftw3_threads m pthread ${MPI_LIBRARIES} ${TBB_LIBRARIES} ${OpenCV_LIBS} ${HDF5_LIBRARIES} cholmod)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(ENABLE_CUDA 1)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories ("${MPI_CXX_INCLUDE_DIRS}")
include_directories ("${HDF5_INCLUDE_DIRS}")
message("FOUND HDF5: ${HDF5_INCLUDE_DIRS}")
include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories ("/usr/include/suitesparse")
cuda_include_directories ("${PROJECT_SOURCE_DIR}/include")

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/independant/*c )
foreach( sourcefile ${APP_SOURCES} )
  string( REPLACE ".c" "_c" libname ${sourcefile} )
  string( REPLACE ".cc" "_cc" libname ${sourcefile} )
  string( REPLACE "src/independant/" "" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  #target_link_libraries(${libname} ${Alllibs})
  list(PREPEND Alllibs ${libname}) 
endforeach( sourcefile ${APP_SOURCES} )

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/vtk/*.cc )
foreach( sourcefile ${APP_SOURCES} )
  string( REPLACE ".cc" "_cc" libname ${sourcefile} )
  string( REPLACE "src/vtk/" "" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  target_link_libraries(${libname} PUBLIC ${VTK_LIBRARIES})
  vtk_module_autoinit(
    TARGETS ${libname}
    MODULES ${VTK_LIBRARIES}
    )
  list(PREPEND Alllibs ${libname}) 
endforeach( sourcefile ${APP_SOURCES} )

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/torch/*.cc )
foreach( sourcefile ${APP_SOURCES} )
  string( REPLACE ".c" "_c" libname ${sourcefile} )
  string( REPLACE "src/torch/" "" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  list(PREPEND Torchlibs ${libname}) 
endforeach( sourcefile ${APP_SOURCES} )

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/common/*c )
file( GLOB APP_SOURCES1 RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu_common/*.cc )
foreach( sourcefile ${APP_SOURCES} ${APP_SOURCES1})
  string( REPLACE ".c" "_c" libname ${sourcefile} )
  string( REPLACE ".cc" "_cc" libname ${sourcefile} )
  string( REPLACE "src/common/" "" libname ${libname} )
  string( REPLACE "src/gpu_common/" "" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  list(PREPEND Alllibs ${libname}) 
endforeach( sourcefile ${APP_SOURCES} )

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-O3")
message(${CUDA_INCLUDE_DIRS})
enable_language(CUDA)
message(${Alllibs})
file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu_common/*.cu )
file( GLOB APP_SOURCES2 RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu_ext/*.cu )
set(libname "cudaLibcommon_cu")
add_library(${libname} SHARED ${APP_SOURCES} ${APP_SOURCES2})
target_link_libraries(${libname} ${Torchlibs})
cuda_add_cufft_to_target(${libname})
set_target_properties( ${libname}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
list(PREPEND Alllibs ${libname}) 

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/post_gpu/*c )
foreach( sourcefile ${APP_SOURCES} )
  string( REPLACE ".c" "_c" libname ${sourcefile} )
  string( REPLACE ".cc" "_cc" libname ${sourcefile} )
  string( REPLACE "src/post_gpu/" "" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  list(PREPEND Alllibs ${libname}) 
endforeach( sourcefile ${APP_SOURCES} )


file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu/*.cu )
foreach( sourcefile ${APP_SOURCES} )
  string( REPLACE ".cu" "_cu_lib" libname ${sourcefile} )
  string( REPLACE "src/gpu/" "" libname ${libname} )
  message(${libname})
  add_library(${libname} SHARED ${sourcefile})
  cuda_add_cufft_to_target(${libname})
  set_target_properties( ${libname}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${libname} ${Alllibs})
endforeach( sourcefile ${APP_SOURCES} )

target_link_libraries(cdi_cu_lib experimentConfig_cu_lib)
target_link_libraries(cdi_cu_lib ${Torchlibs})
target_link_libraries(cdi_cu_lib mnistData_cu_lib)
target_link_libraries(holo_cu_lib cdi_cu_lib)
target_link_libraries(monoChromo_cu_lib experimentConfig_cu_lib)


file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/*.cu )
foreach( sourcefile ${APP_SOURCES} )
  string( REPLACE ".cu" "_cu" exename ${sourcefile} )
  string( REPLACE "util/" "" exename ${exename} )
  add_executable( ${exename} ${sourcefile})
  set_property(TARGET ${exename}
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${exename})
endforeach( sourcefile ${APP_SOURCES} )

target_link_libraries(cdi_cu cdi_cu_lib)
target_link_libraries(ptycho_cu experimentConfig_cu_lib)
target_link_libraries(pulseGen_cu monoChromo_cu_lib)
target_link_libraries(pulseGen_cu cdi_cu_lib)
target_link_libraries(readHj_cu monoChromo_cu_lib)
target_link_libraries(mwlholo_cu monoChromo_cu_lib)
target_link_libraries(mwlholo_cu cdi_cu_lib)
target_link_libraries(holo_cu holo_cu_lib)
target_link_libraries(testFISTA_cu FISTA_cu_lib)
target_link_libraries(testFISTA_cu cdi_cu_lib)


file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/*.cpp )
foreach( sourcefile ${APP_SOURCES} )
  string( REPLACE ".cpp" "_run" exename ${sourcefile} )
  string( REPLACE "util/" "" exename ${exename} )
  add_executable( ${exename} ${sourcefile})
  target_link_libraries(${exename} ${Alllibs} ${CUDA_LIBS})
endforeach( sourcefile ${APP_SOURCES} )

