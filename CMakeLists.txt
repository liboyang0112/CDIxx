cmake_minimum_required(VERSION 3.22)
project(CDIxx LANGUAGES CXX C)
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_COMPILER "clang")
set(TORCH_DIR "/home/boyang/.local/python_venv/lib/python3.11/site-packages/torch")
set(CMAKE_LINK_DEPENDS_NO_SHARED TRUE)
find_package( GSL REQUIRED )
set(CMAKE_CUDA_HOST_COMPILER clang++-15)
set(CMAKE_CUDA_COMPILER nvcc)
enable_language(CUDA)
set(NVHPC_CUDA_VERSION 12.2)
find_package(NVHPC REQUIRED COMPONENTS CUDA MATH)
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)
find_package(PythonLibs REQUIRED)
set(Alllibs)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories ("${MPI_CXX_INCLUDE_DIRS}")
include_directories ("${HDF5_INCLUDE_DIRS}")
include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories (${TORCH_DIR}/include)
include_directories (${TORCH_DIR}/include/torch/csrc/api/include)
include_directories (${PYTHON_INCLUDE_DIRS})
link_directories(${TORCH_DIR}/lib)
link_directories(${PROJECT_SOURCE_DIR}/lib)
set(TORCH_LIBS c10_cuda torch_cpu torch_cuda torch c10)

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/independant/*c )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cc" "_cc" libname ${libname} )
  string( REPLACE ".c" "_c" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  list(PREPEND Alllibs ${libname}) 
endforeach( sourcefile ${APP_SOURCES} )
target_link_libraries(cdilmdb_c lmdb z)
target_link_libraries(videoWriter_cc avformat avutil avcodec)

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/torch/*.cc )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cc" "_cc" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  target_link_libraries(${libname} PUBLIC ${TORCH_LIBS} cdilmdb_c)
endforeach( sourcefile ${APP_SOURCES} )

file( GLOB APP_SOURCES ${PROJECT_SOURCE_DIR}/src/cython/*.pyx )
foreach( sourcefile ${APP_SOURCES} )
  message(${sourcefile})
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".pyx" ".c" cfile ${libname} )
  string( REPLACE ".pyx" "_cython" libname ${libname} )
  add_custom_command(OUTPUT ${cfile} COMMAND cython -3 ${sourcefile} --module-name=lib${libname} -o ${cfile} MAIN_DEPENDENCY ${sourcefile})
  add_library(${libname} SHARED ${cfile})
  target_link_libraries(${libname} PUBLIC cdilmdb_c ${PYTHON_LIBRARIES})
  target_compile_definitions(${libname} PUBLIC -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
endforeach( sourcefile ${APP_SOURCES} )

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/common/*c )
file( GLOB APP_SOURCES1 RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu_common/*.cc )
foreach( sourcefile ${APP_SOURCES} ${APP_SOURCES1})
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".c" "_c" libname ${libname} )
  string( REPLACE ".cc" "_cc" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  list(PREPEND Alllibs ${libname}) 
endforeach( sourcefile ${APP_SOURCES} )

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-O3")
#message(${NVHPC_CUDA_INCLUDE_DIRS})
message(${Alllibs})
file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu_common/*.cu )
file( GLOB APP_SOURCES2 RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu_ext/*.cu )
set(libname "cudaLibcommon_cu")
add_library(${libname} SHARED ${APP_SOURCES} ${APP_SOURCES2})
target_link_libraries(${libname} NVHPC::CUFFT)
set_target_properties( ${libname}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
list(PREPEND Alllibs ${libname}) 
target_link_libraries(cudaLibcommon_cu cuPlotter_cc)
target_link_libraries(cudaLibcommon_cu common_cc)

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/post_gpu/*c )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".c" "_c" libname ${libname} )
  string( REPLACE ".cc" "_cc" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  list(PREPEND Alllibs ${libname}) 
endforeach( sourcefile ${APP_SOURCES} )
target_link_libraries(orthFitter_cc cudaLibcommon_cu matrixInverse_cc)


file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu/*.cu )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cu" "_cu_lib" libname ${libname} )
  message(${libname})
  add_library(${libname} SHARED ${sourcefile})
  target_link_libraries(${libname} NVHPC::CUFFT)
  set_target_properties( ${libname}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${libname} ${Alllibs})
endforeach( sourcefile ${APP_SOURCES} )

target_link_libraries(cuPlotter_cc  videoWriter_cc)
target_link_libraries(common_cc ${GSL_LIBRARIES} tiff png)
target_link_libraries(torchConfig_cc config)


target_link_libraries(cdi_cu_lib experimentConfig_cu_lib)
target_link_libraries(cdi_cu_lib mnistData_cu_lib)
target_link_libraries(cdi_cu_lib FISTA_cu_lib)
target_link_libraries(holo_cu_lib cdi_cu_lib)
target_link_libraries(monoChromo_cu_lib experimentConfig_cu_lib)
target_link_libraries(monoChromo_cu_lib FISTA_cu_lib)


file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/*.cu )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cu" "_cu" exename ${libname} )
  add_executable( ${exename} ${sourcefile})
  set_property(TARGET ${exename}
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endforeach( sourcefile ${APP_SOURCES} )

target_link_libraries(fdtd3d_cu ${Alllibs})
target_link_libraries(fdtd2d_cu ${Alllibs})
target_link_libraries(processimg_cu ${Alllibs})
target_link_libraries(cdi_cu cdi_cu_lib)
target_link_libraries(ptycho_cu experimentConfig_cu_lib)
target_link_libraries(pulseGen_cu cdilmdb_c)
target_link_libraries(pulseGen_cu monoChromo_cu_lib)
target_link_libraries(pulseGen_cu cdi_cu_lib)
target_link_libraries(readHj_cu monoChromo_cu_lib)
target_link_libraries(mwlholo_cu monoChromo_cu_lib)
target_link_libraries(mwlholo_cu cdi_cu_lib)
target_link_libraries(holo_cu holo_cu_lib)
target_link_libraries(testFISTA_cu FISTA_cu_lib)
target_link_libraries(testFISTA_cu cdi_cu_lib)
target_link_libraries(simLineSpectrumImaging_cu cdi_cu_lib)


file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/*.cpp )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cpp" "_run" exename ${libname} )
  add_executable( ${exename} ${sourcefile})
endforeach( sourcefile ${APP_SOURCES} )
target_compile_definitions(unet_run PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(train_cc PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(unet_run train_cc)
target_link_libraries(unet_run torchConfig_cc)
target_link_libraries(testLMDB_run cdilmdb_c)
target_link_libraries(testfitter_run memManager_cc orthFitter_cc)

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
file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/vtk/*.cc )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cc" "_cc" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  target_link_libraries(${libname} PUBLIC ${VTK_LIBRARIES})
  vtk_module_autoinit(
    TARGETS ${libname}
    MODULES ${VTK_LIBRARIES}
    )
endforeach( sourcefile ${APP_SOURCES} )
target_link_libraries(testVtk_run WriteVTI_cc)
