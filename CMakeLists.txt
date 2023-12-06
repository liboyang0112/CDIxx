cmake_minimum_required(VERSION 3.22)
project(CDIxx LANGUAGES CXX C)
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_COMPILER "clang")
set(CMAKE_LINK_DEPENDS_NO_SHARED TRUE)
find_package( GSL REQUIRED )
set(CMAKE_CUDA_HOST_COMPILER clang++-15)
set(CMAKE_CUDA_COMPILER nvcc)
enable_language(CUDA)
set(NVHPC_CUDA_VERSION 12.2)
find_package(NVHPC REQUIRED COMPONENTS CUDA MATH)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK REQUIRED gtk4)
pkg_check_modules(freetype2 REQUIRED freetype2)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories (${GTK_INCLUDE_DIRS})
include_directories (${FREETYPE_INCLUDE_DIRS})
include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories (/usr/include/python3.11)
link_directories(${PROJECT_SOURCE_DIR}/lib)

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu/*c ${PROJECT_SOURCE_DIR}/src/post_gpu/*c ${PROJECT_SOURCE_DIR}/src/independant/*c)
foreach( sourcefile ${APP_SOURCES})
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".c" "_c" libname ${libname} )
  string( REPLACE ".cc" "_cc" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
endforeach( sourcefile ${APP_SOURCES} )

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-O3 --diag-suppress 20040")
file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu/*.cu )
set(libname "cudaLibcommon_cu")
add_library(${libname} SHARED ${APP_SOURCES})
target_link_libraries(${libname} NVHPC::CUFFT)
set_target_properties( ${libname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(FGA_cc monoChromo_cc)
target_link_libraries(freetype_c freetype imgio_c)
target_link_libraries(cdilmdb_c lmdb z)
target_link_libraries(videoWriter_cc avformat avutil avcodec)
target_link_libraries(readConfig_cc config)
target_link_libraries(cuPlotter_cc videoWriter_cc freetype_c)
target_link_libraries(cudaLibcommon_cu cuPlotter_cc memManager_cc imgio_cc)
target_link_libraries(orthFitter_cc cudaLibcommon_cu matrixInverse_cc)
target_link_libraries(experimentConfig_cc cudaLibcommon_cu readConfig_cc)
target_link_libraries(imgio_c tiff png)
target_link_libraries(imgio_cc imgio_c)
target_link_libraries(cdi_cc experimentConfig_cc mnistData_cc FISTA_cc)
target_link_libraries(holo_cc cdi_cc)
target_link_libraries(monoChromo_cc experimentConfig_cc orthFitter_cc gsl FISTA_cc)

file( GLOB APP_SOURCES ${PROJECT_SOURCE_DIR}/src/cython/*.pyx )
foreach( sourcefile ${APP_SOURCES} )
  message(${sourcefile})
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".pyx" ".c" cfile ${libname} )
  string( REPLACE ".pyx" "_cython" libname ${libname} )
  add_custom_command(OUTPUT ${cfile} COMMAND cython -3 ${sourcefile} --module-name=lib${libname} -o ${cfile} MAIN_DEPENDENCY ${sourcefile})
  add_library(${libname} SHARED ${cfile})
  target_compile_definitions(${libname} PUBLIC -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
endforeach( sourcefile ${APP_SOURCES} )
target_link_libraries(unetDataLoader_cython cdilmdb_c)
target_link_libraries(imageIO_cython imgio_c)
target_link_libraries(readHj_cython FGA_cc)

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/*.cu )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cu" "_cu" exename ${libname} )
  add_executable( ${exename} ${sourcefile})
  set_property(TARGET ${exename}
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endforeach( sourcefile ${APP_SOURCES} )

target_link_libraries(fdtd3d_cu cudaLibcommon_cu)
target_link_libraries(fdtd2d_cu cudaLibcommon_cu)

set(resourcefile ${PROJECT_SOURCE_DIR}/src/gui/resource.xml)
file(GLOB UIFILE ${PROJECT_SOURCE_DIR}/src/gui/*.ui)
add_custom_command(OUTPUT resource.c COMMAND glib-compile-resources ${resourcefile} --target=resource.c  --sourcedir=${PROJECT_SOURCE_DIR}/src/gui --generate-source MAIN_DEPENDENCY ${resourcefile} DEPENDS ${UIFILE})
file( GLOB APP_SOURCES ${PROJECT_SOURCE_DIR}/src/gui/*c )
add_library(gui SHARED ${APP_SOURCES} resource.c)
target_link_libraries(gui ${GTK_LIBRARIES})

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/*.cpp )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cpp" "_run" exename ${libname} )
  add_executable( ${exename} ${sourcefile})
endforeach( sourcefile ${APP_SOURCES} )
link_directories(${GTK_LIBRARY_DIRS})
target_link_libraries(freetype_run freetype_c imgio_cc memManager_cc)
target_link_libraries(cdi_run cdi_cc)
target_link_libraries(ptycho_run experimentConfig_cc)
target_link_libraries(frog_run cudaLibcommon_cu)
target_link_libraries(beamGenerator_run memManager_cc imgio_cc)
target_link_libraries(gui_cdi_run gui cdi_cc ${GTK_LIBRARIES})
target_link_libraries(testLMDB_run cdilmdb_c)
target_link_libraries(testfitter_run orthFitter_cc)
target_link_libraries(holo_run holo_cc)
target_link_libraries(propagatorDemo_run cudaLibcommon_cu)
target_link_libraries(fftDemo_run cudaLibcommon_cu)
target_link_libraries(processimg_run cudaLibcommon_cu)
target_link_libraries(pulseGen_run cdilmdb_c monoChromo_cc cdi_cc)
target_link_libraries(testFISTA_run cdi_cc)
target_link_libraries(simLineSpectrumImaging_run cdi_cc monoChromo_cc material_cc)

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
