cmake_minimum_required(VERSION 3.22)
project(CDIxx LANGUAGES CXX C)

#Prerequisite: gsl freetype libpng libtiff
#Optional: Python(Numpy, skbuild, Cython), VTK, GTK

set(CMAKE_LINK_DEPENDS_NO_SHARED TRUE) #We set this to prevent relink after library changes for a faster recompilation
set(CMAKE_CXX_COMPILER "clang++") #compiler, gcc can be alteratives, or install it by `apt install llvm-17`
set(CMAKE_C_COMPILER "clang") #compiler, gcc can be alteratives, or install it by `apt install llvm-17`
set(CMAKE_EXPORT_COMPILE_COMMANDS ON ) #for vim ale grammar check
set(compileVTK TRUE)

set(CMAKE_CUDA_COMPILER_FORCED TRUE)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda") # Update this path if necessary
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
add_compile_options(-Wall -Wextra)

find_package(PkgConfig) #package finder, always equipped in linux system, if not, install it by `apt install pkg-config`
find_package(fmt REQUIRED) #package finder, always equipped in linux system, if not, install it by `apt install pkg-config`
#GTK, a GUI library, always equipped in linux system, if not, install it by `apt install libgtk-4-dev
if(PkgConfig_FOUND)
  pkg_check_modules(GTK gtk4)
  pkg_check_modules(TRACY tracy)
endif(PkgConfig_FOUND)
if(GTK_FOUND)
include_directories (${GTK_INCLUDE_DIRS})
endif()
if(tracy_FOUND)
  include_directories (${tracy_INCLUDE_DIRS})
  message(${tracy_INCLUDE_DIRS})
endif()
#freetype, a font library, always equipped in linux system, if not, install it by `apt install libfreetype-dev`
find_package(Freetype REQUIRED)
include_directories (${FREETYPE_INCLUDE_DIRS})
#GSL is needed for some interpolation algorithm, install it by `apt install libgsl-dev`
find_package( GSL REQUIRED )
find_package(Python REQUIRED COMPONENTS Interpreter NumPy)
if(Python_FOUND)
include_directories(${Python_INCLUDE_DIRS})
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import os, skbuild; print(os.path.dirname(skbuild.__file__))" # pip install scikit-build
    OUTPUT_VARIABLE SKBLD_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())" # pip install numpy
    OUTPUT_VARIABLE NUMPY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("Numpy dir:" ${NUMPY_DIR})
message("PYTHON include:" ${Python_INCLUDE_DIRS})
list(APPEND CMAKE_MODULE_PATH "${SKBLD_DIR}/resources/cmake")
find_package ( Cython REQUIRED )
include_directories(${NUMPY_DIR})
endif(Python_FOUND)
include_directories("/usr/include/ffmpeg") #libav* is in this directory on Fedora

#for debug
#get_property( dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES )
#foreach( dir ${dirs} )
#    message( STATUS "include dir=${dir}" )
#endforeach()
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG") 
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG") 
#set(CMAKE_BUILD_TYPE Release)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories ("${PROJECT_SOURCE_DIR}/include")
link_directories(${PROJECT_SOURCE_DIR}/lib)
link_libraries(fmt)

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/independant/*c)
foreach( sourcefile ${APP_SOURCES})
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".c" "_c" libname ${libname} )
  string( REPLACE ".cc" "_cc" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
endforeach( sourcefile ${APP_SOURCES} )

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cuda-gpu-arch=sm_80")
file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu/*.cu )
set_source_files_properties(${APP_SOURCES} PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-Wno-unused-parameter;-Wno-unknown-cuda-version;-x;cu;--cuda-gpu-arch=sm_86") #check your gpu arch: https://developer.nvidia.com/cuda-gpus
set(libname "cudaLibcommon_cu")
add_library(${libname} SHARED ${APP_SOURCES})
target_link_libraries(${libname} ${CUFFTLIB})
target_link_libraries(cudaLibcommon_cu memManager_cc imgio_cc cudart cufft cusolver)

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/post_gpu/*c)
foreach( sourcefile ${APP_SOURCES})
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".c" "_c" libname ${libname} )
  string( REPLACE ".cc" "_cc" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
  target_link_libraries(${libname} cudaLibcommon_cu)
endforeach( sourcefile ${APP_SOURCES} )

target_link_libraries(cuPlotter_cc videoWriter_cc freetype_c)
target_link_libraries(broadBand_cc gsl gslcblas memManager_cc cuPlotter_cc)
target_link_libraries(spectImaging_cc broadBand_cc cuPlotter_cc misc_cc FISTA_cc)
target_link_libraries(spectPhase_cc broadBand_cc cuPlotter_cc misc_cc)
target_link_libraries(monoChromo_cc broadBand_cc orthFitter_cc FISTA_cc cuPlotter_cc)
target_link_libraries(FGA_cc monoChromo_cc cuPlotter_cc)
target_link_libraries(freetype_c freetype)
target_link_libraries(cdilmdb_c lmdb z)
target_link_libraries(videoWriter_cc avformat avutil avcodec pthread)
target_link_libraries(readConfig_cc config)
target_link_libraries(orthFitter_cc matrixInverse_cc)
target_link_libraries(experimentConfig_cc readConfig_cc)
target_link_libraries(imgio_c tiff png jpeg)
target_link_libraries(imgio_cc imgio_c)
target_link_libraries(cdi_cc experimentConfig_cc mnistData_cc FISTA_cc cuPlotter_cc misc_cc)
target_link_libraries(holo_cc cdi_cc cuPlotter_cc)

if(Python_FOUND)
file( GLOB APP_SOURCES ${PROJECT_SOURCE_DIR}/src/cython/*.pyx )
foreach( sourcefile ${APP_SOURCES} )
  string( REPLACE ".pyx" "" libname ${sourcefile} )
  cmake_path(GET libname FILENAME libname)
  add_cython_target(${libname} ${sourcefile} C)
  add_library(${libname} SHARED ${libname})
  target_compile_definitions(${libname} PUBLIC -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION) #suppress numpy warning
  SET_TARGET_PROPERTIES(${libname} PROPERTIES PREFIX "" SUFFIX ${EXT_SUFFIX}) #standard cpython library name
endforeach( sourcefile ${APP_SOURCES} )
target_link_libraries(cythonLoader cdilmdb_c)
target_link_libraries(imageIO imgio_c)
target_link_libraries(readHj FGA_cc)
endif(Python_FOUND)

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/*.cu )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cu" "_cu" exename ${libname} )
  add_executable( ${exename} ${sourcefile})
  #set_property(TARGET ${exename} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
endforeach( sourcefile ${APP_SOURCES} )

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/*.cpp )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cpp" "_run" exename ${libname} )
  add_executable( ${exename} ${sourcefile})
endforeach( sourcefile ${APP_SOURCES} )

target_link_libraries(fdtd3d_run cuPlotter_cc)
target_link_libraries(TG_run cuPlotter_cc)
target_link_libraries(colorbar_run cuPlotter_cc)
target_link_libraries(testSVD_run cudaLibcommon_cu)
target_link_libraries(freetype_run freetype_c imgio_cc memManager_cc)
target_link_libraries(cdi_run cdi_cc)
target_link_libraries(ptycho_run experimentConfig_cc cuPlotter_cc misc_cc)
target_link_libraries(sesolver_run cuPlotter_cc)
target_link_libraries(frog_run cuPlotter_cc misc_cc)
target_link_libraries(beamGenerator_run memManager_cc imgio_cc)
target_link_libraries(testLMDB_run cdilmdb_c)
target_link_libraries(testfitter_run orthFitter_cc)
target_link_libraries(holo_run holo_cc)
target_link_libraries(propagatorDemo_run cuPlotter_cc misc_cc)
target_link_libraries(fftDemo_run cuPlotter_cc)
target_link_libraries(zoom_run cuPlotter_cc)
target_link_libraries(processimg_run cuPlotter_cc misc_cc)
target_link_libraries(takepsnr_run cuPlotter_cc misc_cc)
target_link_libraries(profile_run cuPlotter_cc misc_cc)
target_link_libraries(pulseGen_run cdilmdb_c monoChromo_cc cdi_cc)
target_link_libraries(mono_run cdilmdb_c monoChromo_cc cdi_cc)
target_link_libraries(testFISTA_run cdi_cc cuPlotter_cc)
target_link_libraries(simLineSpectrumImaging_run cdi_cc spectImaging_cc material_cc)
target_link_libraries(spectImaging_run cudaLibcommon_cu spectImaging_cc material_cc)
target_link_libraries(spectImaging_point_run cudaLibcommon_cu spectImaging_cc material_cc)
target_link_libraries(spectphase_run readConfig_cc spectPhase_cc material_cc)
target_link_libraries(readCCDResponse_run imgio_cc memManager_cc)
target_link_libraries(bandwidthtest_run cudaLibcommon_cu)
target_link_libraries(cup_run cuPlotter_cc FISTA_cc)

if(GTK_FOUND)
file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/gui/*.cpp )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cpp" "_run" exename ${libname} )
  add_executable( ${exename} ${sourcefile})
endforeach( sourcefile ${APP_SOURCES} )
set(resourcefile ${PROJECT_SOURCE_DIR}/src/gui/resource.xml)
file(GLOB UIFILE ${PROJECT_SOURCE_DIR}/src/gui/*.ui)
add_custom_command(OUTPUT resource.c COMMAND glib-compile-resources ${resourcefile} --target=resource.c  --sourcedir=${PROJECT_SOURCE_DIR}/src/gui --generate-source MAIN_DEPENDENCY ${resourcefile} DEPENDS ${UIFILE})
file( GLOB APP_SOURCES ${PROJECT_SOURCE_DIR}/src/gui/*c )
add_library(gui SHARED ${APP_SOURCES} resource.c)
target_link_libraries(gui ${GTK_LIBRARIES} cuPlotter_cc)
link_directories(${GTK_LIBRARY_DIRS})
target_link_libraries(gui_cdi_run gui cdi_cc ${GTK_LIBRARIES})
endif()


if(compileVTK)
#install it by `apt install libvtk9-dev`
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
file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/vtk/*.cpp )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cpp" "_run" exename ${libname} )
  add_executable( ${exename} ${sourcefile})
  message(${exename})
endforeach( sourcefile ${APP_SOURCES} )
target_link_libraries(testVtk_run WriteVTI_cc)
endif()
