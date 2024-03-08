cmake_minimum_required(VERSION 3.22)
project(CDIxx LANGUAGES CXX C)
set(CMAKE_LINK_DEPENDS_NO_SHARED TRUE) #We set this to prevent relink after library changes for a faster recompilation
set(CMAKE_CXX_COMPILER "clang++") #compiler, gcc can be alteratives, or install it by `apt install llvm-17`
set(CMAKE_C_COMPILER "clang") #compiler, gcc can be alteratives, or install it by `apt install llvm-17`
set(CMAKE_CUDA_HOST_COMPILER clang++-15) #compiler, gcc can be alteratives, or install it by `apt install llvm-15` (Higher version is not supported by cuda 12.3)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON ) #for vim ale grammar check
set(compileVTK FALSE)


enable_language(CUDA) #CUDA is required for GPU algorithms, consult nvidia webpage for installation
find_package(NVHPC COMPONENTS CUDA MATH)
if(NOT NVHPC_FOUND)
  find_package ( CUDAToolkit REQUIRED ) #You have this if you installed CUDA, needed for cufft library
  set(CUFFTLIB CUDA::cufft)
else()
  set(NVHPC_CUDA_VERSION 12.2)
  set(CUFFTLIB NVHPC::MATH)
endif()

set(CMAKE_CUDA_COMPILER nvcc) #You have this if you installed CUDA
add_compile_options(-Wall)

find_package(PkgConfig REQUIRED) #package finder, always equipped in linux system, if not, install it by `apt install pkg-config`

#GTK, a GUI library, always equipped in linux system, if not, install it by `apt install libgtk-4-dev
pkg_check_modules(GTK QUIET gtk4)
if(GTK_FOUND)
include_directories (${GTK_INCLUDE_DIRS})
endif()
#freetype, a font library, always equipped in linux system, if not, install it by `apt install libfreetype-dev`
find_package(Freetype REQUIRED)
include_directories (${FREETYPE_INCLUDE_DIRS})
#GSL is needed for some interpolation algorithm, install it by `apt install libgsl-dev`
find_package( GSL REQUIRED )
find_package( Python REQUIRED COMPONENTS NumPy Interpreter) # pip install numpy
include_directories(${Python_INCLUDE_DIRS})
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import os, skbuild; print(os.path.dirname(skbuild.__file__))" # pip install scikit-build
    OUTPUT_VARIABLE SKBLD_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" # pip install scikit-build
    OUTPUT_VARIABLE EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_MODULE_PATH "${SKBLD_DIR}/resources/cmake")
find_package ( Cython REQUIRED )
include_directories("/usr/include/ffmpeg") #libav* is in this directory on Fedora
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories ("${PROJECT_SOURCE_DIR}/include")
link_directories(${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-O3 --diag-suppress 20040")
file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu/*.cu )
set(libname "cudaLibcommon_cu")
add_library(${libname} SHARED ${APP_SOURCES})
target_link_libraries(${libname} ${CUFFTLIB})
set_target_properties( ${libname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/post_gpu/*c ${PROJECT_SOURCE_DIR}/src/independant/*c)
foreach( sourcefile ${APP_SOURCES})
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".c" "_c" libname ${libname} )
  string( REPLACE ".cc" "_cc" libname ${libname} )
  add_library( ${libname} SHARED ${sourcefile})
endforeach( sourcefile ${APP_SOURCES} )

file( GLOB APP_SOURCES ${PROJECT_SOURCE_DIR}/src/cython/*.pyx )
foreach( sourcefile ${APP_SOURCES} )
  string( REPLACE ".pyx" "" libname ${sourcefile} )
  cmake_path(GET libname FILENAME libname)
  add_cython_target(${libname} ${sourcefile} C)
  add_library(${libname} SHARED ${libname})
  target_compile_definitions(${libname} PUBLIC -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION) #suppress numpy warning
  SET_TARGET_PROPERTIES(${libname} PROPERTIES PREFIX "" SUFFIX ${EXT_SUFFIX}) #standard cpython library name
endforeach( sourcefile ${APP_SOURCES} )

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/*.cu )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cu" "_cu" exename ${libname} )
  add_executable( ${exename} ${sourcefile})
  set_property(TARGET ${exename}
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endforeach( sourcefile ${APP_SOURCES} )

file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/util/*.cpp )
foreach( sourcefile ${APP_SOURCES} )
  cmake_path(GET sourcefile FILENAME libname)
  string( REPLACE ".cpp" "_run" exename ${libname} )
  add_executable( ${exename} ${sourcefile})
endforeach( sourcefile ${APP_SOURCES} )

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
target_link_libraries(gui ${GTK_LIBRARIES})
link_directories(${GTK_LIBRARY_DIRS})
target_link_libraries(gui_cdi_run gui cdi_cc ${GTK_LIBRARIES})
endif()


target_link_libraries(broadBand_cc gsl gslcblas memManager_cc cuPlotter_cc cudaLibcommon_cu)
target_link_libraries(spectImaging_cc broadBand_cc)
target_link_libraries(spectPhase_cc broadBand_cc)
target_link_libraries(monoChromo_cc broadBand_cc orthFitter_cc FISTA_cc)
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
target_link_libraries(cythonLoader cdilmdb_c)
target_link_libraries(imageIO imgio_c)
target_link_libraries(readHj FGA_cc)
target_link_libraries(fdtd3d_cu cudaLibcommon_cu)
target_link_libraries(fdtd2d_cu cudaLibcommon_cu)
target_link_libraries(freetype_run freetype_c imgio_cc memManager_cc)
target_link_libraries(cdi_run cdi_cc)
target_link_libraries(ptycho_run experimentConfig_cc)
target_link_libraries(frog_run cudaLibcommon_cu)
target_link_libraries(beamGenerator_run memManager_cc imgio_cc)
target_link_libraries(testLMDB_run cdilmdb_c)
target_link_libraries(testfitter_run orthFitter_cc)
target_link_libraries(holo_run holo_cc)
target_link_libraries(propagatorDemo_run cudaLibcommon_cu)
target_link_libraries(fftDemo_run cudaLibcommon_cu)
target_link_libraries(zoom_run cudaLibcommon_cu)
target_link_libraries(processimg_run cudaLibcommon_cu)
target_link_libraries(pulseGen_run cdilmdb_c monoChromo_cc cdi_cc)
target_link_libraries(mono_run cdilmdb_c monoChromo_cc cdi_cc)
target_link_libraries(testFISTA_run cdi_cc)
target_link_libraries(simLineSpectrumImaging_run cdi_cc spectImaging_cc material_cc)
target_link_libraries(spectphase_run readConfig_cc spectPhase_cc material_cc)
target_link_libraries(readCCDResponse_run imgio_cc memManager_cc)

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
#target_link_libraries(testVtk_run WriteVTI_cc)
endif()
