cmake_minimum_required(VERSION 3.22)
set(CMAKE_LINK_DEPENDS_NO_SHARED TRUE) #We set this to prevent relink after library changes for a faster recompilation
set(CMAKE_CXX_COMPILER "clang++") #compiler, gcc can be alteratives, or install it by `apt install llvm-17`
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc") #compiler, gcc can be alteratives, or install it by `apt install llvm-17`
set(CMAKE_C_COMPILER "clang") #compiler, gcc can be alteratives, or install it by `apt install llvm-17`
set(CMAKE_EXPORT_COMPILE_COMMANDS ON ) #for vim ale grammar check
project(CDIxx LANGUAGES CXX C)

#Prerequisite: gsl freetype libpng libtiff
#Optional: Python(Numpy, skbuild, Cython), VTK, GTK

set(compileVTK FALSE)

add_compile_options(-Wall -Wextra)

find_package(PkgConfig) #package finder, always equipped in linux system, if not, install it by `apt install pkg-config`
include(FetchContent)

find_package(fmt REQUIRED) #package finder, always equipped in linux system, if not, install it by `apt install pkg-config`
#GTK, a GUI library, always equipped in linux system, if not, install it by `apt install libgtk-4-dev
find_package(Tracy)
if(PkgConfig_FOUND)
  pkg_check_modules(GTK gtk4)
  pkg_check_modules(hdf5 hdf5)
endif(PkgConfig_FOUND)
if(GTK_FOUND)
include_directories (${GTK_INCLUDE_DIRS})
endif()
if(tracy_FOUND)
  include_directories (${tracy_INCLUDE_DIRS})
endif()
if(NOT hdf5_FOUND)
  pkg_check_modules(hdf5 hdf5-serial)
endif()
if(hdf5_FOUND)
  include_directories (${hdf5_INCLUDE_DIRS})
  link_directories(${hdf5_LIBRARY_DIRS})
endif()
#freetype, a font library, always equipped in linux system, if not, install it by `apt install libfreetype-dev`
find_package(Freetype REQUIRED)
find_package(Fontconfig REQUIRED)
include_directories (${FREETYPE_INCLUDE_DIRS})
#GSL is needed for some interpolation algorithm, install it by `apt install libgsl-dev`
find_package( GSL REQUIRED )
set(ENV{VIRTUAL_ENV} "/opt/python_venv")
unset(Python_EXECUTABLE)
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
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

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG") 
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG") 
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

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
option(USE_HIP "Build with HIP instead of CUDA" OFF)

set(GPU_LIB_TARGET "")
if(USE_HIP)
    message(STATUS "HIP mode enabled: Converting CUDA to HIP and building with clang++")

    set(CMAKE_HIP_ARCHITECTURES gfx90a)  # Or use native
    set(CMAKE_HIP_COMPILER "amdclang++") #compiler, gcc can be alteratives, or install it by `apt install llvm-17`
    enable_language(HIP)
    # Find hipify-perl
    find_program(HIPIFY_EXECUTABLE NAMES hipify-clang PATHS ${HIP_PATH}/bin /opt/rocm/bin NO_DEFAULT_PATH)
    if(NOT HIPIFY_EXECUTABLE)
        message(FATAL_ERROR "hipify-clang not found. Please install ROCm or set HIP_PATH correctly.")
    endif()

    # ROCm paths
    set(ROCM_PATH /opt/rocm CACHE PATH "Path to ROCm installation")
    include_directories(
        ${ROCM_PATH}/include
        ${ROCM_PATH}/hip/include
    )
    link_directories(${ROCM_PATH}/lib)

    # Set Clang as compiler and enable HIP offload
      # Define input and output directories
      set(CUDA_SRC_DIR ${PROJECT_SOURCE_DIR}/src/gpu)
      set(HIP_OUT_DIR ${PROJECT_BINARY_DIR}/hip_converted)

      # Ensure output directory exists at configure time
      file(MAKE_DIRECTORY ${HIP_OUT_DIR})

      # Gather .cu files and map to output .hip.cpp files
      file(GLOB CUDA_SOURCES RELATIVE ${CUDA_SRC_DIR} ${CUDA_SRC_DIR}/*.cu)
      set(HIP_SOURCES)

      foreach(cu_file IN LISTS CUDA_SOURCES)
        get_filename_component(name_we ${cu_file} NAME_WE)
        set(cu_full_path ${CUDA_SRC_DIR}/${cu_file})
        set(hip_output ${HIP_OUT_DIR}/${name_we}.hip)  # Use .cpp so CMake recognizes it

        add_custom_command(
          OUTPUT ${hip_output}
          COMMAND ${HIPIFY_EXECUTABLE} ${cu_full_path} -o ${hip_output} -I ${PROJECT_SOURCE_DIR}/include --extra-arg=-Wno-unknown-cuda-version
          COMMAND sed -i "s/\\bcub::/hipcub::/g" ${hip_output}
          DEPENDS ${cu_full_path} ${HIPIFY_EXECUTABLE}
          VERBATIM
        )

        list(APPEND HIP_SOURCES ${hip_output})
      endforeach()
      add_custom_target(hipify_target
        DEPENDS ${HIP_SOURCES}
        COMMENT "Running hipify to generate .hip files"
      )

      # Now add_library *after* we have declared all outputs
      set(GPU_LIB_TARGET hipLib)
      message(${HIP_SOURCES})
      add_library(${GPU_LIB_TARGET} SHARED ${HIP_SOURCES})
      set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)
      set_source_files_properties(${HIP_SOURCES} PROPERTIES COMPILE_FLAGS "-x hip")
      add_dependencies(${GPU_LIB_TARGET} hipify_target)
      set_property(TARGET ${GPU_LIB_TARGET} PROPERTY LINKER_LANGUAGE CXX)
      target_include_directories(${GPU_LIB_TARGET} PRIVATE ${CUDA_SRC_DIR} ${ROCM_PATH}/include ${ROCM_PATH}/include/hipsolver)

      target_link_libraries(${GPU_LIB_TARGET} memManager_cc imgio_cc hipfft hiprtc hipsolver)
    else()
      set(CMAKE_CUDA_ARCHITECTURES native)  # Or use native
      set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda") # Update this path if necessary
      include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
      link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
      #set(CMAKE_CUDA_FLAGS "-Wno-unknown-cuda-version") #clang
      #set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler -ccbin=clang++") #nvcc
      enable_language(CUDA)

      file(GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/gpu/*.cu)
      set(GPU_LIB_TARGET "cudaLib")
      add_library(${GPU_LIB_TARGET} SHARED ${APP_SOURCES})
      target_link_libraries(${GPU_LIB_TARGET} memManager_cc imgio_cc cudart cufft cusolver)

    endif()
    file( GLOB APP_SOURCES RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/post_gpu/*c)
    foreach( sourcefile ${APP_SOURCES})
      cmake_path(GET sourcefile FILENAME libname)
      string( REPLACE ".c" "_c" libname ${libname} )
      string( REPLACE ".cc" "_cc" libname ${libname} )
      add_library( ${libname} SHARED ${sourcefile})
      target_link_libraries(${libname} ${GPU_LIB_TARGET})
    endforeach( sourcefile ${APP_SOURCES} )

    target_link_libraries(cuPlotter_cc videoWriter_cc freetype_c)
    target_link_libraries(broadBand_cc gsl gslcblas memManager_cc cuPlotter_cc)
    target_link_libraries(spectImaging_cc broadBand_cc cuPlotter_cc misc_cc FISTA_cc)
    target_link_libraries(spectPhase_cc broadBand_cc cuPlotter_cc misc_cc)
    target_link_libraries(monoChromo_cc broadBand_cc orthFitter_cc FISTA_cc cuPlotter_cc)
    target_link_libraries(FGA_cc monoChromo_cc cuPlotter_cc)
    target_link_libraries(freetype_c freetype fontconfig)
    target_link_libraries(cdilmdb_c lmdb z)
    target_link_libraries(videoWriter_cc avformat avutil avcodec pthread)
    target_link_libraries(readConfig_cc config)
    target_link_libraries(readCXI_cc ${GPU_LIB_TARGET} hdf5 cuPlotter_cc)
    target_link_libraries(orthFitter_cc matrixInverse_cc)
    target_link_libraries(experimentConfig_cc readConfig_cc)
    target_link_libraries(imgio_c tiff png jpeg)
    target_link_libraries(imgio_cc imgio_c)
    target_link_libraries(cdi_cc experimentConfig_cc mnistData_cc FISTA_cc cuPlotter_cc misc_cc)
    target_link_libraries(beamDecomposition_cc ${GPU_LIB_TARGET})
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
    target_link_libraries(testSVD_run ${GPU_LIB_TARGET})
    target_link_libraries(freetype_run freetype_c imgio_cc memManager_cc)
    target_link_libraries(cdi_run cdi_cc)
    target_link_libraries(ptycho_run experimentConfig_cc cuPlotter_cc misc_cc beamDecomposition_cc FISTA_cc)
    target_link_libraries(sesolver_run cuPlotter_cc)
    target_link_libraries(frog_run cuPlotter_cc misc_cc)
    target_link_libraries(beamGenerator_run memManager_cc imgio_cc ${GPU_LIB_TARGET} cuPlotter_cc)
    target_link_libraries(testLMDB_run cdilmdb_c)
    target_link_libraries(testfitter_run orthFitter_cc)
    target_link_libraries(holo_run holo_cc)
    target_link_libraries(propagatorDemo_run cuPlotter_cc misc_cc)
    target_link_libraries(fftDemo_run cuPlotter_cc)
    target_link_libraries(zoom_run cuPlotter_cc)
    target_link_libraries(processimg_run cuPlotter_cc misc_cc)
    target_link_libraries(processimg_ptycho_run cuPlotter_cc misc_cc)
    target_link_libraries(takepsnr_run cuPlotter_cc misc_cc)
    target_link_libraries(profile_run cuPlotter_cc misc_cc)
    target_link_libraries(pulseGen_run cdilmdb_c monoChromo_cc cdi_cc)
    target_link_libraries(mono_run cdilmdb_c monoChromo_cc cdi_cc)
    target_link_libraries(testFISTA_run cdi_cc cuPlotter_cc)
    target_link_libraries(simLineSpectrumImaging_run cdi_cc spectImaging_cc material_cc)
    target_link_libraries(spectImaging_run ${GPU_LIB_TARGET} spectImaging_cc material_cc)
    target_link_libraries(spectImaging_point_run ${GPU_LIB_TARGET} spectImaging_cc material_cc)
    target_link_libraries(spectphase_run readConfig_cc spectPhase_cc material_cc)
    target_link_libraries(bandwidthtest_run ${GPU_LIB_TARGET})
    target_link_libraries(cup_run cuPlotter_cc FISTA_cc)
    target_link_libraries(testCXI_run readCXI_cc)
    target_link_libraries(beamDecomposition_run cuPlotter_cc beamDecomposition_cc)

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
