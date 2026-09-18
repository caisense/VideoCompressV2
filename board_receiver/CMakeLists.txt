cmake_minimum_required(VERSION 3.15)
project(board_receiver LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

option(BOARD_RECEIVER_BUILD_APP "Build the RK3588 receiver application" ON)
option(BOARD_RECEIVER_BUILD_TESTS "Build protocol unit tests" ON)

add_library(board_receiver_protocol STATIC
    src/access_unit_assembler.cc
    src/hevc_depacketizer.cc
    src/profile_switch_gate.cc
    src/receiver_stats.cc
    src/rtp_packet.cc
    src/rtp_reorder_buffer.cc
    src/stream_profile.cc)
target_include_directories(board_receiver_protocol PUBLIC include)
target_compile_options(board_receiver_protocol PRIVATE -Wall -Wextra -Wpedantic)

if(BOARD_RECEIVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BOARD_RECEIVER_BUILD_APP)
    if(NOT CMAKE_SYSROOT)
        message(FATAL_ERROR "BOARD_RECEIVER_BUILD_APP requires the Buildroot sysroot")
    endif()
    find_path(MPP_INCLUDE_DIR rk_mpi.h PATHS
        ${CMAKE_SYSROOT}/usr/include ${CMAKE_SYSROOT}/usr/include/rockchip
        ${CMAKE_SYSROOT}/usr/include/mpp NO_DEFAULT_PATH)
    find_library(MPP_LIBRARY NAMES rockchip_mpp mpp PATHS
        ${CMAKE_SYSROOT}/usr/lib ${CMAKE_SYSROOT}/usr/lib64 ${CMAKE_SYSROOT}/lib
        NO_DEFAULT_PATH)
    if(NOT MPP_INCLUDE_DIR OR NOT MPP_LIBRARY)
        message(FATAL_ERROR "Target Rockchip MPP headers/library not found")
    endif()
    foreach(opencv_lib IN ITEMS core highgui imgproc)
        find_library(OPENCV_${opencv_lib}_LIBRARY NAMES opencv_${opencv_lib}
            PATHS ${CMAKE_SYSROOT}/usr/lib ${CMAKE_SYSROOT}/usr/lib64
            NO_DEFAULT_PATH REQUIRED)
    endforeach()
    add_executable(board_h265_receiver
        src/board_display.cc src/glibc_compat.c src/local_tx_rate_receiver.cc
        src/main.cc src/mpp_hevc_decoder.cc
        src/rtp_receiver.cc)
    target_include_directories(board_h265_receiver PRIVATE
        include ${MPP_INCLUDE_DIR} ${CMAKE_SYSROOT}/usr/include/opencv4)
    target_link_libraries(board_h265_receiver PRIVATE board_receiver_protocol
        ${MPP_LIBRARY} ${OPENCV_core_LIBRARY} ${OPENCV_highgui_LIBRARY}
        ${OPENCV_imgproc_LIBRARY} Threads::Threads)
    target_compile_options(board_h265_receiver PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(board_h265_receiver PROPERTIES
        INSTALL_RPATH "$ORIGIN/lib")
    install(TARGETS board_h265_receiver DESTINATION .)
endif()
