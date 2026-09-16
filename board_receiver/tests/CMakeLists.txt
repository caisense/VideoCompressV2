foreach(test_name IN ITEMS rtp_parser hevc_depacketizer access_unit profile_switch receiver_stats)
    add_executable(test_${test_name} test_${test_name}.cc)
    target_link_libraries(test_${test_name} PRIVATE board_receiver_protocol)
    target_include_directories(test_${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${test_name} COMMAND test_${test_name})
endforeach()

add_executable(test_sender_packetizer
    test_sender_packetizer.cc
    ../../cpp/transport/packetizer.cc)
target_include_directories(test_sender_packetizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp)
target_link_libraries(test_sender_packetizer PRIVATE board_receiver_protocol)
add_test(NAME sender_packetizer_differential COMMAND test_sender_packetizer)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME python_cpp_protocol_differential
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_python_cpp_differential.py
                $<TARGET_FILE:test_sender_packetizer>
                ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/live_h265_hud.py)
endif()
