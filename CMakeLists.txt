cmake_minimum_required(VERSION 3.16)
project(DominionZero VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_SANITIZERS "Enable ASan/UBSan for v2 targets" OFF)
option(BUILD_PYTHON "Build v2 Python bindings" OFF)

function(configure_v2_target target_name)
    target_compile_features(${target_name} PUBLIC cxx_std_20)
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

    if(ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            target_compile_options(${target_name} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target_name} PRIVATE -fsanitize=address,undefined)
        else()
            message(WARNING "ENABLE_SANITIZERS is only wired for Clang/GNU v2 targets")
        endif()
    endif()
endfunction()

# --- v2 engine ---
add_library(dominion_v2 STATIC
    src/v2/cards/base.cpp
    src/v2/core/actions.cpp
    src/v2/core/determinize.cpp
    src/v2/core/game.cpp
    src/v2/core/interp.cpp
    src/v2/core/moves.cpp
    src/v2/core/score.cpp
    src/v2/core/setup.cpp
    src/v2/core/triggers.cpp
    src/v2/core/turns.cpp
)
target_include_directories(dominion_v2 PUBLIC src)
configure_v2_target(dominion_v2)

add_library(dominion_v2_drivers STATIC
    src/v2/drivers/bots.cpp
)
target_link_libraries(dominion_v2_drivers PUBLIC dominion_v2)
configure_v2_target(dominion_v2_drivers)

add_library(dominion_v2_encode STATIC
    src/v2/encode/encoder.cpp
)
target_link_libraries(dominion_v2_encode PUBLIC dominion_v2)
target_include_directories(dominion_v2_encode PUBLIC src)
configure_v2_target(dominion_v2_encode)

add_library(dominion_v2_observe STATIC
    src/v2/observe/card_text.cpp
)
target_link_libraries(dominion_v2_observe PUBLIC dominion_v2)
target_include_directories(dominion_v2_observe PUBLIC src)
configure_v2_target(dominion_v2_observe)

add_executable(v2_dump_defs
    src/v2/observe/dump_defs.cpp
)
target_link_libraries(v2_dump_defs PRIVATE dominion_v2_observe)
configure_v2_target(v2_dump_defs)

set(V2_DEFS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/src/v2/web/client-data/defs.gen.json")
add_custom_command(
    OUTPUT "${V2_DEFS_JSON}"
    COMMAND "$<TARGET_FILE:v2_dump_defs>" "${V2_DEFS_JSON}"
    DEPENDS v2_dump_defs
    COMMENT "Generating v2 web card definition JSON"
    VERBATIM
)
add_custom_target(v2_client_defs ALL DEPENDS "${V2_DEFS_JSON}")

add_executable(v2_bench
    src/v2/bench/bench.cpp
)
target_link_libraries(v2_bench PRIVATE dominion_v2_drivers)
configure_v2_target(v2_bench)

add_executable(dominion_v2_play
    src/v2/drivers/tui.cpp
)
target_link_libraries(dominion_v2_play PRIVATE dominion_v2_drivers dominion_v2_observe)
configure_v2_target(dominion_v2_play)

add_executable(v2_fuzz
    tests/v2/fuzz_harness.cpp
    tests/v2/fuzz_main.cpp
)
target_link_libraries(v2_fuzz PRIVATE dominion_v2_drivers)
configure_v2_target(v2_fuzz)

if(BUILD_PYTHON)
    if(NOT Python3_EXECUTABLE)
        set(_local_python "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
        if(EXISTS "${_local_python}")
            set(Python3_EXECUTABLE "${_local_python}" CACHE FILEPATH "Python interpreter for v2 bindings")
        endif()
    endif()
    set(Python3_FIND_UNVERSIONED_NAMES FIRST)
    set(Python3_FIND_STRATEGY LOCATION)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    if(NOT pybind11_DIR)
        set(_local_pybind11_dir "${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
        if(EXISTS "${_local_pybind11_dir}")
            set(pybind11_DIR "${_local_pybind11_dir}" CACHE PATH "pybind11 CMake config directory")
        endif()
    endif()
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(dominion_v2_py
        src/v2/py/module.cpp
    )
    target_link_libraries(dominion_v2_py PRIVATE dominion_v2_drivers dominion_v2_encode)
    target_include_directories(dominion_v2_py PRIVATE src)
    add_dependencies(dominion_v2_py v2_client_defs)
    configure_v2_target(dominion_v2_py)
endif()

# --- Tests ---
option(BUILD_TESTS "Build test suite" ON)
if(BUILD_TESTS)
    include(CTest)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(v2_tests
        tests/v2/fuzz_harness.cpp
        tests/v2/spec_harness.cpp
        tests/v2/test_types.cpp
        tests/v2/test_rng.cpp
        tests/v2/test_state.cpp
        tests/v2/test_setup.cpp
        tests/v2/test_game_flow.cpp
        tests/v2/test_determinism.cpp
        tests/v2/test_conservation.cpp
        tests/v2/test_alloc.cpp
        tests/v2/test_fuzz.cpp
        tests/v2/test_golden.cpp
        tests/v2/test_encoder.cpp
        tests/v2/test_determinize.cpp
        tests/v2/test_bots.cpp
        tests/v2/test_specs_base.cpp
        tests/v2/test_attacks_triggers.cpp
    )
    target_link_libraries(v2_tests PRIVATE dominion_v2_drivers dominion_v2_encode Catch2::Catch2WithMain)
    target_compile_definitions(v2_tests PRIVATE V2_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/v2/golden")
    add_dependencies(v2_tests v2_client_defs)
    configure_v2_target(v2_tests)

    include(Catch)
    catch_discover_tests(v2_tests TEST_PREFIX "v2." PROPERTIES TIMEOUT 10)
endif()
