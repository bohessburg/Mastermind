cmake_minimum_required(VERSION 3.16)
project(DominionZero VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Core engine library ---
add_library(dominion_engine
    src/game/card.cpp
    src/game/player.cpp
    src/game/supply.cpp
    src/game/game_state.cpp
    src/game/cards/base_cards.cpp
    src/game/cards/level_1_cards.cpp
    src/game/cards/level_2_cards.cpp
    src/game/cards/level_3_cards.cpp
    src/game/cards/level_4_cards.cpp
    src/game/cards/level_5_cards.cpp
    src/game/cards/level_6_cards.cpp
    src/game/cards/level_7_cards.cpp
    src/game/cards/level_8_cards.cpp
    src/game/cards/level_9_cards.cpp
    src/game/cards/level_10_cards.cpp
    src/engine/action_space.cpp
    src/engine/rules.cpp
    src/engine/agents.cpp
    src/engine/game_runner.cpp
)
target_include_directories(dominion_engine PUBLIC src)
target_compile_options(dominion_engine PRIVATE -Wall -Wextra -Wpedantic)

# --- Main executable (benchmark) ---
add_executable(dominion_zero src/main.cpp)
target_link_libraries(dominion_zero PRIVATE dominion_engine)

# --- Interactive play (vs bot) ---
add_executable(dominion_play src/interactive.cpp)
target_link_libraries(dominion_play PRIVATE dominion_engine)

# --- Local 2-player ---
add_executable(dominion_2p src/local_multiplayer.cpp)
target_link_libraries(dominion_2p PRIVATE dominion_engine)

# --- Stress test ---
add_executable(dominion_stress src/stress_test.cpp)
target_link_libraries(dominion_stress PRIVATE dominion_engine)

# --- Tests ---
option(BUILD_TESTS "Build test suite" ON)
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(dominion_tests
        tests/test_helpers.cpp
        tests/test_player.cpp
        tests/test_supply.cpp
        tests/test_game_state.cpp
        tests/test_cards_base.cpp
        tests/test_card_interactions.cpp
        tests/test_attack_reactions.cpp
        tests/test_action_space.cpp
        tests/test_invariants.cpp
    )
    target_link_libraries(dominion_tests PRIVATE dominion_engine Catch2::Catch2WithMain)
    target_compile_options(dominion_tests PRIVATE -Wall -Wextra)

    include(CTest)
    include(Catch)
    catch_discover_tests(dominion_tests)
endif()
