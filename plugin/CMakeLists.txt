cmake_minimum_required(VERSION 3.20)
project(SinhalaASRPlugin LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra -pthread -Wno-template-body)
endif()

add_compile_definitions(SOL_ALL_SAFETIES_ON=1 SOL_STD_OPTIONAL=1)

include(FetchContent)

# 1. Fetch Lua 5.4 source if not found on system
find_package(Lua QUIET)
if(NOT LUA_FOUND)
    message(STATUS "System Lua headers not found. Building Lua via FetchContent...")
    FetchContent_Declare(
        lua_src
        GIT_REPOSITORY https://github.com/lua/lua.git
        GIT_TAG v5.4.6
    )
    FetchContent_GetProperties(lua_src)
    if(NOT lua_src_POPULATED)
        FetchContent_Populate(lua_src)
        # Gather all Lua source files except lua.c and luac.c (which have main())
        file(GLOB LUA_SOURCES "${lua_src_SOURCE_DIR}/*.c")
        list(FILTER LUA_SOURCES EXCLUDE REGEX ".*(lua|luac)\\.c$")
        add_library(lua_lib STATIC ${LUA_SOURCES})
        target_include_directories(lua_lib PUBLIC ${lua_src_SOURCE_DIR})
        set(LUA_LIBRARIES lua_lib)
        set(LUA_INCLUDE_DIR ${lua_src_SOURCE_DIR})
    endif()
endif()

# 2. Fetch sol2
message(STATUS "Fetching sol2 via FetchContent...")
FetchContent_Declare(
    sol2
    GIT_REPOSITORY https://github.com/ThePhD/sol2.git
    GIT_TAG v3.3.0
)
FetchContent_MakeAvailable(sol2)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/moodycamel
    ${LUA_INCLUDE_DIR}
)

# Core Plugin Executable / Harness
add_executable(sinhala_asr_engine
    src/main.cpp
)

target_link_libraries(sinhala_asr_engine PRIVATE
    sol2::sol2
    ${LUA_LIBRARIES}
    pthread
    dl
)

message(STATUS "Configured Sinhala ASR Plugin Engine with C++20 and Lua.")
