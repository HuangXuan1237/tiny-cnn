cmake_minimum_required(VERSION 3.16)

project(tiny-cnn LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)

add_compile_options(-O3 -flto=auto -mavx2 -mfma)

# Debug Only
# add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
# add_link_options(-fsanitize=address)

find_package(OpenMP REQUIRED)
find_library(OPENBLAS_LIB NAMES openblas)
find_library(MATH_LIB m)

include_directories(include third_party)

file(
    GLOB_RECURSE SOURCES
    "src/*.cpp" "src/*.c"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(
    tiny-cnn 
    PRIVATE 
    ${OPENBLAS_LIB} 
    ${MATH_LIB} 
    OpenMP::OpenMP_C
)