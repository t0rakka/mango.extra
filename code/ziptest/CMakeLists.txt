cmake_minimum_required(VERSION 3.14)
project(ziptest
        VERSION 1.0.0
        DESCRIPTION "ZIP container test"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ziptest "source/ziptest.cpp")

find_package(mango REQUIRED)
target_link_libraries(ziptest PUBLIC mango::mango)
