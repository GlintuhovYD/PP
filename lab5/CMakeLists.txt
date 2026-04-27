cmake_minimum_required(VERSION 3.15)
project(MatrixMult_Lab3_MPI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Находим MPI с помощью встроенного модуля find_package
find_package(MPI REQUIRED)

# Создаём исполняемый файл
add_executable(lab1 lab1.cpp)

# Подключаем заголовки и библиотеки MPI
target_include_directories(lab1 PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(lab1 PRIVATE ${MPI_CXX_LIBRARIES})

# Для Windows MSVC можно добавить опции компиляции (опционально)
if(MSVC)
    target_compile_options(lab1 PRIVATE /EHsc /Zi)
    set_property(TARGET lab1 PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Для отображения найденных путей (полезно для диагностики)
message(STATUS "MPI include dirs: ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI libraries: ${MPI_CXX_LIBRARIES}")