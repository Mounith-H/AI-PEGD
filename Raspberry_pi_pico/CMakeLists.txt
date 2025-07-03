# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.1.1)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.1.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(Raspberry_pi_pico C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Note: TensorFlow Lite Micro integration temporarily disabled due to version conflicts
# add_subdirectory(lib/pico-tflmicro pico-tflmicro)

# Add executable. Default name is the project name, version 0.1
add_executable(Raspberry_pi_pico
    Raspberry_pi_pico.cpp 
    lib/ml_model/model_handler.c
    lib/NRF24/NRF24.cpp
    lib/gps_handler/gps_handler.cpp
)


# Generate PIO header
pico_generate_pio_header(Raspberry_pi_pico ${CMAKE_CURRENT_LIST_DIR}/inmp441.pio)
pico_generate_pio_header(Raspberry_pi_pico ${CMAKE_CURRENT_LIST_DIR}/power_control.pio)



pico_set_program_name(Raspberry_pi_pico "Raspberry_pi_pico")
pico_set_program_version(Raspberry_pi_pico "0.1")

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(Raspberry_pi_pico 0)
pico_enable_stdio_usb(Raspberry_pi_pico 0)

# Add minmea library
add_library(minmea STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/minmea/minmea.c
)

# Set up minmea includes
target_include_directories(minmea PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/minmea
    ${CMAKE_CURRENT_SOURCE_DIR}  # For time_utils.h
)

# Add compiler definitions for minmea
target_compile_definitions(minmea PRIVATE
    timegm=mktime  # Use mktime instead of timegm
)

# Add the standard library to the build
target_link_libraries(Raspberry_pi_pico
        pico_stdlib
        pico_multicore
        hardware_spi
        hardware_pio
        hardware_dma
        hardware_uart
        minmea
)

# Add the standard include files to the build
target_include_directories(Raspberry_pi_pico PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/minmea
        ${CMAKE_CURRENT_LIST_DIR}/lib/NRF24
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/ml_model
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/pico-tflmicro/src
)


# Configure UART/USB settings and generate output files
pico_enable_stdio_uart(Raspberry_pi_pico 1)  # Enable UART for printf
pico_enable_stdio_usb(Raspberry_pi_pico 0)   # Disable USB
pico_add_extra_outputs(Raspberry_pi_pico)     # Generate bin/hex/uf2 files

