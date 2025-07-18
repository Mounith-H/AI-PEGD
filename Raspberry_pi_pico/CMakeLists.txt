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

project(app C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1
add_executable(app
    Raspberry_pi_pico.cpp 
    lib/NRF24/NRF24.cpp
)

# Generate PIO header for power_control 
pico_generate_pio_header(app ${CMAKE_CURRENT_LIST_DIR}/power_control.pio)

pico_set_program_name(app "Raspberry_pi_pico")
pico_set_program_version(app "0.1")

add_subdirectory(lib/pico_voice_recognize)
add_subdirectory(lib/pico_audio_recorder)

# Add the standard library to the build
target_link_libraries(app
        pico_stdlib
        pico_multicore
        hardware_spi
        hardware_pio
        hardware_adc
        hardware_dma
        hardware_uart
        hardware_rtc 
        pico_util
        pico_audio_recorder
)

# Add the standard include files to the build
target_include_directories(app PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/lib/NRF24
)


# Configure UART/USB settings and generate output files
pico_enable_stdio_uart(app 1)  # Enable UART for printf
pico_enable_stdio_usb(app 0)   # Disable USB
pico_add_extra_outputs(app)     # Generate bin/hex/uf2 files


