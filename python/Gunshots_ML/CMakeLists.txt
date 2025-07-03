# CMakeLists.txt for RP2040 Gunshot Detection
cmake_minimum_required(VERSION 3.13)

# Include pico-sdk
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(gunshot_detection)

# Initialize the SDK
pico_sdk_init()

# Add executable
add_executable(gunshot_detection
    rp2040_gunshot_detection.cpp
)

# Create map/bin/hex/uf2 files
pico_add_extra_outputs(gunshot_detection)

# Link libraries
target_link_libraries(gunshot_detection
    pico_stdlib
    pico_tflmicro
)

# Enable USB output, disable UART output
pico_enable_stdio_usb(gunshot_detection 1)
pico_enable_stdio_uart(gunshot_detection 0)

# Compiler flags for optimization
target_compile_options(gunshot_detection PRIVATE
    -Os                    # Optimize for size
    -DPICO_USE_MALLOC_MUTEX=0
    -DPICO_TIME_DEFAULT_ALARM_POOL_DISABLED=1
)

# Include directories
target_include_directories(gunshot_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
