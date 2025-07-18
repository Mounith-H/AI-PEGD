
add_library(pico_audio_recorder INTERFACE)
pico_generate_pio_header(pico_audio_recorder ${CMAKE_CURRENT_LIST_DIR}/pico_audio_recorder.pio)
target_sources(pico_audio_recorder INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/pico_audio_recorder.c
)

target_include_directories(pico_audio_recorder INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(pico_audio_recorder INTERFACE
        hardware_dma
        hardware_pio
        hardware_rtc
        pico_stdlib
        hardware_irq    
)