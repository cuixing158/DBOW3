add_library(loopDatabase_x86_64 SHARED loopDatabase_x86_64.cpp opencvAPI.cpp)
target_include_directories(loopDatabase_x86_64 PUBLIC ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(loopDatabase_x86_64 ${PROJECT_NAME} )

install(TARGETS  loopDatabase_x86_64) 