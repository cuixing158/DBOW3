add_library(loopDatabase_x86_64 STATIC loopDatabase_x86_64.cpp)
target_include_directories(loopDatabase_x86_64 PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(loopDatabase_x86_64 ${PROJECT_NAME} )