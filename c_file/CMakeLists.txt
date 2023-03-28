add_library(loopDatabase_x86_64 SHARED loopDatabase_x86_64.cpp opencvAPI.cpp)
target_include_directories(loopDatabase_x86_64 PUBLIC ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(loopDatabase_x86_64 ${OpenCV_LIBS} )

option(BUILD_demo	"Set to OFF to not build demo_general_query" ON)
if(BUILD_demo)
    add_executable(demo_general_query  ${CMAKE_CURRENT_SOURCE_DIR}/loopDatabase_x86_64.cpp ${CMAKE_CURRENT_SOURCE_DIR}/demo_general_query.cpp)
    target_include_directories(demo_general_query PUBLIC ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(demo_general_query ${OpenCV_LIBS} ${PROJECT_NAME} )
endif(BUILD_demo)

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR})
install(TARGETS  loopDatabase_x86_64) 