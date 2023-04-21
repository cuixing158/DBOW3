
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src )
LINK_LIBRARIES(${PROJECT_NAME})
message(STATUS "PROJECT_NAME,link library:${PROJECT_NAME}")
IF(OPENCV_VERSION_3)
ADD_EXECUTABLE(demo_general demo_general.cpp)
ADD_EXECUTABLE(create_voc_step0 create_voc_step0.cpp)
ADD_EXECUTABLE(create_voc_step1 create_voc_step1.cpp)
add_executable(demo_queryImg demo_queryImg.cpp)
add_executable(demo_cereal_bench demo_cereal_bench.cpp)
target_include_directories(demo_cereal_bench PUBLIC ${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS}) # 需要链接opencv库，已经在本文件上面LINK_LIBRARIES了

INSTALL(TARGETS  demo_general  create_voc_step0  create_voc_step1  demo_queryImg RUNTIME DESTINATION bin)
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR})
ENDIF()
