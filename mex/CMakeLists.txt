
# Matlab package
find_package(Matlab) # https://cmake.org/cmake/help/latest/module/FindMatlab.html

if(Matlab_FOUND)
    message(STATUS "matlab have found")
else()
    # message(STATUS "can't use find_package to find matlab")
    if(WIN32)
        set(Matlab_ROOT_DIR "C:/Program Files/MATLAB/R2022b")
    else()
        set(Matlab_ROOT_DIR "/opt_disk2/rd22946/MATLAB/R2023a")
    endif(WIN32)

    find_package(Matlab REQUIRED)
    if(NOT Matlab_FOUND)
        message(FATAL "can't find matlab")
    endif(NOT Matlab_FOUND)
    
endif(Matlab_FOUND)

if(OPENCV_VERSION_3)
set(mainCreateDatabase createDatabase)
message(STATUS "我的子mex文件名字：${mainCreateDatabase}")

add_library(${mainCreateDatabase} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/loopDatabase.cpp)
target_include_directories(${mainCreateDatabase} PUBLIC ${PROJECT_SOURCE_DIR}/src ${Matlab_INCLUDE_DIRS})
target_link_libraries(${mainCreateDatabase} ${PROJECT_NAME} ${MATLAB_LIBRARIES})
message(STATUS "链接的库名为：${PROJECT_NAME},matlab包含的头文件有:${Matlab_INCLUDE_DIRS}")
    if(WIN32)
        set_target_properties(${mainCreateDatabase} PROPERTIES OUTPUT_NAME "${mainCreateDatabase}.mexw64"
            PREFIX "" # 前缀为空
            SUFFIX "") # 后缀为空
    elseif(UNIX)
        set_target_properties(${mainCreateDatabase} PROPERTIES OUTPUT_NAME "${mainCreateDatabase}.mexa64"
            PREFIX ""
            SUFFIX "")
    elseif(APPLE)
        set_target_properties(${mainCreateDatabase} PROPERTIES OUTPUT_NAME "${mainCreateDatabase}.mexmaci64"
            PREFIX ""
            SUFFIX "")
    else()
        message(FATAL_ERROR "Now this platform is not support.")
    endif()

install(TARGETS ${mainCreateDatabase}
        EXPORT ${PROJECT_NAME}_Targets # 生成.cmake文件
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    message(STATUS "执行make install, 安装路径为：${PROJECT_SOURCE_DIR}/install")
endif()

