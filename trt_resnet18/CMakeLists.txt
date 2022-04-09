cmake_minimum_required(VERSION 3.1) 
project(Demo VERSION 1.0)        
set(target_name Demo)    
enable_language(CUDA)

set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wno-write-strings -fpermissive")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")


#cuda
#${CUDA_INCLUDE_DIRS}
#${CUDA_LIBRARIES}

find_package(CUDA REQUIRED)

#opencv
#find_package(OpenCV REQUIRED)

set(LOCAL_BUILD_OPENCV_INCLUDE  /home/wending/user/00_3d_package/opencv-3.4.5/release/include/)
set(LOCAL_BUILD_OPENCV_LIB  /home/wending/user/00_3d_package/opencv-3.4.5/release/lib/)
#TensorRT
set(LOCAL_BUILD_TensorRT /home/wending/user/00_3d_package/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6/TensorRT-7.0.0.11)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(
        /usr/local/lib
        /usr/lib
        ${LOCAL_BUILD_TensorRT}/lib
        )
#sample
add_executable(${target_name} demo/main.cpp src/tensorrt.cpp)				  
target_include_directories(${target_name} PRIVATE include/tensorrt.hpp
                                                ${LOCAL_BUILD_OPENCV_INCLUDE}
                                                ${LOCAL_BUILD_TensorRT}/include 
                                                ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(${target_name} ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvparsers "stdc++fs")

