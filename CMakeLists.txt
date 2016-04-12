cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project( ImageSegmentation )
find_package( OpenCV REQUIRED )
add_executable( ImageSegmentation ECSE415-Project/lbp.cpp
                                  ECSE415-Project/main.cpp
                                  ECSE415-Project/util.cpp
)
target_link_libraries( ImageSegmentation ${OpenCV_LIBS} )
