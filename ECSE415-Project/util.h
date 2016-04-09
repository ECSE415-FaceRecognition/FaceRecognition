#pragma once

const char QMUL_DIR[] = "QMUL/";
const char POSE_DIR[] = "HeadPoseImageDatabase/";

#include <string>
#include <vector>

std::string get_image_pose(int id, int serie, int number, std::string tilt, std::string pan);
std::string get_image_qmul(std::string person, int tilt, int angle);
std::vector<std::string> open_all_poses();