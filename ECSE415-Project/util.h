#pragma once

const char QMUL_DIR[] = "QMUL/";
const char POSE_DIR[] = "HeadPoseImageDatabase/";

#include <string>
#include <vector>

std::string get_image_hpid(int id, int serie, int number, std::string tilt, std::string pan);
std::string get_image_qmul(std::string person, int tilt, int pose);

std::vector<std::vector<std::string>> open_all_qmul_by_person(std::vector<std::string> people);
std::vector<std::vector<std::string>> open_all_qmul_by_pose(std::vector<std::string> people);
