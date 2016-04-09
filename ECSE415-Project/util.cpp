#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

#include "util.h"

std::string tilts[] = {"-90", "-60", "-30", "-15", "0", "+15", "+30", "+60", "+90"};
std::string pans[] = {"-90", "-75", "-60", "-45", "-30", "-15", "0", "+15", "+30", "+45", "+60", "+75", "+90"};
std::string get_image_pose(int id, int serie, int number, std::string tilt, std::string pan) {
	std::stringstream s, id_ss, number_ss;
	
	id_ss << std::setfill('0') << std::setw(2) << id;
	number_ss << std::setfill('0') << std::setw(2) << number;

	s << POSE_DIR << "Person" << id_ss.str() << "/" << "person" << id_ss.str() << serie << number_ss.str() << tilt << pan << ".jpg";
	return s.str();
}

std::string get_image_qmul(std::string person, int tilt, int angle) {
	std::stringstream s, tilt_ss, angle_ss;
	tilt_ss << std::setfill('0') << std::setw(3) << tilt;
	angle_ss << std::setfill('0') << std::setw(3) << angle;

	s << QMUL_DIR << person << "/" << person.substr(0, person.size() - 4) << "_"  << tilt_ss.str() << "_" << angle_ss.str() << ".ras";
	return s.str();
}

std::vector<std::string> open_all_poses() {
	std::vector<std::string> names;
	for (int id=0; id <= 15; id++) {
		for (int serie=1; serie <= 2; serie++) {
			for (int number=0; number <= 92; number++) {
				for (int til=0; til <= 9; til++) {
					for (int pan=0; pan <= 9; pan++) {
						std::string pose_name = get_image_pose(1, 1, 00, "-90", "+0");
						names.push_back(pose_name);
					}
				}
			}
		}
	}
	return names;
}