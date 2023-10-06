#pragma once

#include <iostream>
//#include <direct.h>
#include <sys/stat.h>
#include <sys/types.h>

inline void makedirs(const char *path) {

    if(strlen(path) > PATH_MAX) { //_MAX_PATH) {
        throw "path length over MAX_LENGTH";
    }

    int path_length = strlen(path);
    int leave_length = 0;
    int created_length = 0;
    char size_path_temp[PATH_MAX] = {0};
    while (true) {
        int pos = -1;
        if (NULL != strchr(path + created_length, '\\')) {
            pos = strlen(strchr(path + created_length, '\\')) - 1;
        } else if (NULL != strchr(path + created_length, '/')) {
            pos = strlen(strchr(path + created_length, '/')) - 1;
        } else {
            break;
        }
        leave_length = pos;
        created_length = path_length - leave_length;
        strncpy(size_path_temp, path, created_length);
        mkdir(size_path_temp, 0755);
    }
    if (created_length < path_length) {
    	std::cout << "path: " << path << '\n';
        mkdir(path, 0755);
    }
}

