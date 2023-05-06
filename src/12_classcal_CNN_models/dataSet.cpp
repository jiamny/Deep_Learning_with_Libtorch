#include "dataSet.h"


/* Returns a list of files in a directory (except the ones that begin with a dot) */

std::vector<std::string> GetFilesInDirectory(const std::string &directory) {
	std::vector<std::string> out;
#ifdef WINDOWS
    HANDLE dir;
    WIN32_FIND_DATA file_data;

    if ((dir = FindFirstFile((directory + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
        return; /* No files found */

    do {
        const string file_name = file_data.cFileName;
        const string full_file_name = directory + "/" + file_name;
        const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

        if (file_name[0] == '.')
            continue;

        if (is_directory)
            continue;

        out.push_back(full_file_name);
    } while (FindNextFile(dir, &file_data));

    FindClose(dir);
#else
    DIR *dir;
    class dirent *ent;
    class stat st;

    dir = opendir(directory.c_str());
    while ((ent = readdir(dir)) != NULL) {
        const std::string file_name = ent->d_name;
        const std::string full_file_name = directory + "/" + file_name;

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory)
            continue;

        out.push_back(full_file_name);
    }
    closedir(dir);
#endif
    return(out);
} // GetFilesInDirectory


//遍历该目录下的.jpg图片
void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label) {
	struct stat s;
	DIR* root_dir;
	struct dirent *dirs;
	if(path.back() != '/') {
		path.push_back('/');
	}

    if((root_dir = opendir(path.c_str())) != NULL) {
    	while ((dirs = readdir(root_dir))) {
        	std::string fd(dirs->d_name);
        	std::string fdpath = path + fd;
 //       	std::cout << fdpath << std::endl;

        	if (fd[0] == '.')
        	   continue;
        	//it's a directory
        	if( stat(fdpath.c_str(), &s) == 0 ) {
        		if( s.st_mode & S_IFDIR ){
        			auto files = GetFilesInDirectory(fdpath);
        			for(auto it = files.begin(); it != files.end(); ++it) {
        				std::string filename = *it;
        				//if( label == 0 ) std::cout <<  filename << std::endl;

        				if(filename.length() > 4 ) {
        					// check image format type
        				    if( filename.find(type.c_str()) ){
        				       cv::Mat img = cv::imread(filename.c_str(), 1);
        				       //std::cout << "empty = " << ( ! img.empty() ) << std::endl;
        				       if( ! img.empty() ) {
        				    	   list_images.push_back(filename);
        				    	   list_labels.push_back(label);
        				        }
        				    }
        				}
        			}
        			label += 1;
        		}
        	}
    	}
    }

    closedir(root_dir);
}

//遍历该目录下的.jpg图片
void load_data_and_classes_from_folder(std::string path, std::string type, std::vector<std::string> &list_images,
										std::vector<int> &list_labels, std::vector<std::string> &classes, int label) {
	struct stat s;
	DIR* root_dir;
	struct dirent *dirs;
	if(path.back() != '/') {
		path.push_back('/');
	}

    if((root_dir = opendir(path.c_str())) != NULL) {
    	while ((dirs = readdir(root_dir))) {
        	std::string fd(dirs->d_name);
        	std::string fdpath = path + fd;
//        	std::cout << fdpath << std::endl;

        	if (fd[0] == '.')
        	   continue;
        	//it's a directory
        	if( stat(fdpath.c_str(), &s) == 0 ) {
        		if( s.st_mode & S_IFDIR ){
        			auto files = GetFilesInDirectory(fdpath);
        			for(auto it = files.begin(); it != files.end(); ++it) {
        				std::string filename = *it;
        				//if( label == 0 ) std::cout <<  filename << std::endl;

        				if(filename.length() > 4 ) {
        					// check image format type
        				    if( filename.find(type.c_str()) ){
        				       cv::Mat img = cv::imread(filename.c_str(), 1);
        				       //std::cout << "empty = " << ( ! img.empty() ) << std::endl;
        				       if( ! img.empty() ) {
        				    	   list_images.push_back(filename);
        				    	   list_labels.push_back(label);
        				        }
        				    }
        				}
        			}
        			classes.push_back(fd);
        			label += 1;
        		}
        	}
    	}
    }

    closedir(root_dir);
}

void load_data_by_classes_from_folder(std::string path, std::string type, std::vector<std::string> &list_images,
										std::vector<int> &list_labels, std::string names[], int num_classes) {
	struct stat s;
	DIR* root_dir;
	struct dirent *dirs;
	if(path.back() != '/') {
		path.push_back('/');
	}

    if((root_dir = opendir(path.c_str())) != NULL) {

    	for( int i = 0; i < num_classes; i++) {
        	std::string fd(names[i]);
        	std::string fdpath = path + fd;
        	std::cout << fdpath << std::endl;

        	//it's a directory
        	if( stat(fdpath.c_str(), &s) == 0 ) {
        		if( s.st_mode & S_IFDIR ){
        			auto files = GetFilesInDirectory(fdpath);
        			for(auto it = files.begin(); it != files.end(); ++it) {
        				std::string filename = *it;
        				//if( label == 0 ) std::cout <<  filename << std::endl;
        				if(filename.length() > 4 ) {
        					// check image format type
        				    if( filename.find(type.c_str()) ){
        				       cv::Mat img = cv::imread(filename.c_str(), 1);
        				       //std::cout << "empty = " << ( ! img.empty() ) << std::endl;
        				       if( ! img.empty() ) {
        				    	   list_images.push_back(filename);
        				    	   list_labels.push_back(i);
        				        }
        				    }
        				}
        			}
        		}
        	}
    	}
    }

    closedir(root_dir);
}

