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

// --------------------------------------------------------------------------------------------
// read data already splited
void readSplittedDataInfo(std::string dir, std::string infoFilePath, std::string load_type, std::vector<std::string> &list_images,
							std::vector<int> &list_labels) {
	if(dir.back() != '/') {
		dir.push_back('/');
	}

	std::ifstream stream(infoFilePath.c_str());
	  assert(stream.is_open());

	  long label;
	  std::string path, type;

	  while (true) {
	    stream >> path >> label >> type;
	//    std::cout << path << " " << label << " " << type << std::endl;
	    if (load_type == type) {
	    	// check if path containg dir ???
	    	if( path.length() > dir.length() ) {
	    		list_images.push_back(path);
	    		list_labels.push_back(label);
	    	} else {
	    		list_images.push_back( dir + path);
	    		list_labels.push_back(label);
	    	}
	    }

	    if (stream.eof())
	      break;
	  }

//  std::random_shuffle(train.begin(), train.end());
//  std::random_shuffle(test.begin(), test.end());
}

// ------------------------------------------------------------------------------------------
// get from a single folder and separate train and test data sets by ratio
void load_data_from_folder_and_split(std::string file_root, float train_pct, std::vector<std::string> &train_image_paths,
										std::vector<int> &train_labels, std::vector<std::string> &test_image_paths,
										std::vector<int> &test_labels, std::map<int, std::string> &label_names) {
	if(file_root.back() != '/') {
		file_root.push_back('/');
	}

    //std::cout << "root = " << file_root << "\n" << file_root.c_str() << std::endl;
    struct stat s;
    DIR* root_dir;
    struct dirent *dirs;

    int label = 0;

    if((root_dir = opendir(file_root.c_str())) != NULL) {

    	while ((dirs = readdir(root_dir))) {
        	std::string fd(dirs->d_name);
        	std::string fdpath = file_root + fd;
        	//std::cout << fdpath << std::endl;
        	//it's a directory
        	if (fd[0] == '.')
        	    continue;

        	if( stat(fdpath.c_str(), &s) == 0 ) {
        		if( s.st_mode & S_IFDIR ){
        			auto files = GetFilesInDirectory(fdpath);
        			int train_count = static_cast<int>(files.size()*train_pct);
        			//std::cout << fd << " num_files = " << files.size() << " label = " << label << " train_cnt = " << train_count << std::endl;
        			int cnt = 0;
        			for(auto it = files.begin(); it != files.end(); ++it) {
        				std::string filename = *it;
        				//if( label == 0 ) std::cout <<  filename << std::endl;

        				if(filename.length() > 4 ) {
        				    if( filename.find(".jpg") || filename.find(".png") || filename.find(".jpeg")){
        				       cv::Mat img = cv::imread(filename.c_str(), 1);
        				       //std::cout << "empty = " << ( ! img.empty() ) << std::endl;
        				       if( ! img.empty() ) {
        				    	   // train data set
        				    	   if( cnt < train_count ) {
        				    		   train_image_paths.push_back(filename);
        				    		   train_labels.push_back(label);
        				    	   } else {
        				    	   // teat data set
        				    		   test_image_paths.push_back(filename);
        				    		   test_labels.push_back(label);
        				    	   }
        				    	   cnt += 1;
        				        }
        				    }
        				}
        			}
        		}
        	}
        	label_names.insert(std::make_pair(label, fd));
        	label += 1;
    	}
    }

    closedir(root_dir);
// randomize the data items
//    std::random_shuffle(train_list.begin(), train_list.end());
//    std::random_shuffle(test_list.begin(), test_list.end());

//    return std::make_pair(train_list, test_list);
}


//遍历该目录下的.jpg图片 --------------------------------------------------------------------------
void load_data_from_split_folder(std::string path, std::vector<std::string> &image_paths,
									std::vector<int> &labels, std::map<int, std::string> &label_names) {

	struct stat s;
	DIR* root_dir;
	struct dirent *dirs;
	if(path.back() != '/') {
		path.push_back('/');
	}

	int label = 0;
	if((root_dir = opendir(path.c_str())) != NULL) {
	    while ((dirs = readdir(root_dir))) {
	        std::string fd(dirs->d_name);
	        std::string fdpath = path + fd;
	        std::cout << fdpath << std::endl;

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
	        				if( filename.find(".jpg") || filename.find(".png") || filename.find(".jpeg") ){
	        				   cv::Mat img = cv::imread(filename.c_str(), 1);
	        				   //std::cout << "empty = " << ( ! img.empty() ) << std::endl;
	        				   if( ! img.empty() ) {
	        				       image_paths.push_back(filename);
	        				       labels.push_back(label);
	        				    }
	        				}
	        			}
	        		}
	        	}
	        }
	        label_names.insert(std::make_pair(label, fd));
	        label += 1;
	    }
	}
	closedir(root_dir);
}

void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images,
							std::vector<int> &list_labels, std::map<int, std::string> &label_names) {
	struct stat s;
	DIR* root_dir;
	struct dirent *dirs;
	if(path.back() != '/') {
		path.push_back('/');
	}

    int label = 0;
    if((root_dir = opendir(path.c_str())) != NULL) {
    	while ((dirs = readdir(root_dir))) {
        	std::string fd(dirs->d_name);
        	std::string fdpath = path + fd;
        	std::cout << fdpath << std::endl;

        	if (fd[0] == '.')
        	   continue;
        	//it's a directory
        	if( stat(fdpath.c_str(), &s) == 0 ) {
        		if( s.st_mode & S_IFDIR ){
        			auto files = GetFilesInDirectory(fdpath);
        			for(auto it = files.begin(); it != files.end(); ++it) {
        				std::string filename = *it;

        				if(filename.length() > 4 ) {
        					// check image format type
        				    //if( filename.find(type.c_str()) ){
        					if( filename.find(".jpg") || filename.find(".png") || filename.find(".jpeg") ) {
        				       cv::Mat img = cv::imread(filename.c_str(), 1);
        				       //std::cout << "empty = " << ( ! img.empty() ) << std::endl;
        				       if( ! img.empty() ) {
        				    	   list_images.push_back(filename);
        				    	   list_labels.push_back(label);
        				        }
        				    }
        				}
        			}
        		}
        	}
        	label_names.insert(std::make_pair(label, fd));
        	label += 1;
    	}
    }

    closedir(root_dir);
}


void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels) {
	struct stat s;
	DIR* root_dir;
	struct dirent *dirs;
	if(path.back() != '/') {
		path.push_back('/');
	}

    int label = 0;
    if((root_dir = opendir(path.c_str())) != NULL) {
    	while ((dirs = readdir(root_dir))) {
        	std::string fd(dirs->d_name);
        	std::string fdpath = path + fd;
        	std::cout << fdpath << std::endl;

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
        		}
        	}
        	label += 1;
    	}
    }

    closedir(root_dir);

/*
    long long hFile = 0; //句柄
    struct _finddata_t fileInfo;
    std::string pathName;
    if ((hFile = _findfirst(pathName.assign(path).append("\\*.*").c_str(), &fileInfo)) == -1){
        return;
    }
    do
    {
        const char* s = fileInfo.name;
        const char* t = type.data();

        if (fileInfo.attrib&_A_SUBDIR) //是子文件夹
        {
            //遍历子文件夹中的文件(夹)
            if (strcmp(s, ".") == 0 || strcmp(s, "..") == 0) //子文件夹目录是.或者..
                continue;
            std::string sub_path = path + "\\" + fileInfo.name;
            label++;
            load_data_from_folder(sub_path, type, list_images, list_labels, label);

        }
        else //判断是不是后缀为type文件
        {
            if (strstr(s, t))
            {
                std::string image_path = path + "\\" + fileInfo.name;
                list_images.push_back(image_path);
                list_labels.push_back(label);
            }
        }
    } while (_findnext(hFile, &fileInfo) == 0);
    return;
    */
}

