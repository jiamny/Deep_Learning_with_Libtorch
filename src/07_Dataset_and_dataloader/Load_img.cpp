#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>
#include <sys/stat.h>
#include <dirent.h>


int main() {

	std::vector<std::string> ff;
	ff.push_back("./data/group.jpg");
	ff.push_back("./data/cat_image.jpg");

	for( int i = 0; i < ff.size(); i++ ) {
		// Read input image
		std::cout << ff[i] << std::endl;

		cv::Mat image= cv::imread(ff[i].c_str(), 1);
		if (image.empty()) {  // error handling
			// no image has been created...
			// possibly display an error message
			// and quit the application
			std::cout << "Error reading image..." << std::endl;
			return 0;
		} else {
			// Display the image
			/*
			cv::namedWindow("Original Image");
			cv::imshow("Original Image",image);
			cv::waitKey(0);
			*/
		}
	}
//	std::string dir = "ttfolder";
//	if( mkdir(dir.c_str(), 0777) != -1) std::cout << "Directory created\n";
/*
	struct dirent *entry;
	DIR *dp;
	std::string path = "./data/Caltech_101";

	dp = opendir(path.c_str());
	if (dp == NULL) {
	   std::cout << "opendir: Path does not exist or could not be read.\n";
	   return -1;
	}

	struct stat s;

	while ((entry = readdir(dp))) {
		std::string fd(entry->d_name);
		std::cout << fd << " " << fd.length() << std::endl;

		DIR *fdp;
		struct dirent *fentry;
		std::string fpath = path + "/" + fd;

		if( stat(fpath.c_str(), &s) == 0 ) {
			   if( s.st_mode & S_IFDIR ){
			   //it's a directory
			   fdp = opendir(fpath.c_str());

			   while ((fentry = readdir(fdp))) {
			    	std::string fn(fentry->d_name);
			    	std::string ffpath = path + "/" + fd + "/" + fn;
			    	if( stat(ffpath.c_str(), &s) == 0 ) {
			    		if( s.st_mode & S_IFREG ){
			    			//it's a file
			    			std::cout << "fname = " << ffpath << " label = " << fd << std::endl;
			    		}
			    	}
			    }
			    closedir(fdp);
			}
		}
	}

	closedir(dp);
*/


	return 0;
}
