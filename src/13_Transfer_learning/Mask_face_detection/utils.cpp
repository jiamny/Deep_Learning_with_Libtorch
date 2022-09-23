#include "utils.hpp"

//Global variables for normalization
std::vector<double> norm_mean = {0.485, 0.456, 0.406};
std::vector<double> norm_std = {0.229, 0.224, 0.225};


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
std::vector<std::string> listContents(const std::string& dirName, const std::vector<std::string>& extensions) {
    std::vector<std::string> fileNames;

    for (auto &p : std::filesystem::directory_iterator(dirName))

    	if( std::find(extensions.begin(), extensions.end(), p.path().extension()) != extensions.end()) {
    		fileNames.push_back(p.path().string());
    	}
    //std::cout << fileNames.size() << std::endl;
    return fileNames;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
std::vector<std::string> listFolders(const std::string& dirName) {
    std::vector<std::string> folders;
    for(auto& p : std::filesystem::recursive_directory_iterator(dirName))
        if (p.is_directory())
            folders.push_back(p.path().string()+"/");

    // if current folder no subfolder(s)
    if(folders.size() < 1)
    	folders.push_back(dirName+"/");

    return folders;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
torch::Tensor read_image(const std::string& imageName) {
    cv::Mat img = cv::imread(imageName);

    cv::resize(img, img, cv::Size(224,224));

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    img.convertTo( img, CV_32FC3, 1/255.0 );

    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat);
    img_tensor = img_tensor.permute({2, 0, 1});

    img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);

    return img_tensor.clone();
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
torch::Tensor read_label(int label) {
    torch::Tensor label_tensor = torch::full({1}, label, torch::kInt64);
    return label_tensor.clone();
}
