#include "folder.h"
#include "../utils/hand.h"
#include "../utils/readfile.h"

void load_imgs_from_folder(std::string folder, std::string image_type, std::vector<std::string> &list_images) {

    for_each_file(folder, [&](const char *path, const char *name) {
        auto full_path = std::string(path).append({file_sepator()}).append(name);
        std::string lower_name = tolower1(name);

        if (end_with(lower_name, image_type)) {
            list_images.push_back(full_path);
        }
        return false;
    }, true);
}

