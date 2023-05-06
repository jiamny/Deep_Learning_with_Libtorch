#include <iostream>                    // std::cout, std::flush
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <utility>                     // std::pair
// For External Library
#include <torch/torch.h>               // torch
//#include <boost/program_options.hpp> // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "options.hpp"
#include "networks.hpp"               // MC_Discriminator

// Define Namespace
namespace fs = std::filesystem;
//namespace po = boost::program_options;

// Function Prototype
void valid(Option_Arguments &vm, DataLoader::ImageFolderClassesWithPaths &valid_dataloader, torch::Device &device, Loss &criterion, MC_Discriminator &model, const std::vector<std::string> class_names, const size_t epoch);

// -------------------
// Training Function
// -------------------
void train(Option_Arguments &vm, torch::Device &device, MC_Discriminator &model, std::vector<transforms_Compose> &transform, const std::vector<std::string> class_names){

    constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
    constexpr size_t train_workers = 2;  // the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 2;  // the number of workers to retrieve data from the validation dataset

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t epoch;
    size_t total_iter;
    size_t start_epoch, total_epoch;
    std::string date, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, path;
    std::string dataroot, valid_dataroot;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, valid_dataloader;
//    visualizer::graph train_loss, valid_loss, valid_accuracy, valid_each_accuracy;
//    progress::display *show_progress;
//    progress::irregular irreg_progress;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    dataroot = vm.dataroot + "/" + vm.dataset + "/" + vm.train_dir;
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, vm.batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // (2) Get Validation Dataset
    if(vm.valid){
        valid_dataroot = vm.dataroot + "/" + vm.dataset + "/" + vm.valid_dir;
        valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
        valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, vm.valid_batch_size, /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

    // (3) Set Optimizer Method
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(vm.lr).betas({vm.beta1, vm.beta2}));

    // (4) Set Loss Function
    auto criterion = Loss();

    // (5) Make Directories
    checkpoint_dir = "checkpoints/" + vm.dataset;
    path = checkpoint_dir + "/models";  fs::create_directories(path);
    path = checkpoint_dir + "/optims";  fs::create_directories(path);
    path = checkpoint_dir + "/log";  fs::create_directories(path);

    // (6) Set Training Loss for Graph
    path = checkpoint_dir + "/graph";
//    train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{"Classification"});
//    if(vm.valid){
//        valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{"Classification"});
//        valid_accuracy = visualizer::graph(path, /*gname_=*/"valid_accuracy", /*label_=*/{"Accuracy"});
//        valid_each_accuracy = visualizer::graph(path, /*gname_=*/"valid_each_accuracy", /*label_=*/class_names);
//    }

    // (7) Get Weights and File Processing
    if(vm.train_load_epoch == ""){
        model->init();
        ofs.open(checkpoint_dir + "/log/train.txt", std::ios::out);
        if(vm.valid){
            init.open(checkpoint_dir + "/log/valid.txt", std::ios::trunc);
            init.close();
        }
        start_epoch = 0;
    } else{
        path = checkpoint_dir + "/models/epoch_" + vm.train_load_epoch + ".pth";  torch::load(model, path, device);
        path = checkpoint_dir + "/optims/epoch_" + vm.train_load_epoch + ".pth";  torch::load(optimizer, path, device);
        ofs.open(checkpoint_dir + "/log/train.txt", std::ios::app);
        ofs << std::endl << std::endl;
        if (vm.train_load_epoch == "latest"){
            infoi.open(checkpoint_dir + "/models/info.txt", std::ios::in);
            std::getline(infoi, buff);
            infoi.close();
            latest = "";
            for (auto &c : buff){
                if (('0' <= c) && (c <= '9')){
                    latest += c;
                }
            }
            start_epoch = std::stoi(latest);
        }
        else{
            start_epoch = std::stoi(vm.train_load_epoch);
        }
    }

    // (8) Display Date
    //date = progress::current_date();
    //date = progress::separator_center("Train Loss (" + date + ")");
    //std::cout << std::endl << std::endl << date << std::endl;
    //ofs << date << std::endl;


    // -----------------------------------
    // a2. Training Model
    // -----------------------------------
    
    // (1) Set Parameters
    start_epoch++;
    total_iter = dataloader.get_count_max();
    total_epoch = vm.epochs;

    std::vector<float> loss_ave;

    // (2) Training per Epoch
//    irreg_progress.restart(start_epoch - 1, total_epoch);
    for (epoch = start_epoch; epoch <= total_epoch; epoch++){

        model->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        //std::cout << "epoch:" << epoch << ' / ' << total_epoch << std::endl;
        //show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{epoch, total_epoch}, /*loss_=*/{"classify"});

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        float loss_sum = 0.0;
        while (dataloader(mini_batch)){

            // -----------------------------------
            // c1. Discriminator Training Phase
            // -----------------------------------
            image = std::get<0>(mini_batch).to(device);
            label = std::get<1>(mini_batch).to(device);
            output = model->forward(image);
            loss = criterion(output, label);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // -----------------------------------
            // c2. Record Loss (iteration)
            // -----------------------------------
            //show_progress->increment(/*loss_value=*/{loss.item<float>()});
            //ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            //ofs << "classify:" << loss.item<float>() << "(ave:" <<  show_progress->get_ave(0) << ')' << std::endl;
            loss_sum += loss.item<float>();
        }

        loss_ave.push_back(loss_sum/total_iter);
        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        //train_loss.plot(/*base=*/epoch, /*value=*/loss_ave);
        std::cout << "epoch:" << epoch << " / "  << total_epoch << ", avg_loss: " << (loss_sum/total_iter) << std::endl;
        //delete show_progress;
        
        // -----------------------------------
        // b3. Validation Mode
        // -----------------------------------
        if(vm.valid && ((epoch - 1) % vm.valid_freq == 0)){
            valid(vm, valid_dataloader, device, criterion, model, class_names, epoch);
        }

        // -----------------------------------
        // b4. Save Model Weights and Optimizer Parameters
        // -----------------------------------
        if (epoch % vm.save_epoch == 0){
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + ".pth";  torch::save(model, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + ".pth";  torch::save(optimizer, path);
        }
        path = checkpoint_dir + "/models/epoch_latest.pth";  torch::save(model, path);
        path = checkpoint_dir + "/optims/epoch_latest.pth";  torch::save(optimizer, path);
        infoo.open(checkpoint_dir + "/models/info.txt", std::ios::out);
        infoo << "latest = " << epoch << std::endl;
        infoo.close();

        // -----------------------------------
        // b5. Show Elapsed Time
        // -----------------------------------
        //if (epoch % 2 == 0){

            // -----------------------------------
            // c1. Get Output String
            // -----------------------------------
            //ss.str(""); ss.clear(std::stringstream::goodbit);
            //irreg_progress.nab(epoch);
            //ss << "elapsed = " << irreg_progress.get_elap() << '(' << irreg_progress.get_sec_per() << "sec/epoch)   ";
            //ss << "remaining = " << irreg_progress.get_rem() << "   ";
            //ss << "now = " << irreg_progress.get_date() << "   ";
            //ss << "finish = " << irreg_progress.get_date_fin();
            //date_out = ss.str();

            // -----------------------------------
            // c2. Terminal Output
            // -----------------------------------
            //std::cout << date_out << std::endl << progress::separator() << std::endl;
            //ofs << date_out << std::endl << progress::separator() << std::endl;
        //}
    }

    // Post Processing
    ofs.close();

    // End Processing
    return;
}
