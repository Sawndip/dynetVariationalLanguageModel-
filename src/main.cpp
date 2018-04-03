#include "ptbReader.h"
#include "variationalLm.h"
#include "rnnLm.h"

#include "dynet/training.h"
#include "dynet/io.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/gru.h"
#include "dynet/dict.h"

#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <cassert> 

std::string PROJECT_PATH = "/home/shantanu/Programming/dynetCppProjects/vaeLm/";
const std::string PTB_TRAIN_FILE = PROJECT_PATH + "src/ptb_data/ptb_train.txt";
const std::string PTB_VALID_FILE = PROJECT_PATH + "src/ptb_data/ptb_valid.txt";  
const std::string UNK            = "<unk>"; // as defined in ptb train file
const unsigned int LAYERS        = 1;
const unsigned int IMPUT_DIM     = 64;
const unsigned int HIDDEN_DIM    = 128;
const unsigned int HIDDEN2_DIM   = 32;
const unsigned int LATENT_DIM    = 10;
const unsigned int NOISE_SAMPLES = 1;
const unsigned int MAX_EPOCHS    = 10;
const unsigned int BATCH_SIZE    = 16;



void run_vaelm(std::vector<std::vector<int> >* pt_ptb_train_data,
               std::vector<std::vector<int> >* pt_ptb_valid_data, 
               const dynet::Dict& dict)
{
    std::cout << "running vaeLm" << std::endl;
    
    // dynet model
    std::shared_ptr<dynet::ParameterCollection> sp_model = 
                          std::make_shared<dynet::ParameterCollection>();
    VariationalLm vaeLm(sp_model, 
                        LAYERS, 
                        IMPUT_DIM, 
                        HIDDEN_DIM, 
                        HIDDEN2_DIM,
                        LATENT_DIM,
                        dict.size());
    vaeLm.train(pt_ptb_train_data, pt_ptb_valid_data, MAX_EPOCHS, BATCH_SIZE); 
    return;
}

int main(int argc, char** argv)
{
     // Initialize dynet
    dynet::DynetParams dyparams = dynet::extract_dynet_params(argc, argv); 
    dynet::initialize(dyparams);

    // Read training data and construct dict {word: word_idx}
    dynet::Dict dict;
    std::vector<std::vector<int> > ptb_train_data;
    PtbReader::get_ptb_data(&ptb_train_data, &dict, PTB_TRAIN_FILE);
    // Freeze dict and set unk
    dict.freeze();
    dict.set_unk(UNK);
    PtbReader::log_data_stats(ptb_train_data, dict, "Training data");
  
    // Validation data
    std::vector<std::vector<int> > ptb_valid_data;
    PtbReader::get_ptb_data(&ptb_valid_data, &dict, PTB_VALID_FILE);
    PtbReader::log_data_stats(ptb_train_data, dict, "Validation data");

    run_vaelm(&ptb_train_data, &ptb_valid_data, dict);
}
