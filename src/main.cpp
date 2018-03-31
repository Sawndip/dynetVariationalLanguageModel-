#include "ptbReader.h"
#include "variationalLm.h"

#include "dynet/training.h"
#include "dynet/io.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/gru.h"
#include "dynet/dict.h"

#include <iostream>
#include <vector>
#include <memory>


std::string PROJECT_PATH = "/home/shantanu/Programming/dynetCppProjects/vaeLm/";
const std::string PTB_TRAIN_FILE = PROJECT_PATH + "src/ptb_data/ptb_train.txt"; 
const std::string UNK            = "<unk>"; // as defined in ptb train file
const unsigned int LAYERS        = 1;
const unsigned int IMPUT_DIM     = 64;
const unsigned int HIDDEN_DIM    = 128;
const unsigned int HIDDEN2_DIM   = 32;
const unsigned int LATENT_DIM    = 10;
const unsigned int NOISE_SAMPLES = 10;
const unsigned int MAX_EPOCHS    = 2;



int main(int argc, char** argv)
{
    // Read data
    std::vector<std::vector<int> > ptb_train_data;
    dynet::Dict dict;
    VaeLm::PtbReader::get_ptb_data(&ptb_train_data, &dict, PTB_TRAIN_FILE);
    dict.freeze();
    dict.set_unk(UNK);
    unsigned int vocab_size = dict.size();

    // Initialize dynet
    dynet::DynetParams dyparams = dynet::extract_dynet_params(argc, argv); 
    dynet::initialize(dyparams);

    // dynet model
    std::shared_ptr<dynet::ParameterCollection> sp_model = 
                          std::make_shared<dynet::ParameterCollection>();
    dynet::AdamTrainer trainer(*sp_model);

    // variational language model instance
    VaeLm::VariationalLm vaeLm(sp_model, 
                               LAYERS, 
                               IMPUT_DIM, 
                               HIDDEN_DIM, 
                               HIDDEN2_DIM,
                               LATENT_DIM,
                               NOISE_SAMPLES, 
                               vocab_size);

    std::shared_ptr<dynet::Expression> sp_mu = std::make_shared<dynet::Expression>();
    std::shared_ptr<dynet::Expression> sp_logvar = std::make_shared<dynet::Expression>();
    std::shared_ptr<dynet::Expression> sp_err = std::make_shared<dynet::Expression>();

    unsigned int current_epoc = 0;
    while(current_epoc<MAX_EPOCHS){
        
        for(unsigned int i=0; i<)
        
        ++current_epoch;
    }

    for(int i=0; i<100; ++i){

        std::shared_ptr<dynet::ComputationGraph> sp_cg = 
                             std::make_shared<dynet::ComputationGraph>();
        vaeLm.forward(sp_cg, sp_mu, sp_logvar, sp_err, ptb_train_data[i]);
    }
    
    
}
