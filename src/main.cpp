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
#include <math.h>
#include <cassert> 

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
    std::cout << "dict.size() = " << dict.size() << std::endl;
    std::cout << "ptb_train_data.size() = " << ptb_train_data.size() << std::endl;
   
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

    std::shared_ptr<dynet::Expression> sp_err = std::make_shared<dynet::Expression>();
    unsigned int current_epoch = 0;
    unsigned int report_every = 50; // log every report_every sentences 
    unsigned int lines = 0; // lines seen in current epoch
    while(current_epoch<MAX_EPOCHS){
        
        unsigned int num_words = 0;
        double loss = 0.0; 
        for(unsigned int i=0; 
                i<report_every && lines<ptb_train_data.size();
                ++i){

            std::shared_ptr<dynet::ComputationGraph> sp_cg = 
                             std::make_shared<dynet::ComputationGraph>();
            vaeLm.forward(sp_cg, sp_err, ptb_train_data[lines]);
            ++lines;
            num_words += ptb_train_data[i].size();
            loss += dynet::as_scalar(sp_cg->forward(*sp_err));
            sp_cg->backward(*sp_err);
            trainer.update();
        }
        
        trainer.status();
         
        assert(num_words != 0); 
        
        if(isnan(loss/num_words)){
            std::cout << "loss = " << loss
                      << " num_words = " << num_words
                      << std::endl;
        }
        std::cout << " E = " << (loss / num_words ) 
                  << " ppl = " << std::exp(loss / num_words) 
                  << std::endl;

        if(lines == ptb_train_data.size()){
            lines = 0;  
            ++current_epoch; 
        }
    }
    
}
