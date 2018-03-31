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
const std::string UNK          = "<unk>"; // as defined in ptb train file
const unsigned int LAYERS      = 1;
const unsigned int IMPUT_DIM   = 64;
const unsigned int HIDDEN_DIM  = 128;
const unsigned int HIDDEN2_DIM = 32;
const unsigned int LATENT_DIM  = 10;


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
                               vocab_size);

    std::shared_ptr<dynet::ComputationGraph> sp_cg = 
                             std::make_shared<dynet::ComputationGraph>();
    std::shared_ptr<dynet::Expression> sp_mu = std::make_shared<dynet::Expression>();
    std::shared_ptr<dynet::Expression> sp_logvar = std::make_shared<dynet::Expression>();
    std::shared_ptr<dynet::Expression> sp_x_recon = std::make_shared<dynet::Expression>();
    vaeLm.forward(sp_cg, sp_mu, sp_logvar, sp_x_recon, ptb_train_data[0]);
 
    
    
     

    
    
    /*
    // Create layers
    std::shared_ptr<std::vector<MlpLayer> > sp_layers = 
                          std::make_shared<std::vector<MlpLayer> >();  
    sp_layers->push_back(MlpLayer(sp_model, 
                                  784, 512, 
                                  MlpLayer::Activation::RELU));
    sp_layers->push_back(MlpLayer(sp_model, 
                                  512, 512, 
                                  MlpLayer::Activation::RELU)); 
    sp_layers->push_back(MlpLayer(sp_model, 
                                  512, 10, 
                                  MlpLayer::Activation::LINEAR)); 
    
    // Create mlp
    Mlp mlp(sp_model, sp_layers);
    
    double loss = 0.0;
    for(size_t i=0; i<mnist_train.size(); ++i){
        std::shared_ptr<dynet::ComputationGraph> sp_cg = 
                             std::make_shared<dynet::ComputationGraph>(); 
        dynet::Expression x_input = dynet::input(*sp_cg, 
                                                 {784}, 
                                                 mnist_train[i]);
        dynet::Expression loss_expr = mlp.get_neg_log_like(sp_cg, 
                                                           x_input, 
                                                           mnist_train_labels[i]);
        loss += dynet::as_scalar(sp_cg->forward(loss_expr));
        sp_cg->backward(loss_expr);
        trainer.update();  
        
        if(i && i%100==0){
            std::cout <<"i = " << i 
                      << " loss / i = " << loss / i 
                      << std::endl;
        }  
    }

    unsigned int num_correct_pred = 0;
    for(size_t i=0; i<mnist_train.size(); ++i){
        
        std::shared_ptr<dynet::ComputationGraph> sp_cg = 
                             std::make_shared<dynet::ComputationGraph>(); 
        dynet::Expression x_input = dynet::input(*sp_cg, 
                                                 {784}, 
                                                 mnist_train[i]);
        if(mlp.predict(sp_cg, x_input) == mnist_train_labels[i]){
            ++num_correct_pred;
        }
        if(i && i%100==0){
            std::cout <<"i = " << i 
                      << " accuracy / i = " << num_correct_pred * 1.0 / i 
                      << std::endl;
        }  
    }
    */
}
