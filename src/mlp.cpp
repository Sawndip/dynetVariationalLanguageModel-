#include "mlp.h"

#include "dynet/io.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/training.h"

#include <vector>
#include <stdexcept>
#include <memory>
#include <cassert>

Mlp::Mlp(std::shared_ptr<dynet::ParameterCollection> sp_model, 
         std::shared_ptr<std::vector<MlpLayer> > sp_layers)
    : d_sp_model(sp_model) 
      , d_sp_layers(sp_layers)
{
    if(!this->is_valid_seq_of_layers(*d_sp_layers)){
        throw(std::invalid_argument("Layer dimentions are not correct."));
    }
}

dynet::Expression Mlp::forward(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                               const dynet::Expression& x)
{
    dynet::ComputationGraph& cg = *sp_cg;
    dynet::Expression current_exp = x;
    std::vector<MlpLayer>& layers = *d_sp_layers;    
 
    for(size_t l_id=0; l_id<layers.size(); ++l_id){
        
        dynet::Parameter& pW = layers[l_id].get_pW();    
        dynet::Parameter& pB = layers[l_id].get_pB();    
                 
        dynet::Expression W = dynet::parameter(cg, pW); 
        dynet::Expression b = dynet::parameter(cg, pB); 
        
        current_exp = layers[l_id].activation(W*current_exp + b);
    }
    
    return current_exp;
}

dynet::Expression Mlp::get_neg_log_like(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                                        const dynet::Expression& x,
                                        const std::vector<unsigned int>& labels)
{
    dynet::Expression y_pred = this->forward(sp_cg, x);
    return dynet::sum_batches(dynet::pickneglogsoftmax(y_pred, labels)); // This is assuming that the label is in range 0-9
}


dynet::Expression Mlp::get_neg_log_like(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                                        const dynet::Expression& x,
                                        const unsigned int& label)
{
    dynet::Expression y_pred = this->forward(sp_cg, x);
    return dynet::pickneglogsoftmax(y_pred, label); // This is assuming that the label is in range 0-9
}


unsigned int Mlp::predict(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                          const dynet::Expression& x)
{
    dynet::Expression y_pred = this->forward(sp_cg, x);
    
    //std::vector<float> probs = 
          //dynet::as_vector(sp_cg->forward(dynet::softmax(y_pred)));
    
    std::vector<float> scores = dynet::as_vector(sp_cg->forward(y_pred));
    unsigned int argmax = 0;
    for(size_t i=0; i<scores.size(); ++i){
        argmax = scores[i] > argmax ? i : argmax;
    }
    return argmax;
}


bool Mlp::is_valid_seq_of_layers() const
{
    return this->is_valid_seq_of_layers(*d_sp_layers);
}

bool Mlp::is_valid_seq_of_layers(const std::vector<MlpLayer>& layers) const
{
    /* 
    Checks if the output_dim of current layer 
    is the same as the input_dim of the next layer. 
    If they do not match, the function returns false.
    */
    for(size_t l_id=0; l_id<layers.size() - 1; ++l_id){
        size_t current_layer_idx = l_id;
        size_t next_layer_idx = current_layer_idx + 1;
        
        size_t next_layer_input_dim = layers[next_layer_idx].get_input_dim();
        size_t current_layer_output_dim = layers[current_layer_idx].get_output_dim();   
        
        if(next_layer_input_dim != current_layer_output_dim){
            return false; 
        }
        
        ++current_layer_idx;
    }
    return true;
}
