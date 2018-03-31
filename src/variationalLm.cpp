#include "variationalLm.h"
#include "dynet/gru.h"

#include <iostream>
#include <cassert>

namespace VaeLm{

VariationalLm::VariationalLm(std::shared_ptr<dynet::ParameterCollection> sp_model,
                             unsigned int layers, 
                             unsigned int input_dim,
                             unsigned int hidden_dim,
                             unsigned int hidden2_dim,
                             unsigned int latent_dim,
                             unsigned int vocab_size)
    : d_sp_model(sp_model)
      , d_layers(layers)
      , d_input_dim(input_dim)
      , d_hidden_dim(hidden_dim)
      , d_hidden2_dim(hidden2_dim)
      , d_latent_dim(latent_dim)
      , d_vocab_size(vocab_size)
{
    std::cout << "blah" << std::endl;
    if(d_sp_model==NULL){
        std::cout << "d_sp_model cannot be null. Aborting." << std::endl;
    } 
    assert(d_sp_model!=NULL);
    
    d_p_W_hh2 = d_sp_model->add_parameters({d_hidden2_dim, d_hidden_dim});
    d_p_b_h2 = d_sp_model->add_parameters({d_hidden2_dim});

    d_p_W_h2m = d_sp_model->add_parameters({d_latent_dim, d_hidden2_dim});
    d_p_b_m = d_sp_model->add_parameters({d_latent_dim});

    d_p_W_h2s = d_sp_model->add_parameters({d_latent_dim, d_hidden2_dim});
    d_p_b_s = d_sp_model->add_parameters({d_latent_dim});

    d_p_W_hv = d_sp_model->add_parameters({d_vocab_size, d_hidden_dim});
    d_p_b_v = d_sp_model->add_parameters({d_vocab_size});

    d_p_lookup = d_sp_model->add_lookup_parameters(d_vocab_size, {d_input_dim});  
}

dynet::Expression VariationalLm::forward(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                                         const std::vector<int>& sent)
{
   return dynet::parameter(*sp_cg, d_p_b_v); 
}



} // VaeLm
