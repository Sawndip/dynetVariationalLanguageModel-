#include "variationalLm.h"

#include "dynet/io.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/training.h"
#include "dynet/gru.h"
#include "dynet/dict.h"

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
      , d_source_rnn(layers, input_dim, hidden_dim, *d_sp_model)
      , d_target_rnn(layers, input_dim, hidden_dim, *d_sp_model)
{    
    d_p_W_hh2 = d_sp_model->add_parameters({d_hidden2_dim, d_hidden_dim});
    d_p_b_h2 = d_sp_model->add_parameters({d_hidden2_dim});

    d_p_W_h2m = d_sp_model->add_parameters({d_latent_dim, d_hidden2_dim});
    d_p_b_m = d_sp_model->add_parameters({d_latent_dim});

    d_p_W_h2s = d_sp_model->add_parameters({d_latent_dim, d_hidden2_dim});
    d_p_b_s = d_sp_model->add_parameters({d_latent_dim});

    d_p_W_zh0 = d_sp_model->add_parameters({d_hidden_dim, d_latent_dim});
    d_p_b_h0 = d_sp_model->add_parameters({d_hidden_dim});

    d_p_W_hv = d_sp_model->add_parameters({d_vocab_size, d_hidden_dim});
    d_p_b_v = d_sp_model->add_parameters({d_vocab_size});

    d_p_lookup = d_sp_model->add_lookup_parameters(d_vocab_size, 
                                                   {d_input_dim});  
}


void VariationalLm::forward(std::shared_ptr<dynet::ComputationGraph> sp_cg,
                            std::shared_ptr<dynet::Expression> sp_mu,
                            std::shared_ptr<dynet::Expression> sp_logvar,
                            std::shared_ptr<dynet::Expression> sp_x_recon,
                            const std::vector<int>& sent)
{
    /*
    * Given a sent, the function evaluates 
    * 1) mean of latent variable
    * 2) logvar of laent variable
    * 3) The reconstructed sentence 
    */

    // encode: x --> {mu, logvar} --> z
    this->encode(sp_cg, sp_mu, sp_logvar, sent);
    
    std::shared_ptr<dynet::Expression> sp_z = std::make_shared<dynet::Expression>();
    this->reparameterize(sp_cg, sp_z, sp_mu, sp_logvar);
    
    // decode z --> x
    this->decode(sp_cg, sp_z);
    return; 
}


void VariationalLm::encode(std::shared_ptr<dynet::ComputationGraph> sp_cg,
                           std::shared_ptr<dynet::Expression> sp_mu,
                           std::shared_ptr<dynet::Expression> sp_logvar,
                           const std::vector<int>& sent)
{
    /*
     * Evaluates the mu and lagvar of the latent variable
    */

    d_source_rnn.new_graph(*sp_cg);
    d_source_rnn.start_new_sequence();
    for(size_t i=0; i<sent.size(); ++i){
        dynet::Expression word_exp = dynet::lookup(*sp_cg, d_p_lookup, sent[i]);
        d_source_rnn.add_input(word_exp);
    }

    // h-->h2
    dynet::Expression e_W_hh2 = dynet::parameter(*sp_cg, d_p_W_hh2);
    dynet::Expression e_b_h2 = dynet::parameter(*sp_cg, d_p_b_h2);
    dynet::Expression e_h2 = dynet::tanh(e_W_hh2 * d_source_rnn.back() + e_b_h2);

    // h2-->m
    dynet::Expression e_W_h2m = dynet::parameter(*sp_cg, d_p_W_h2m);
    dynet::Expression e_b_m = dynet::parameter(*sp_cg, d_p_b_m);
    *sp_mu = e_W_h2m * e_h2 + e_b_m;

    // h2-->s
    dynet::Expression e_W_h2s = dynet::parameter(*sp_cg, d_p_W_h2s);
    dynet::Expression e_b_s = dynet::parameter(*sp_cg, d_p_b_s);
    *sp_logvar = e_W_h2s * e_h2 + e_b_s;    

    return;

}

void VariationalLm::reparameterize(std::shared_ptr<dynet::ComputationGraph> sp_cg,
                                   std::shared_ptr<dynet::Expression> sp_z,
                                   std::shared_ptr<dynet::Expression> sp_mu,
                                   std::shared_ptr<dynet::Expression> sp_logvar) 
{
    // reparameterization will be different in training and testing
    dynet::Expression std = dynet::exp((*sp_logvar) * 0.5);
    dynet::Expression eps = dynet::random_normal(*sp_cg, std.dim());
    *sp_z = dynet::cmult(std, eps) + (*sp_mu);
}

void VariationalLm::decode(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                           std::shared_ptr<dynet::Expression> sp_z,
                           const std::vector<int>& sent)
{
    d_target_rnn.new_graph(*sp_cg);    

    // z-->h0
    dynet::Expression e_W_zh0 = dynet::parameter(*sp_cg, d_p_W_zh0);
    dynet::Expression e_b_h0 = dynet::parameter(*sp_cg, d_p_b_h0);
    dynet::Expression e_h0 = e_W_zh0 * (*sp_z) + e_b_h0;

    // Layers == 1. multi layers not yet supported 
    std::vector<dynet::Expression> h0s(d_layers);   
    d_target_rnn.start_new_sequence(h0s);

     
}


} // VaeLm
