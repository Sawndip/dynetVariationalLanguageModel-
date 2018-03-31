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
                             unsigned int noise_samples,
                             unsigned int vocab_size)
    : d_sp_model(sp_model)
      , d_layers(layers)
      , d_input_dim(input_dim)
      , d_hidden_dim(hidden_dim)
      , d_hidden2_dim(hidden2_dim)
      , d_latent_dim(latent_dim)
      , d_noise_samples(noise_samples)
      , d_vocab_size(vocab_size)
      , d_source_rnn(layers, input_dim, hidden_dim, *sp_model)
      , d_target_rnn(layers, input_dim, hidden_dim, *sp_model)
{  
    std::cout << "VariationalLm constructor" << std::endl; 
    
    d_p_W_hh2 = d_sp_model->add_parameters({d_hidden2_dim, d_hidden_dim});
    d_p_b_h2 = d_sp_model->add_parameters({d_hidden2_dim});

    d_p_W_h2m = d_sp_model->add_parameters({d_latent_dim, d_hidden2_dim});
    d_p_b_m = d_sp_model->add_parameters({d_latent_dim});

    d_p_W_h2s = d_sp_model->add_parameters({d_latent_dim, d_hidden2_dim});
    d_p_b_s = d_sp_model->add_parameters({d_latent_dim});

    d_p_W_zh0 = d_sp_model->add_parameters({d_hidden_dim * d_layers, 
                                            d_latent_dim});
    d_p_b_h0 = d_sp_model->add_parameters({d_hidden_dim * d_layers});

    d_p_W_hv = d_sp_model->add_parameters({d_vocab_size, d_hidden_dim});
    d_p_b_v = d_sp_model->add_parameters({d_vocab_size});

    d_p_lookup = d_sp_model->add_lookup_parameters(d_vocab_size, 
                                                   {d_input_dim});  
}


void VariationalLm::forward(std::shared_ptr<dynet::ComputationGraph> sp_cg,
                            std::shared_ptr<dynet::Expression> sp_error,
                            const std::vector<int>& sent)
{
    /*
    * Given a sent, the function evaluates ** change this description ** 
    * 1) mean of latent variable
    * 2) logvar of laent variable
    * 3) The reconstructed sentence 
    */

    // encode: x --> {mu, logvar} --> z
    std::shared_ptr<dynet::Expression> sp_mu = std::make_shared<dynet::Expression>();
    std::shared_ptr<dynet::Expression> sp_logvar = std::make_shared<dynet::Expression>();
    std::shared_ptr<dynet::Expression> sp_enc_error = std::make_shared<dynet::Expression>(); // KL
    this->encode(sp_cg, sp_mu, sp_logvar, sp_enc_error, sent);
   
    std::vector<dynet::Expression> dec_errors;
    for(unsigned int i=0; i<d_noise_samples; ++i){
         
        // Reparameterize
        std::shared_ptr<dynet::Expression> sp_z = std::make_shared<dynet::Expression>();
        this->reparameterize(sp_cg, sp_z, sp_mu, sp_logvar);
        
        // decode z --> x
        std::shared_ptr<dynet::Expression> sp_dec_error = std::make_shared<dynet::Expression>();
        this->decode(sp_cg, sp_z, sp_dec_error, sent); 
        dec_errors.push_back(*sp_dec_error);
    } 

    *sp_error = (*sp_enc_error) + dynet::sum(dec_errors) / d_noise_samples;
    
    return; 
}


void VariationalLm::encode(std::shared_ptr<dynet::ComputationGraph> sp_cg,
                           std::shared_ptr<dynet::Expression> sp_mu,
                           std::shared_ptr<dynet::Expression> sp_logvar,
                           std::shared_ptr<dynet::Expression> sp_enc_error,
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
    *sp_mu = dynet::affine_transform({e_b_m, e_W_h2m, e_h2});

    // h2-->s
    dynet::Expression e_W_h2s = dynet::parameter(*sp_cg, d_p_W_h2s);
    dynet::Expression e_b_s = dynet::parameter(*sp_cg, d_p_b_s);
    *sp_logvar = dynet::affine_transform({e_b_s, e_W_h2s, e_h2}); 

    // KL Error: See Doersch's paper
    *sp_enc_error = 0.5 * dynet::sum_elems(dynet::exp(*sp_logvar)
                                           + dynet::square(*sp_mu) 
                                           -1 
                                           - *sp_logvar); 

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

    return;
}

void VariationalLm::decode(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                           std::shared_ptr<dynet::Expression> sp_z,
                           std::shared_ptr<dynet::Expression> sp_dec_error,
                           const std::vector<int>& sent)
{
    d_target_rnn.new_graph(*sp_cg);    

    // z-->h0
    dynet::Expression e_W_zh0 = dynet::parameter(*sp_cg, d_p_W_zh0);
    dynet::Expression e_b_h0 = dynet::parameter(*sp_cg, d_p_b_h0);
    dynet::Expression e_h0 = dynet::affine_transform({e_b_h0, e_W_zh0, *sp_z});

    std::vector<dynet::Expression> h0s;  
    h0s.push_back(e_h0); // multi layers not yet supported  
    d_target_rnn.start_new_sequence(h0s);

    std::vector<dynet::Expression> errors; 
    for(size_t t=0; t<sent.size()-1; ++t){
        // Note the range of t
        // The max value of t = sent.size() - 1 
        // See net_word_id as a reason of this range
        
        int current_word_id = sent[t];
        int next_word_id = sent[t+1];          

        dynet::Expression x_t = dynet::lookup(*sp_cg, d_p_lookup, current_word_id);
        dynet::Expression h_t = d_target_rnn.add_input(x_t);

        // h_t-->v
        dynet::Expression e_W_hv = dynet::parameter(*sp_cg, d_p_W_hv);
        dynet::Expression e_b_v = dynet::parameter(*sp_cg, d_p_b_v);
        dynet::Expression e_v = dynet::affine_transform({e_b_v, e_W_hv, h_t});

        // Beam search not yet supported
        errors.push_back(dynet::pickneglogsoftmax(e_v, next_word_id)); 
    }
    
    *sp_dec_error = dynet::sum(errors);
    
    return;
}


} // VaeLm
