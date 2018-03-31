#ifndef VARIATIONALLM_H
#define VARIATIONALLM_H

#include "dynet/io.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/training.h"
#include "dynet/gru.h"
#include "dynet/dict.h"

namespace VaeLm{

class VariationalLm{

public:

explicit VariationalLm(std::shared_ptr<dynet::ParameterCollection> sp_model,
                       unsigned int layers, 
                       unsigned int input_dim,
                       unsigned int hidden_dim,
                       unsigned int hidden2_dim,
                       unsigned int latent_dim,
                       unsigned int vocab_size);

~VariationalLm(){}

void forward(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
             std::shared_ptr<dynet::Expression> sp_mu,
             std::shared_ptr<dynet::Expression> sp_logvar,
             std::shared_ptr<dynet::Expression> sp_x_recon,
             const std::vector<int>& sent);

void encode(std::shared_ptr<dynet::ComputationGraph> sp_cg,
            std::shared_ptr<dynet::Expression> sp_mu,
            std::shared_ptr<dynet::Expression> sp_logvar,
            const std::vector<int>& sent);

void decode(std::shared_ptr<dynet::ComputationGraph> sp_cg,
            std::shared_ptr<dynet::Expression> sp_z);

void reparameterize(std::shared_ptr<dynet::ComputationGraph> sp_cg,
                    std::shared_ptr<dynet::Expression> sp_z,
                    std::shared_ptr<dynet::Expression> sp_mu,
                    std::shared_ptr<dynet::Expression> sp_logvar);



private:

// dynet model
std::shared_ptr<dynet::ParameterCollection> d_sp_model;

// dimentions
unsigned int d_layers;      // rnn layers
unsigned int d_input_dim;   // rnn input dim
unsigned int d_hidden_dim;  // rnn hidden state dim
unsigned int d_hidden2_dim; // layer between rnn output and latent var  
unsigned int d_latent_dim;  // latent var dim 
unsigned int d_vocab_size;

// model parameters
dynet::Parameter d_p_W_hh2; // matrix h --> h2
dynet::Parameter d_p_b_h2;  // bias h2

dynet::Parameter d_p_W_h2m; // matrix h2 --> mean
dynet::Parameter d_p_b_m;   // bias mean

dynet::Parameter d_p_W_h2s; // matrix h2 --> st. dev.
dynet::Parameter d_p_b_s;   // bias st. dev.

dynet::Parameter d_p_W_zh0;  // matrix z --> h0
dynet::Parameter d_p_b_h0;  // matrix z --> h0


dynet::Parameter d_p_W_hv; // matrix h --> vocab size
dynet::Parameter d_p_b_v;   // bias vocab size

// rnn builders
dynet::GRUBuilder d_source_rnn; // encodes the sent 
dynet::GRUBuilder d_target_rnn; // decodes the sent

dynet::LookupParameter d_p_lookup; // vocab embed   

};

}

#endif
