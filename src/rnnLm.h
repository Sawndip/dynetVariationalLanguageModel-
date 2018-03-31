#ifndef RNNLM_H
#define RNNLM_H

#include "dynet/io.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/training.h"
#include "dynet/gru.h"
#include "dynet/dict.h"

class RnnLm{

public:

explicit RnnLm(std::shared_ptr<dynet::ParameterCollection> sp_model,
               unsigned int layers, 
               unsigned int input_dim,
               unsigned int hidden_dim,
               unsigned int vocab_size);

~RnnLm(){}

void forward(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
             std::shared_ptr<dynet::Expression> sp_error,
             const std::vector<int>& sent);

private:

// dynet model
std::shared_ptr<dynet::ParameterCollection> d_sp_model;

// dimentions
unsigned int d_layers;      // rnn layers
unsigned int d_input_dim;   // rnn input dim
unsigned int d_hidden_dim;  // rnn hidden state dim
unsigned int d_vocab_size;

// model parameters
dynet::Parameter d_p_W_hv; // matrix h --> vocab size
dynet::Parameter d_p_b_v;   // bias vocab size
dynet::GRUBuilder d_rnn;  
dynet::LookupParameter d_p_lookup; // vocab embed   
};


#endif

