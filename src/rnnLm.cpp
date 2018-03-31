#include "rnnLm.h"

#include "dynet/io.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/training.h"
#include "dynet/gru.h"
#include "dynet/dict.h"


RnnLm::RnnLm(std::shared_ptr<dynet::ParameterCollection> sp_model,
             unsigned int layers, 
             unsigned int input_dim,
             unsigned int hidden_dim,
             unsigned int vocab_size)
    : d_sp_model(sp_model)
      , d_layers(layers)
      , d_input_dim(input_dim)
      , d_hidden_dim(hidden_dim)
      , d_vocab_size(vocab_size)
      , d_rnn(layers, input_dim, hidden_dim, *sp_model)
{
    if(d_layers!=1){
        std::cout << "multi layer rnn not implemented" << std::endl;
        abort();
    }

    d_p_W_hv = d_sp_model->add_parameters({d_vocab_size, d_hidden_dim});
    d_p_b_v = d_sp_model->add_parameters({d_vocab_size});

    d_p_lookup = d_sp_model->add_lookup_parameters(d_vocab_size, 
                                                   {d_input_dim});
}

void RnnLm::forward(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                    std::shared_ptr<dynet::Expression> sp_error,
                    const std::vector<int>& sent)
{

    d_rnn.new_graph(*sp_cg);
    d_rnn.start_new_sequence();
    std::vector<dynet::Expression> errors; 
    for(size_t t=0; t<sent.size()-1; ++t){
        // Note the range of t
        // The max value of t = sent.size() - 1 
        // See net_word_id as a reason of this range

        int current_word_id = sent[t];
        int next_word_id = sent[t+1];          

        dynet::Expression x_t = dynet::lookup(*sp_cg, d_p_lookup, current_word_id);
        dynet::Expression h_t = d_rnn.add_input(x_t);

        // h_t-->v
        dynet::Expression e_W_hv = dynet::parameter(*sp_cg, d_p_W_hv);
        dynet::Expression e_b_v = dynet::parameter(*sp_cg, d_p_b_v);
        dynet::Expression e_v = dynet::affine_transform({e_b_v, e_W_hv, h_t});
        
        errors.push_back(dynet::pickneglogsoftmax(e_v, next_word_id)); 
    }
    *sp_error = dynet::sum(errors);    
    return;
}
