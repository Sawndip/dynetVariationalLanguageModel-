#include "variationalLm.h"
#include "ptbReader.h"

#include "dynet/io.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/training.h"
#include "dynet/gru.h"
#include "dynet/dict.h"

#include <iostream>
#include <cassert>
#include <algorithm>

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
      , d_source_rnn(layers, input_dim, hidden_dim, *sp_model)
      , d_target_rnn(layers, input_dim, hidden_dim, *sp_model)
{  
    if(d_layers!=1){
        std::cout << "multi layer rnn not implemented" << std::endl;
        abort();
    } 
    
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
                            std::shared_ptr<dynet::Expression> sp_enc_error,
                            std::shared_ptr<dynet::Expression> sp_dec_error,
                            const std::vector<int>& sent)
{
    /*
    * Has two part: 
    * 1) encode: encodes x to z
    * 2) decode: decodes z to x
    * Evaluates the error in encode and decode steps  
    */

    // encode: x --> {mu, logvar} --> z
    std::shared_ptr<dynet::Expression> sp_mu = std::make_shared<dynet::Expression>();
    std::shared_ptr<dynet::Expression> sp_logvar = std::make_shared<dynet::Expression>();
    this->encode(sp_cg, sp_mu, sp_logvar, sp_enc_error, sent);
    
    // Reparameterize
    std::shared_ptr<dynet::Expression> sp_z = std::make_shared<dynet::Expression>();
    this->reparameterize(sp_cg, sp_z, sp_mu, sp_logvar);
        
    // decode z --> x
    this->decode(sp_cg, sp_z, sp_dec_error, sent); 

    return; 
}


void VariationalLm::forward(std::shared_ptr<dynet::ComputationGraph> sp_cg,
                            std::shared_ptr<dynet::Expression> sp_error,
                            const std::vector<int>& sent)
{
    std::shared_ptr<dynet::Expression> sp_enc_error = std::make_shared<dynet::Expression>();
    std::shared_ptr<dynet::Expression> sp_dec_error = std::make_shared<dynet::Expression>();
    this->forward(sp_cg, sp_enc_error, sp_dec_error, sent);

    // Error = sum of encode/decode errors
    *sp_error = (*sp_enc_error) + (*sp_dec_error);
    
}


void VariationalLm::encode(std::shared_ptr<dynet::ComputationGraph> sp_cg,
                           std::shared_ptr<dynet::Expression> sp_mu,
                           std::shared_ptr<dynet::Expression> sp_logvar,
                           std::shared_ptr<dynet::Expression> sp_enc_error,
                           const std::vector<int>& sent)
{
    /*
     * Evaluates the mu and logvar of the latent variable
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
    *sp_enc_error = 0.5 * dynet::sum_elems(dynet::exp(*sp_logvar) + dynet::square(*sp_mu) -1 - *sp_logvar); 
  
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
        // The max value of t = sent.size() - 2 
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

void VariationalLm::train(std::vector<std::vector<int> >* pt_train_data,
                          std::vector<std::vector<int> >* pt_valid_data,
                          const unsigned int& max_epochs,
                          const unsigned int& batch_size)
{ 
 
    // Not doing explicit batching
    // For explicit batching, look at code by duyvuleo/Transformer-DyNet
 
    // Prepare train data for batching
    std::vector<std::vector<int> >& train_data = *pt_train_data;
    PtbReader::sort_data_in_ascending_length(&train_data);
    std::vector<PtbReader::BATCH_INDEX_t> batchIndexListTrain;
    PtbReader::create_batches(&batchIndexListTrain, train_data, batch_size);

    // Prepare valid data for batching
    std::vector<std::vector<int> >& valid_data = *pt_valid_data;
    PtbReader::sort_data_in_ascending_length(&valid_data);
    std::vector<PtbReader::BATCH_INDEX_t> batchIndexListValid;
    PtbReader::create_batches(&batchIndexListValid, valid_data, batch_size);
 
    dynet::AdamTrainer trainer(*d_sp_model);
    std::shared_ptr<dynet::Expression> sp_enc_err = std::make_shared<dynet::Expression>();
    std::shared_ptr<dynet::Expression> sp_dec_err = std::make_shared<dynet::Expression>();
    for(unsigned int current_epoch=0; current_epoch<max_epochs; ++current_epoch){
        unsigned int train_words = 0;
        double train_loss = 0.0;
        double dec_loss = 0.0;
        unsigned int train_samples = 0;
        std::random_shuffle(batchIndexListTrain.begin(), batchIndexListTrain.end());
        for(unsigned int batch_id=0; batch_id<batchIndexListTrain.size();++batch_id){
            
            std::shared_ptr<dynet::ComputationGraph> sp_cg = 
                                 std::make_shared<dynet::ComputationGraph>();
            std::vector<dynet::Expression> tot_losses_expression;
            std::vector<dynet::Expression> dec_losses_expression;
            const PtbReader::BATCH_INDEX_t& batchIndex =  batchIndexListTrain[batch_id];
            for(unsigned int sent_idx=batchIndex.batch_begin_idx; 
                    sent_idx<batchIndex.batch_begin_idx + batchIndex.batch_num_elements; 
                    ++sent_idx){
                this->forward(sp_cg, sp_enc_err, sp_dec_err, train_data[sent_idx]);
                tot_losses_expression.push_back((*sp_enc_err) + (*sp_dec_err));
                dec_losses_expression.push_back(*sp_dec_err);
                train_words += train_data[sent_idx].size();
            }      
            
            // Calculate the loss and update trainer
            dynet::Expression tot_loss_expression = dynet::sum(tot_losses_expression);
            train_loss += dynet::as_scalar(sp_cg->forward(tot_loss_expression));
            sp_cg->backward(tot_loss_expression);
            trainer.update();
           
            // Calculate the dec loss
            dynet::Expression dec_loss_expression = dynet::sum(dec_losses_expression);
            //dec_loss += dynet::as_scalar(dec_loss_expression);
            dec_loss += dynet::as_scalar(dec_loss_expression.value());
   
            train_samples += batchIndexListTrain[batch_id].batch_num_elements;

            std::cout << " Total E = " << (train_loss / train_words )
                      << " Decoder E = " << (dec_loss / train_words) 
                      << " ppl = " << std::exp(dec_loss / train_words) 
                      << " lines = " << train_samples
                      << " current_epoch = " << current_epoch
                      << std::endl;

        } // batch_id
    } // current_epoch
} // train
