#ifndef MLP_H 
#define MLP_H

#include "mlpLayer.h"

#include "dynet/io.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/training.h"

#include <vector>
//#include <unordered_map>
#include <string>
#include <memory>

class Mlp{

public:
explicit Mlp(std::shared_ptr<dynet::ParameterCollection> sp_model, 
             std::shared_ptr<std::vector<MlpLayer> > sp_layers);
~Mlp(){}

dynet::Expression forward(std::shared_ptr<dynet::ComputationGraph> sp_cg,
                          const dynet::Expression& x);


dynet::Expression get_neg_log_like(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                                   const dynet::Expression& x,
                                   const std::vector<unsigned int>& labels);


dynet::Expression get_neg_log_like(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                                   const dynet::Expression& x,
                                   const unsigned int& labels);

unsigned int predict(std::shared_ptr<dynet::ComputationGraph> sp_cg, 
                     const dynet::Expression& x);


private:

// Validation functions
bool is_valid_seq_of_layers() const;
bool is_valid_seq_of_layers(const std::vector<MlpLayer>& layers) const;

// Data members
// Mlp itself should not have the sole ownership of model
std::shared_ptr<dynet::ParameterCollection> d_sp_model; 
std::shared_ptr<std::vector<MlpLayer> > d_sp_layers;
}; 

#endif
