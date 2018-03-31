#include "mlpLayer.h"

#include <iostream>
#include <cassert>

MlpLayer::MlpLayer(std::shared_ptr<dynet::ParameterCollection> sp_model,
                   unsigned int input_dim, 
                   unsigned int output_dim, 
                   Activation activation,                  
                   float dropout_rate)
    : d_input_dim(input_dim)
      , d_output_dim(output_dim)
      , d_dropout_rate(dropout_rate)
      , d_activation(activation)
      , d_sp_model(sp_model)
{
    if(d_sp_model==NULL){
        assert(d_sp_model!=NULL);
    }
    
    d_pW = d_sp_model->add_parameters({d_output_dim, d_input_dim});
    d_pB = d_sp_model->add_parameters({d_output_dim});
}

dynet::Expression MlpLayer::activation(const dynet::Expression& exp)
{
    switch(d_activation){
        case SIGMOID: return dynet::logistic(exp);
        case TANH: return dynet::tanh(exp);
        case LINEAR: return exp;
        case RELU: return dynet::rectify(exp);
        default:
            std::cout << "ERROR: Input activation not supported returning linear" << std::endl; 
            return exp;  
    }
    
    return exp;
}

