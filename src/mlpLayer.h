#ifndef MLPLAYER_H
#define MLPLAYER_H

#include "dynet/io.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/training.h"

#include <memory>

class MlpLayer{

public:

enum Activation{
    SIGMOID
    , TANH
    , LINEAR
    , RELU
};

// Ctors
explicit MlpLayer(std::shared_ptr<dynet::ParameterCollection> sp_model,
                  unsigned int input_dim, 
                  unsigned int output_dim,
                  Activation activation, 
                  float dorput_rate=0.0);
~MlpLayer(){}

// Accessors
inline unsigned int get_input_dim() const { return d_input_dim; }
inline unsigned int get_output_dim() const { return d_output_dim; }
inline float get_dropout_rate() const { return d_dropout_rate; }

dynet::Expression activation(const dynet::Expression& exp);

// Manipulators
inline void set_input_dim(unsigned int input_dim) { d_input_dim = input_dim; }
inline void set_output_dim(unsigned int output_dim) { d_output_dim = output_dim; }
inline void set_dropout_rate(float dropout_rate) { d_dropout_rate = dropout_rate; }
inline dynet::Parameter& get_pW() { return d_pW; }
inline dynet::Parameter& get_pB() { return d_pB; }

private:
unsigned int d_input_dim;
unsigned int d_output_dim;
float d_dropout_rate;
Activation d_activation;

std::shared_ptr<dynet::ParameterCollection> d_sp_model;
dynet::Parameter d_pW;
dynet::Parameter d_pB;

};

#endif
