//
//  Layers.hpp
//  Deep Learning
//
//  Created by Moss on 9/1/21.
//

#ifndef Layers_hpp
#define Layers_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include "LinearAlgebra.hpp"

using namespace std;

class Layer
{
public:
    std::function<double(double)> activation;
    std::function<double(double)> derived_activation;
    std::function<double(Matrix, Matrix)> activation_loss;
    std::function<Matrix(Matrix, Matrix)> derived_activation_loss;
    std::function<Matrix(Matrix)> activation_softmax;
    std::function<Matrix(Matrix)> derived_activation_softmax;
    vector<Matrix> gradients;
    Layer(string activ);
    string LayerType = "";
    virtual ~Layer();
};

class Dense : public Layer
{
public:
    Dense(int neuron_number, string activ);
    Matrix W = Matrix::ones(1, 1);
    Matrix b = Matrix::ones(1, 1);
    Matrix inputs = Matrix::ones(1, 1);
    int neuron_number;
    bool is_softmax=false;
    Matrix evaluate(Matrix input);
    void rand_init(int prev_neuron_number);
    
    
};

class Loss : Layer
{
public:
    Matrix inputs = Matrix::ones(1, 1);
    Matrix target = Matrix::ones(1, 1);
    Loss(string activ);
};


#endif /* Layers_hpp */
