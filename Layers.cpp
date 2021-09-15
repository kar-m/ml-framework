//
//  Layers.cpp
//  Deep Learning
//
//  Created by Moss on 9/1/21.
//

#include "Layers.hpp"
#include <vector>
#include <iostream>
#include <math.h>
#include "LinearAlgebra.hpp"
#include <random>
using namespace std;

double ReLu(double input)   {return (abs(input)+input)/2;}
double ReLu_Derived(double input)   {return 0.5+abs(input)/(2*input);}

double Sigmoid(double input)    {return 1/(1+exp(-1*input));}
double Sigmoid_Derived(double input)    {return Sigmoid(input)*(1-Sigmoid(input));}

double Tanh(double input)   {return tanh(input);}
double Tanh_Derived(double input)   {return 1/(cosh(input)*cosh(input));}

Matrix Softmax(Matrix input)
{
    /* input: a matrix containing the features which need to be normalized along the 0th axis. */
    vector<vector<double>> ans_values;
    for(int i = 0; i < input.shape[1]; i++)
    {
        Matrix temp = input.loc(":", i);
        double exp_current_sum = 0;
        for(int j = 0; j < input.shape[0]; j++)
        {
            exp_current_sum += exp(temp.values[j][0]);
        }
        vector<double> ans_temp;
        for(int j = 0; j < input.shape[0]; j++)
        {
            double x = exp(temp.values[j][0])/exp_current_sum;
            ans_temp.push_back(x);
        }
        ans_values.push_back(ans_temp);
    }
    Matrix ans = Matrix::create_matrix(ans_values, input.shape[1], input.shape[0]);
    ans = ans.T();
    return ans;
}
Matrix Softmax_Derived(Matrix input)
{
    /* input: a vector containing the values which are going into softmax */
    Matrix temp = Softmax(input);
    vector<vector<double>> ans_values;
    for(int i = 0; i < input.shape[0]; i++)
    {
        vector<double> ans_temp;
        for(int j = 0; j < input.shape[0]; j++)
        {
            if(i==j)
                ans_temp.push_back(temp.values[i][0]*(1-temp.values[j][0]));
            else
                ans_temp.push_back(-1*temp.values[i][0]*temp.values[j][0]);
        }
    }
    Matrix ans = Matrix::create_matrix(ans_values, input.shape[0], input.shape[0]);
    return ans;
}

double MSE(Matrix input, Matrix target)
{
    Matrix diff = input - target;
    diff = diff*diff;
    return diff.sum();
}

Matrix MSE_Derived(Matrix input, Matrix target)
{
    Matrix diff = input-target;
    diff = diff+diff;
    diff = diff.T();
    return diff;
}


Layer::Layer(string activ)
{
    string sl = activ;
    transform(sl.begin(), sl.end(), sl.begin(), ::tolower);
    
    if(sl == "relu"){
        this->activation = ReLu;
        this->derived_activation = ReLu_Derived;
    }
    else if(sl == "sigmoid"){
        this->activation = Sigmoid;
        this->derived_activation = Sigmoid_Derived;
    }
    else if(sl == "tanh"){
        this->activation = Tanh;
        this->derived_activation = Tanh_Derived;
    }
    else if(sl=="loss_mse")
    {
        this->activation_loss = MSE;
        this->derived_activation_loss = MSE_Derived;
    }
    else if(sl=="softmax")
    {
        this->activation_softmax = Softmax;
        this->derived_activation_softmax = Softmax_Derived;
    }
    else
        throw std::invalid_argument("No such actiavtion function found!");
}

Layer::~Layer(){}

Dense::Dense(int neuron_number, string activ) : Layer(activ)
{
    string sl = activ;
    transform(sl.begin(), sl.end(), sl.begin(), ::tolower);
    vector<vector<double>> random = {{0, 1}, {2, 3}};
    this->neuron_number = neuron_number;
    LayerType = "Dense";
    if(sl=="softmax")
        this->is_softmax=true;
}

Matrix Dense::evaluate(Matrix input)
{
    Matrix ans = this->W^input;
    ans = ans + this->b;
    
    for(int i = 0; i < ans.shape[0]; i++)
        for(int j = 0; j < ans.shape[1]; j++)
            ans.values[i][j] = this->activation(ans.values[i][j]);
    
    return ans;
}

void Dense::rand_init(int prev_neuron_number)
{
    this->b = Matrix::zeros(neuron_number, 1);
    
    vector<vector<double>> values;
    for(int i = 0; i < neuron_number; i++)
    {
        vector<double> temp;
        for(int j = 0; j < prev_neuron_number; j++)
        {
            default_random_engine generator;
            normal_distribution<double> distribution(0,(1/sqrt(prev_neuron_number)));
            double number = distribution(generator);
            temp.push_back(number);
        }
        values.push_back(temp);
    }
    this->W = Matrix::create_matrix(values, neuron_number, prev_neuron_number);
    
}


Loss::Loss(string activ) : Layer(activ)
{
    this->LayerType = "Loss";
}
