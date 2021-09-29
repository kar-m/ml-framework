//
//  NeuralNetwork.hpp
//  Deep Learning
//
//  Created by Moss on 9/6/21.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include <queue>
#include "Layers.hpp"
#include "LinearAlgebra.hpp"
using namespace std;

class LayerNode
{
public:
    static LayerNode BuildTree(vector<queue<LayerNode>> tree_seed, shared_ptr<LayerNode> parent, int input_size, int current_depth=0, int current_branch=0, bool isRoot=false);
    vector<LayerNode> child;
    shared_ptr<LayerNode> parent = NULL;
    shared_ptr<Layer> value = NULL;
    int BranchNumber;
    int TreeDepth;
    bool isRoot=false;
    bool isLeaf=false;
    string LayerName;
    static LayerNode create_node(Layer* LayerType, int BranchNumber, int TreeDepth, string LayerName);
    static void print_tree(LayerNode root);
    static LayerNode get_node(LayerNode root, int BranchNumber, int TreeDepth);
    LayerNode();
    
private:
    LayerNode(Layer* LayerType, int BranchNumber, int TreeDepth, string LayerName);
};

class NeuralNetwork
{
private:
    NeuralNetwork(vector<queue<LayerNode>> Layers, int input_size, int batch_size, double learning_rate=0.01, string optimizer="gradient_descent");
    int batch_size;
    double learning_rate;
    string optimizer;
    LayerNode tree_root;
    vector<LayerNode> tree_leafs;
    vector<Matrix> gradients;
    vector<Matrix> flattened_weights;
    static void get_leafs(vector<LayerNode> &knownLeafs, LayerNode node);
    Matrix calculate_gradient_hidden(int BranchNumber, int TreeDepth);
    Matrix calculate_gradient_loss(int BranchNumber);
    Matrix calculate_gradient_input(int BranchNumber);
    Matrix calculate_gradient(int BranchNumber);
    void calculate_gradient_shapes();
    Matrix get_flattened_weights();
    Matrix stochastic_gradient(int BranchNumber, Matrix x, vector<Matrix> y);
    Matrix mini_batch_gradient(int BranchNumber, Matrix X, vector<Matrix> Y);
    void update_inputs(Matrix x, vector<Matrix> y, LayerNode child);
    void update_weights(Matrix flat_change, int BranchNumber);
    void gradient_descent(int batch_size, double learning_rate, Matrix X, vector<Matrix> Y);
    
public:    
    static NeuralNetwork createModel(vector<queue<LayerNode>> Layers, int input_size, int batch_size, double learning_rate=0.01, string optimizer="gradient_descent");
    void fit(Matrix X, vector<Matrix> Y);
    Matrix predict(Matrix X, int BranchNumber=0);
    void export_model();
};



#endif /* NeuralNetwork_hpp */
