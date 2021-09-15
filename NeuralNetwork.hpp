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
    
private:
    LayerNode(Layer* LayerType, int BranchNumber, int TreeDepth, string LayerName);
};

class NeuralNetwork
{
private:
    NeuralNetwork(vector<queue<LayerNode>> Layers, int batch_size, double learning_rate=0.01, string optimizer="gradient_descent");
    LayerNode tree_root;
    
public:    
    NeuralNetwork createModel(vector<queue<LayerNode>> Layers, int input_size, int batch_size, double learning_rate=0.01, string optimizer="gradient_descent", string loss="cross_entropy");
    void fit(Matrix x, Matrix y);
    void predict(Matrix x);
    void export_model();
};



#endif /* NeuralNetwork_hpp */
