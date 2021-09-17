//
//  NeuralNetwork.cpp
//  Deep Learning
//
//  Created by Moss on 9/6/21.
//

#include "NeuralNetwork.hpp"
#include <vector>
#include <string>
#include <queue>
#include <iostream>
#include "Layers.hpp"
#include "LinearAlgebra.hpp"
using namespace std;

LayerNode::LayerNode()
{}

LayerNode::LayerNode(Layer* LayerType, int BranchNumber, int TreeDepth, string LayerName)
{
    this->value = std::shared_ptr<Layer>(LayerType);
    this->BranchNumber = BranchNumber;
    this->TreeDepth = TreeDepth;
    string sl = LayerName;
    transform(sl.begin(), sl.end(), sl.begin(), ::tolower);
    this->LayerName = sl;
}

LayerNode LayerNode::create_node(Layer* LayerType, int BranchNumber, int TreeDepth, string LayerName)
{
    unique_ptr<LayerNode> ans = unique_ptr<LayerNode>(new LayerNode(LayerType, BranchNumber, TreeDepth, LayerName));
    LayerNode ret = *ans;
    return ret;
}


LayerNode LayerNode::BuildTree(vector<queue<LayerNode>> tree_seed, shared_ptr<LayerNode> parent, int input_size, int current_depth, int current_branch, bool isRoot)
{
    LayerNode this_node = tree_seed[current_branch].front();
    this_node.parent = parent;
    this_node.isRoot = isRoot;
    
    tree_seed[current_branch].pop();
    
    vector<LayerNode> current_child;
    
    for(int i = 0; i < tree_seed.size(); i++)
    {
        if(!tree_seed[i].empty()){
            LayerNode next_node = tree_seed[i].front();
            if(next_node.TreeDepth==(current_depth+1))
            {
                if((!tree_seed[0].empty() && this_node.BranchNumber==0) || next_node.BranchNumber==current_branch)
                    current_child.push_back(next_node);
            }
        }
    }
    
    if(current_child.size()==0)
        this_node.isLeaf = true;
    
    for(int i = 0; i < current_child.size(); i++)
    {
        if(this_node.LayerName == "dense")
        {
            shared_ptr<Dense> value_spec = dynamic_pointer_cast<Dense>(this_node.value);
            value_spec->rand_init(input_size);
            current_child[i] = BuildTree(tree_seed, make_shared<LayerNode>(this_node), value_spec->neuron_number, current_depth+1, current_child[i].BranchNumber, false);
        }
        
    }
    this_node.child = current_child;
    
    cout << "The node of branch " << current_branch << " at depth " << current_depth << " has " << current_child.size() << " children" << endl;
    
    return this_node;
}

void LayerNode::print_tree(LayerNode root)
{
    cout << root.LayerName << endl << "|" << endl;
    for(int i = 0; i < root.child.size(); i++)
    {
        for(int j = 0; j < root.child[i].BranchNumber; j++)
            cout << " ";
        print_tree(root.child[i]);
    }
}

void NeuralNetwork::get_leafs(vector<LayerNode> &knownLeafs, LayerNode node)
{
    vector<LayerNode> children = node.child;
    
    if(node.isLeaf)
        knownLeafs.push_back(node);
    
    for(int i = 0; i < children.size(); i++)
    {
        get_leafs(knownLeafs, node.child[i]);
    }
}

NeuralNetwork::NeuralNetwork(vector<queue<LayerNode>> Layers, int input_size, int batch_size, double learning_rate, string optimizer)
{
    //...
    this->tree_root = LayerNode::BuildTree(Layers, NULL, input_size);
    NeuralNetwork::get_leafs(this->tree_leafs, this->tree_root);
    
    //...
}

LayerNode LayerNode::get_node(LayerNode root, int BranchNumber, int TreeDepth)
{
    LayerNode my_node = root;
    while(my_node.child.size() != 0)
    {
        if(my_node.BranchNumber == BranchNumber && my_node.TreeDepth == TreeDepth)
            return my_node;
        
        bool contains_my_number = false;
        
        for(int i = 0; i < my_node.child.size(); i++)
            contains_my_number = (contains_my_number || (my_node.child[i].BranchNumber == BranchNumber));
            
        if(contains_my_number)
            my_node = my_node.child[BranchNumber];
        
        else
            my_node = my_node.child[0];
    }
    throw(std::invalid_argument("No such Node found in the tree"));
}

Matrix NeuralNetwork::calculate_gradient_hidden(int BranchNumber, int TreeDepth)
{
    /* calculates the gradient for a single instance */
    LayerNode my_node = LayerNode::get_node(this->tree_root, BranchNumber, TreeDepth);
    LayerNode my_parent = *my_node.parent;
    Matrix output = Matrix::ones(1, 1);
    
    if(my_node.LayerName == "dense")
    {
        if(my_parent.LayerName == "dense")
        {
            shared_ptr<Dense> value_spec = dynamic_pointer_cast<Dense>(my_node.value);
            shared_ptr<Dense> value_spec_parent = dynamic_pointer_cast<Dense>(my_parent.value);
            output = Matrix::ones(value_spec->neuron_number, value_spec_parent->neuron_number);
            if(value_spec->LayerType != "softmax")
            {
                Matrix temp = value_spec->W^value_spec->inputs;
                temp = temp + value_spec->b;
                for(int i = 0; i < output.shape[0]; i++)
                    for(int j = 0; j < output.shape[1]; j++)
                        output.values[i][j] = value_spec->W.values[i][j]*value_spec->derived_activation(temp.values[i][0]);
            }
            else
            {
                Matrix temp = value_spec->W^value_spec->inputs;
                temp = temp + value_spec->b;
                output = value_spec->derived_activation_softmax(temp);
                output = value_spec->W * output;
            }
        }
    }
    return output;
}

Matrix NeuralNetwork::calculate_gradient_loss(int BranchNumber)
{
    Matrix output = Matrix::ones(1, 1);
    LayerNode my_loss = this->tree_leafs[BranchNumber];
    
    shared_ptr<Loss> value_spec = dynamic_pointer_cast<Loss>(my_loss.value);
    output = value_spec->derived_activation_loss(value_spec->inputs, value_spec->target);
    
    return output;
}

Matrix NeuralNetwork::calculate_input_loss(int BranchNumber)
{
    Matrix output = Matrix::ones(1, 1);
    
    // Calculate the shape of the flattened complete W weights array
        
        // go up the given branch of the tree and sum all the sizes of the weights (don't forget the biases)
    
    
    // Calculate dF/dW for the input layer
    
        //Follow the math (don't forget the biases)
    
    return output;
}
