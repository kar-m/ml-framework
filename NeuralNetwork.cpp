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
    
    cout << current_child.size() << " " << current_branch << " " << current_depth << endl;
    
    return this_node;
}

void LayerNode::print_tree(LayerNode root)
{
    cout << root.LayerName << endl << "|" << endl;
    for(int i = 0; i < root.child.size(); i++)
    {
        print_tree(root.child[i]);
    }
}
