//
//  main.cpp
//  Deep Learning
//
//  Created by Moss on 8/30/21.
//

#include <iostream>
#include <vector>
#include <queue>
#include "LinearAlgebra.hpp"
#include "Layers.hpp"
#include "NeuralNetwork.hpp"
using namespace std;

int main(int argc, const char * argv[]) {
    queue<LayerNode> branch0;
    for(int i = 0; i < 5; i++)
    {
        Dense* dense = new Dense(10, "ReLu");
        branch0.push(LayerNode::create_node(dense, 0, i, "Dense"));
    }
    queue<LayerNode> branch1;
    for(int i = 3; i < 7; i++)
    {
        Dense* dense = new Dense(7, "ReLu");
        branch1.push(LayerNode::create_node(dense, 1, i, "Dense"));
    }
    
    queue<LayerNode> branch2;
    for(int i = 3; i < 5; i++)
    {
        Dense* dense = new Dense(5, "ReLu");
        branch2.push(LayerNode::create_node(dense, 2, i, "Dense"));
    }
    
    vector<queue<LayerNode>> seed = {branch0, branch1, branch2};
    
    LayerNode root_tree = LayerNode::BuildTree(seed, NULL, 20, 0, 0, true);
    vector<LayerNode> child = root_tree.child;
    
    cout << child.size();
    
    cout << endl << endl << "HI" << endl;
}
