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
#include <numeric>
#include <algorithm>
#include <random>
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
    NeuralNetwork::calculate_gradient_shapes();
    this->batch_size = batch_size;
    this->learning_rate = learning_rate;
    this->optimizer = optimizer;
    //...
}

NeuralNetwork NeuralNetwork::createModel(vector<queue<LayerNode>> Layers, int input_size, int batch_size, double learning_rate, string optimizer)
{
    unique_ptr<NeuralNetwork> ans = unique_ptr<NeuralNetwork>(new NeuralNetwork(Layers, input_size, batch_size, learning_rate, optimizer));
    return *ans;
}

LayerNode LayerNode::get_node(LayerNode root, int BranchNumber, int TreeDepth)
{
    LayerNode my_node = root;

    if(my_node.BranchNumber == BranchNumber && my_node.TreeDepth == TreeDepth)
        return my_node;
        
        
    for(int i = 0; i < my_node.child.size(); i++)
        if(my_node.child[i].BranchNumber == BranchNumber)
        {
            return get_node(my_node.child[i], BranchNumber, TreeDepth);
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

void NeuralNetwork::calculate_gradient_shapes()
{
    for(int i = 0; i < this->tree_leafs.size(); i++)
    {
        int size = 0;
        LayerNode current_leaf = this->tree_leafs[i];
        LayerNode current_node = *current_leaf.parent;
        while(current_node.parent != NULL)
        {
            if(current_node.LayerName == "dense")
            {
                shared_ptr<Dense> value_spec = dynamic_pointer_cast<Dense>(current_node.value);
                size += value_spec->W.shape[0]*value_spec->W.shape[1] + value_spec->b.shape[0];
            }
            current_node = *current_node.parent;
        }
        Matrix gradient = Matrix::zeros(size, 1);

        this->gradients.push_back(gradient);
    }
}

Matrix NeuralNetwork::calculate_gradient_input(int BranchNumber)
{
    Matrix temp;
    
    if(this->tree_root.LayerName == "dense")
    {
        shared_ptr<Dense> value_spec = dynamic_pointer_cast<Dense>(this->tree_root.value);
        temp = Matrix::zeros(value_spec->neuron_number, this->gradients[BranchNumber].shape[0]);
        int m, n;
        
        m = value_spec->neuron_number;
        n = value_spec->W.shape[1];
        
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                int current_index = i*n+j;
                temp.values[i][current_index] = value_spec->inputs.values[j][0];
            }
        }
        
        for(int i = 0; i < n; i++)
        {
            int current_index = m*n + i;
            temp.values[i][current_index] = 1;
        }
        
    }

    return temp;
}

Matrix NeuralNetwork::calculate_gradient(int BranchNumber)
{
    LayerNode my_node = this->tree_leafs[BranchNumber];
    
    Matrix output;
    
    while(my_node.parent != NULL)
    {
        if(my_node.LayerName == "loss")
        {
            output = calculate_gradient_loss(BranchNumber);
        }
        
        else
        {
            Matrix hidden_gradient = calculate_gradient_hidden(BranchNumber, my_node.TreeDepth);
            output = output^hidden_gradient;
        }
        
        my_node = *my_node.parent;
    }
    
    Matrix input_gradient = calculate_gradient_input(BranchNumber);
    output = output^input_gradient;
    
    return output;
}


void NeuralNetwork::update_inputs(Matrix input, vector<Matrix> y, LayerNode child)
{
    LayerNode my_node = child;
    
    if(my_node.isLeaf == false)
    {
        Matrix current_input = input;
    
        my_node.value->inputs = current_input;
    
        vector<LayerNode> children = my_node.child;
    
        for(int i = 0; i < children.size(); i++)
        {
            if(my_node.LayerName == "dense")
            {
                shared_ptr<Dense> value_spec = dynamic_pointer_cast<Dense>(my_node.value);
                current_input = value_spec->evaluate(current_input);
                update_inputs(current_input, y, children[i]);
            }
        }
    }
    
    else
    {
        shared_ptr<Loss> value_spec = dynamic_pointer_cast<Loss>(my_node.value);
        value_spec->target = y[my_node.BranchNumber];
    }
}

Matrix NeuralNetwork::stochastic_gradient(int BranchNumber, Matrix x, vector<Matrix> y)
{
    update_inputs(x, y, this->tree_root);
    
    Matrix output = calculate_gradient(BranchNumber);
    
    return output;
}

vector<Matrix> generate_y(vector<Matrix> Y, int data_point)
{
    vector<Matrix> output;
    
    for(int i = 0; i < Y.size(); i++)
    {
        Matrix temp = Y[i].loc(":", data_point);
        output.push_back(temp);
    }
    
    return output;
}

Matrix NeuralNetwork::mini_batch_gradient(int BranchNumber, Matrix X, vector<Matrix> Y)
{
    Matrix output;
    
    Matrix X_first = X.loc(":", 0);
    
    vector<Matrix> y_first = generate_y(Y, 0);

    output = stochastic_gradient(BranchNumber, X_first, y_first);
    
    for(int i = 1; i < X.shape[1]; i++)
    {
        Matrix current_x = X.loc(":", i);
        vector<Matrix> current_y = generate_y(Y, i);
        
        Matrix grad_temp = stochastic_gradient(BranchNumber, current_x, current_y);
        
        Matrix modify = output + grad_temp;
        
        output = modify;
    }
    
    double c = 1/(X.shape[1]);
    
    Matrix modify = output * c;
    
    return modify;
}

void NeuralNetwork::update_weights(Matrix flat_change, int BranchNumber)
{
    LayerNode my_node = this->tree_root;
    int current_index = 0;
    while(my_node.child.size() != 0)
    {
        //updating the current branch's weights
        
        if(my_node.LayerName == "dense")
        {
            shared_ptr<Dense> value_spec = dynamic_pointer_cast<Dense>(my_node.value);
            
            int m = value_spec->W.shape[0];
            int n = value_spec->W.shape[1];
            
            int lim = m * n;
            
            // for bugs look here, maybe swap the coordinates
            for(int i = 0; i < lim; i++)
                value_spec->W.values[i % m][i / n] += flat_change.values[0][current_index + i];
            
            current_index += lim;
            
            for(int i = 0; i < value_spec->b.shape[0]; i++)
                value_spec->b.values[i][0] += flat_change.values[0][current_index + i];
            
            current_index += value_spec->b.shape[0];
            
        }
        
        //traversing down the given branch
        
        //bool contains_my_number = false;
        int branch_id = 0;
        
        for(int i = 0; i < my_node.child.size(); i++)
            if(my_node.child[i].BranchNumber == BranchNumber)
            {
                //contains_my_number = true;
                branch_id = i;
            }

        my_node = my_node.child[branch_id];

    }
}

void NeuralNetwork::gradient_descent(int batch_size, double learning_rate, Matrix X, vector<Matrix> Y)
{
    // this one will be a doozie to debug
    
    int barWidth = 70;
    float progress = 0.0;
    
    int Branches = this->tree_leafs.size();
    int data_size = X.shape[1];
    
    std::vector<int> indexes(data_size);
    std::iota(std::begin(indexes), std::end(indexes), 0);

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(indexes), std::end(indexes), rng);
    
    
    for(int i = 0; i < data_size / batch_size; i++)
    {
        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
        
        float progress_max = data_size/batch_size;
        
        progress = float(i) / progress_max;
        
        int base = batch_size*i;
        Matrix batch = X.loc(":", indexes[base]);
        vector<Matrix> y_batch = generate_y(Y, 0);
        
        for(int j = 1; j < batch_size; j++)
        {
            Matrix temp = X.loc(":", indexes[base+j]);
            Matrix temp_prime = batch.hstack(temp);
            
            batch = temp_prime;
            
            vector<Matrix> current_y = generate_y(Y, j);
            
            for(int k = 0; k < Branches; k++)
            {
                Matrix y_temp = y_batch[k].hstack(current_y[k]);
                y_batch[k] = y_temp;
            }
        }
        
        for(int BranchNumber = 0; BranchNumber < Branches; BranchNumber++)
        {
            Matrix gradient = mini_batch_gradient(BranchNumber, batch, y_batch);
            gradient = gradient * learning_rate;
            
            double neg_one = double(-1);
            gradient = gradient * neg_one;
            
            update_weights(gradient, BranchNumber);
        }
    }
    
    int base = data_size - (data_size % batch_size);
    Matrix batch = X.loc(":", indexes[base]);
    vector<Matrix> y_batch = generate_y(Y, 0);
    
    for(int j = 1; j < (data_size % batch_size); j++)
    {
        Matrix temp = X.loc(":", indexes[base+j]);
        Matrix temp_prime = batch.hstack(temp);
        
        batch = temp_prime;
        
        vector<Matrix> current_y = generate_y(Y, j);
        
        for(int k = 0; k < Branches; k++)
        {
            Matrix y_temp = y_batch[k].hstack(current_y[k]);
            y_batch[k] = y_temp;
        }
    }
    
    for(int BranchNumber = 0; BranchNumber < Branches; BranchNumber++)
    {
        Matrix gradient = mini_batch_gradient(BranchNumber, batch, y_batch);
        gradient = gradient * learning_rate;
        
        double neg_one = double(-1);
        gradient = gradient * neg_one;
        
        update_weights(gradient, BranchNumber);
    }
}

void NeuralNetwork::fit(Matrix X, vector<Matrix> Y, int epoch_number)
{
    for(int i = 0; i < epoch_number; i++){
        cout << "Epoch " << i;
        gradient_descent(this->batch_size, this->learning_rate, X, Y);
    }
    
}
