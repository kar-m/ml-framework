//
//  LinearAlgebra.hpp
//  Deep Learning
//
//  Created by Moss on 8/30/21.
//

#ifndef LinearAlgebra_hpp
#define LinearAlgebra_hpp

#include <stdio.h>
#include <vector>
using namespace std;
int vector_sum();

class Matrix
{
public:
    int shape[2];
    vector<vector<double>> values;
    double loc(int x, int y);
    Matrix loc(string a, int y);
    Matrix loc(int x, string b);
    static Matrix create_matrix(vector<vector<double>> given_values, int m, int n);
    Matrix operator+(Matrix& b);
    Matrix operator-(Matrix& b);
    Matrix operator*(Matrix& b);
    Matrix operator^(Matrix& b);
    Matrix T();
    void print();
    static Matrix zeros(int m, int n);
    static Matrix ones(int m, int n);
    Matrix hstack(Matrix b);
    Matrix vstack(Matrix b);
    Matrix flatten();
    double sum();
private:
    Matrix(vector<vector<double>> given_values, int m, int n);
    
};


#endif /* LinearAlgebra_hpp */
