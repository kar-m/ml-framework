//
//  LinearAlgebra.cpp
//  Deep Learning
//
//  Created by Moss on 8/30/21.
//

#include "LinearAlgebra.hpp"
#include <iostream>
using namespace std;

double vector_sum(vector<double> a, vector<double> b, int n)
{
    double ans = 0;
    
    for(int i = 0; i < n; i++)
        ans += a[i]*b[i];
    
    return ans;
}


/*
    Matrix::int shape[2];
Matrix::int** values;*/
    
Matrix::Matrix(vector<vector<double>> given_values, int m, int n)
{
    shape[0] = m;
    shape[1] = n;

    values = given_values;
}

Matrix Matrix::create_matrix(vector<vector<double>> given_values, int m, int n)
{
    vector<vector<double>> temp_values;
    for(int i = 0; i < m; i++)
    {
        vector<double> temp;
        for(int j = 0; j < n; j++)
        {
            temp.push_back(given_values[i][j]);
        }
        temp_values.push_back(temp);
    }
    unique_ptr<Matrix> ans = unique_ptr<Matrix>(new Matrix(given_values, m, n));
    return *ans;
}

Matrix Matrix::operator+(Matrix& b)
{
    if(this->shape[0] != b.shape[0] && this->shape[1] != b.shape[1])
        throw invalid_argument("Mismatched shape error");
    int m = this->shape[0];
    int n = this->shape[1];
    unique_ptr<Matrix> a = make_unique<Matrix>(b);
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            a->values[i][j] += this->values[i][j];
    return *a;
}
    
Matrix Matrix::operator-(Matrix& b)
{
    if(this->shape[0] != b.shape[0] && this->shape[1] != b.shape[1])
        throw invalid_argument("Mismatched shape error");
    int m = this->shape[0];
    int n = this->shape[1];
    unique_ptr<Matrix> a = make_unique<Matrix>(b);
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++){
            a->values[i][j] -= this->values[i][j];
            a->values[i][j] *= -1;
        }
    return *a;
}
    
Matrix Matrix::operator*(Matrix& b)
{
    if(this->shape[0] != b.shape[0] && this->shape[1] != b.shape[1])
        throw invalid_argument("Mismatched shape error");
    int m = this->shape[0];
    int n = this->shape[1];
    unique_ptr<Matrix> a = make_unique<Matrix>(b);
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            a->values[i][j] *= this->values[i][j];
    return *a;
}
    
Matrix Matrix::operator^(Matrix& b)
{
    if(this->shape[1] != b.shape[0])
        throw invalid_argument("Mismatched shape error");
    
    int m = this->shape[0];
    int n = this->shape[1];
    int q = b.shape[1];
    
    vector<vector<double>> temp_values;
    for(int i = 0; i < m; i++)
    {
        vector<double> temp;
        for(int j = 0; j < q; j++)
        {
            temp.push_back(0);
        }
        temp_values.push_back(temp);
    }
        
    unique_ptr<Matrix> temp = unique_ptr<Matrix>(new Matrix(temp_values, m, q));
    
    for(int i = 0; i < m; i++)
        for(int j = 0; j < q; j++){
            vector<double> temporary;
            for(int k = 0; k < n; k++)
                temporary.push_back(b.values[k][j]);
            temp->values[i][j] = vector_sum(this->values[i], temporary, n);
        }
    
    return *temp;
}
    
Matrix Matrix::T()
{
    int m = this->shape[0];
    int n = this->shape[1];
        
    vector<vector<double>> temp_values;
    for(int i = 0; i < n; i++)
    {
        vector<double> temp;
        for(int j = 0; j < m; j++)
        {
            temp.push_back(0);
        }
        temp_values.push_back(temp);
    }
        
      
    unique_ptr<Matrix> temp = unique_ptr<Matrix>(new Matrix(temp_values, n, m));
    
    for(int i = 0; i < this->shape[1]; i++)
    {
        for(int j = 0; j < this->shape[0]; j++)
        {
            temp->values[i][j] = this->values[j][i];
        }
    }
    return *temp;
}
void Matrix::print()
{
    for(int i = 0; i < this->shape[0]; i++)
    {
        cout << "| ";
        for(int j = 0; j < this->shape[1]; j++)
            cout << this->values[i][j] << " ";
        cout << "|" << endl;
    }
    cout << endl;
}
Matrix Matrix::zeros(int m, int n)
{
    vector<vector<double>> temp_values;
    for(int i = 0; i < m; i++)
    {
        vector<double> temp;
        for(int j = 0; j < n; j++)
        {
            temp.push_back(0);
        }
        temp_values.push_back(temp);
    }
        
      
    unique_ptr<Matrix> temp = unique_ptr<Matrix>(new Matrix(temp_values, m, n));
    
    return *temp;
}

Matrix Matrix::ones(int m, int n)
{
    vector<vector<double>> temp_values;
    for(int i = 0; i < m; i++)
    {
        vector<double> temp;
        for(int j = 0; j < n; j++)
        {
            temp.push_back(1);
        }
        temp_values.push_back(temp);
    }
        
    unique_ptr<Matrix> temp = unique_ptr<Matrix>(new Matrix(temp_values, m, n));

    return *temp;
}

Matrix Matrix::hstack(Matrix b)
{
    int m = this->shape[0];
    int n = this->shape[1];
    int p = b.shape[0];
    int q = b.shape[1];
    
    if(m != p)
        throw invalid_argument("Mismatched shape error");

    vector<vector<double>> temp_values;
    for(int i = 0; i < m; i++)
    {
        vector<double> temp;
        for(int j = 0; j < (n+q); j++)
        {
            if(j<n)
                temp.push_back(this->values[i][j]);
            else
                temp.push_back(b.values[i][j-n]);
        }
        temp_values.push_back(temp);
    }
        
      
    unique_ptr<Matrix> temp = unique_ptr<Matrix>(new Matrix(temp_values, m, (n+q)));
    
    
    return *temp;
}

Matrix Matrix::vstack(Matrix b)
{
    int m = this->shape[0];
    int n = this->shape[1];
    int p = b.shape[0];
    int q = b.shape[1];
    
    if(n != q)
        throw invalid_argument("Mismatched shape error");

    vector<vector<double >> temp_values;
    for(int i = 0; i < (m+p); i++)
    {
        vector<double > temp;
        for(int j = 0; j < n; j++)
        {
            if(i<m)
                temp.push_back(this->values[i][j]);
            else
                temp.push_back(b.values[i-m][j]);
        }
        temp_values.push_back(temp);
    }
        
      
    Matrix temp = create_matrix(temp_values, (m+p), n);
    
    
    return temp;
}

double Matrix::loc(int x, int y)
{
    return this->values[x][y];
}

Matrix Matrix::loc(string a, int y)
{
    vector<vector<double>> temporary_values;
    vector<double> temporary;
    int m = this->shape[0];
    for(int i = 0; i < m; i++)
        temporary.push_back(this->values[i][y]);
    temporary_values.push_back(temporary);
    Matrix ans = create_matrix(temporary_values, 1, m);
    ans = ans.T();
    return ans;
}

Matrix Matrix::loc(int x, string b)
{
    vector<vector<double>> temporary_values;
    vector<double> temporary;
    int n = this->shape[1];
    for(int i = 0; i < n; i++)
        temporary.push_back(this->values[x][i]);
    temporary_values.push_back(temporary);
    Matrix ans = create_matrix(temporary_values, 1, n);
    ans = ans.T();
    return ans;
}

Matrix Matrix::flatten()
{
    //In terms of the Dense layer, this will make it so each neuron's coefficients in the layer are next to each other row by row
    
    Matrix ans = this->loc(0, ":");
    for(int i = 1; i < this->shape[0]; i++)
    {
        Matrix temp = this->loc(i, ":");
        ans = ans.vstack(temp);
    }
    return ans;
}
double Matrix::sum()
{
    double ans = 0;
    for(int i = 0; i < this->shape[0]; i++)
        for(int j = 0; j < this->shape[1]; j++)
            ans += this->values[i][j];
    return ans;
}
