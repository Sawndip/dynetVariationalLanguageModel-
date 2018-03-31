#ifndef MNISTREADER_H
#define MNISTREADER_H

#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>

using namespace std;

namespace MnistReader{

void read_mnist(string data_file,  vector<vector<float> > &arr);
void read_mnist_labels(string label_file, vector<unsigned> &labels); 

}

#endif
