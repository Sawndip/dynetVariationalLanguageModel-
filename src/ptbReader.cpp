#include "ptbReader.h"
#include "dynet/dict.h"

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <string.h>

namespace VaeLm{

void PtbReader::get_ptb_data(std::vector<std::vector<int> >* pt_ptb_data,
                             dynet::Dict* pt_dict, 
                             const std::string& file_path,
                             const std::string& bos,
                             const std::string& eos)
{
    std::ifstream ifs;
    ifs.open(file_path, std::ifstream::in);
    
    if(ifs.fail()){
        std::cout << "Could not open file" << file_path << std::endl;
        exit(1);
    }
    
    std::string sent;
    std::vector<int> sent_ids;
    while(std::getline(ifs, sent)){
        
        sent = bos + " " + sent + " " + eos;
        
        sent_ids = dynet::read_sentence(sent, *pt_dict); // dict is modified in this call 
        pt_ptb_data->push_back(sent_ids);
    }

    ifs.close();
    return;
}

} // VaeLm
