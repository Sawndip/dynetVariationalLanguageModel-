#ifndef PTB_READER_H
#define PTB_READER_H

#include <vector>
#include "dynet/dict.h"
#include <string>

namespace VaeLm{
namespace PtbReader{

void get_ptb_data(std::vector<std::vector<int> >* pt_ptb_data,
                  dynet::Dict* pt_dict, 
                  const std::string& file_path,
                  const std::string& bos="<bos>",
                  const std::string& eos="<eos>");

} // PtbReader
} // VaeLm

#endif
