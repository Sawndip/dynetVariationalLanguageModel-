#ifndef PTB_READER_H
#define PTB_READER_H

#include <vector>
#include "dynet/dict.h"
#include <string>

namespace PtbReader{

typedef struct BatchIndex{
    unsigned int batch_begin_idx;
    unsigned int batch_num_elements;
} BATCH_INDEX_t;

void get_ptb_data(std::vector<std::vector<int> >* pt_ptb_data,
                  dynet::Dict* pt_dict, 
                  const std::string& file_path,
                  const std::string& bos="<bos>",
                  const std::string& eos="<eos>");

void log_data_stats(const std::vector<std::vector<int> >& ptb_data, 
                    const dynet::Dict& dict, 
                    const std::string& data_type="");

void sort_data_in_ascending_length(std::vector<std::vector<int> >* pt_data);

void create_batches(std::vector<BATCH_INDEX_t>* pt_batchIndexList,
                    const std::vector<std::vector<int> >& data, 
                    const unsigned int& max_batch_size);
 

} // PtbReader

#endif
