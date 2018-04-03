#include "ptbReader.h"
#include "dynet/dict.h"

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <map>
#include <algorithm>
#include <utility>

namespace{

bool comparator_to_sort_in_ascending_length(const std::vector<int>& lhs, const std::vector<int>& rhs)
{
    return (lhs.size() < rhs.size());
}

void testCreate_batches(const std::vector<PtbReader::BATCH_INDEX_t>& batchIndexList,
                        const std::vector<std::vector<int> >& data, 
                        const unsigned int& max_batch_size)
{
    // test batches have equal length
    bool has_batching_error = false;
    for(unsigned int i=0; i<batchIndexList.size();++i){
        const PtbReader::BATCH_INDEX_t& current_batch_index = batchIndexList[i];
        unsigned int batch_for_length = data[current_batch_index.batch_begin_idx].size();
        for(unsigned int j=0; j<current_batch_index.batch_num_elements; ++j){
            if(data[current_batch_index.batch_begin_idx + j].size()!=batch_for_length){
                has_batching_error = true;
                std::cout << "ERROR: length of data in batch does not match" << std::endl;
            }
        } 
    }
    
    if(!has_batching_error){
        std::cout << "Test passed. Each batches has data of equal length" << std::endl;
    }
    
    // test the sum of all batch sizes is equal to the data size
    unsigned int num_of_elements_in_all_batches = 0;
    for(unsigned int i=0; i<batchIndexList.size();++i){
        const PtbReader::BATCH_INDEX_t& current_batch_index = batchIndexList[i];
        num_of_elements_in_all_batches += current_batch_index.batch_num_elements;  
    }
    if(num_of_elements_in_all_batches!=data.size()){
        std::cout << "ERROR: sum of all batch sizes is not equal to the data size" 
                  << " : num_of_elements_in_all_batches = " << num_of_elements_in_all_batches
                  << " data.size() = " << data.size() 
                  << std::endl;
        for(unsigned int i=0; i<batchIndexList.size();++i){
            const PtbReader::BATCH_INDEX_t& current_batch_index = batchIndexList[i];
            std::cout << "batch_for_length = " << data[current_batch_index.batch_begin_idx].size()
                      << " : begin_index = " << current_batch_index.batch_begin_idx
                      << " num_elements = " << current_batch_index.batch_num_elements
                      << std::endl;

            has_batching_error = true; 
        } 
    }else{
        std::cout << "sum of all batch sizes is equal to the data size" << std::endl;
    }

    if(has_batching_error){
        abort();
    }
}


}

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

void PtbReader::log_data_stats(const std::vector<std::vector<int> >& ptb_data, 
                               const dynet::Dict& dict,
                               const std::string& data_type)
{ 
    /*
    Function to log:
        1) Size of the dict
        2) The number of sentences as a function of sentence length 
    */

    std::cout << "logging for data_type = " << data_type << std::endl;
    
    std::cout << "dict.size() = " << dict.size() << std::endl;
    std::cout << "ptb_train_data.size() = " << ptb_data.size() << std::endl;

    std::map<unsigned int, int> size_count;
    for(size_t i=0; i<ptb_data.size(); ++i){
        if(size_count.find(ptb_data[i].size()) == size_count.end()){
            size_count[ptb_data[i].size()] = 1;
        }else{
            size_count[ptb_data[i].size()] += 1;
        }
    }

    for(std::map<unsigned int, int>::const_iterator it=size_count.begin();
           it!=size_count.end(); ++it){
        std::cout << "size = " << it->first
                  << " count = " << it->second
                  << std::endl;
    }
    
    std::cout << "Done logging for data_type = " << data_type << std::endl;
}

void PtbReader::sort_data_in_ascending_length(std::vector<std::vector<int> >* pt_data)
{
    /*
    * data = < <1, 32, 12, -1>, <23, 1, 0>, <32, 56, 1, 8, 9>, <45> >
    * after sorting: 
    * data = < <45>, <23, 1, 0>, <1, 32, 12, -1>, <32, 56, 1, 8, 9> >
    */
    
    std::vector<std::vector<int> >& data = *pt_data;
    std::sort(data.begin(), data.end(), comparator_to_sort_in_ascending_length);
}

void PtbReader::create_batches(std::vector<PtbReader::BATCH_INDEX_t>* pt_batchIndexList,
                               const std::vector<std::vector<int> >& data, 
                               const unsigned int& max_batch_size) 
{
    /* Creates batches where all elements in the batch have the same length
    *
    * data must be sorted in increasing ordered of length.
    * i.e. we must have
    * data[i].size() <= data[j].size() for i<j
    *
    * example:  
    * data = < <23>,           <45>,       <3>,     <12>, 
               <31, 2>,        <5, 32>,    <37, 1>, <23, 87>, <5, 65>,
               <2, 4, 43>,     <18, 45, 6>,
               <34, 45, 65, 76> >
    * max_batch_size = 3
    * batchIndexList: <(0, 3), (3, 1), (4, 3), (7, 2), (9, 2), (11, 1)>
    */  
    
    if(data.empty()){
        std::cout << "Cannot create batches for empty data" << std::endl;
        abort();
    }
 
    if(max_batch_size == 0){
       std::cout << "max_batch_size cannot be zero" << std::endl;
       abort();
    }
  
    PtbReader::BATCH_INDEX_t batch_index; 
    batch_index.batch_begin_idx = 0;
    batch_index.batch_num_elements = 1;
    pt_batchIndexList->push_back(batch_index);
 
    unsigned int current_batch_size = 1;
    unsigned int batch_for_length = data[0].size();
    for(unsigned int i=1; i<data.size(); ++i){
        if(current_batch_size==max_batch_size || data[i].size()!=batch_for_length){
            pt_batchIndexList->back().batch_num_elements = current_batch_size;
            
            batch_index.batch_begin_idx = i;
            batch_index.batch_num_elements = 1;
            pt_batchIndexList->push_back(batch_index);             
 
            current_batch_size = 1;
            batch_for_length = data[i].size();
        }else{
            ++current_batch_size;
        }
    }

    testCreate_batches(*pt_batchIndexList, data, max_batch_size);
}
