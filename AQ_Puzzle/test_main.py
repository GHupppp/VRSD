from create_vector_db import embedding_corpus
from create_vector_db import read_OpenBookQA
from create_vector_db import read_ARC
from create_vector_db import read_BoolQ
from create_vector_db import read_Puzzle
from create_vector_db import read_OpenBookQA_NoOption
from txt_sim_div_retriever import test_main1_txt
from sim_div_retriever import test_main1
from mmr_retriever import test_main2

from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
torch.cuda.empty_cache()

def main():
    HuggingFace_embedding = HuggingFaceEmbeddings()  


    embedding_model = HuggingFace_embedding

    # all_data_list=read_OpenBookQA()  
    # all_data_list=read_ARC()  
    all_data_list=read_Puzzle()
    print(len(all_data_list))

    vectordb, query_list, answer_list = embedding_corpus(all_data_list, embedding_model)  
    print('Vector database is OK')
    
    test_main1('SDR', embedding_model, vectordb, query_list, answer_list, batch=20, k=4)  
    print('SDR is OK')

    test_main2('MMR00', embedding_model, vectordb, query_list, answer_list, k=4, lambda_mult=0)  
    print('MMR00 is OK')
    test_main2('MMR05', embedding_model, vectordb, query_list, answer_list, k=4, lambda_mult=0.5)  
    print('MMR05 is OK')
    test_main2('MMR10', embedding_model, vectordb, query_list, answer_list, k=4, lambda_mult=1)  
    print('MMR10 is OK')

main()
