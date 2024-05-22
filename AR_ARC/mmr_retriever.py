from vec_ret_sim_div import cos_similarity_list
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import openai
import os
from openai import OpenAI
from evaluate import load
import evaluate
from tqdm import tqdm
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import json

def mistral_answer(question):
    os.environ["MISTRAL_API_KEY"] = "Your Own Key"
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "open-mistral-7b"

    client = MistralClient(api_key=api_key)

    chat_response = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content=question)],
    )
    return chat_response.choices[0].message.content

def chatgpt_answer(question, gpt_model="gpt-3.5-turbo"):
    os.environ["OPENAI_API_KEY"] = "Your Own key"
    openai.api_key = os.environ["OPENAI_API_KEY"]

    client = OpenAI()
    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return completion.choices[0].message.content


def MMR(embedding, vectordb, query, k=4, lambda_mult=0.5):
    margi_search = vectordb.max_marginal_relevance_search(query, k,
                                                          lambda_mult=lambda_mult)  

    texts_list = [docu.page_content for docu in margi_search]  

    embed_texts = embedding.embed_documents(texts_list)  

    embed_query = embedding.embed_query(query)  

    cos_sim_list = cos_similarity_list(embed_query, embed_texts)  

    return (texts_list, cos_sim_list)


def test_main2(file_name, embedding, vectordb, query_list, answer_list, k=4, lambda_mult=0.5):
    new_answer_list = []
    list_of_list = []
    for query in tqdm(query_list):
        mmr, cosSim = MMR(embedding, vectordb, query, k, lambda_mult)
        #cosSim_csv = [cs[0][0] for cs in cosSim]  
        mmr_text_sum = ('\n'.join(mmr))  

        prompt = mmr_text_sum + "Based on these, what is the answer of the following question: " + query + ". Just give me the answer directly, no other words." #replace
        new_answer_list.append(mistral_answer(prompt)) #replace
    with open(file_name + "Ans" + '.json', 'w', encoding='utf-8') as file: #replace
        json.dump(new_answer_list, file, ensure_ascii=False, indent=4) #replace
        #embed_mmr_text_sum = embedding.embed_documents([mmr_text_sum])  
        #embed_query = embedding.embed_query(query)
        #cos_query_mmr_text_sum = cosine_similarity(np.array(embed_query).reshape(1, len(embed_query)),
        #                                           np.array(embed_mmr_text_sum[0]).reshape(1, len(embed_query)))
        # print('cos_vsd',cos_query_mmr_text_sum)
        #cosSim_csv.append(cos_query_mmr_text_sum[0][0])

        #list_of_list.append(cosSim_csv)
        # print(cosSim_csv)
        # print('\n')

    #col_name = ['Vector1', 'Vector2', 'Vector3', 'Vector4', 'Sum_vector', 'Mmr_text_sum']
    
    #df = pd.DataFrame(columns=col_name, data=list_of_list)
    
    #df.to_csv(file_name + '.csv', encoding='utf-8', index=False)

# test_main2('MMR00', embedding_model, vectordb, query_list, k=4, lambda_mult=0)
# test_main2('MMR05', embedding_model, vectordb, query_list, k=4, lambda_mult=0.5)
# test_main2('MMR10', embedding_model, vectordb, query_list, k=4, lambda_mult=1)
