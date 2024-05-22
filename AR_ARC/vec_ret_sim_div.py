# Vector retrieve: similarity and diversity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  



def sum_vectors(vectors):
    s = vectors[0]
    for i in range(1, len(vectors)):
        s = np.add(s, vectors[i])
    return s


def cos_similarity_list(query_vector, vector_list):
    row = 1
    col = len(query_vector)  
    cos_sim_list = [cosine_similarity(np.array(query_vector).reshape(row, col), np.array(vec).reshape(row, col)) for vec
                    in vector_list]

    sum_v = sum_vectors(vector_list)
    cos_query_sum = cosine_similarity(np.array(query_vector).reshape(1, len(query_vector)),
                                      np.array(sum_v).reshape(1, len(query_vector)))
    cos_sim_list.append(cos_query_sum)

    return cos_sim_list


def VecRetSimDiv(vectors, query, k=4):
    query_dim = len(query)
    results = []
    results_index = []  
    max_index = 0
    for i in range(k):
        results.append(vectors[max_index])  
        results_index.append(max_index)  

        sum_results = sum_vectors(results)  

        max_simi = 0
        for j in range(len(vectors)):
            if j not in results_index:  
                v = np.add(sum_results, vectors[j])
                cos_simi = cosine_similarity(np.array(v).reshape(1, query_dim),
                                             np.array(query).reshape(1, query_dim))  
                if cos_simi > max_simi:
                    max_simi = cos_simi
                    max_index = j
            else:
                pass
    return results, results_index


def VecRetSimDivTxt(txt_list, query, embedding_model, k=4):
    embed_query = embedding_model.embed_query(query)  
    query_dim = len(embed_query)

    results = []  
    max_index = 0  
    for i in range(k):
        results.append(txt_list[max_index]) 
        del txt_list[max_index]  

        sum_results = ('\n'.join(results))  

        max_simi = 0
        for j in range(len(txt_list)):
            tmp_sum = sum_results + '\n' + txt_list[j]
            embed_tmp_sum = embedding_model.embed_documents([tmp_sum])  

            cos_simi = cosine_similarity(np.array(embed_tmp_sum).reshape(1, query_dim),
                                         np.array(embed_query).reshape(1, query_dim))  
            if cos_simi > max_simi:
                max_simi = cos_simi
                max_index = j
    return results

'''
L = [[5, 2, 6], [1, 2, 3], [3, 2, 6], [1, 5, 6], [3, 8, 1], [2, 26, 6]]
q = [1, 1, 1]
print([cosine_similarity(np.array(q).reshape(1, 3), np.array(v).reshape(1, 3)) for v in L])

res, res_index = VecRetSimDiv(L, q, k=3)
sum_v = sum_vectors(res)
print(res, sum_v)
print(res_index)
cosine_similarity(np.array(sum_v).reshape(1, 3), np.array(q).reshape(1, 3))
'''
