# -------------------------------------- 调用 VecRetSimDivTxt --------------------------------------
# --------------------------------------------------------------------------------------------------
from vec_ret_sim_div import VecRetSimDivTxt
from vec_ret_sim_div import cos_similarity_list
from sklearn.metrics.pairwise import cosine_similarity  # 使用余弦相似度函数
import numpy as np
import pandas as pd
def SimDivRetrieverTxt(embedding, vectordb, query, batch=20, k=4):
    # embedding: 嵌入模型
    # vectordb: 已生成的向量数据库
    # query: 查询字符串
    # batch: 使用相似查询时得到 batch 个最相似的文档
    # k: 从 batch 个最相似向量中找出 k 个满足 simi 和 diversity 要求的向量

    # 1------使用Langchain自带的 Similarity Search 进行检索，得到初始的 batch 个相似度最高的向量
    simi_search = vectordb.similarity_search(query, batch)

    texts_list = [docu.page_content for docu in simi_search]  # 将文档列表转换为字符串列表

    # 2------调用兼具相似性和多样性特点的向量检索算法 VetRetSimDivTxt 从候选例子集中找出 k 个符合要求的例子
    res = VecRetSimDivTxt(texts_list, query, embedding, k)  # 从 batch 个最相似例子中找出 k 个满足 simi 和 diversity 要求的例子

    embed_res = embedding.embed_documents(res)  # 由字符串列表生成嵌入向量
    embed_query = embedding.embed_query(query)  # 生成查询串的嵌入向量

    cos_sim_list = cos_similarity_list(embed_query, embed_res)  # 计算查询向量与查得的k个向量间的余弦相似性，以及查询向量与和向量间的余弦相似性

    return res, cos_sim_list


def test_main1_txt(file_name, embedding, vectordb, query_list, batch=20, k=4):
    list_of_list = []
    for query in query_list:
        vsd, cosSim = SimDivRetrieverTxt(embedding, vectordb, query, batch, k)
        cosSim_csv = [cs[0][0] for cs in cosSim]  # cosSim中的元素是np.array类型的二维数组

        vsd_text_sum = ('\n'.join(vsd))  # 将所有查询到的 k 个例子的原始文本用'\n'拼接在一起
        embed_vsd_text_sum = embedding.embed_documents([vsd_text_sum])  # 生成嵌入向量
        embed_query = embedding.embed_query(query)  # 生成查询串的嵌入向量
        cos_query_vsd_text_sum = cosine_similarity(np.array(embed_query).reshape(1, len(embed_query)),
                                                   np.array(embed_vsd_text_sum[0]).reshape(1, len(embed_query)))
        # print('cos_vsd',cos_query_vsd_text_sum)
        cosSim_csv.append(cos_query_vsd_text_sum[0][0])

        list_of_list.append(cosSim_csv)
        # print(cosSim_csv)
        # print('\n')

    col_name = ['Vector1', 'Vector2', 'Vector3', 'Vector4', 'Sum_vector', 'Vsd_text_sum']
    # 先转为DataFrame格式
    df = pd.DataFrame(columns=col_name, data=list_of_list)
    # index=False表示存储csv时没有默认的id索引
    df.to_csv(file_name + '.csv', encoding='utf-8', index=False)

#embedding_model, vectordb, query_list = eModel_vectordb_queryList()  #约5分钟
#test_main1('SDR', embedding_model, vectordb, query_list, batch=20, k=4)  #约15分钟