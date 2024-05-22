# VRSD: Rethinking Similarity and Diversity for Retrieval in Large Language Models

# ----------------------------------------------------------------------
# Usage
**pip install -r requirements.txt**

# Experiment results

### The results of retrieval quality

###### To evaluate the "retrieval quality" of VRSD in dataset ARC-DA, go into the directory RQ_ARC, run the test_main.py first, and then run the analysis_result.py.

###### To evaluate the "retrieval quality" of VRSD in dataset OpenBookQA, go into the directory RQ_OpenBookQA, run the test_main.py first, and then run the analysis_result.py.

###### To evaluate the "retrieval quality" of VRSD in dataset Puzzle, go into the directory RQ_Puzzle, run the test_main.py first, and then run the analysis_result.py.

### The results of answer quality

**Before you run the code to get the results of "answer quality", first create you own api key if in the website of Mistral and OpenAI. Once you get your keys, replace them in the mmr_retriever.py and sim_div_retriever.py at the position "Your Own key" under each directory**

**If you want to see the results of the algorithm(including VRSD and MMR) with the open source model "open-mistral-7b", just follow the instructions below and run the program directly; If you want to see the results of the algorithm(including VRSD and MMR) with the close source model "gpt-3.5-turbo", before you follow the below instructions, go into the files mmr_retriever.py and sim_dive_retriever.py and replace the function name mistral_answer with the name chatgpt_answer in line 66 and 71 respectively.**

###### To evaluate the "answer quality" of VRSD in dataset ARC-DA, go into the directory AQ_ARC, run the test_main.py first, and then run the metrics.py.

###### To evaluate the "answer quality" of VRSD in dataset OpenBookQA, go into the directory AQ_OpenBookQA, run the test_main.py first, and then run the metrics.py.

###### To evaluate the "answer quality" of VRSD in dataset Puzzle, go into the directory AQ_Puzzle, run the test_main.py first, and then run the metrics.py.
