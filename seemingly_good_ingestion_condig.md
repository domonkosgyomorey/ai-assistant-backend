# OCR Based PDF loader
- OCR Based Loader
- 1000 Chunk len and 200 overlap size

- Result: It was definitly have some improvements with this kinda ingestion. The tables that generated are okay ( most important retrieved the important things from the table )

# RAGAS based testset generation
- RAGAS usually creates test cases with a lot of text both for questions and reference answers, so it can be good for many use cases, however for information retiever systems are usually recieves low quality or shorter inputs, so the result for this method could be a bit missleading.
- The testset generation is a bit hard and long, however it does some tricks that can be very good for some use cases ( writer AI ).
- RAGAS caching the most of the states, but there are some redundancy when I want to generate more test cases.
- RAGAS can use the langchain Documents directly

# Giskard
- Harder to install due some very annoying package like griffe or docstring.
- To create knowledge base I need to parse the Documents into dataframes.
- Because of the previous statement the Giskard can found topics and generate questions and reference answer much faster, however the quality is much worst.
- For a RAG system the question quality does not matter at all, but the context and the reference answer should be good enough.
- Giskard also supports multiple kind of questions and it can figure out different voulnabilities.