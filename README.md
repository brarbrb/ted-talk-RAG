# ted-talk-RAG
TED Talk RAG Assistant

**Deployed at Vercel: [link](https://ted-talk-rag-ctwz.vercel.app/)**

In the file `embeding_RAG_tuning.ipynb` you can find everything implmented to acreate the embedings for ted talks, the upserting to the pinecone index and creating overline the rag flow (architecture). I run few manuak grid searches and evaluations. The final indexing that worked the best is subitted with fully indexed `ted_talks_en.csv` file.


Overall to run the code create virtual python enviroment `.venv` (it's recommended) and run this line to install the dependencies: 
```bash
pip install -r requirements.txt
```

Note: The final  selected RAG hyperparameters are:
- chunk_size = 758
- overlap_ratio = 0.1
- top_k = 5