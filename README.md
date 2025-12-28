# ted-talk-RAG
TED Talk RAG Assistant

**Deployed at Vercel!**

In the file `embeding_RAG_tuning.ipynb` you can find everything implmented to acreate the embedings for ted talks, the upserting to the pinecone index and creating overline the rag flow (architecture). I run few manuak grid searches and evaluations. The final indexing that worked the best is subitted with fully indexed `ted_talks_en.csv` file.


Overall to run the `app.py` create virtual python enviroment `.venv` (it's recommended) and run this line to install the dependencies: 
```bash
pip install -r requirements.txt
```

Note: The final  selected RAG hyperparameters are:
- chunk_size = 758 (smaller chunk size more precision for smaller details)
- overlap_ratio = 0.1 (smaller overlap)
- top_k = 20 (but much higher k retrieved docs)

To run the `embeding_RAG_tuning.ipynb` heavier dependencies are neded (I put them in different file), Run this line:
```bash
pip install -r heavier_requirements.txt
```