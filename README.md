# medic.ai
medic.ai is a medical chatbot based on Retrieval Augmented Fine Tuning ( RAFT ), RAG, Finetuning 


To-Do : 
Completed chatbot locally, might try dockerization 


Couldn't complete RAFT, out of time and Colab creds.


Outcome: 


Finally completed the chatbot Although I couldn't dockerize it, I successfully used NGrok to expose the app publicly by hosting it on my machine as a server.


**How To Run Medic on Your Machine ?**

1. Clone this repository into your machine :
   ```
   git clone https://github.com/VamshiKrsna/medic.ai/
   ```

2. Start an instance of your favorite LLM on local LLM Inference platforms like LM Studio, Ollama, etc.
3. cd into src directory of the project
4. run the following command :
   ```
   streamlit run chatuimedic.py
   ```  
