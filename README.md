# Chat-with-PDF-using-LangChain
## <b> Chat with your PDF data using using LangChain Framework. </b>

## System Setup:
Step 1. Create a virtual environment
  > conda create -p myenv python=3.9 -y

step 2. Activate the environment:
  > conda activate myenv/

step 3. Install all the requirements:
  > pip install -r requirements.txt
 
step 4. start writing the code, with standard file name as main.py

### <b> High Level Architecture </b>
![image](https://github.com/Chandrakant817/Chat-with-PDF-using-LangChain/assets/69152112/885d549e-3d21-4e95-a927-70b31a02c08d)

### <b> Steps: </b>
1. Upload the PDF Files
2. Extract the content from the PDF
3. Split content into Chunk
4. Do Embeddings of the data (Download embeddings from the OpenAI)
5. Store Data into Vector Store (eg: FAISS and Croma)
6. User can pass a Prompt
7. Based on User query, Similarity search will apply
8. Get the Output.

## <b> Output </b>
![image](https://github.com/Chandrakant817/Chat-with-PDF-using-LangChain/assets/69152112/e203a8a7-0388-418b-b11b-e8e2a18e9d74)

### <b> References </b>
1. https://youtu.be/5Ghv-F1wF_0?si=7Y2yEV6k6y89VOeo


