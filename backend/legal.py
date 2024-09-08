# indian_law_rag_agent.py
import json
from openai import OpenAI
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class IndianLawRAGAgent:
    def __init__(self, portkey_api_key, portkey_virtual_key, bert_model_name, knowledge_base_path):
        self.client = OpenAI(
            api_key="dummy",
            base_url=PORTKEY_GATEWAY_URL,
            default_headers=createHeaders(
                provider="openai",
                api_key=portkey_api_key,
                virtual_key=portkey_virtual_key
            )
        )

        # Load BERT model and tokenizer from Hugging Face
        self.bert_model = AutoModelForQuestionAnswering.from_pretrained(bert_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Load knowledge base
        with open(knowledge_base_path, 'r') as f:
            data = json.load(f)
            self.knowledge_base = data["knowledge_base"]

        # Encode documents
        self.document_embeddings = self.sentence_model.encode([doc['content'] for doc in self.knowledge_base])

    def retrieve_relevant_documents(self, query, top_k=3):
        query_embedding = self.sentence_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.knowledge_base[i] for i in top_indices]

    def answer_legal_question(self, question, context):
        inputs = self.bert_tokenizer(question, context, return_tensors="pt")
        outputs = self.bert_model(**inputs)

        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax() + 1
        answer = self.bert_tokenizer.convert_tokens_to_string(
            self.bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
        return answer

    def generate_legal_advice(self, startup_type, situation):
        relevant_docs = self.retrieve_relevant_documents(f"{startup_type} {situation}")
        context = "\n".join([doc['content'] for doc in relevant_docs])

        prompt = f"""As a legal expert, provide advice for an Indian {startup_type} startup in the following situation: {situation}

        Relevant legal information:
        {context}

        Advice:"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an AI legal assistant specializing in Indian startup law."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def summarize_law(self, law_name):
        relevant_docs = self.retrieve_relevant_documents(law_name)
        context = "\n".join([doc['content'] for doc in relevant_docs])

        prompt = f"""Summarize the key points of the Indian law: {law_name}, especially as it pertains to startups.

        Relevant legal information:
        {context}

        Summary:"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an AI legal assistant specializing in Indian law summaries."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def check_compliance(self, startup_description):
        relevant_docs = self.retrieve_relevant_documents(startup_description)
        context = "\n".join([doc['content'] for doc in relevant_docs])

        prompt = f"""Given the following Indian startup description, list the key compliance requirements and potential legal issues to be aware of: {startup_description}

        Relevant legal information:
        {context}

        Compliance requirements and potential legal issues:"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an AI legal assistant specializing in compliance for Indian startups."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
