import os
import time
import arxiv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatZhipuAI


# 核心业务逻辑类
class ResearchEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.db = None
        self.loaded_files = []

    def process_pdf(self, file_path, file_name):
        """处理 PDF 并存入向量库"""
        if not self.api_key:
            raise ValueError("API Key is missing")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_paper'] = file_name

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        valid_chunks = [c for c in chunks if len(c.page_content.strip()) > 20]

        embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=self.api_key)

        # 构建或更新向量库
        if self.db is None:
            self.db = FAISS.from_documents(valid_chunks, embeddings)
        else:
            self.db.add_documents(valid_chunks)

        if file_name not in self.loaded_files:
            self.loaded_files.append(file_name)

        return len(valid_chunks)

    def search_arxiv(self, query, max_results=10):
        """搜索 ArXiv"""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        return list(search.results())

    def generate_summary(self):
        """生成全量综述"""
        if not self.db or not self.loaded_files:
            return "No data available."

        aggregated_context = ""
        for filename in self.loaded_files:
            sub_docs = self.db.similarity_search(
                "Abstract, methodology, main contribution, conclusion",
                k=2,
                filter={"source_paper": filename}
            )
            if sub_docs:
                content = "\n".join([d.page_content for d in sub_docs])
                aggregated_context += f"\n=== Paper: {filename} ===\n{content}\n"

        llm = ChatZhipuAI(model="glm-4", api_key=self.api_key, temperature=0.1)
        prompt = f"""
        请分析以下 {len(self.loaded_files)} 篇论文，生成 Markdown 对比表格。
        包含：论文名称 | 核心创新 | 方法 | 结论。
        内容：{aggregated_context}
        """
        return llm.invoke(prompt).content

    def query_bot(self, prompt, context):
        """调用 LLM 回答问题"""
        llm = ChatZhipuAI(model="glm-4", api_key=self.api_key, temperature=0.1)
        response = llm.invoke(prompt)
        return response.content
