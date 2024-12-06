
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional, Union

from FlagEmbedding import FlagAutoModel
from pydantic import BaseModel, Field

from config import DEVICE, MODEL_TIMEOUT, logger
from exception import ModelNotLoadedError
from singleton import Singleton


# 定义请求体模型
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Input text to embed, can be a string or an array of strings.")
    model: str = Field(..., description="ID of the model to use for generating embeddings.")
    encoding_format: Optional[str] = Field("float", description="The format to return the embeddings in. Can be 'float' or 'base64'.")
    dimensions: Optional[int] = Field(None, description="The number of dimensions for the output embeddings. Only for text-embedding-3 and later models.")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user for monitoring and abuse detection.")

# 定义响应体模型
class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]
    
# 定义响应体模型
class EmbeddingResponse(BaseModel):
    data: List[EmbeddingObject]
    model: str
    object: str = "list"
    usage: dict = {
        "prompt_tokens": 0,
        "total_tokens": 0
    }
    

class EmbeddingModel(metaclass=Singleton):
    def __init__(self):
        self.embed = None
        self.last_used = None
        self.model_name = None
        self.tokenizer = None
        self._lock = threading.Lock()  # 防止并发问题

    def load_model(self, model_name):
        """Loads the reranker model if it's not already loaded or if the path has changed."""
        with self._lock:
            if self.embed is None or self.model_name != model_name:
                logger.info(f"开始加载模型: {model_name} to {DEVICE}")
                self.embed = FlagAutoModel.from_finetuned(
                        model_name,
                        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                        devices=DEVICE,   # if not specified, will use all available gpus or cpu when no gpu available
                    )
                self.tokenizer = self.embed.tokenizer
                self.model_name = model_name
            self.last_used = datetime.now()
    
    def embedding(self, sentences: Union[list[str], str], dimensions=None) -> EmbeddingResponse:
        """生成嵌入"""
        with self._lock:
            if self.embed is None:
                raise ModelNotLoadedError()
            
            self.last_used = datetime.now()
            if sentences and isinstance(sentences, str):
                    sentences = [sentences]

            # 计算 prompt_tokens
            if self.tokenizer:
                prompt_tokens = sum(len(self.tokenizer.encode(sentence)) for sentence in sentences)
            else:
                # 如果没有加载 tokenizer，使用字符长度作为一个粗略的估计
                prompt_tokens = sum(len(sentence) for sentence in sentences)
                
            # 生成嵌入
            embeddings = self.embed.encode(
                sentences, 
                batch_size=12, 
                max_length=dimensions,
            )['dense_vecs']
            

            # total_tokens 在嵌入模型中等于 prompt_tokens
            total_tokens = prompt_tokens

            # 创建 EmbeddingObject 列表
            embedding_objects = [EmbeddingObject(object="embedding", index=i, embedding=emb.tolist()) for i, emb in enumerate(embeddings)]
            # 返回 EmbeddingResponse
            return EmbeddingResponse(
                data=embedding_objects,
                model=self.model_name,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": total_tokens
                }
            )

        
    def check_timeout(self):
        """检查模型是否超时，并在超时时释放模型"""
        while True:
            with self._lock:
                if (
                    self.embed
                    and self.last_used
                    and (datetime.now() - self.last_used)
                    > timedelta(minutes=MODEL_TIMEOUT)
                ):
                    self.embed = None
                    self.model_name = None
                    logger.info(f"Model released due to timeout at {datetime.now()}")
            time.sleep(60)  # 每分钟检查一次
            


if __name__ == "__main__":
    model_name = 'BAAI/bge-m3'
    embed = EmbeddingModel()
    embed.load_model(model_name)
    sentences_1 = "What is BGE M3?"
    sentences_2 = ["What is BGE M3?", "What is BGE M4?"]
    sentences_3 = ["What is BGE M3?", "What is BGE M4?", "What is BGE M5?"]
    # 验证归一化
    import numpy as np
    def verify_normalization(embedding):
        norm = np.linalg.norm(embedding)
        print(f"向量长度: {norm}")
        print(f"是否接近 1: {np.isclose(norm, 1.0)}")
    embedding = embed.embedding(sentences_1)
    print(embedding)
    verify_normalization(np.array(embedding.data[0].embedding))
    embedding = embed.embed.encode(
                sentences_3, 
                batch_size=12, 
                max_length=None,
            )['dense_vecs']
    print(embedding)
    verify_normalization(np.array(embedding[0]))

