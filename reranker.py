
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from FlagEmbedding import FlagReranker
from pydantic import BaseModel, Field

from config import DEVICE, MODEL_TIMEOUT, logger
from exception import ModelNotLoadedError
from singleton import Singleton


# 定义请求体模型
class RerankRequest(BaseModel):
    model: Optional[str] = Field("BAAI/bge-reranker-base", description="模型路径，如果未提供则使用默认模型")
    query: str
    documents: List[str]


# 定义响应体模型
class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str


class RerankResponse(BaseModel):
    id: str
    results: List[RerankResult]


class ReRanker(metaclass=Singleton):
    def __init__(self):
        self.reranker = None
        self.last_used = None
        self.model_name = None
        self._lock = threading.Lock()  # 防止并发问题

    def load_model(self, model_name):
        """Loads the reranker model if it's not already loaded or if the path has changed."""
        with self._lock:
            if self.reranker is None or self.model_name != model_name:
                logger.info(f"开始加载模型: {model_name} to {DEVICE}")
                self.reranker = FlagReranker(model_name, use_fp16=True, devices=DEVICE)
                self.model_name = model_name
            self.last_used = datetime.now()

    def compute_score(self, pairs: List[Tuple[str, str]]) -> RerankResponse:
        """Compute the relevance scores for document-query pairs and return a RerankResponse."""
        with self._lock:
            if self.reranker is None:
                raise ModelNotLoadedError()
            self.last_used = datetime.now()
            
            # 计算得分
            scores = self.reranker.compute_score(pairs, normalize=True)
            # 确保 pairs 和 scores 的长度匹配
            if len(pairs) != len(scores):
                raise ValueError("The length of pairs and computed scores do not match.")

            # 创建 RerankResult 列表
            results = [
                RerankResult(
                    index=i,
                    relevance_score=score,
                    document=pair[1]  # 假设 pair 是 (query, document)
                )
                for i, (pair, score) in enumerate(zip(pairs, scores))
            ]

            # 生成一个唯一的 id
            import uuid
            response_id = str(uuid.uuid4())

            # 返回 RerankResponse
            return RerankResponse(id=response_id, results=results)

    def check_timeout(self):
        """检查模型是否超时，并在超时时释放模型"""
        while True:
            with self._lock:
                if (
                    self.reranker
                    and self.last_used
                    and (datetime.now() - self.last_used)
                    > timedelta(minutes=MODEL_TIMEOUT)
                ):
                    self.reranker = None
                    self.model_name = None
                    logger.info(f"Model released due to timeout at {datetime.now()}")
            time.sleep(60)  # 每分钟检查一次


if __name__ == "__main__":
    reranker = ReRanker()
    reranker.load_model("BAAI/bge-reranker-large")
    pairs = [
    ['what is panda?', 'Today is a sunny day'], 
    ['what is panda?', 'The tiger (Panthera tigris) is a member of the genus Panthera and the largest living cat species native to Asia.'],
    ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
    ]
    scores = reranker.compute_score(pairs)
    print(scores)