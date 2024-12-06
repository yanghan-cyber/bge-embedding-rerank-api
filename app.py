import argparse
import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from FlagEmbedding import FlagReranker
from pydantic import BaseModel, Field

from config import ACCESS_TOKEN, DEVICE, MODEL_TIMEOUT, logger, logging_config
from embedding import EmbeddingModel, EmbeddingRequest, EmbeddingResponse
from reranker import ReRanker, RerankRequest, RerankResponse

app = FastAPI()
security = HTTPBearer()


def check_model():
    ReRanker().check_timeout()
    EmbeddingModel().check_timeout()


# 启动一个线程来检查模型超时
threading.Thread(target=check_model, daemon=True).start()


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    request: RerankRequest,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    # 直接获取并打印请求的JSON内容
    request_data = request.json()
    logger.info(f"请求JSON内容:\n{json.dumps(request_data, indent=2)}")

    token = credentials.credentials
    if ACCESS_TOKEN is not None and token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:

        reranker = ReRanker()
        reranker.load_model(request.model)
        pairs = [[request.query, doc] for doc in request.documents]
        reranker_response = reranker.compute_score(pairs)
        # results.sort(key=lambda x: x.relevance_score, reverse=True)
        return reranker_response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"错误发生: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Reranking failed")


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embedding(
    request: EmbeddingRequest,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    # 直接获取并打印请求的JSON内容
    request_data = request.json()
    logger.info(f"请求JSON内容:\n{json.dumps(request_data, indent=2)}")

    token = credentials.credentials
    if ACCESS_TOKEN is not None and token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:

        embed = EmbeddingModel()
        embed.load_model(request.model)
        input_str = request.input
        embedding_response = embed.embedding(input_str)
        # results.sort(key=lambda x: x.relevance_score, reverse=True)
        return embedding_response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"错误发生: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding failed")


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="Your script description")

    # 添加 ACCESS_TOKEN 参数
    parser.add_argument(
        "--access_token",
        type=str,
        default=os.getenv("ACCESS_TOKEN", None),
        help="Access token for authentication",
    )

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    ACCESS_TOKEN = args.access_token

    logger.info("API正在启动...")
    logger.info(f"ACESS_TOKEN={ACCESS_TOKEN}")
    try:
        uvicorn.run(app, host="0.0.0.0", port=7701, log_config=logging_config)
    except Exception as e:
        logger.error(f"API启动失败！\n错误：{e}")
