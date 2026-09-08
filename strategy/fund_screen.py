
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
import yaml
from pathlib import Path



import logging
logger = logging.getLogger(__name__) 





# 加载 .env 文件中的环境变量
load_dotenv()
# 读取环境变量
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")







config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

fund_screen_threshold = config['FUND_SCREEN_THRESHOLD']





def business_prosperity_analysis(code: str, date: str) -> str:
    client = OpenAI(
    api_key=api_key,
    base_url=base_url)
    logger.info(f'business prosperity analysis for stock code: {code}')
    response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
                {"role": "system", "content": "你是一位A股市场股票投资专家"},
                {"role": "user", "content": f"{date}时期, 分析{code}的业务及其所在行业的景气度如何？给出0-10分的评价，评分前后用==分隔"}
            ],
                temperature=0,
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}}
    )
    response_text = response.choices[0].message.content
    logger.info(f'Response for {code}:\n {response_text}')
    return response_text




def policy_relevance_analysis(code: str, date: str) -> str:

    client = OpenAI(
        api_key=api_key,
        base_url=base_url)
    
    logger.info(f'policy relevance analysis for stock code: {code}')
    response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
            {"role": "system", "content": "你是一位A股市场股票投资专家"},
            {"role": "user", "content": f"{date}时期，国家政策激励的产业方向是什么？{code}的业务与政策热点贴合度如何？给出0-10分的评价，评分前后用==分隔"}
            ],
            temperature=0,
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
    )
    response_text = response.choices[0].message.content
    logger.info(f'Response for {code}:\n {response_text}')
    return response_text


def business_advantage_analysis(code: str, date: str) -> str:
    
    client = OpenAI(
            api_key=api_key,
            base_url=base_url)
    logger.info(f'business advantage analysis for stock code: {code}')
    response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
                {"role": "system", "content": "你是一位A股市场股票投资专家"},
                {"role": "user", "content": f"{date}时期，基于供给侧优势、需求侧优势、无形资产与壁垒、管理效率等角度分析{code}的独家优势，并给出0-10分的评价，评分前后用==分隔"}
                ],
                temperature=0,
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}}
    )
    response_text = response.choices[0].message.content
    logger.info(f'Response for {code}:\n {response_text}')
    return response_text


def get_fundamental_score(code: str, date: str) -> dict:
    """
    获取股票的基本面评分，包括政策相关性、业务优势和业务景气度。
    """
    policy_score_text = policy_relevance_analysis(code, date)
    business_advantage_score_text = business_advantage_analysis(code, date)
    business_prosperity_score_text = business_prosperity_analysis(code, date)
    from utils.text_utils import extract_score
    policy_score = extract_score(policy_score_text)
    business_advantage_score = extract_score(business_advantage_score_text)
    business_prosperity_score = extract_score(business_prosperity_score_text)

    return {
        "policy_relevance": policy_score[0] if policy_score else 0,
        "business_advantage": business_advantage_score[0] if business_advantage_score else 0,
        "business_prosperity": business_prosperity_score[0] if business_prosperity_score else 0
    }





def screen_stocks_by_fundamentals(code_list: list, date: str) -> list:
    """
    根据基本面评分筛选股票。
    """
    selected_code_list = []
    for code in code_list:
        score = get_fundamental_score(code, date)
        total_score = sum(score.values())
        if total_score > fund_screen_threshold:
            selected_code_list.append(code)
    return selected_code_list



if __name__ == "__main__":
    pass
