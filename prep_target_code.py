# import logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('app.log', mode='w'),
# #        logging.StreamHandler()
#     ]
# )

from utils.text_utils import read_hs300_constituents, read_business_description
from strategy.industry_chain_classifier import is_ai_compute
target_code_dict = {}
for date in ['2021-01-01', 
             '2021-06-30', '2021-12-31',
             '2022-06-30', '2022-12-31',
             '2023-06-30', '2023-12-31',
             '2024-06-30', '2024-12-31',
             '2025-06-30', '2025-12-31']:
    hs300_code_list = read_hs300_constituents('data/hs300_constituents_2021_2026.csv', date)
    target_code = ""
    for code in hs300_code_list:
        des = read_business_description(stock_code = code, csv_path = 'data/stock_main_business/main_business.csv')
        if is_ai_compute(business_description=des):
            target_code = target_code + code + " "
    target_code_dict[date] = target_code
import csv
with open("target_code.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["query_date", "target_code"])
    for k, v in target_code_dict.items():
        writer.writerow([k, v])

