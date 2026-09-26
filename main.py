import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='w'),
#        logging.StreamHandler()
    ]
)

# from utils.text_utils import fund_screen_prosperity, fund_screen_policy_match
# fund_screen_prosperity()

# from utils.text_utils import read_screened_company_codes
# code_list = read_screened_company_codes(
#         "policy_match_results.csv",
#         "prosperity_results.csv")
# print(code_list)
# from data.fetch_daily_k import get_daily_data
# all_df = get_daily_data(code_list)


from strategy.tech_analysis import get_stock_data, compute_trend
from pathlib import Path
data_dict = {}
folder = Path("stock_daily_k/individual")  # 例如 r"C:\data" 或 "/home/user/data"
for file in folder.glob("*.csv"):
    code = file.name.replace("_", ".")[:-4]
    data_dict[code] = compute_trend(get_stock_data(file), adx_threshold=20, adx_mode="shrink")

# print(data_dict)

from strategy.multi_stock_cross_section import PositionConfig, backtest_pool, pool_report
cfg = PositionConfig(
    top_n=50,
    min_mult=0.15,          # 打分 < 0.15 不进池
    base_risk_pct=0.008,    # 池化后单笔风险略降
    max_position_pct=0.02,  # 单票上限 = 1/top_n 
    cooldown_bars=3,
)

trades, equity = backtest_pool(data_dict, cfg, initial_equity=1_000_000)
pool_report(trades, equity, 1_000_000)





