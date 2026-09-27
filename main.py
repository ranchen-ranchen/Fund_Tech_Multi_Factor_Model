# import logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('app.log', mode='w'),
# #        logging.StreamHandler()
#     ]
# )

from utils.text_utils import read_hs300_constituents
df_hs300 = read_hs300_constituents('data/hs300_constituents_2021_2026.csv')
hs300_list = df_hs300.loc[df_hs300['query_date']=='2021-01-01']['code'].tolist()

# from data.fetch_daily_k import get_daily_data
# all_df = get_daily_data(hs300_list)

from utils.text_utils import read_screened_company_codes
screened_list = read_screened_company_codes(
        "policy_match_results.csv",
        "prosperity_results.csv")


def backtest(code_list):
    from strategy.multi_stock_cross_section import PositionConfig, backtest_pool, pool_report
    from strategy.tech_analysis import get_stock_data, compute_trend
    data_dict = {}
    for code in code_list:
        if code.startswith(("60", "68", "51", "58", "11", "90", "50")):
            filename = f"stock_daily_k/individual/sh_{code}.csv"
        if code.startswith(("00", "30", "12", "15", "16", "18", "20", "39", "13")):
            filename = f"stock_daily_k/individual/sz_{code}.csv"
        if code.startswith(("43", "83", "87", "88", "92")):
            filename = f"stock_daily_k/individual/bj_{code}.csv"

        data_dict[code] = compute_trend(get_stock_data(filename), adx_threshold=20, adx_mode="shrink")
    
    trades, equity = backtest_pool(data_dict, initial_equity=1_000_000)
    pool_report(trades, equity, 1_000_000)


print('=='*20)
print('stock pool of all hs300')
backtest(hs300_list)
print('=='*20)

# print('=='*20)
# print('stock pool of screened stocks')
# backtest(screened_list)
# print('=='*20)

