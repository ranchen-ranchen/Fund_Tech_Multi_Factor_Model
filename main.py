import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='w'),
#        logging.StreamHandler()
    ]
)



    

# from data.query_data import query_stock_code_bs, query_stock_price_bs
# date = '2023-06-30'
# code = query_stock_code_bs(date)['code']
# test_code = code[0]

# data = query_stock_price_bs(code = test_code, start_date = '2023-01-01', end_date = '2023-06-30')

######

# from strategy.fund_screen import screen_stocks_by_fundamentals
# test_code_list = ['sh.60000', 'sz.300308', 'sh.600001', 'sz.000709']
# selected_code_list = screen_stocks_by_fundamentals(test_code_list, date='2023-03-31', threshold=20)


def test(filename):
    from utils.text_utils import read_from_csv
    date_series, open_series, close_series, high_series, low_series, amount_series = read_from_csv(filename)
    # print(close_series.shape)

    from strategy.tech_signal import change_position_by_tech_analysis
    position_series = change_position_by_tech_analysis(date_series, close_series, high_series, low_series, amount_series)
    # print(position_series.shape)

    from evaluation.eval_perform import vector_backtest
    cum_price_change_series, cum_return_series = vector_backtest(filename, close_series, position_series)
    # print(cum_price_change_series.shape, cum_return_series.shape)


####
# filename = 'data/data_sh.688589_stock_price.csv'
# filename = 'data/data_sh.600584_stock_price.csv'
# filename = 'data/data_sh.603986_stock_price.csv'
# test(filename)

import os 
for filename in os.listdir("data"):
    if filename.startswith('data') and filename.endswith('.csv'):
        test('data/' + filename)




# for i in range(len(date_series)):
#     logging.info(f'Date: {date_series.iloc[i]} | cum price change : {cum_price_change_series.iloc[i]} | cum return : {cum_return_series.iloc[i]}')



