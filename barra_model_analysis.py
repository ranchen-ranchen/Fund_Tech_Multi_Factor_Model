from utils.text_utils import get_hs300etf_close, read_hs300_constituents
df_bench = get_hs300etf_close('hs300etf_510300_performance.csv')
import pandas as pd
from pprint import pprint

date_list = ['2021-01-01', '2021-06-30', '2021-12-31', 
        '2022-06-30', '2022-12-31', 
        '2023-06-30', '2023-12-31', 
        '2024-06-30', '2024-12-31',
        '2025-06-30', '2025-12-31',]
all_code_list = []
for date in date_list:
    all_code_list.append(read_hs300_constituents(query_date = date, csv_file = 'data/hs300_constituents_2021_2026.csv'))


weight_file_list = [
    'hs300_weight_2021h1.csv', 'hs300_weight_2021h2.csv',
    'hs300_weight_2022h1.csv', 'hs300_weight_2022h2.csv',
    'hs300_weight_2023h1.csv', 'hs300_weight_2023h2.csv',
    'hs300_weight_2024h1.csv', 'hs300_weight_2024h2.csv',
    'hs300_weight_2025h1.csv', 'hs300_weight_2025h2.csv']


# # data_code_list = list(dict.fromkeys(x for sub in all_code_list for x in sub))
# # from data.fetch_daily_k import get_daily_data
# # get_daily_data(codes = data_code_list,
# #                 adjustflag = '3', ## 补充不复权的数据
# #                 start_date = date_list[0],
# #                 end_date = date_list[-1],
# #                 out_dir = "stock_daily_k_bfq/") 
# # from utils.text_utils import query_pb_mrq
# # for i in range(len(date_list)):
# #     print(date_list[i])
# #     df_pb = query_pb_mrq(csv_path = 'data/hs300_valuation_6m.csv',
# #         stock_codes = all_code_list[i],
# #         target_date = date_list[i])
# #     print(df_pb)

from utils.text_utils import normalize_code, get_stock_data, query_pb_mrq, read_hs300_weight, read_target_code_weight
target_code_weight_df_list = read_target_code_weight('target_code.csv')

periods = []
for i in range(len(date_list)-1):
    start_date = date_list[i]
    end_date = date_list[i+1]
    code_list = [normalize_code(code) for code in all_code_list[i]]
    file_hfq_list = [f"stock_daily_k_hfq/individual/{code.replace('.', '_')}.csv" for code in code_list]
    df_hfq_list = [get_stock_data(file) for file in file_hfq_list]
    file_bfq_list = [f"stock_daily_k_bfq/individual/{code.replace('.', '_')}.csv" for code in code_list]
    df_bfq_list = [get_stock_data(file) for file in file_bfq_list]
    ## compile data into a dictionary for each period
    tmp_dict = {}
    tmp_dict['name'] = f"period_{i+1}"
    tmp_dict['df_hfq_list'] = df_hfq_list
    tmp_dict['df_bfq_list'] = df_bfq_list
    tmp_dict['code_list'] = code_list
    tmp_dict['start_date'] = start_date
    tmp_dict['end_date'] = end_date
    tmp_dict['df_pb'] = query_pb_mrq(csv_path = 'data/hs300_valuation_6m.csv',
    stock_codes = code_list,
    target_date = end_date)
    ## calculate the weight difference between the strategy and the benchmark
    hs300_weight_df = read_hs300_weight(weight_file_list[i])
    target_weight_df = target_code_weight_df_list[i]
    weight_diff_df = pd.DataFrame(0.0, index=hs300_weight_df.index, 
                                  columns=hs300_weight_df.columns)
    for idx in weight_diff_df.index:
        target_weight = target_weight_df.at[idx, 'weight'] if idx in target_weight_df.index else 0.0
        hs300_weight = hs300_weight_df.at[idx, 'weight'] / 100.0
        weight_diff_df.at[idx, 'weight'] = target_weight - hs300_weight

    # print(f"The shape of the weight difference matrix: {weight_diff_df.shape}")

    tmp_dict['weight_diff_df'] = weight_diff_df
    periods.append(tmp_dict)


from strategy.calc_volatility_exposure import compute_volatility_exposure
from strategy.calc_momentum_exposure import momentum_exposure_histogram
from strategy.calc_size_factor_exposure import calc_size_factor_exposure
from strategy.calc_excess_return import calc_excess_return_distribution
from strategy.calc_value_exposure import value_factor_exposure

###############
## test the nav calculation of the strategy is correct ##
###############
# test_nav = 1.0
# for i in range(len(target_code_weight_df_list)):
#     df_excess_return, ax = calc_excess_return_distribution(
#         benchmark_df=df_bench, 
#         stock_dfs = periods[i]['df_hfq_list'], 
#         start_date=periods[i]['start_date'], 
#         end_date=periods[i]['end_date'], 
#         stock_names=periods[i]['code_list'], 
#         return_df=True)
#     df_excess_return = df_excess_return.set_index('stock')
#     print(f"Period {i+1}: {periods[i]['start_date']} to {periods[i]['end_date']}")
#     tmp = 0.0
#     for idx in target_code_weight_df_list[i].index:
#         normalized_idx = normalize_code(idx)
#         weight = target_code_weight_df_list[i].loc[idx, 'weight']
#         stock_return = df_excess_return.loc[normalized_idx, 'stock_return']
#         tmp += weight * (1 + stock_return)
#     test_nav *= tmp
#     print(f"Test NAV after period {i+1}: {test_nav:.4f}")


###########
## calculate the factor exposure and factor return
###########
periods_exposure_return = []

for i in range(len(periods)):
    df_value_factor = value_factor_exposure(
        df = periods[i]['df_pb'],
        target_date = periods[i]['end_date'])
    df_value_factor = df_value_factor.set_index('code')

    df_volatility_factor, fig = compute_volatility_exposure(
        stock_data=periods[i]['df_hfq_list'],
        start_date=periods[i]['start_date'],
        end_date=periods[i]['end_date'],
        stock_names=periods[i]['code_list'],
        annualize=False)
    df_volatility_factor = df_volatility_factor.set_index('stock')

    df_momentum_factor = momentum_exposure_histogram(
        price_dfs=periods[i]['df_hfq_list'],
        start_date=periods[i]['start_date'],
        end_date=periods[i]['end_date'],
        names=periods[i]['code_list'],
        skip_days=0,        # 若想剔除最近 5 个交易日，可设为 5
        winsorize=True,
        return_fig=False)
    df_momentum_factor = df_momentum_factor.set_index('stock')

    df_size_factor, ax = calc_size_factor_exposure(
        data = {code: df for code, df in zip(periods[i]['code_list'], periods[i]['df_bfq_list'])},
        date = periods[i]['end_date'])
    df_size_factor = df_size_factor.set_index('code')

    df_excess_return, ax = calc_excess_return_distribution(
        benchmark_df=df_bench, 
        stock_dfs = periods[i]['df_hfq_list'], 
        start_date=periods[i]['start_date'], 
        end_date=periods[i]['end_date'], 
        stock_names=periods[i]['code_list'], 
        return_df=True)
    df_excess_return = df_excess_return.set_index('stock')


    ret = pd.DataFrame(0.0, index=periods[i]['code_list'], columns=['return'])
    mcap = pd.DataFrame(0.0, index=periods[i]['code_list'], columns=['market_cap'])
    X = pd.DataFrame(0.0, index=periods[i]['code_list'], columns=['size', 'value', 'momentum', 'volatility'])

    for j in range(len(periods[i]['code_list'])):
        code = periods[i]['code_list'][j]
        # print(f"Code: {code}")
        ret.loc[code, 'return'] = df_excess_return.loc[code, 'excess_return']
        mcap.loc[code, 'market_cap'] = df_size_factor.loc[code, 'float_mv']
        X.loc[code, 'size'] = df_size_factor.loc[code, 'size_exposure']
        X.loc[code, 'value'] = df_value_factor.loc[code, 'value_z']
        X.loc[code, 'momentum'] = df_momentum_factor.loc[code, 'momentum_z']
        X.loc[code, 'volatility'] = df_volatility_factor.loc[code, 'vol_exposure']

    from strategy.factor_return_cross_sec_regression import build_risk_model_cross_section
    fac_ret, spec_ret = build_risk_model_cross_section(
            market_cap=mcap,
            stock_return=ret,
            factor_exposure=X,
            weight_power=0.5)

    tmp_dict = {}
    tmp_dict['factor_exposure'] = X
    tmp_dict['fac_ret'] = fac_ret
    tmp_dict['spec_ret'] = spec_ret
    periods_exposure_return.append(tmp_dict)

    # print(f"The shape of factor exposure matrix: {X.shape}")
    # print(f"The shape of factor return matrix: {fac_ret.shape}")
    # print(f"The shape of speccific return matrix: {spec_ret.shape}")
###################
### partition the excess return of the strategy to factor contributions
###################
periods_contrib = []
for i in range(len(periods)):
    print(f"Period {i+1}: {periods[i]['start_date']} to {periods[i]['end_date']}")
    A = periods[i]['weight_diff_df']
    B = periods_exposure_return[i]['factor_exposure']
    strategy_factor_exposure = pd.DataFrame(
        A.to_numpy().T @ B.to_numpy(),
        columns=B.columns)
    pprint(strategy_factor_exposure)
    pprint(periods_exposure_return[i]['fac_ret'])
    contrib = strategy_factor_exposure.mul(periods_exposure_return[i]["fac_ret"]['factor_return'], axis=1)
    pprint(contrib)
    print(f"factor contribution in total: {contrib.sum(axis=1)}")


    df_excess_return, ax = calc_excess_return_distribution(
        benchmark_df=df_bench, 
        stock_dfs = periods[i]['df_hfq_list'], 
        start_date=periods[i]['start_date'], 
        end_date=periods[i]['end_date'], 
        stock_names=periods[i]['code_list'], 
        return_df=True)
    df_excess_return = df_excess_return.set_index('stock')
    excess_return = 0.0
    for idx in target_code_weight_df_list[i].index:
        normalized_idx = normalize_code(idx)
        weight = target_code_weight_df_list[i].loc[idx, 'weight']
        tmp_excess_return = df_excess_return.loc[normalized_idx, 'excess_return']
        excess_return += weight * tmp_excess_return
    print(f"the total excess return of the strategy: {excess_return: .4f}")

    contrib['total'] = excess_return
    pprint(contrib)
    periods_contrib.append(contrib.iloc[0].tolist())


from utils.plot_utils import plot_clustered_bars
plot_clustered_bars(data=periods_contrib, output_path='factor_contrib_barplot.png')








### previous test ###

# from strategy.calc_value_exposure import value_factor_exposure
# import matplotlib.pyplot as plt
# results = []
# for i in range(len(periods)):
#     result_df = value_factor_exposure(
#     df = periods[i]['df_pb'],
#     target_date = periods[i]['end_date'],
#     )
#     results.append(result_df)
#     plt.savefig(f"value_exposure_{i+1}.png", dpi=150)
# all_result = pd.concat(
#     results, axis=0,
#     keys=[f"batch_{i}" for i in range(len(results))],
#     names=["batch", "stock"],
# )
# print(all_result)


# from strategy.calc_volatility_exposure import compute_volatility_exposure
# import matplotlib.pyplot as plt
# results = []
# for i in range(len(periods)):
#     result_df, fig = compute_volatility_exposure(
#     stock_data=periods[i]['df_list'],
#     start_date=periods[i]['start_date'],
#     end_date=periods[i]['end_date'],
#     stock_names=periods[i]['code_list'],
#     annualize=False,
#     standardize=True,
#     plot=True,
#     bins=20,
#     )
#     results.append(result_df)
#     plt.savefig(f"volatility_exposure_{i+1}.png", dpi=150)
# all_result = pd.concat(
#     results, axis=0,
#     keys=[f"batch_{i}" for i in range(len(results))],
#     names=["batch", "stock"],
# )
# print(all_result)





# from strategy.calc_momentum_exposure import momentum_exposure_histogram
# import matplotlib.pyplot as plt
# results = []
# for i in range(len(periods)):
#     result_df, fig = momentum_exposure_histogram(
#         price_dfs=periods[i]['df_list'],
#         start_date=periods[i]['start_date'],
#         end_date=periods[i]['end_date'],
#         names=periods[i]['code_list'],
#         skip_days=0,        # 若想剔除最近 5 个交易日，可设为 5
#         winsorize=True,
#         bins=10,
#         figsize=(10, 6),
#         return_fig=True,
#     )
#     results.append(result_df)
#     plt.savefig(f"momentum_exposure_{i+1}.png", dpi=150)

# all_result = pd.concat(
#     results, axis=0,
#     keys=[f"batch_{i}" for i in range(len(results))],
#     names=["batch", "stock"],
# )
# print(all_result)


# from strategy.calc_size_factor_exposure import calc_size_factor_exposure
# import matplotlib.pyplot as plt
# results = []
# for i in range(len(periods)):
#     result_df, ax = calc_size_factor_exposure(data = {code: df for code, df in zip(periods[i]['code_list'], periods[i]['df_list'])},
#         date = periods[i]['end_date'])
#     results.append(result_df)
#     plt.tight_layout()
#     plt.savefig(f"size_factor_exposure_{i+1}.png", dpi=150)

# all_result = pd.concat(
#     results, axis=0,
#     keys=[f"batch_{i}" for i in range(len(results))],
#     names=["batch", "stock"],
# )
# print(all_result)



# import matplotlib.pyplot as plt
# from strategy.calc_excess_return import calc_excess_return_distribution
# results = []
# for i in range(len(periods)):
#     result_df, ax = calc_excess_return_distribution(benchmark_df=df_bench, stock_dfs = periods[i]['df_list'], 
#         start_date=periods[i]['start_date'], end_date=periods[i]['end_date'], stock_names=periods[i]['code_list'], return_df=True)
#     results.append(result_df)
#     plt.tight_layout()
#     plt.savefig(f"excess_return_hist_{i+1}.png", dpi=150)
# #    plt.show()

# # 不同批次股票名可能重复，加 keys 区分批次
# all_result = pd.concat(
#     results, axis=0,
#     keys=[f"batch_{i}" for i in range(len(results))],
#     names=["batch", "stock"],
# )
# print(all_result)









