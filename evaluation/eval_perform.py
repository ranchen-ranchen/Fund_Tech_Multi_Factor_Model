
import pandas as pd
import numpy as np


import logging
logger = logging.getLogger(__name__) 



def vector_backtest(code: str, close_series: pd.Series, position_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    price_change_series = close_series.pct_change()
    return_series = position_series.shift(1) * price_change_series 
    logging.info(f'Stock Code: {code}')

    for i in range(len(close_series)):
        logging.info(f'{price_change_series.iloc[i]} | {position_series.iloc[i]} | {return_series.iloc[i]}')
    
    
    cum_price_change_series = (1 + price_change_series).cumprod()
    cum_return_series = (1 + return_series).cumprod()

    # 总收益率
    total_return = cum_return_series.iloc[-1] - 1
    # 日收益率序列
    cleaned_return_series = return_series.dropna()
    # 夏普比率（无风险利率为0,全年252个交易日）
    sharpe_ratio = np.sqrt(252) * cleaned_return_series.mean() / cleaned_return_series.std()
    # 最大回撤（向量化）
    running_max = cum_return_series.expanding().max()
    drawdown = (cum_return_series - running_max) / running_max
    max_drawdown = drawdown.min()
    # 胜率
    win_rate = (cleaned_return_series > 0).mean()
    # 输出结果
    logger.info("========== 回测绩效 ==========")
    logger.info(f"总收益率:        {total_return:.2%}")
    logger.info(f"夏普比率:        {sharpe_ratio:.2f}")
    logger.info(f"最大回撤:        {max_drawdown:.2%}")
    logger.info(f"胜率:            {win_rate:.2%}")
    logger.info("===============================")
    plot_dual_axis(x=close_series.index, y1=close_series, y2=cum_return_series, label1="close price", label2="cumulative return", title=code)
    return cum_price_change_series, cum_return_series







def plot_dual_axis(x, y1, y2,
                   label1="曲线 1", label2="曲线 2",
                   color1="tab:blue", color2="tab:red",
                   xlabel="X", ylabel1="Y1", ylabel2="Y2",
                   title='Price vs Cumulative Return',
                   linestyle1="-", linestyle2="-",
                   linewidth=1.5,
                   grid=True,
                   figsize=(8, 5),
                   save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    """
    在同一幅图上绘制两条曲线，分别使用左右两个 Y 轴。

    参数
    ----
    x        : 横坐标数据 (array-like)
    y1, y2   : 两条曲线的纵坐标数据 (array-like)
    label1   : 左轴曲线的图例名
    label2   : 右轴曲线的图例名
    color1   : 左轴曲线颜色
    color2   : 右轴曲线颜色
    xlabel   : X 轴标签
    ylabel1  : 左 Y 轴标签
    ylabel2  : 右 Y 轴标签
    title    : 图标题
    linestyle1/2 : 线型 ('-', '--', '-.', ':')
    linewidth: 线宽
    grid     : 是否显示网格
    figsize  : 图像大小
    save_path: 保存路径 (为 None 时不保存)
    """

    fig, ax1 = plt.subplots(figsize=figsize)

    # ---- 左轴曲线 ----
    line1, = ax1.plot(x, y1, color=color1, linestyle=linestyle1,
                      linewidth=linewidth, label=label1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # ---- 右轴曲线 ----
    ax2 = ax1.twinx()
    line2, = ax2.plot(x, y2, color=color2, linestyle=linestyle2,
                      linewidth=linewidth, label=label2)
    ax2.set_ylabel(ylabel2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # ---- 网格、标题、图例 ----
    if grid:
        ax1.grid(True, linestyle='--', alpha=0.5)

    plt.title(title)
    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines],
               loc='best', frameon=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
#    return fig, ax1, ax2




if __name__ == "__main__":
    pass
