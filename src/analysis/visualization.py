"""可视化模块"""

from typing import Optional, List, Dict
import pandas as pd
import numpy as np


class ChartGenerator:
    """图表生成器"""

    def __init__(self):
        self._plt = None
        self._sns = None

    @property
    def plt(self):
        """延迟导入matplotlib"""
        if self._plt is None:
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
            plt.rcParams["axes.unicode_minus"] = False
            self._plt = plt
        return self._plt

    @property
    def sns(self):
        """延迟导入seaborn"""
        if self._sns is None:
            import seaborn as sns
            self._sns = sns
        return self._sns

    def plot_returns(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = "累计收益曲线",
        figsize: tuple = (12, 6),
    ):
        """绘制收益曲线"""
        fig, ax = self.plt.subplots(figsize=figsize)

        # 计算累计收益
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label="策略", linewidth=2)

        if benchmark is not None:
            benchmark_cum = (1 + benchmark).cumprod()
            ax.plot(benchmark_cum.index, benchmark_cum.values,
                   label="基准", linewidth=2, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("日期")
        ax.set_ylabel("累计收益")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_drawdown(
        self,
        returns: pd.Series,
        title: str = "回撤曲线",
        figsize: tuple = (12, 4),
    ):
        """绘制回撤曲线"""
        fig, ax = self.plt.subplots(figsize=figsize)

        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        ax.fill_between(drawdown.index, drawdown.values, 0,
                       color="red", alpha=0.3)
        ax.plot(drawdown.index, drawdown.values, color="red", linewidth=1)

        ax.set_title(title)
        ax.set_xlabel("日期")
        ax.set_ylabel("回撤")
        ax.grid(True, alpha=0.3)

        # 标注最大回撤
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown[max_dd_idx]
        ax.annotate(f"最大回撤: {max_dd_val:.2%}",
                   xy=(max_dd_idx, max_dd_val),
                   xytext=(10, 10), textcoords="offset points")

        return fig

    def plot_monthly_returns(
        self,
        returns: pd.Series,
        title: str = "月度收益热力图",
        figsize: tuple = (12, 8),
    ):
        """绘制月度收益热力图"""
        # 计算月度收益
        monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame(monthly)
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        monthly_df.columns = ["return", "year", "month"]

        # 创建透视表
        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        pivot.columns = [f"{m}月" for m in pivot.columns]

        fig, ax = self.plt.subplots(figsize=figsize)
        self.sns.heatmap(pivot, annot=True, fmt=".1%", center=0,
                        cmap="RdYlGn", ax=ax)
        ax.set_title(title)

        return fig

    def plot_distribution(
        self,
        returns: pd.Series,
        title: str = "收益分布",
        figsize: tuple = (10, 6),
    ):
        """绘制收益分布图"""
        fig, ax = self.plt.subplots(figsize=figsize)

        self.sns.histplot(returns, kde=True, ax=ax)

        # 添加统计信息
        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()

        stats_text = f"均值: {mean:.4f}\n标准差: {std:.4f}\n偏度: {skew:.2f}\n峰度: {kurt:.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment="top", horizontalalignment="right",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_title(title)
        ax.set_xlabel("日收益率")
        ax.set_ylabel("频数")

        return fig

    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 60,
        title: str = "滚动指标",
        figsize: tuple = (12, 10),
    ):
        """绘制滚动指标"""
        fig, axes = self.plt.subplots(3, 1, figsize=figsize)

        # 滚动年化收益
        rolling_return = returns.rolling(window).apply(
            lambda x: (1 + x).prod() ** (252 / len(x)) - 1
        )
        axes[0].plot(rolling_return.index, rolling_return.values)
        axes[0].set_title(f"{window}日滚动年化收益")
        axes[0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[0].grid(True, alpha=0.3)

        # 滚动波动率
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        axes[1].plot(rolling_vol.index, rolling_vol.values)
        axes[1].set_title(f"{window}日滚动波动率")
        axes[1].grid(True, alpha=0.3)

        # 滚动夏普
        rolling_sharpe = rolling_return / rolling_vol
        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[2].set_title(f"{window}日滚动夏普比率")
        axes[2].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[2].grid(True, alpha=0.3)

        self.plt.tight_layout()
        return fig

    def plot_positions(
        self,
        positions: pd.DataFrame,
        title: str = "持仓分布",
        figsize: tuple = (10, 6),
    ):
        """绘制持仓分布"""
        if positions.empty:
            return None

        # 最新持仓
        latest = positions.iloc[-1]
        latest = latest[latest > 0].sort_values(ascending=False)

        fig, ax = self.plt.subplots(figsize=figsize)
        ax.barh(latest.index, latest.values)
        ax.set_title(title)
        ax.set_xlabel("市值")

        return fig

    def create_report(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        save_path: Optional[str] = None,
    ):
        """生成完整报告"""
        fig = self.plt.figure(figsize=(16, 20))

        # 累计收益
        ax1 = fig.add_subplot(4, 2, 1)
        cumulative = (1 + returns).cumprod()
        ax1.plot(cumulative.index, cumulative.values, label="策略")
        if benchmark is not None:
            benchmark_cum = (1 + benchmark).cumprod()
            ax1.plot(benchmark_cum.index, benchmark_cum.values, label="基准")
        ax1.set_title("累计收益")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 回撤
        ax2 = fig.add_subplot(4, 2, 2)
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
        ax2.set_title("回撤")
        ax2.grid(True, alpha=0.3)

        # 日收益分布
        ax3 = fig.add_subplot(4, 2, 3)
        ax3.hist(returns, bins=50, density=True, alpha=0.7)
        ax3.set_title("日收益分布")

        # 滚动夏普
        ax4 = fig.add_subplot(4, 2, 4)
        rolling_return = returns.rolling(60).mean() * 252
        rolling_vol = returns.rolling(60).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values)
        ax4.axhline(y=0, color="r", linestyle="--")
        ax4.set_title("60日滚动夏普")
        ax4.grid(True, alpha=0.3)

        self.plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
