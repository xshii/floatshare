#!/usr/bin/env python3
"""测试各数据源接口是否可用"""

import sys
from datetime import date, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_akshare():
    """测试 AKShare 数据源"""
    print("\n" + "=" * 50)
    print("测试 AKShare")
    print("=" * 50)

    try:
        import akshare as ak

        print(f"✓ akshare 版本: {ak.__version__}")

        # 测试股票列表
        print("\n[1] 获取A股列表...")
        df = ak.stock_zh_a_spot_em()
        print(f"✓ 获取到 {len(df)} 只股票")
        print(f"  示例: {df['代码'].head(3).tolist()}")

        # 测试日线数据
        print("\n[2] 获取日线数据 (000001)...")
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        df = ak.stock_zh_a_hist(
            symbol="000001",
            period="daily",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust=""
        )
        print(f"✓ 获取到 {len(df)} 条日线")
        if not df.empty:
            print(f"  最新: {df.iloc[-1]['日期']} 收盘 {df.iloc[-1]['收盘']}")

        # 测试指数成分股
        print("\n[3] 获取沪深300成分股...")
        df = ak.index_stock_cons_csindex(symbol="000300")
        print(f"✓ 获取到 {len(df)} 只成分股")

        # 测试分红数据
        print("\n[4] 获取分红数据 (000001)...")
        df = ak.stock_fhps_em(symbol="000001")
        print(f"✓ 获取到 {len(df)} 条分红记录")

        print("\n✅ AKShare 测试通过!")
        return True

    except ImportError:
        print("✗ akshare 未安装，运行: pip install akshare")
        return False
    except Exception as e:
        print(f"✗ AKShare 测试失败: {e}")
        return False


def test_tushare():
    """测试 Tushare 数据源"""
    print("\n" + "=" * 50)
    print("测试 Tushare")
    print("=" * 50)

    try:
        import tushare as ts

        print(f"✓ tushare 版本: {ts.__version__}")

        # 检查 token
        try:
            from config.settings import settings
            token = settings.TUSHARE_TOKEN
        except Exception:
            token = None

        if not token:
            print("⚠ 未配置 TUSHARE_TOKEN，跳过 Tushare 测试")
            print("  请在 config/settings.py 中设置 TUSHARE_TOKEN")
            return None

        pro = ts.pro_api(token)

        # 测试股票列表
        print("\n[1] 获取股票列表...")
        df = pro.stock_basic(exchange='', list_status='L')
        print(f"✓ 获取到 {len(df)} 只股票")

        # 测试日线数据
        print("\n[2] 获取日线数据 (000001.SZ)...")
        end_date = date.today().strftime("%Y%m%d")
        start_date = (date.today() - timedelta(days=30)).strftime("%Y%m%d")
        df = pro.daily(ts_code='000001.SZ', start_date=start_date, end_date=end_date)
        print(f"✓ 获取到 {len(df)} 条日线")

        # 测试复权因子
        print("\n[3] 获取复权因子...")
        df = pro.adj_factor(ts_code='000001.SZ', start_date=start_date, end_date=end_date)
        print(f"✓ 获取到 {len(df)} 条复权因子")

        print("\n✅ Tushare 测试通过!")
        return True

    except ImportError:
        print("✗ tushare 未安装，运行: pip install tushare")
        return False
    except Exception as e:
        print(f"✗ Tushare 测试失败: {e}")
        return False


def test_eastmoney():
    """测试东方财富数据源 (通过 AKShare)"""
    print("\n" + "=" * 50)
    print("测试 EastMoney (via AKShare)")
    print("=" * 50)

    try:
        import akshare as ak

        # 东方财富实时行情
        print("\n[1] 获取实时行情...")
        df = ak.stock_zh_a_spot_em()
        print(f"✓ 获取到 {len(df)} 只股票实时行情")

        # 东方财富历史数据
        print("\n[2] 获取历史K线 (sz000001)...")
        df = ak.stock_zh_a_hist(
            symbol="000001",
            period="daily",
            adjust="hfq"
        )
        print(f"✓ 获取到 {len(df)} 条历史K线")

        print("\n✅ EastMoney 测试通过!")
        return True

    except Exception as e:
        print(f"✗ EastMoney 测试失败: {e}")
        return False


def test_data_loader():
    """测试统一数据加载器"""
    print("\n" + "=" * 50)
    print("测试 DataLoader")
    print("=" * 50)

    try:
        from src.data.loader import DataLoader

        # 测试 AKShare 源
        print("\n[1] 通过 DataLoader 获取数据 (akshare)...")
        loader = DataLoader(source="akshare")

        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        df = loader.get_daily(
            code="000001.SZ",
            start_date=start_date,
            end_date=end_date
        )
        print(f"✓ 获取到 {len(df)} 条日线")
        if not df.empty:
            print(f"  列: {list(df.columns)}")

        # 测试股票列表
        print("\n[2] 获取股票列表...")
        df = loader.get_stock_list()
        print(f"✓ 获取到 {len(df)} 只股票")

        print("\n✅ DataLoader 测试通过!")
        return True

    except Exception as e:
        print(f"✗ DataLoader 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_source_pool():
    """测试多数据源池"""
    print("\n" + "=" * 50)
    print("测试 SourcePool (多数据源)")
    print("=" * 50)

    try:
        from src.data.syncer import SourcePool

        print("\n[1] 初始化多数据源...")
        pool = SourcePool(
            sources=["akshare", "eastmoney"],
            parallel=False
        )
        print(f"✓ 可用数据源: {pool.get_available_sources()}")

        print("\n[2] 获取数据 (自动降级)...")
        end_date = date.today()
        start_date = end_date - timedelta(days=10)

        df, source = pool.fetch_daily(
            code="000001.SZ",
            start_date=start_date,
            end_date=end_date
        )
        print(f"✓ 从 {source} 获取到 {len(df)} 条数据")

        print("\n[3] 数据源健康报告:")
        report = pool.get_health_report()
        for name, health in report.items():
            print(f"  {name}: 成功率={health['success_rate']}, 可用={health['is_available']}")

        print("\n✅ SourcePool 测试通过!")
        return True

    except Exception as e:
        print(f"✗ SourcePool 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 50)
    print("FloatShare 数据源接口测试")
    print("=" * 50)

    results = {}

    # 测试各数据源
    results["AKShare"] = test_akshare()
    results["Tushare"] = test_tushare()
    results["EastMoney"] = test_eastmoney()
    results["DataLoader"] = test_data_loader()
    results["SourcePool"] = test_source_pool()

    # 汇总
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    for name, result in results.items():
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⚠️ 跳过"
        print(f"  {name}: {status}")

    # 返回是否全部通过
    failures = [k for k, v in results.items() if v is False]
    if failures:
        print(f"\n❌ 有 {len(failures)} 项测试失败")
        return 1
    else:
        print("\n✅ 所有测试通过!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
