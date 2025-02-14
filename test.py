from backtester.database_handler import borsdata_database, price_earnings_joiner
from backtester.strategies.strategy import TestStrategy
from datetime import datetime


def main():
    data_params = {
        "sector": ["Industri","Energi"]
    }
    db = borsdata_database(data_params=data_params)
    dfs_sverige = db.load_data(country="Sverige",data_type="price",update=False)
    dfs_norge = db.load_data(country="Norge",data_type="price",update=False)

    dfs_report_sv = db.load_data(country="Sverige",data_type="report",update=False)
    dfs_report_no = db.load_data(country="Norge",data_type="report",update=False)

    dfs_sverige = price_earnings_joiner(price_dfs=dfs_sverige,report_dfs=dfs_report_sv,select_cols=["earningsPerShare","numberOfShares"],rel_change="earningsPerShare")
    dfs_norge = price_earnings_joiner(price_dfs=dfs_norge,report_dfs=dfs_report_no,select_cols=["earningsPerShare","numberOfShares"],rel_change="earningsPerShare")

    dfs = dfs_sverige | dfs_norge

    st = TestStrategy(
        insts=list(dfs.keys()),
        dfs=dfs,
        start=datetime(2015,1,1),
        end=datetime(2024,1,1),
        use_portfolio_opt=True, 
        max_leverage=1.5
    )
    st.run_simulation()
    st.get_perf_stats(plot=True,show=True)

if __name__ == "__main__":
    main()
    