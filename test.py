from backtester.database_handler import borsdata_database
from backtester.strategies.strategy import TestStrategy
from datetime import datetime

def main():
    db = borsdata_database()
    dfs = db.load_data(country="Sverige",data_type="price",update=False)
    st = TestStrategy(insts=list(dfs.keys()),dfs=dfs,start=datetime(2015,1,1),end=datetime(2024,1,1))
    st.run_simulation()
    st.get_perf_stats(plot=True,show=True)
    

if __name__ == "__main__":
    main()
    