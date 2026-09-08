# The struture of the program
quant_program/
├── main.py
├── data/
│   ├── query_data.py          
│   └── save_update_data.py    
├── strategy/
│   ├── fund_screen.py         # load fundamental data and screen stocks -> a list of screened stocks
│   └── tech_signal.py         # load technical/market data, calculate factors, generate signals(buy, sell, hold) -> log files of trading signals for the screened stocks
├── evaluation/
│   └── eval_perform.py        # load the log files, calculate gain / loss -> the performace report
├── utils/
│   ├── logger.py              
│   └── config_loader.py       
└── tests/
    ├── test_fund_screen.py
    └── test_tech_signal.py






