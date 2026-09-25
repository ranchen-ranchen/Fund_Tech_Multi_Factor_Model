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

from utils.text_utils import read_screened_company_codes
code_list = read_screened_company_codes(
        "policy_match_results.csv",
        "prosperity_results.csv")

from data.fetch_daily_k import get_daily_data

all_df = get_daily_data(code_list)







