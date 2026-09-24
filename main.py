import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', mode='w'),
#        logging.StreamHandler()
    ]
)

from utils.text_utils import fund_screen_prosperity, fund_screen_policy_match





