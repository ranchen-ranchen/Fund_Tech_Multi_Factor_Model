
from utils.text_utils import read_screened_company_codes
overlap = read_screened_company_codes(
        "policy_match_results.csv",
        "prosperity_results.csv")
print(overlap)
print("重叠数量:", len(overlap))