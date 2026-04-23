"""本地 NLP — 不走外部 API, 用于新闻联播关键字匹配等.

cctv_local     : 加载 data/news/industry_keywords.json + 对文本做行业匹配
keyword_config : 关键词词典的 Python API (get_keywords / lint_dictionary 等)
"""

from floatshare.infrastructure.nlp.cctv_local import (
    INDUSTRY_BASELINE_DIR as INDUSTRY_BASELINE_DIR,
)
from floatshare.infrastructure.nlp.cctv_local import (
    INDUSTRY_BASELINE_PATH as INDUSTRY_BASELINE_PATH,
)
from floatshare.infrastructure.nlp.cctv_local import (
    INDUSTRY_KEYWORDS_PATH as INDUSTRY_KEYWORDS_PATH,
)
from floatshare.infrastructure.nlp.cctv_local import (
    extract_industry_mentions as extract_industry_mentions,
)
from floatshare.infrastructure.nlp.cctv_local import (
    load_industry_baseline as load_industry_baseline,
)
from floatshare.infrastructure.nlp.cctv_local import (
    load_industry_keywords as load_industry_keywords,
)
from floatshare.infrastructure.nlp.keyword_config import (
    KeywordLintIssue as KeywordLintIssue,
)
from floatshare.infrastructure.nlp.keyword_config import (
    find_conflicts as find_conflicts,
)
from floatshare.infrastructure.nlp.keyword_config import (
    get_industry_name as get_industry_name,
)
from floatshare.infrastructure.nlp.keyword_config import (
    get_keywords as get_keywords,
)
from floatshare.infrastructure.nlp.keyword_config import (
    lint_dictionary as lint_dictionary,
)
from floatshare.infrastructure.nlp.keyword_config import (
    list_industries as list_industries,
)
