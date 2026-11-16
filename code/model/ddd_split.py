# DDD 留来源划分（训练/验证共用）: 固定种子随机留出 6 段困倦 + 6 段清醒来源视频
# 用法: from ddd_split import heldout_sources
import random

SEED = 42


def heldout_sources():
    drowsy_src = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                  'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'ZA', 'ZB', 'ZC']
    alert_src = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'x', 'y', 'za', 'zb', 'zc']
    random.seed(SEED)
    held = set(random.sample(drowsy_src, 6)) | set(random.sample(alert_src, 6))
    return held
