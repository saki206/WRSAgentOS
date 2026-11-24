"""为抓取示例统一选择 Zenoh 地址、环境与本地凭据会话。"""

import os

PROFILES = {
    "wrs-demo": ("tcp/127.0.0.1:7449", "wrs-demo", "wrs"),
    "voice-goal": ("tcp/127.0.0.1:7451", "voice-goal", "voice"),
}


def connection_settings():
    """默认使用独立抓取示例，需要复用语音环境时显式选择对应配置。"""
    profile = os.environ.get("WRS_GRASP_PROFILE", "wrs-demo")
    if profile not in PROFILES:
        raise SystemExit("WRS_GRASP_PROFILE 应为 wrs-demo 或 voice-goal。")
    return PROFILES[profile]
