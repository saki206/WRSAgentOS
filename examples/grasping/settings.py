"""Shared Zenoh endpoint, environment, and local token session for grasping examples."""

import os

PROFILES = {
    "wrs-demo": ("tcp/127.0.0.1:7449", "wrs-demo", "wrs"),
    "voice-goal": ("tcp/127.0.0.1:7451", "voice-goal", "voice"),
}


def connection_settings():
    """Use the standalone grasping profile unless the voice profile is selected."""
    profile = os.environ.get("WRS_GRASP_PROFILE", "wrs-demo")
    if profile not in PROFILES:
        raise SystemExit("WRS_GRASP_PROFILE 应为 wrs-demo 或 voice-goal。")
    return PROFILES[profile]
