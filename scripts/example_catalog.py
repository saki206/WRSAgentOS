"""列出仓库的示例文件和 verify.py 检查的预期输出。"""

# 这些示例有可核对的终端输出，验收脚本据此判断运行结果。
WRS = {
    "nodes/06_late_node.py": ["直接调用： SUCCEEDED", "Agent 调度： SUCCEEDED"],
    "beginner/01_action.py": ["动作结果： SUCCEEDED", "当前位置： B"],
    "beginner/02_skills.py": ["move_named_pose 版本 1", "speak 版本 1"],
    "beginner/04_invalid_input.py": ["错误代码： invalid_arguments"],
    "tasks/01_sequence.py": ["任务结果： SUCCEEDED", "TCP 位置："],
    "tasks/02_parallel.py": ["正在执行的动作数量： 2", "任务结果： SUCCEEDED"],
    "tasks/03_cancel.py": ["旧任务： CANCELLED", "新任务： SUCCEEDED", "旧句柄仍指向： CANCELLED"],
    "tasks/04_watch.py": ["任务状态： RUNNING", "任务状态： SUCCEEDED"],
    "voice/01_text_stop_task.py": ["停止受理： True", "任务结果： CANCELLED"],
    "voice/02_text_stop_speech.py": ["播报结果： CANCELLED", "机器人结果： SUCCEEDED"],
    "voice/03_text_query.py": ["识别意图： query", "任务结果： SUCCEEDED"],
    "transport/01_query.py": ["节点： wrs", "当前位置： home"],
    "transport/02_events.py": ["收到事件：", "动作结果： SUCCEEDED"],
    "transport/03_submit.py": ["动作编号：", "动作结果： SUCCEEDED"],
    "transport/04_timeout.py": ["查询超时，没有取消任何动作。", "仍可查询机器人： home"],
    "wrs/01_move.py": ["动作结果： SUCCEEDED", "关节角：", "TCP 位置："],
    "wrs/02_cancel.py": ["动作结果： CANCELLED", "停止已确认： True"],
    "wrs/03_new_action_after_cancel.py": [
        "旧动作： CANCELLED",
        "允许新动作： True",
        "新动作： SUCCEEDED",
    ],
    "wrs/08_scene.py": ["场景坐标系： world", "物体： table", "物体： A", "碰撞规划已验证： False"],
    "wrs/04_move_relative.py": ["上 SUCCEEDED", "下 SUCCEEDED", "左 SUCCEEDED", "右 SUCCEEDED"],
}
# 服务端和客户端成对运行，由 test_developer_examples.py 一起验证。
PAIRED = {
    "wrs/05_start_node.py",
    "wrs/06_control_arm.py",
    "wrs/07_viewer.py",
    "nodes/00_start_router.py",
    "nodes/01_start_speaker.py",
    "nodes/02_start_agent.py",
    "nodes/03_call_skill.py",
    "nodes/04_task.py",
    "nodes/05_cancel.py",
    "camera/00_start_router.py", "camera/01_start_camera.py",
    "camera/04_start_agent.py", "camera/05_speak_objects.py",
}
# 需要音频、模型、设备或交互的示例由使用者显式启动，默认验收跳过。
LIVE = {
    "grasping/00_start_router.py", "grasping/01_start_camera.py",
    "grasping/02_start_locator.py", "grasping/03_start_fastsam.py",
    "grasping/04_start_graspnet.py", "grasping/05_start_agent.py", "grasping/06_propose_grasps.py",
    "models/01_plan.py", "models/02_execute.py",
    "voice/05_start_wrs_voice.py", "voice/06_push_to_talk.py",
    "voice/07_start_llm_voice.py", "voice/08_listen_goals.py", "voice/09_viewer.py",
    "tts/01_start_node.py", "tts/02_speak.py", "tts/03_cancel.py",
    "camera/06_read_frame.py", "camera/02_start_yolo.py", "camera/03_start_tts.py",
}
# 公共配置和辅助模块纳入文件清单，供其他示例导入。
SUPPORT = {"nodes/greet_skill.py", "voice/commands.py", "grasping/settings.py", "_session.py"}


def check_catalog(root):
    """检查示例目录和分类清单是否一致，防止新增文件漏登记。"""
    actual = {str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*.py")}
    expected = set(WRS) | PAIRED | LIVE | SUPPORT
    if actual != expected:
        raise ValueError(f"Example catalog mismatch: {actual ^ expected}")
