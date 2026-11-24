"""Agent node: own the model and Runtime, bind their existing RPC contract."""

import os

from pydantic import model_validator

from wrs_agent.nodes.agent.rpc import register_runtime
from wrs_agent.nodes.node import Node
from wrs_agent.planner import ModelPlanner
from wrs_agent.runtime import Runtime
from wrs_agent.schemas import Boundary


class AgentOptions(Boundary):
    live_model: bool = False
    voice_part_grasp: bool = False

    @model_validator(mode="after")
    def check_voice_part_grasp(self):
        if self.voice_part_grasp and not self.live_model:
            raise ValueError("voice_part_grasp_requires_live_model")
        return self


class AgentNode(Node):
    node_type = "agent"
    launch_options = {"live_model": "live_model", "voice_part_grasp": "voice_part_grasp"}
    options_type = AgentOptions
    features = ("task.coordinate",)

    async def setup(self):
        planner = None
        if self.options.live_model:
            from wrs_agent.planner.providers.llm import LLMClient, LLMConfig

            model = LLMClient(LLMConfig.from_env(), live_model=True)
            self.on_close(model.aclose)
            planner = ModelPlanner(model)
        if self.options.voice_part_grasp:
            from wrs_agent.planner.grasp_parts import VoiceGraspPlanner
            from wrs_agent.planner.providers.part_grounding import (
                PartGroundingClient,
                part_model_config,
                part_model_key,
                part_settings,
            )

            settings = part_settings(os.environ.get("WRS_PART_LLM_ENV_FILE"))
            config = part_model_config(settings)
            part_client = PartGroundingClient(config, part_model_key(config, settings))
            self.on_close(part_client.aclose)
            planner = VoiceGraspPlanner(part_client, planner)

        registry = self.discover()
        registry.local[self.node_id] = self.info
        runtime = Runtime(
            registry.clients,
            self.skill_bindings,
            planner=planner,
            registry=registry,
        )
        self.on_close(runtime.close)
        register_runtime(self.transport, runtime)
