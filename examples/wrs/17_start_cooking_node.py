"""Start design's Runtime/dashboard with the original kitchen and atomic object skills."""
import asyncio
from pathlib import Path

from examples._session import use_local_token
from wrs_agent import System
from wrs_agent.env.wrs import ROOT


async def main():
    async with System.launch(
        backend="wrs", scene=ROOT / "wrs_agent/assets/kitchen/scene.toml",
        bindings=Path(__file__).parents[1] / "ui/bindings.toml",
        port=7451, env_id="wrs-composed", ui_port=8014, ui_viewer_port=8004,
    ) as system:
        names = list((await system.nodes())["wrs"]["skills"])
        print("基础技能：", names, flush=True)
        print("看板 http://127.0.0.1:8014/ · 仿真 http://127.0.0.1:8004/", flush=True)
        await asyncio.Event().wait()


if __name__ == "__main__":
    use_local_token("wrs", create=True)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
