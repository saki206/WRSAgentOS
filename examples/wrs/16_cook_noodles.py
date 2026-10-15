"""Run the noodle recipe as nine design-contract steps through the Runtime."""
import argparse
from pathlib import Path

from examples._session import use_local_token
from wrs_agent import connect
from wrs_agent.recipes.noodles import RECIPE_TIMEOUT, noodle_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--motion-speed", type=float, default=1.0)
    parser.add_argument("--group", choices=["all", "load", "cook", "serve"], default="all")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    use_local_token("wrs")
    with connect("tcp/127.0.0.1:7451", env_id="wrs-composed",
                 bindings=Path(__file__).with_name("bindings.toml")) as system:
        snapshot = system.snapshot(node="wrs")
        if snapshot.admission == "HELD" and snapshot.stop_confirmed:
            assert system.allow_actions(node="wrs").accepted
        if args.reset:
            assert system.action("reset_scene").wait().state == "SUCCEEDED"
        steps = noodle_steps(system.snapshot(node="wrs").data.objects,
                             seconds=args.seconds, motion_speed=args.motion_speed, group=args.group)
        result = system.start(*steps).wait(timeout=RECIPE_TIMEOUT)
        print("组合动作结果：", result.state)
        print("各步骤：", result.steps)
        if result.state != "SUCCEEDED":
            raise RuntimeError(result.reason)


if __name__ == "__main__":
    main()
