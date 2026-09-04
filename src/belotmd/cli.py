"""
cli.py — `belot-bot`, or `python -m belotmd`.
"""

import argparse
import asyncio
import sys

from .agents import available
from .config import Config


def _agent_kwarg(text):
    """Parse a --agent-arg KEY=VALUE pair. Values stay strings; an agent that
    wants a number can coerce it, and guessing types here would be worse."""
    key, sep, value = text.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError(
            f"expected KEY=VALUE, got {text!r}"
        )
    return key.strip(), value


def build_parser():
    known = ", ".join(available()) or "(none installed)"
    p = argparse.ArgumentParser(
        prog="belot-bot",
        description="Play Belot on belot.md with a pluggable agent.",
        epilog=(
            f"Installed agents: {known}. Agents from other packages are "
            f"discovered through the 'belotmd.agents' entry-point group -- "
            f"install the package and its name appears here. Credentials "
            f"come from BELOT_COOKIES; see .env.example, which also documents "
            f"the BELOT_* equivalent of every flag below."
        ),
    )
    p.add_argument("--agent", default=None, metavar="NAME",
                   help=f"decision-maker to use (default: random). "
                        f"Currently installed: {known}")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="convenience alias for --agent-arg checkpoint=PATH")
    p.add_argument("--agent-arg", action="append", default=[],
                   type=_agent_kwarg, metavar="KEY=VALUE",
                   help="pass an option through to the agent's factory; "
                        "repeatable")
    p.add_argument("--frames", dest="frames_path", default=None, metavar="PATH",
                   help="where to record raw state frames (default frames.jsonl)")

    audit = p.add_mutually_exclusive_group()
    audit.add_argument("--audit", dest="audit", action="store_true", default=None,
                       help="record frames and run the invariant auditor (default)")
    audit.add_argument("--no-audit", dest="audit", action="store_false",
                       help="run without recording or auditing")

    p.add_argument("--once", dest="reconnect", action="store_false",
                   default=None,
                   help="play one table and exit, instead of looking for "
                        "another when the table dissolves")
    p.add_argument("--retry-delay", dest="retry_delay_s", type=float,
                   default=None, metavar="SECONDS",
                   help="wait this long when there is nothing to join -- no "
                        "open tables, or we were kicked (default 300)")
    p.add_argument("--rejoin-delay", dest="rejoin_delay_s", type=float,
                   default=None, metavar="SECONDS",
                   help="wait this long after a match ends before finding "
                        "another table (default 5)")

    p.add_argument("--list-agents", action="store_true",
                   help="print the agents this environment can build, and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_agents:
        names = available()
        if not names:
            print("no agents installed", file=sys.stderr)
            return 1
        for name in names:
            print(name)
        return 0

    agent_kwargs = dict(args.agent_arg)
    if args.checkpoint is not None:
        agent_kwargs["checkpoint"] = args.checkpoint

    cfg = Config.from_env(
        agent=args.agent,
        frames_path=args.frames_path,
        audit=args.audit,
        reconnect=args.reconnect,
        retry_delay_s=args.retry_delay_s,
        rejoin_delay_s=args.rejoin_delay_s,
    )
    cfg.require_cookies()

    # Imported here so --help and --list-agents work without websockets.
    from .bot import LiveBelotBot

    try:
        bot = LiveBelotBot(config=cfg, agent_kwargs=agent_kwargs)
    except (ValueError, ImportError) as exc:
        print(f"belot-bot: {exc}", file=sys.stderr)
        return 2

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n[Bot] Interrupted; leaving the table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
