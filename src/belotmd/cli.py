"""
cli.py — `belot-bot`, or `python -m belotmd`.
"""

import argparse
import asyncio
import sys

from .agents import available, get_agent
from .config import Config


def build_parser():
    p = argparse.ArgumentParser(
        prog="belot-bot",
        description="Play Belot on belot.md with a pluggable agent.",
        epilog="Credentials come from BELOT_COOKIES (see .env.example). "
               "Every flag below also has a BELOT_* environment variable.",
    )
    p.add_argument("--agent", choices=available(), default=None,
                   help="decision-maker to use (default: ppo, or $BELOT_AGENT)")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="weights for a learned agent; searched in "
                        "checkpoints/ when omitted")
    p.add_argument("--frames", dest="frames_path", default=None, metavar="PATH",
                   help="where to record raw state frames (default frames.jsonl)")

    audit = p.add_mutually_exclusive_group()
    audit.add_argument("--audit", dest="audit", action="store_true", default=None,
                       help="record frames and run the invariant auditor (default)")
    audit.add_argument("--no-audit", dest="audit", action="store_false",
                       help="run without recording or auditing")

    p.add_argument("--list-agents", action="store_true",
                   help="print the registered agents and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_agents:
        for name in available():
            print(name)
        return 0

    cfg = Config.from_env(
        agent=args.agent,
        checkpoint=args.checkpoint,
        frames_path=args.frames_path,
        audit=args.audit,
    )
    cfg.require_cookies()

    # Imported here so `--list-agents` and `--help` work without websockets.
    from .bot import LiveBelotBot

    bot = LiveBelotBot(config=cfg)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n[Bot] Interrupted; leaving the table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
