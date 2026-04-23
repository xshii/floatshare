"""CLI wrapper — `floatshare-eval-ppo`. 负责 metrics bootstrap, 实际逻辑走 ml.eval.run_eval.

分层: ml 层不可 import application, 所以 bootstrap 放在这里 (cli 层).
"""

from __future__ import annotations

import os


def main() -> None:
    from floatshare.application.bootstrap import cli_metrics_run
    from floatshare.ml.eval import build_parser, run_eval

    args = build_parser().parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with cli_metrics_run():
        run_eval(args)


if __name__ == "__main__":
    main()
