"""agent-flight-recorder: a black box for agents."""

from flight_recorder.diff import RunDiff, diff_runs
from flight_recorder.fake import ScriptedClient
from flight_recorder.log import Call, Run, read_run
from flight_recorder.recorder import RecordingClient
from flight_recorder.replay import RecordedAPIError, ReplayClient, ReplayDivergence

__all__ = [
    "Call",
    "RecordedAPIError",
    "RecordingClient",
    "ReplayClient",
    "ReplayDivergence",
    "Run",
    "RunDiff",
    "ScriptedClient",
    "diff_runs",
    "read_run",
]
__version__ = "0.1.0"
