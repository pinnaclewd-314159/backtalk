# Restarts backtalk's voice line (backtalk.main) from OUTSIDE its own
# process tree, so it survives even when triggered by killing the very
# process this agent is running inside of (backtalk's WarmBrain spawns
# the Claude Code session that runs Jarvis - killing it mid-restart
# would otherwise take this script down too if it were a child of that
# tree). Always launched via a detached one-shot Scheduled Task, never
# as a direct child process - see JarvisVault/07 - Resources/Backtalk
# Voice Line.md ("Restarting mid-session") for why.
#
# Waits for the single-instance lock port to free up (main.py's
# _claim_single_instance binds 127.0.0.1:8791 and only releases it when
# the old process is fully gone, however it died), then relaunches in a
# new visible interactive window - backtalk needs desktop access for
# its mic, speaker, and global PTT hotkey, so this must NOT run headless
# (S4U) the way the backend listener tasks do.
param(
    [int]$LockPort = 8791,
    [string]$WorkDir = "C:\Users\JARVIS\my-agent\backtalk",
    [int]$TimeoutSec = 30
)

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse("127.0.0.1"), $LockPort)
        $listener.Start()
        $listener.Stop()
        break
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

Start-Process cmd.exe -ArgumentList '/c "uv run python -m backtalk.main"' -WorkingDirectory $WorkDir

try {
    Unregister-ScheduledTask -TaskName 'Jarvis - Backtalk Restart Helper' -Confirm:$false -ErrorAction Stop
} catch {}
