import os, sys, time, signal
import subprocess
from threading import Timer

process_pids = []

def CreateProcess(cmd, t_process, t_output=1):
    proc = subprocess.Popen(cmd, shell=True,
    # proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("New subprocess (pid = %d) is created, terminate in %d seconds." %(proc.pid, t_process))
    print("Command: %s" %(cmd))
    process_pids.append(proc.pid)
    proc_timer = Timer(t_process, KillProcess, [proc.pid])
    proc_timer.start()
    output_timer = Timer(t_output, Output, [proc])
    output_timer.start()

def Output(proc):
    outs, _ = proc.communicate()
    # print('== subprocess exited with rc =', proc.returncode)
    print(outs.decode('utf-8'))

def KillProcess(proc_pid):
    try:
        os.killpg(proc_pid, signal.SIGTERM)
        print("Subprocess %d terminated." %(proc_pid))
    except OSError as e:
        print("Subprocess %d is already terminated." %(proc_pid))
    process_pids.remove(proc_pid)

def Exiting():
    # Cleanup subprocess is important!
    print("Cleaning ... ")
    for pid in process_pids:
        KillProcess(proc_pid=pid)