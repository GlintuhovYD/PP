import os
import subprocess
import sys
import random
import time

sizes = [250, 500, 1000, 1500, 2000]
procs_list = [1, 2, 4, 8]
exe = "lab1.exe"
py_script = "lab1.py"
results = []

def generate_matrices(n):
    with open("mat1.txt", "w") as f1, open("mat2.txt", "w") as f2:
        for _ in range(n):
            row1 = " ".join(str(random.randint(0, 99)) for _ in range(n))
            row2 = " ".join(str(random.randint(0, 99)) for _ in range(n))
            f1.write(row1 + "\n")
            f2.write(row2 + "\n")
    print("Generated {}x{} matrices".format(n, n))

def run_mpi(procs):
    subprocess.run(["srun", "-n", str(procs), exe], check=True)
    print("MPI finished with {} procs".format(procs))

def run_python():
    subprocess.run([sys.executable, py_script], check=True)
    print("Python finished")

def compare():
    def load(fname):
        with open(fname) as f:
            return [[float(x) for x in line.split()] for line in f if line.strip()]
    a = load("PYresult.txt")
    b = load("CPPresult.txt")
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        print("Mismatch: different dimensions")
        return False
    max_diff = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            diff = abs(a[i][j] - b[i][j])
            if diff > max_diff:
                max_diff = diff
    if max_diff < 1e-10:
        print("Match")
        return True
    else:
        print("Mismatch, max diff =", max_diff)
        return False

def get_time():
    import re
    with open("info.txt", "rb") as f:
        data = f.read()
    numbers = re.findall(rb"[-+]?\d*\.\d+|\d+", data)
    if numbers:
        return float(numbers[-1])
    return None

for size in sizes:
    for procs in procs_list:
        print("\n=== Size {} Processes {} ===".format(size, procs))
        generate_matrices(size)
        run_mpi(procs)
        run_python()
        if not compare():
            sys.exit(1)
        t = get_time()
        results.append((size, procs, t))
        time.sleep(0.5)

with open("summary.txt", "w") as f:
    f.write("Size\tProcesses\tTime(s)\n")
    for s, p, t in results:
        f.write("{}\t{}\t{}\n".format(s, p, t))
print("\nAll done. Results in summary.txt")