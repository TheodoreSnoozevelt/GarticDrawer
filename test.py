from subprocess import run
index = 0
levels = [ 1, 1, 1 ]
msg = f"Level {index+1}/{len(levels)}"
run(["echo", msg, "|", "xclip", "-i", "-selection", "clipboard"])

