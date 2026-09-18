# HARNESS NOTES (opencode)

You are running inside opencode, without a human in the loop. Nobody will answer questions; make your own decisions and keep going until the task is complete.

Your working directory is `{{RUN_DIR}}`. Work only inside it, and do not read or search other directories.

Always run Python with `{{PYTHON}} <script.py>`. It already has numpy, scipy, OpenCV, scikit-image, Pillow, matplotlib and other scientific libraries installed. Do not install packages and do not use another interpreter.

To look at an image with your own vision, open the image file with the `read` tool. This is the only way to actually see an image; reading its pixel values in Python is not visual inspection.

This complete prompt is saved as `PROMPT.md` in the run directory.
