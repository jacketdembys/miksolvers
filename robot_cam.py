import mujoco
from mujoco import viewer
import time

model = mujoco.MjModel.from_xml_path("results_models/mujoco_menagerie/franka_emika_panda/panda.xml")
data = mujoco.MjData(model)

# set a fixed pose
data.qpos[:7] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
mujoco.mj_forward(model, data)

with viewer.launch_passive(model, data) as v:
    print("\nMove the mouse to set the view you like.")
    print("Then press Ctrl+C in the terminal to print the current camera and exit.\n")

    try:
        while v.is_running():
            v.sync()
            time.sleep(1/60)
    except KeyboardInterrupt:
        cam = v.cam  # this is an mjvCamera
        print("\n=== COPY THESE INTO YOUR RENDER SCRIPT ===")
        print("azimuth =", cam.azimuth)
        print("elevation =", cam.elevation)
        print("distance =", cam.distance)
        print("lookat =", list(cam.lookat))
        print("=========================================\n")