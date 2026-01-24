import numpy as np
import mujoco
from mujoco import viewer
import time
from PIL import Image



# Load model
model = mujoco.MjModel.from_xml_path("results_models/mujoco_menagerie/franka_emika_panda/panda.xml")
data = mujoco.MjData(model)

"""
# Set ONE fixed joint configuration (radians)
data.qpos[:7] = [
    0.0,
    -0.785,
    0.0,
    -2.356,
    0.0,
    1.571,
    0.785
]

# Forward kinematics
mujoco.mj_forward(model, data)

# Launch viewer and keep it open
with viewer.launch_passive(model, data) as v:
    while v.is_running():
        v.sync()
        time.sleep(1/60)   # keep UI responsive
"""


"""
# Fake test solutions (5)
q_nominal = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
solutions = q_nominal + 0.25 * np.random.randn(5, 7)
solutions = np.clip(solutions, model.jnt_range[:7, 0], model.jnt_range[:7, 1])

# Start at first solution
data.qpos[:7] = solutions[0]
mujoco.mj_forward(model, data)

with viewer.launch_passive(model, data) as v:
    i = 0
    while v.is_running():
        # Cycle through solutions so you can see diversity
        data.qpos[:7] = solutions[i % len(solutions)]
        mujoco.mj_forward(model, data)
        v.sync()
        time.sleep(0.8)
        i += 1




def alpha_composite(images, alpha=0.20):
    acc = np.zeros_like(images[0], dtype=np.float32)
    for im in images:
        acc = (1 - alpha) * acc + alpha * im.astype(np.float32)
    return np.clip(acc, 0, 255).astype(np.uint8)



# ---- Paste your camera values here (from the preview script output) ----
cam = mujoco.MjvCamera()
cam.azimuth = 135.0
cam.elevation = -20.0
cam.distance = 2.5
cam.lookat[:] = np.array([0.0, 0.0, 0.5])
# ----------------------------------------------------------------------

renderer = mujoco.Renderer(model, width=1024, height=768)
#renderer = mujoco.Renderer(model, width=640, height=480)

renderer.update_scene(data)

img = renderer.render()
print("min/max pixel:", img.min(), img.max(), "mean:", img.mean())

# 5 test solutions
q_nominal = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
solutions = q_nominal + 0.25 * np.random.randn(5, 7)
solutions = np.clip(solutions, model.jnt_range[:7,0], model.jnt_range[:7,1])

imgs = []
for q in solutions:
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)  # <- use the same camera
    imgs.append(renderer.render())

out = alpha_composite(imgs, alpha=0.20)
Image.fromarray(out).save("panda_overlay_5.png")
print("Saved panda_overlay_5.png")
"""




def overlay_sum(images, bg=255):
    acc = np.zeros_like(images[0], dtype=np.float32)
    for im in images:
        acc += (bg - im.astype(np.float32))
    out = bg - acc / len(images)
    return np.clip(out, 0, 255).astype(np.uint8)


renderer = mujoco.Renderer(model, width=1024, height=768)

# camera (set yours here)
cam = mujoco.MjvCamera()
cam.azimuth = 135
cam.elevation = -25
cam.distance = 2.5
cam.lookat[:] = np.array([0.0, 0.0, 0.4])

# 5 test solutions (replace later with MDN mu[0])
q_nominal = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
solutions = q_nominal + 0.25*np.random.randn(5, 7)
solutions = np.clip(solutions, model.jnt_range[:7,0], model.jnt_range[:7,1])

imgs = []
for q in solutions:
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    imgs.append(renderer.render())

out = overlay_sum(imgs, bg=255)
Image.fromarray(out).save("panda_overlay_5.png")
print("Saved panda_overlay_5.png")