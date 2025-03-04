import mujoco.viewer

m = mujoco.MjModel.from_xml_path('../../../../mujoco/mujoco_menagerie/unitree_go1/scene.xml')
v = mujoco.viewer.launch(m)
