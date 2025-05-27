from controller import Supervisor
import numpy as np
import sys, os

image_script = os.path.abspath(os.path.join(__file__, "..", "..", "..", ".."))
sys.path.append(image_script)

from create_map import create_png

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
pheromone_map = np.memmap('../main_skript/pheromone_array.dat', dtype='float32', mode='r+', shape=(10,10,2))
pheromone_image = robot.getFromDef("PheromoneImage").getField("url")
deadend_map = np.memmap('../main_skript/deadend_array.dat', dtype='uint8', mode='r+', shape=(10,10,4))
goal_pos = robot.getFromDef("Goal").getPosition()[:2]
sec_img = True

if __name__ == "__main__":
    simulation_step = 0
    while robot.step(timestep) != -1:
        simulation_step += 1
        if not simulation_step%10:
            pheromone_map.flush()
        if not simulation_step%100:
            create_png(pheromone_map, deadend_map, 10, (goal_pos[0]*2, goal_pos[1]*2), sec_img)
            pheromone_image.setMFString(0, f"../controllers/main_skript/maps/pheromone_map{1 if sec_img else 0}.png")
            sec_img = not sec_img
        pass