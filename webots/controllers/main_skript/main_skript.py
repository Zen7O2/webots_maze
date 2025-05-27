from controller import Supervisor
from controller import Emitter
from controller import Receiver
import math
import numpy as np
import portalocker
import random

# initialize and get robot informations
robot = Supervisor()
bot_id = int(str(robot.getName()).removeprefix("Thymio"))
bot = robot.getFromDef(robot.getName())
goal_pos = robot.getFromDef("Goal").getPosition()[:2]
bot_rotation_field = bot.getField("rotation")
bodySlot_field = bot.getField("bodySlot")
pheromone_map = np.memmap('pheromone_array.dat', dtype='float32', mode='r+', shape=(10,10,2))
deadend_map = np.memmap('deadend_array.dat', dtype='uint8', mode='r+', shape=(10,10,4))

# shortcut functions
bot_pos = bot.getPosition
goal_dist = lambda x: math.dist(goal_pos, x)
bot_rotation = bot_rotation_field.getSFRotation

# global variables
timestep = int(robot.getBasicTimeStep())
cell_size = 0.5 # sidelength of square pheromone cell in meters
max_speed = 9.53
decay = 0.01
sensor_offsets = 0.337 # angle (in deg) between sensors
sensor_angles = [0.674,0.337,0,-0.337,-0.674]
random.seed(1)

# completely defective parts (no input or output)
# every relevant function first checks if it should execute or not
left_motor_failure = bot_id in []
right_motor_failure = bot_id in []
phero_reader_failure = bot_id in []
phero_writer_failure = bot_id in []
wall_sensor_failure = bot_id in []
IR_emitter_failure = bot_id in []
IR_receiver_failure = bot_id in []

# parts with gaussian failure probability
# every relevant function first checks if it should execute normally or apply gaussian inaccuracy to the real outcome
left_motor_inaccuracy = bot_id in []
right_motor_inaccuracy = bot_id in []
phero_reader_inaccuracy = bot_id in []
phero_writer_inaccuracy = bot_id in []
wall_sensor_inaccuracy = bot_id in []
IR_emitter_inaccuracy = bot_id in []
IR_receiver_inaccuracy = bot_id in []

# get and activate sensors/motors
sensors = []
for i in range(7):
    sensor = robot.getDevice(f"prox.horizontal.{i}")
    sensor.enable(timestep)
    sensors.append(sensor)
wheels = []
for i in range(2):
    wheel = robot.getDevice(f"motor.{'left' if i==0 else 'right'}")
    wheel.setPosition(float("inf"))
    wheels.append(wheel)
emitter = Emitter(f"emitter{bot_id}")
receiver = Receiver(f"receiver{bot_id}")
receiver.enable(timestep)

class Lock:
    def __init__(self, path):
        self.file = open(path, "w")

    def __enter__(self):
        portalocker.lock(self.file, portalocker.LOCK_EX)

    def __exit__(self, exc_type, exc_val, exc_tb):
        portalocker.unlock(self.file)
        self.file.close()

class Thymio_bot():
    def __init__(self):
        self.__last_position = bot_pos()[:2]
        self.__state = "wander" # states: wander (by pheromone), dodge (other bot), turn (towards wall)
        self.__timeout = 0
        self.__motor_speeds = [0,0]
        self.__last_dir = None

    def __str__(self):
        return f"state: {self.__state}\ntimeout: {self.__timeout}\nmotors: {self.__motor_speeds}\nsensors: {self.__get_sensors()}"

    """ -- helper functions -- """
    def update_position(self) -> None:
        self.__last_position = bot_pos()[:2]
    def __get_sensors(self) -> list[float]:
        """ returns values between 0 (at ~5.7cm) and ~4350 (at contact) """
        noises = [0] * 5
        if wall_sensor_failure: return noises # on complete sensor failure, return zeros
        if wall_sensor_inaccuracy: # on partial failure, add noise to each sensor
            for i in noises:
                i = np.clip(i+np.random.normal(0,1), 0, 4500)
        return [sensor.getValue()+noise for sensor,noise in zip(sensors, noises)][:5]
    def get_back_sensors(self) -> list[float]:
        """ returns values between 0 (at ~5.7cm) and ~4350 (at contact) """
        noises = [0] * 2
        if wall_sensor_failure: return noises # on complete sensor failure, return zeros
        if wall_sensor_inaccuracy: # on partial failure, add noise to each sensor
            for i in noises:
                i = np.clip(i+np.random.normal(0,1), 0, 4500)
        return [sensor.getValue()+noise for sensor,noise in zip(sensors[5:], noises)]
    def __get_dir_prio(self, vector: list[float]):
        prios = [-1] * 4
        x,y = vector
        if abs(x) > abs(y):         # x is primary direction
            if x>0: prios[0] = 0    # right is primary direction
            else: prios[0] = 2      # left is primary direction
            if y>0: prios[1] = 1    # top is secondary direction
            else: prios[1] = 3      # bottom is secondary direction
        else:                       # y is primary direction
            if y>0: prios[0] = 1    # top is primary direction
            else: prios[0] = 3      # bottom is primary direction
            if x>0: prios[1] = 0    # left is secondary direction
            else: prios[1] = 2      # right is secondary direction
        prios[2] = (prios[1]+2)%4   # set tertiary direction
        prios[3] = (prios[0]+2)%4   # set quaternary direction
        return prios
    """ deadend functions """
    def __get_deadend_p(self) -> list[int]:
        noises = [0] * 4
        if phero_reader_failure: return noises
        if phero_reader_inaccuracy:
            for i in noises:
                i = np.clip(i+np.random.normal(0,1), 0, 100)
        return [phero+noise for phero,noise in zip(deadend_map[int(bot_pos()[1]*2), int(bot_pos()[0]*2)],noises)]
    def __set_deadend_p(self, direction) -> None:
        if phero_writer_failure: return
        if 0 <= bot_pos()[0] < 5 and 0 <= bot_pos()[1] < 5:
            noise = 0
            if phero_writer_inaccuracy: noise = np.clip(np.random.normal(0,1), 0, 100)
            global deadend_map
            with Lock("shared_array.lock"):
                deadend_map[int(bot_pos()[1]*2), int(bot_pos()[0]*2), direction] = 100 + noise
    def __get_main_rotation(self):
        """ helper function to determine approach side of a wall """
        robot_angle = bot_rotation()
        if robot_angle[2]<0: robot_angle[3] *= -1
        active_sensor_angles = [angle for angle,sensor in zip(sensor_angles[1:4], self.__get_sensors()[1:4]) if sensor>0]
        sensor_avg_angle = sum(active_sensor_angles) / len(active_sensor_angles)
        wall_direction = (sensor_avg_angle+robot_angle[3]+np.pi)%(2*np.pi)-np.pi
        if 3*np.pi/4 >= wall_direction > np.pi/4:
            return 1 # top
        if np.pi/4 >= wall_direction > -np.pi/4:
            return 0 # right
        if -np.pi/4 >= wall_direction > -3*np.pi/4:
            return 3 # bottom
        return 2 # left
    def __get_turn_direction(self, sensors: list[int]):
        """ helper function to determine direction of potential wall """
        if all(sensors): return "found"
        if not sensors[0] and not sensors[4]: return "none"
        if sensors[0]: # either left or none
            if any(sensors[sensors.index(0):]):
                return "none" # hole
            else: return "left"
        else: # either right or none
            sensor_copy = sensors[:]
            sensor_copy.reverse()
            if any(sensor_copy[sensor_copy.index(0):]):
                return "none" # hole
            else: return "right"
    """ pheromone functions """
    def get_pheromone_reading(self) -> list[float]:
        if phero_reader_failure: return
        if 0 <= bot_pos()[0] < 5 and 0 <= bot_pos()[1] < 5:
            noises = [0] * 2
            for i in noises:
                i = np.clip(i+np.random.normal(0,1), 0, 100)
            return [phero+noise for phero,noise in zip(pheromone_map[int(bot_pos()[1]*2), int(bot_pos()[0]*2)],noises)]
    def __set_pheromone_reading(self, value) -> None:
        """ helper function to set the pheromone of the current cell to a new value considering decay rules """
        if phero_writer_failure or value is None: return
        noise = [0] * 2
        if phero_writer_inaccuracy:
            for i in noise:
                i = np.clip(i+np.random.normal(0,1), -100, 100)
        try: new_value = np.multiply((1-decay),self.get_pheromone_reading())+np.multiply(decay,value)+noise
        except: return # safety measure for defective pheromone reader
        with Lock("shared_array.lock"):
            pheromone_map[int(bot_pos()[1]*2), int(bot_pos()[0]*2)] = new_value
    def __get_ideal_value(self):
        """ helper function to determine pheromone information gained in the last step """
        if self.__last_position == None: return None
        xb_xa = np.subtract(bot_pos()[:2], self.__last_position[:2])
        x_ab =  np.linalg.norm(xb_xa)
        if x_ab <= 0.001: return None # return if robot didn't move
        direction = xb_xa/x_ab # normalize movement (vec2)
        fb_fa = goal_dist(self.__last_position[:2]) - goal_dist(bot_pos()[:2]) # scale difference in signal strength (float)
        strength = fb_fa/x_ab
        return direction*10*strength
    """ motor control functions """
    def __get_rotation(self, reference):
        """ helper function to determine angle difference between current and ideal rotation """
        if reference is None or np.linalg.norm(reference) == 0: return None # no pheromone was read
        robot_angle = bot_rotation()
        if robot_angle[2]<0: robot_angle[3] *= -1
        pheromone_angle = np.arctan2(reference[1], reference[0])
        angle_diff = pheromone_angle-robot_angle[3]
        # 0->keep going
        # +-pi->opposite
        # -pi/2->toomuchleft
        # pi/2->toomuchright
        return (angle_diff+np.pi)%(2*np.pi)-np.pi
    def test_deadend_transfer(self):
        """ tests if the cell was changed during the last step and returns the direction of the old cell if it is determined to be a deadend """
        # skip testing if the cell didn't change
        if int(bot_pos()[1]*2) == int(self.__last_position[1]*2) and \
            int(bot_pos()[0]*2) == int(self.__last_position[0]*2): return
        if int(bot_pos()[0]*2) > int(self.__last_position[0]*2): self.__last_dir = 2
        if int(bot_pos()[0]*2) < int(self.__last_position[0]*2): self.__last_dir = 0
        if int(bot_pos()[1]*2) > int(self.__last_position[1]*2): self.__last_dir = 3
        if int(bot_pos()[1]*2) < int(self.__last_position[1]*2): self.__last_dir = 1
        old_de_p = list(deadend_map[int(self.__last_position[1]*2), int(self.__last_position[0]*2)])
        # previous cell is not yet determined to be a deadend
        if len([x for x in old_de_p if x!=0])<3: return
        # return the direction the robot came from
        return (old_de_p.index(0)+2)%4
    def __invoke_special_case(self, direction, blocked):
        """ the robot can get stuck in two cases in cells with two available directions.
         1: opposite deadends (i.e. in a corridor) with a goal pheromone
            pointing towards one will get the robot stuck on that wall
         2: adjacent deadends will get the robot stuck on a wall wedge """
        #print("special case")
        pos = bot_pos()
        offset_x = (cell_size*int(pos[0]*2)+cell_size/2) - pos[0]
        offset_y = (cell_size*int(pos[1]*2)+cell_size/2) - pos[1]
        # determine the special case
        adj = False
        for i in range(len(blocked)):
            if blocked[i] and blocked[(i+1)%4]: adj = True
        #print(f"adj: {adj}")
        # handle L-turn-case
        if adj:
            dir_prio = self.__get_dir_prio(self.get_pheromone_reading())
            #print(f"dir_prio: {dir_prio}")
            for i in dir_prio:
                if not blocked[i]:
                    goal_dir = i
                    break
            #print(f"goal dir: {goal_dir}")
            match goal_dir:
                case 0: # steer towards center right
                    direction[0] = offset_x+cell_size/2
                    direction[1] = offset_y
                case 1: # steer towards center top
                    direction[0] = offset_x
                    direction[1] = offset_y+cell_size/2
                case 2: # steer towards center left
                    direction[0] = offset_x-cell_size/2
                    direction[1] = offset_y
                case 3: # steer towards center bottom
                    direction[0] = offset_x
                    direction[1] = offset_y-cell_size/2
        elif blocked[0]: # horizontal corridor
            #print(f"blocked hor")
            direction[0] = offset_x # steer towards vertical center line
        else: # vertical corridor
            #print(f"blocked ver")
            direction[1] = offset_y # steer towards horizontal center line
        #print(f"dir: {direction}\nxxxxxxx")
        return direction
    def calculate_pheromone_speed(self):
        """ the main function to calculate motor speeds in the general case.
        this considers dead ends, pheromones and their special cases """
        if self.__state != "wander": return
        # get base direction
        direction = [0,0]
        if not self.__timeout:
            reading_p = self.get_pheromone_reading()
            if reading_p: direction = reading_p
        # modify direction based on dead ends
        reading_d = self.__get_deadend_p()
        de_amount = len(list(filter(lambda x: x, reading_d)))
        if de_amount == 3: self.__last_dir = None
        blocked_directions = list(map(lambda x: bool(x), reading_d))
        try: blocked_directions[self.__last_dir] = True
        except: pass
        # handle edge cases with two opposite or two adjacent dead ends
        if len(list(filter(lambda x: x, blocked_directions))) == 2:
            direction = self.__invoke_special_case(direction,blocked_directions)
        else:
            if reading_d[0] or self.__last_dir == 0: direction[0] -= 10
            if reading_d[2] or self.__last_dir == 2: direction[0] += 10
            if reading_d[1] or self.__last_dir == 1: direction[1] -= 10
            if reading_d[3] or self.__last_dir == 3: direction[1] += 10
        rotation = self.__get_rotation(direction)
        if not self.__timeout: # catch error when no pheromone was read
            if rotation:
                self.__motor_speeds[0] = np.clip(max_speed-(rotation*max_speed*2/np.pi), -max_speed, max_speed)
                self.__motor_speeds[1] = np.clip(max_speed-(rotation*-max_speed*2/np.pi), -max_speed, max_speed)
            else:
                self.__motor_speeds[0] = 0.5*max_speed
                self.__motor_speeds[1] = 0.5*max_speed
    def emergency_collision_avoidance(self):
        # emergency collision avoidance forward
        ffs_emergency = len([s for s in self.__get_sensors() if s > 4200]) # front facing sensors
        bfs_emergency = len([s for s in self.get_back_sensors() if s > 4200]) # back facing sensors
        move_bw = len([m for m in self.__motor_speeds if m<0])
        if all([ffs_emergency, not bfs_emergency]):
            self.__motor_speeds = [-0.5*max_speed, -0.4*max_speed]
            self.__timeout = 100
        elif all([not ffs_emergency, bfs_emergency, move_bw]):
            self.__motor_speeds = [0.4*max_speed, 0.5*max_speed]
            self.__timeout = 0
        elif all([ffs_emergency, bfs_emergency]):
            # a simple complete-halt until obstacle is gone can lead to robots getting stuck when driving backwards into a corner
            obstacle = self.__get_turn_direction(self.__get_sensors())
            match obstacle:
                case "left":
                    self.__motor_speeds = [0.2*max_speed,0.01*max_speed]
                    self.__timeout = 0
                case "right":
                    self.__motor_speeds = [0.01*max_speed,0.2*max_speed]
                    self.__timeout = 0
                case _:
                    self.__motor_speeds = [0,0]
                    self.__timeout = 0
        # cases forwards/backwards-free and forwards-free/no-bw-movement lead to default movement

    """ -- state entry functions -- """
    def __wander_to_dodge(self):
        """ initiate turn right (1) """
        self.__motor_speeds = [0.2*max_speed,-0.2*max_speed]
    def __wander_to_turn(self, direction):
        """ initiate turn towards wall (2) """
        if direction == "left": self.__motor_speeds = [-0.05*max_speed, 0.15*max_speed]
        else: self.__motor_speeds = [0.15*max_speed, -0.05*max_speed]
    def __dodge_to_wander(self):
        """ initiate forwards left turn and timeout (6) """
        self.__motor_speeds = [0.1*max_speed, 0.3*max_speed]
        self.__timeout = 100
    def __turn_to_wander_wall(self):
        """ initiate printing of wall repellent (7) """
        self.__set_deadend_p(self.__get_main_rotation())
        self.__motor_speeds = [-0.5*max_speed, 0.5*max_speed]
        self.__timeout = 25
    def __turn_to_wander_tip(self):
        """ initiate timeout (9) """
        self.__timeout = 100

    def stateswitcher(self):
        """ executes at the beginning of every activation cycle. matches the current state and changes it depending on sensor values and calls state-entry functions. legend to numbering can be found in the stateswitcher fsm diagram """
        self.__timeout = max(0,self.__timeout-1)
        ff_sensor_values = self.__get_sensors()
        ir_beacon = receiver.getQueueLength()
        match self.__state:
            case "wander":
                if any(ff_sensor_values[1:4]):
                    if ir_beacon:
                        """ enter dodge (1) """
                        self.__wander_to_dodge()
                        self.__state = "dodge"
                        self.__last_dir = None
                    else:
                        if not self.__get_deadend_p()[self.__get_main_rotation()]:
                            direction = self.__get_turn_direction(ff_sensor_values)
                            match direction:
                                case "found":
                                    """ special case where the robot approaches a wall so straight, that all sensors flip simultaneously -> print wall pheromone (7s) """
                                    self.__turn_to_wander_wall()
                                case "none":
                                    """ a very slim obstacle or hole was found -> start timeout and stay in wander (9s)"""
                                    self.__turn_to_wander_tip()
                                    if self.__motor_speeds == [0,0]:
                                        # there is a special case where the robots stands still if it spawns badly in front of a wall
                                        # i.e. sensor case [0,0,1,0,0]
                                        self.__motor_speeds = [0.1*max_speed, -0.1*max_speed]
                                case _:
                                    """ search for wall -> enter turn (2) """
                                    self.__wander_to_turn(direction)
                                    self.__state = "turn"
                    """ dep already exists -> stay in wander (3) """
                """ no sensor active -> stay in wander (4) """
            case "dodge":
                if not (any(ff_sensor_values) or ir_beacon):
                    """ initiate forward left turn, start timeout and enter wander (6) """
                    self.__dodge_to_wander()
                    self.__state = "wander"
                """ a sensor is still active -> stay in dodge (5) """
            case "turn":
                direction = self.__get_turn_direction(ff_sensor_values)
                match direction:
                    case "found":
                        """ print wall pheromone and enter wander (7) """
                        self.__turn_to_wander_wall()
                        self.__state = "wander"
                    case "none":
                        """ start timeout and enter wander (9) """
                        self.__turn_to_wander_tip()
                        self.__state = "wander"
                    case _:
                        self.__wander_to_turn(direction)
                """ not clear if wall or tip -> stay in turn (8) """

    def update_pheromones(self):
        """ calculate and apply new pheromone intelligence and adjust motor speeds based on new pheromone map if no timeout is set """
        deadend_transfer_direction = self.test_deadend_transfer()
        if deadend_transfer_direction != None:
            self.__set_deadend_p(deadend_transfer_direction)
        if self.__state != "wander": return
        self.__set_pheromone_reading(self.__get_ideal_value())

    def apply_speeds(self):
        """ at the end of every activation cycle, the motor speeds must be updated to the newly determined values. This function does just that. """
        self.emergency_collision_avoidance()
        if not left_motor_failure: # check for failure
            noise = 0
            if left_motor_inaccuracy: # check for inaccuracy
                noise = np.random.normal(0,1)
            wheels[0].setVelocity(self.__motor_speeds[0]+noise) # apply speed
        if not right_motor_failure: # check for failure
            noise = 0
            if right_motor_inaccuracy: # check for inaccuracy
                noise = np.random.normal(0,1)
            wheels[1].setVelocity(self.__motor_speeds[1]+noise) # apply speed

def main():
    thymio.stateswitcher()
    thymio.update_pheromones()
    thymio.calculate_pheromone_speed()
    thymio.apply_speeds()
    thymio.update_position()
    #print(thymio)
    #print(thymio.get_back_sensors())
    #print("--------------")

if __name__ == "__main__":
    thymio = Thymio_bot()
    simulation_step = 0
    wheels[0].setVelocity(0)
    wheels[1].setVelocity(0)
    while robot.step(timestep) != -1:
        simulation_step += 1
        if not IR_emitter_failure:
            emitter.send(bytes(1)) # 1 = "i am here"-signal
        if not ((simulation_step+bot_id))%1:
            main()
        queuesize = receiver.getQueueLength()
        for _ in range(queuesize): # clear out unused IR beacon information
            receiver.nextPacket()
        if goal_dist(bot_pos()[:2]) < 0.2:
            bot.remove() # remove bot is goal is reached
            break
    while robot.step(timestep) != -1:
        pass # prevent forced webots restart