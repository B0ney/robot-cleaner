from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

import pygame
import tkinter.simpledialog as simpledialog
import tkinter.messagebox as messagebox
import random

## ---- ENVIRONMENT ---- ##

CLEAN = True
WASTE = False


@dataclass
class Point:
    """Represents a coordinate"""

    x: int = 0
    y: int = 0


class World:
    width: int
    height: int
    entities: List[bool]

    def __init__(self, size: int):
        self.height = size
        self.width = size

        self.entities = [CLEAN] * self.width * self.height

    def within_bounds(self, x: int, y: int) -> bool:
        return (x >= 0 and x < self.width) and (y >= 0 and y < self.height)

    def modify(self, x: int, y: int, entity: bool) -> bool:
        already_has_entity = False

        if self.within_bounds(x, y):
            if self.get_state(x, y) == entity:
                already_has_entity = True

            self.entities[x + y * self.width] = entity

        return already_has_entity

    def add_waste(self, x: int, y: int) -> bool:
        return self.modify(x, y, WASTE)

    def clean_waste(self, x: int, y: int):
        self.modify(x, y, CLEAN)

    def get_state(self, x: int, y: int) -> Optional[bool]:
        if self.within_bounds(x, y):
            return self.entities[x + y * self.width]
        else:
            return None

    def has_waste(self) -> bool:
        return WASTE in self.entities


## ---- Robot cleaner ---- ##


class State(Enum):
    """The state of the robot cleaner."""

    INIT = 0
    CLEANING = 1
    DONE = 2


class Action(Enum):
    """A set of possible actions."""

    COLLECT_WASTE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    DO_NOTHING = 5


class Result(Enum):
    """A set of possible results when an action has been performed."""

    Nothing = 0
    AllCleaned = 1


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """https://xlinux.nist.gov/dads/HTML/manhattanDistance.html"""

    return abs(x1 - x2) + abs(y1 - y2)


class RobotCleaner:
    """Agent that does waste management."""

    position: Point
    state: State
    actions: List[Action]

    def __init__(self, x: int, y: int) -> None:
        self.position = Point(x, y)
        self.state = State.INIT
        self.actions = []

    def x(self) -> int:
        return self.position.x

    def y(self) -> int:
        return self.position.y

    def process(self, environment: World) -> List[Action]:
        """The agent's processing mechanism.

        Generates a list of actions to find
        and clean waste that is closest to the cleaner.
        """

        # The agent might be placed in an invalid location.
        self.check_bounds(environment)

        print("Scanning environment...")

        # scan environment
        waste_locations: List[Point] = []

        for j in range(environment.height):
            for i in range(environment.width):
                if environment.get_state(i, j) == WASTE:
                    waste_locations.append(Point(i, j))

        # If there's no waste, do nothing.
        if len(waste_locations) == 0:
            return [Action.DO_NOTHING]

        # Sort the waste locations by distance.
        # We use the manhattan distance because the agent can only move four directions. (up, down, left, right)
        waste_locations.sort(
            key=lambda waste: manhattan_distance(self.x(), self.y(), waste.x, waste.y),
            reverse=True,
        )

        # obtain the closest waste
        closest_waste = waste_locations.pop()

        # Determine if the robot should move left or right.
        # If the distance is negative, the waste is located on the left hand side.
        # Otherwise, the waste is located on the right.
        horizontal_distance = closest_waste.x - self.x()

        if horizontal_distance < 0:
            horizontal_movement = [Action.MOVE_LEFT] * abs(horizontal_distance)
        else:
            horizontal_movement = [Action.MOVE_RIGHT] * abs(horizontal_distance)

        # Determine if the robot should move up or down.
        # If the distance is negative, the waste is located above it.
        # Otherwise, the waste is located below.
        vertical_distance = closest_waste.y - self.y()

        if vertical_distance < 0:
            vertical_movement = [Action.MOVE_UP] * abs(vertical_distance)
        else:
            vertical_movement = [Action.MOVE_DOWN] * abs(vertical_distance)

        # Randomly decide if it should move horizontally or vertically first.
        if random.randint(0, 1) == 0:
            commands = horizontal_movement + vertical_movement
        else:
            commands = vertical_movement + horizontal_movement

        # Once the agent has reached the location we add the collect waste command
        commands.append(Action.COLLECT_WASTE)

        return commands

    def determine_action(self, environment: World) -> Action:
        """Use the current state and the environment
        to determine the most appropriate action
        """

        # If there are no more actions to perform,
        # the agent will do some processing to regenerate that list.
        if len(self.actions) == 0:
            self.actions = self.process(environment)

        # If the list of actions are still empty,
        # then the agent has nothing to do
        if len(self.actions) == 0:
            return Action.DO_NOTHING

        # Otherwise, we slowly consume the list of actions
        else:
            return self.actions.pop(0)

    def check_bounds(self, world: World):
        """Clamp the position to fit within the environment's boundaries."""

        if self.position.x < 0:
            self.position.x = 0

        if self.position.y < 0:
            self.position.y = 0

        if self.position.x >= world.width:
            self.position.x = world.width - 1

        if self.position.y >= world.height:
            self.position.y = world.height - 1

    def perform_action(self, world: World, action: Action) -> Result:
        """Uses the given action to update the environment.
        Returns the result of that action.
        """

        result: Result = Result.Nothing

        match action:
            case Action.COLLECT_WASTE:
                print("collecting waste...")
                world.clean_waste(self.x(), self.y())

            case Action.MOVE_LEFT:
                print("moving left...")
                self.position.x -= 1

            case Action.MOVE_RIGHT:
                print("moving right...")
                self.position.x += 1

            case Action.MOVE_UP:
                print("moving up...")
                self.position.y -= 1

            case Action.MOVE_DOWN:
                print("moving down...")
                self.position.y += 1

            case Action.DO_NOTHING:
                print("Nothing to do...")

        # If the world is clean, we can set the result to all clean
        if not world.has_waste():
            print("all clean!")
            result = Result.AllCleaned

        self.check_bounds(world)
        return result

    def update_state(self, result: Result):
        """Use the given result to update the state of the agent."""

        match result:
            case Result.Nothing:
                pass

            case Result.AllCleaned:
                self.state = State.DONE

    def goal_state_reached(self) -> bool:
        return self.state == self.goal_state()

    def goal_state(self) -> State:
        return State.DONE


def tick_simulation(world: World, agent: RobotCleaner):
    next_action = agent.determine_action(world)
    result = agent.perform_action(world, next_action)
    agent.update_state(result)


## ---- GRAPHICAL DISPLAY ---- ##

RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

SIMULATION_FPS = 5
WINDOW_SIZE = (480, 640)
WINDOW_FPS = 60
WINDOW_TITLE = "Smart Vacuum Simulation"


def gui(world: World, agent: RobotCleaner, simulation_speed: int = SIMULATION_FPS):
    """A graphical runtime for the simulation"""

    # Initialize pygame
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption(WINDOW_TITLE)

    # create some text
    font = pygame.font.Font(pygame.font.get_default_font(), 36)
    info = font.render("Red = Robot Cleaner", True, WHITE)
    info2 = font.render("Green = Waste", True, WHITE)

    clock = pygame.time.Clock()
    window = pygame.display.set_mode(WINDOW_SIZE)
    canvas = TiledCanvas(
        window, (WINDOW_SIZE[0], WINDOW_SIZE[0]), (world.width, world.height)
    )
    tick = 0

    while True:
        # view
        canvas.draw_background(BLACK)
        draw_environment(canvas, world)
        draw_agent(canvas, agent)

        # Display some information
        window.blit(info, (0, 480 + 10))
        window.blit(info2, (0, 480 + 10 + 40))

        # poll pygame's event queue.
        # While we're not using it directly, the event queue
        # needs to be emptied to allow pygame to handle windowing events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # stop the program if the agent has reached its goal state
        if agent.goal_state_reached():
            messagebox.showinfo(
                "Goal state reached", "The robot has reached its goal state"
            )
            pygame.quit()
            return

        # The pygame event loop runs at WINDOW_FPS times per second.
        # But this is too fast to observe the simulation.
        # So we update the simulation at a slower interval.
        if tick > WINDOW_FPS / simulation_speed:
            tick_simulation(world, agent)
            tick = 0

        pygame.display.update()
        window.fill(BLACK)
        clock.tick(max(WINDOW_FPS, simulation_speed))
        tick += 1


@dataclass
class TiledCanvas:
    """A class to display a tiled grid on a pygame window"""

    window: pygame.Surface
    canvas_size: Tuple[int, int]
    tile_size: Tuple[int, int]
    canvas_position: Tuple[int, int] = (0, 0)  # Top left

    def within_bounds(self, x: int, y: int) -> bool:
        """Check that the given coordinates within the tile's boundaries"""

        [tile_width, tile_height] = self.tile_size

        return (x >= 0 and x < tile_width) and (y >= 0 and y < tile_height)

    def draw_background(self, color: Tuple[int, int, int]):
        """Provide a background for the canvas"""

        [x, y] = self.canvas_position
        [canvas_width, canvas_height] = self.canvas_size

        pygame.draw.rect(
            self.window,
            color,
            pygame.Rect(
                x,
                y,
                canvas_width,
                canvas_height,
            ),
        )

    def draw_tile(self, x: int, y: int, color: Tuple[int, int, int], fill: bool = True):
        """Display a coloured tile on the canvas."""

        if not self.within_bounds(x, y):
            return

        [tile_width, tile_height] = self.tile_size
        [canvas_width, canvas_height] = self.canvas_size
        [offset_x, offset_y] = self.canvas_position

        # scale the tiles to fit inside the canvas
        scale_x = canvas_width / tile_width
        scale_y = canvas_height / tile_height

        new_x = int(x * scale_x)
        new_y = int(y * scale_y)

        pygame.draw.rect(
            self.window,
            color,
            pygame.Rect(offset_x + new_x, offset_y + new_y, scale_x, scale_y),
            int(fill),
        )


def draw_agent(canvas: TiledCanvas, robot: RobotCleaner):
    canvas.draw_tile(robot.x(), robot.y(), RED, fill=False)


def draw_environment(canvas: TiledCanvas, world: World):
    for x in range(world.width):
        for y in range(world.height):
            if world.get_state(x, y) == WASTE:
                canvas.draw_tile(x, y, GREEN, fill=False)
            else:
                canvas.draw_tile(x, y, WHITE)


## ---- GET USER INPUTS ---- ##

MAX_WORLD_SIZE = 15
MIN_WORLD_SIZE = 1


def get_world_size() -> int:
    answer = None
    while answer is None:
        answer = simpledialog.askinteger(
            "Let's create the environment",
            f"What is the size of the environment? {MIN_WORLD_SIZE}-{MAX_WORLD_SIZE}",
            minvalue=MIN_WORLD_SIZE,
            maxvalue=MAX_WORLD_SIZE,
        )

    return answer


def get_waste_locations(world: World):
    max_width = world.width - 1
    max_height = world.height - 1
    error_message = ""

    while True:
        answer = simpledialog.askstring(
            "Let's add some waste",
            f"{error_message}Input coordinates to put WASTE.\nEsc - Stop adding waste\n\nHint:\nx,y\n0,0 = top left\n{max_width},{max_height} = bottom right",
        )

        if answer is None:
            break

        coordinate = parse_coords(answer)

        if coordinate is not None:
            [x, y] = coordinate

            if not world.within_bounds(x, y):
                error_message = "The coordinates are not within bounds.\n"
            else:
                already_has_waste = world.add_waste(x, y)
                if already_has_waste:
                    error_message = (
                        "The coordinates you entered already contains waste.\n"
                    )
                else:
                    error_message = ""
        else:
            error_message = "The input is not valid.\n"


def get_agent_start_position(world: World) -> Tuple[int, int]:
    max_width = world.width - 1
    max_height = world.height - 1
    error_message = ""

    while True:
        user_input = simpledialog.askstring(
            "Time to deploy the agent",
            f"{error_message}Input the start location for the cleaning ROBOT.\n\nHint:\n0,0 = top left\n{max_width},{max_height} = bottom right",
        )

        if user_input is None:
            error_message = "The input is not valid.\n"
            continue

        coords = parse_coords(user_input)

        if coords is None:
            error_message = "The input is not valid.\n"
            continue

        [x, y] = coords
        if not world.within_bounds(x, y):
            error_message = "The coordinates are not within bounds.\n"
        else:
            return coords


def generate_random_waste() -> bool:
    answer = messagebox.askquestion(
        "Randomly generate waste", "Would you like me to randomly add waste?"
    )
    return answer == "yes"


def parse_coords(point: str) -> Optional[Tuple[int, int]]:
    """Converts user input into a valid coordinate."""

    import re

    split = list(map(strip_whitespace, re.split(",", point)))

    if len(split) != 2:
        return None

    [x, y] = split

    if not (x.isdigit() and y.isdigit()):
        return None

    return (int(x), int(y))


def strip_whitespace(s: str) -> str:
    return s.strip()


## ---- MAIN APPLICATION ---- ##


def main():
    # Ask the user to input the world size
    size = get_world_size()
    world = World(size)

    # Ask the user if they want the waste to be randomly placed.
    if generate_random_waste():
        for _ in range(random.randint(size, (size * size))):
            world.add_waste(random.randrange(0, size), random.randrange(0, size))

    else:
        # Ask the user to input locations to put waste
        get_waste_locations(world)

    # Ask the user for the start posision
    [x, y] = get_agent_start_position(world)
    agent = RobotCleaner(x, y)

    # run the simulation
    gui(world, agent)


if __name__ == "__main__":
    main()
