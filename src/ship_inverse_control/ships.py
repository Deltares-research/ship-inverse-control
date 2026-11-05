import copy
import datetime
import itertools
import uuid
from typing import Optional, Tuple, TypeVar

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
import pyrsistent
import python_vehicle_simulator.lib
import python_vehicle_simulator.vehicles as vehicles
import shapely

# create generic types for Vector, Matrix, Tensor
T = TypeVar("T", bound=np.generic, covariant=True)
Vector = np.ndarray[Tuple[int], np.dtype[T]]
Matrix = np.ndarray[Tuple[int, int], np.dtype[T]]
Tensor = np.ndarray[Tuple[int, ...], np.dtype[T]]


class Ship:
    def __init__(
        self,
        starting_point: Optional[shapely.Point] = None,
        speed_over_ground: float = 0.0,
        course_over_ground: float = 0.0,
        heading: Optional[float] = None,
        L: float = 100.0,
        B: float = 7.0,
        T: float = 3.0,
        V_current: float = 0.0,
        beta_current: float = 0.0,
        start_time: datetime.datetime = datetime.datetime(1970, 1, 1),
        id: Optional[str] = None,
        initial_rudder_angle: Optional[float] = None,
        initial_tau_X: Optional[float] = None,
    ):
        """
        Create a ship with a starting point and a course over ground (based on AIS based parameters)
        :param starting_point: starting point of the ship (default: (0, 0)), theta_en in Fossen
        :param speed_over_ground: speed over ground in m/s
        :param course_over_ground: course over ground in radians, clockwise, 0 north
        :param heading: heading in radians, clockwise, 0 north.
        """
        if starting_point is None:
            starting_point = shapely.Point(0, 0)

        # starting point of the ship
        self.starting_point = starting_point
        self.position = starting_point
        # course over ground in radians, clockwise, 0 north
        self.course_over_ground = course_over_ground
        self.heading = course_over_ground
        # speed over ground in knots
        self.speed_over_ground = speed_over_ground
        self.heading = heading
        # length of the ship
        self.L = L
        # beam of the ship
        self.B = B
        # draft of the ship
        self.T = T
        # water current speed
        self.V_current = V_current
        # water current direction
        self.beta_current = beta_current
        # unique id IF not given
        if id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = id

        if initial_tau_X is not None:
            self.initial_tau_X = initial_tau_X
        else:
            self.initial_tau_X = 0
        if initial_rudder_angle is not None:
            self.initial_rudder_angle = initial_rudder_angle
        else:
            self.initial_rudder_angle = 0

        self.epoch = start_time

        if self.heading is not None:
            # If heading is provided, use it for the ship's orientation.
            # Heading is the direction the bow is pointing, in radians, clockwise from North.
            self.psi = self.heading
        else:
            # If no heading is provided, assume it's the same as the course over ground.
            # This simplification is valid for initialization at t=0, assuming no
            # initial drift from wind or currents.
            self.psi = self.course_over_ground

        # Resolve the velocity vector (defined by SOG and COG) into the
        # ship's body-fixed coordinate system (defined by heading).

        # Angle difference between the ship's heading and its actual course.
        if self.heading is not None and self.course_over_ground is not None:
            drift_angle = self.course_over_ground - self.heading
        else:
            drift_angle = 0

        # If drift angle is 0, we're going straight ahead, so cos(drift_angle) == 1
        # Surge (u) is the forward velocity component.
        u = self.speed_over_ground * np.cos(drift_angle)

        # If drift angle is 0, we're going straight ahead, so sin(drift_angle) == 0
        # Sway (v) is the sideways velocity component.
        v = self.speed_over_ground * np.sin(drift_angle)

        # import ipdb
        # ipdb.set_trace()
        # The initial velocity vector for the 3-DOF model [surge, sway, yaw_rate].
        self.nu_3df = [u, v, 0]

        # position, velocity, yaw angle
        self.eta_3df = [self.starting_point.y, self.starting_point.x, self.psi]

        self.logbook = []

        self.initialize()

    def initialize(self):
        """
        Define a ship and the initial state variables (similar [1])
        [1] https://github.com/cybergalactic/PythonVehicleSimulator/blob/master/src/python_vehicle_simulator/lib/mainLoop.py#L62
        """

        # TODO: how can I initialize with both cog and heading

        # TODO: initialize tau_X and rudder_angle (so that rudder_angle starts in correct position)
        vehicle = vehicles.shipClarke83(
            L=self.L,
            B=self.B,
            T=self.T,
            V_current=self.V_current,
            beta_current=self.beta_current,
            tau_X=self.initial_tau_X,
        )
        starting_point = self.starting_point

        # degrees of freedom
        DOF = 6
        # initial simulation time
        t = 0

        # position/attitude, user editable
        eta = np.zeros(6, dtype=float)
        eta[0] = self.eta_3df[0]
        eta[1] = self.eta_3df[1]
        eta[5] = self.eta_3df[2]

        self.position = shapely.Point([self.eta_3df[1], self.eta_3df[0]])
        self.heading = eta[5]

        # velocity, defined by vehicle class
        nu = vehicle.nu
        nu[0] = self.nu_3df[0]
        nu[1] = self.nu_3df[1]
        # yaw rate
        nu[5] = self.nu_3df[2]

        # actual inputs, defined by vehicle class
        u_actual = vehicle.u_actual
        # overwrite with initial_rudder_angle so rudder starts in correct position (it takes time to move the rudder)
        u_actual[0] = self.initial_rudder_angle
        vehicle.tau_X = self.initial_tau_X

        simData = np.zeros([0, 2 * DOF + 2 * vehicle.dimU], dtype=float)

        u_control = np.array([self.initial_rudder_angle], dtype=float)

        # Store simulation data in simData
        signals = np.append(np.append(np.append(eta, nu), u_control), u_actual)
        simData = np.vstack([simData, signals])
        # collect all local variables
        state = {
            "vehicle": vehicle,
            "DOF": DOF,
            "timestamp": t,
            "eta": eta,
            "nu": nu,
            "u_actual": u_actual,
            "simData": simData,
            "u_control": u_control,
            "signals": signals,
            "epoch": self.epoch,
            "rudder_angle": self.initial_rudder_angle,
            "tau_X": self.initial_tau_X,
        }
        # return the state
        state = self.post_process_state(state)
        self.state = state
        # self.logbook = pd.DataFrame(state)
        self.logbook.append(copy.deepcopy(state))

    @staticmethod
    def post_process_state(state):

        # Position in NED frame (meters)
        # eta[0] represents the position a long the North-South axis (y).
        # eta[1] represents the position along the East-West axis (x).
        # eta[2] represents the position along the Down-Up axis (z).
        # ....
        state["x"] = state["eta"][1]
        state["y"] = state["eta"][0]
        state["z"] = state["eta"][2]
        state["geometry"] = shapely.Point(state["x"], state["y"])

        # Orientation (attitude) in radians
        # phi (roll), theta (pitch) are counter-clockwise. psi (yaw) is clockwise from North.
        state["phi"] = state["eta"][3]
        state["theta"] = state["eta"][4]

        psi = state["eta"][5]
        # for readability
        heading = psi
        state["heading"] = heading

        # Velocities in body-fixed frame [u (surge), v (sway), r (yaw rate)]
        # u: m/s, v: m/s, r: rad/s
        surge_velocity = state["nu"][0]
        sway_velocity = state["nu"][1]
        state["surge_velocity"] = surge_velocity
        state["sway_velocity"] = sway_velocity
        state["yaw_rate"] = state["nu"][5]

        # These variables show the contribution of the ship's forward motion
        # along the North-East axes, independent of sway.
        # u_surge_east: Eastward component of surge velocity (m/s)
        # v_surge_north: Northward component of surge velocity (m/s)
        heading = state["heading"]
        u_surge_east = np.sin(heading) * surge_velocity
        v_surge_north = np.cos(heading) * surge_velocity
        state["u_surge_east"] = u_surge_east
        state["v_surge_north"] = v_surge_north
        state["u"] = u_surge_east
        state["v"] = v_surge_north

        # --- Explicit Calculation of Course and Speed Over Ground ---
        # Rotate body-fixed velocities to the NED frame to get velocity over ground
        v_north = surge_velocity * np.cos(psi) - sway_velocity * np.sin(psi)
        v_east = surge_velocity * np.sin(psi) + sway_velocity * np.cos(psi)

        # Speed Over Ground (SOG) is the magnitude of the NED velocity vector (m/s)
        state["speed_over_ground"] = np.sqrt(v_north**2 + v_east**2)

        # Course Over Ground (COG) is the direction of the NED velocity vector (radians, clockwise from North)
        # We use atan2 to handle all quadrants correctly. The arguments are (east, north).
        state["course_over_ground_rad"] = np.arctan2(v_east, v_north)

        # Store NED velocity components for reference v_north: Represents how
        # fast the ship is moving northward (if positive) or southward (if
        # negative), in meters per second.
        # v_east: Represents how fast the ship
        # is moving eastward (if positive) or westward (if negative), in meters
        # per second.
        state["v_north"] = v_north
        state["v_east"] = v_east

        try:
            state["date"] = state["epoch"] + datetime.timedelta(
                seconds=state["timestamp"]
            )
        except ValueError:
            state["date"] = None

        return state

    def update(self, dt: float = 0.05, rudder_angle: float = 0, tau_X: float = 10000):
        """update vehicle for timestep dt [s] with a new rudder_angle [rad, 0 forward] and tau_X [N]"""
        state = self.state

        state["rudder_angle"] = rudder_angle
        state["tau_X"] = tau_X

        # rudder angle (in rad, 0 forward)
        u_control = np.array([rudder_angle])

        eta = state["eta"]

        # timestamp
        sampleTime = dt

        t = state["timestamp"]

        # get the vessel
        vehicle = state["vehicle"]

        # set the engine order
        vehicle.tau_X = tau_X

        # get the actual u, eta and nu
        u_actual = state["u_actual"]
        eta = state["eta"]
        nu = state["nu"]

        [nu, u_actual] = vehicle.dynamics(eta, nu, u_actual, u_control, sampleTime)
        # update the position of the ship
        eta = python_vehicle_simulator.lib.attitudeEuler(eta, nu, sampleTime)

        # new simulation time
        t += sampleTime

        self.position = shapely.Point([eta[1], eta[0]])
        self.heading = eta[5]

        # make a new state (no deep copy so we have the same ship and other memory each time)
        # state = copy.deepcopy(state)
        # store the new state
        state["nu"] = nu
        state["timestamp"] = t
        state["u_actual"] = u_actual
        state["u_control"] = u_control
        state["eta"] = copy.deepcopy(eta)

        state = self.post_process_state(state)
        self.state = state
        # self.logbook[len(self.logbook)] = state
        self.logbook.append(pyrsistent.pmap(state))
        return state

    def estimate_dt(
        self,
        max_courant_number=0.1,
        min_speed_threshold=0.1,
        default_dt=0.2,
        min_dt=0.05,
        max_dt=5.0,
    ):
        """Estimates a stable simulation timestep 'dt' based on the ship's
        velocity using the Courant-Friedrichs-Lewy (CFL) condition.

        If the ship's speed is below a minimum threshold, a default timestep
        is used.

        Args:

        - max_courant_number (float): The maximum allowable Courant number (C).
          A value <= 1 is normally required for stability. Default is 0.1 for a
          safety (shipClarke is not so stable).

        - min_speed_threshold (float): The minimum speed (m/s) below which the
          default_dt will be used.

        - default_dt (float): A fallback timestep for low-speed orstationary
          conditions.

        Returns:
            float: The estimated stable timestep 'dt' in seconds.

        """
        ship_speed = self.state.get("speed_over_ground", 0.0)

        # If speed is below the threshold, use the default dt
        if ship_speed < min_speed_threshold:
            return default_dt

        # --- Calculate dt based on the Courant (CFL) Condition ---
        # The CFL condition is dt <= C * dx / u, where:
        #   C = Courant Number
        #   dx = characteristic length (self.L)
        #   u = ship's speed
        dt_from_cfl = (max_courant_number * self.L) / ship_speed

        # Ensure the calculated dt is not excessively small
        dt = max(dt_from_cfl, min_dt)
        # Don't make dt too big, e.g. if we need output at a certain time
        dt = min(dt, max_dt)
        return dt

    def plot(self, ax):
        ax.plot(self.state["x"], self.state["y"], "o", label="ship")
        ax.quiver(
            self.state["x"],
            self.state["y"],
            np.cos(self.state["v_z"]) * self.state["v_x"],
            np.sin(self.state["v_z"]) * self.state["v_x"],
            angles="xy",
            label="velocity",
            scale=0.1,
        )

    @property
    def trajectory(self):
        """extract trajectory from ship logbook"""
        ship = self
        keys = [
            "x",
            "y",
            "eta",
            "nu",
            "date",
            "timestamp",
            "u",
            "v",
            "rudder_angle",
            "tau_X",
            "course_over_ground_rad",
        ]
        rows = []
        for state in ship.logbook:
            row = {}
            for key in keys:
                row[key] = state[key]
            rows.append(row)

        traj_df = pd.DataFrame(rows)
        traj_df["psi"] = traj_df.apply(lambda row: row["eta"][5], axis=1)
        traj_df["heading"] = np.rad2deg(traj_df["psi"])
        traj_df["cog"] = np.rad2deg(traj_df["course_over_ground_rad"])
        traj_df["velocity"] = traj_df.apply(lambda row: row["nu"][0], axis=1)

        # TODO store EPSG in ship
        traj_gdf = gpd.GeoDataFrame(
            traj_df, geometry=gpd.points_from_xy(traj_df.x, traj_df.y, crs="EPSG:28992")
        )

        traj = mpd.Trajectory(traj_gdf.set_index("date"), traj_id=ship.id + "_logbook")
        traj.add_speed(overwrite=True)
        traj.add_direction(overwrite=True)
        return traj

    def create_ship_polygon(self):
        """
        Create a polygon around the ship's position. The ship's position is taken as the middle back of the rectangle
        representing the ship. The rectangle is oriented along the ship's heading and has a width and length equal to the ship's width and length.
        """

        # Get the ship's position and heading
        ship_pos = self.position
        ship_heading = np.pi / 2 - self.heading
        # Get the ship's dimensions
        ship_width = self.B
        ship_length = self.L
        # Calculate the coordinates of the rectangle's corners
        x_min = ship_pos.x
        y_min = ship_pos.y - ship_width / 2
        x_max = ship_pos.x + ship_length
        y_max = ship_pos.y + ship_width / 2
        # Create the polygon
        try:
            ship_polygon = shapely.Polygon(
                [
                    (x_min, y_min),
                    (x_max, y_min),
                    (x_max, y_max),
                    (x_min, y_max),
                    (x_min, y_min),
                ]
            )
        except:
            return ship_pos

        ship_polygon = shapely.affinity.rotate(
            ship_polygon,
            ship_heading,
            origin=(ship_pos.x, ship_pos.y),
            use_radians=True,
        )

        return ship_polygon
