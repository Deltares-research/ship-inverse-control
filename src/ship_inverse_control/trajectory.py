import copy
import datetime
import functools
import logging
import math
import uuid
from typing import Optional, Callable

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
import scipy.interpolate
import shapely
from scipy.optimize import minimize
from tqdm.auto import tqdm

from ship_inverse_control.ships import Ship

logger = logging.getLogger(__name__)


def create_trajectory_interpolator(traj_i, columns_to_interpolate, circular_columns):
    df = traj_i.df.copy()

    old_idx = df.index
    old_idx_float = np.array([x.timestamp() for x in old_idx])

    for column in circular_columns:
        df[f"{column}_u"] = df[column].apply(lambda x: np.cos(np.deg2rad(x)))
        df[f"{column}_v"] = df[column].apply(lambda x: np.sin(np.deg2rad(x)))

        columns_to_interpolate.append(f"{column}_u")
        columns_to_interpolate.append(f"{column}_v")

    values_to_interpolate = df[columns_to_interpolate].values
    F = scipy.interpolate.interp1d(
        old_idx_float, values_to_interpolate, assume_sorted=True, axis=0
    )
    return F, columns_to_interpolate


def create_ship_from_trajectory(
    traj: mpd.Trajectory,
    initial_tau_X: Optional[float] = None,
    initial_rudder_angle: Optional[float] = None,
    get_current_for_trajectory: Optional[Callable[[mpd.Trajectory], tuple]] = None,
) -> Ship:
    """Creates and initializes a Ship object from a MovingPandas trajectory."""
    start_row = traj.df.iloc[0]
    start_time = traj.get_start_time()
    start_location = traj.get_start_location()

    # Extract properties, providing defaults if they don't exist
    speed_ms = start_row.get("speed", 0)
    course_over_ground_deg = start_row.get("cog", 0)
    course_over_ground_rad = np.deg2rad(course_over_ground_deg)

    heading_deg = start_row.get("heading", course_over_ground_deg)
    # Convert heading from degrees (CW, 0 North) to radians (CW, 0 North)
    heading_rad = np.deg2rad(heading_deg)
    sog_knots = start_row.get("sog", speed_ms * 1.943844)  # speed over ground in knots
    sog_ms = sog_knots / 1.943844  # speed over ground in m/s

    ship_length = start_row.get("ship_length", 50)
    if pd.isna(ship_length) or ship_length == 0:
        ship_length = 50

    # Extract current data if provided
    if get_current_for_trajectory is not None:
        V_current, beta_current = get_current_for_trajectory(trajectory=traj)
    else:
        V_current, beta_current = 0, 0

    return Ship(
        starting_point=start_location,
        course_over_ground=course_over_ground_rad,
        speed_over_ground=sog_ms,
        heading=heading_rad,
        L=float(ship_length),
        V_current=V_current,
        beta_current=beta_current,
        start_time=start_time,
        id=traj.id,
        initial_tau_X=initial_tau_X,
        initial_rudder_angle=initial_rudder_angle,
    )


def extrapolate_trajectory(
    traj_gdf: gpd.GeoDataFrame,
    time_delta_seconds: float,
    direction_column="direction",
) -> gpd.GeoDataFrame:
    """
    Projects all points in a trajectory forward in time.

    Args:
        traj_gdf: A GeoDataFrame representing the trajectory.
        time_delta_seconds: The time in seconds to project forward.

    Returns:
        A new GeoDataFrame with the projected trajectory.
    """
    if traj_gdf.empty or time_delta_seconds <= 0:
        return traj_gdf.copy()

    projected_gdf = traj_gdf.copy()

    # Ensure speed and direction are available, with defaults
    assert (
        "speed" in projected_gdf.columns
    ), "speed is not in traj_gdf columns, can't extrapolate"
    assert (
        direction_column in projected_gdf.columns
    ), f"{direction_column} is not in traj_gdf columns, can't extrapolate"

    direction_column_rad = f"{direction_column}_rad"
    projected_gdf[direction_column_rad] = np.deg2rad(projected_gdf[direction_column])

    delta_x = (
        projected_gdf["speed"]
        * time_delta_seconds
        * np.sin(projected_gdf[direction_column_rad])
    )
    delta_y = (
        projected_gdf["speed"]
        * time_delta_seconds
        * np.cos(projected_gdf[direction_column_rad])
    )

    projected_gdf["geometry"] = projected_gdf.apply(
        lambda row: shapely.Point(
            row.geometry.x + delta_x.loc[row.name],
            row.geometry.y + delta_y.loc[row.name],
        ),
        axis=1,
    )

    projected_gdf = projected_gdf.drop(columns=[direction_column_rad])

    return projected_gdf


def resample_trajectory(
    traj_to_resample: mpd.Trajectory, target_index: pd.DatetimeIndex, crs
) -> gpd.GeoDataFrame:
    """
    Resamples a trajectory to a new time index using linear interpolation.

    This function handles both geometry (by interpolating x and y coordinates)
    and other numeric attributes like speed and direction.

    Args:
        traj_to_resample: The MovingPandas Trajectory to be resampled.
        target_index: The target DatetimeIndex for the new trajectory.
        crs: The coordinate reference system of the trajectory.

    Returns:
        A new GeoDataFrame resampled to the target index.
    """
    if traj_to_resample.df.empty:
        return gpd.GeoDataFrame()

    df = traj_to_resample.df.copy()

    # Decompose geometry into x and y for interpolation
    df["x"] = df.geometry.x
    df["y"] = df.geometry.y

    # Ensure direction and speed are present for interpolation
    if "speed" not in df.columns:
        traj_to_resample.add_speed(overwrite=True)
        df["speed"] = traj_to_resample.df["speed"]
    if "direction" not in df.columns:
        traj_to_resample.add_direction(overwrite=True)
        df["direction"] = traj_to_resample.df["direction"]

    circular_columns = ["direction", "heading"]
    for circular_column in circular_columns:
        df[f"{circular_column}_rad"] = np.deg2rad(df[circular_column])
        df[f"{circular_column}_u"] = np.cos(df[f"{circular_column}_rad"])
        df[f"{circular_column}_v"] = np.sin(df[f"{circular_column}_rad"])

    columns_to_interpolate = ["x", "y", "speed"]
    for circular_column in circular_columns:
        columns_to_interpolate.append(f"{circular_column}_u")
        columns_to_interpolate.append(f"{circular_column}_v")

    # Use the numeric representation of the timestamp for interpolation
    source_time_numeric = df.index.astype("datetime64[us]").astype(np.int64)
    target_time_numeric = target_index.astype("datetime64[us]").astype(np.int64)

    # Create an interpolator for all required columns
    interpolator = scipy.interpolate.interp1d(
        source_time_numeric,
        df[columns_to_interpolate].values,
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",  # Handle edge cases
    )

    # Calculate the interpolated values at the target timestamps
    interpolated_values = interpolator(target_time_numeric)

    resampled_df = pd.DataFrame(
        interpolated_values, index=target_index, columns=columns_to_interpolate
    )

    # reconstruct interpolated columns from linear components
    for circular_column in circular_columns:
        resampled_df[f"{circular_column}_rad"] = np.arctan2(
            resampled_df[f"{circular_column}_v"], resampled_df[f"{circular_column}_u"]
        )
        resampled_df[circular_column] = np.rad2deg(
            resampled_df[f"{circular_column}_rad"]
        )

    # drop columns that we no longer need
    columns_to_drop = []
    for circular_column in circular_columns:
        columns_to_drop.append(f"{circular_column}_rad")
        columns_to_drop.append(f"{circular_column}_u")
        columns_to_drop.append(f"{circular_column}_v")
    resampled_df = resampled_df.drop(columns=columns_to_drop)

    # Reconstruct the GeoDataFrame with the new geometries
    resampled_gdf = gpd.GeoDataFrame(
        resampled_df,
        geometry=gpd.points_from_xy(resampled_df.x, resampled_df.y),
        crs=crs,
    )

    return resampled_gdf


def calculate_trajectory_error(
    observed: mpd.Trajectory,
    expected: mpd.Trajectory,
    # extrapolate heading and direction for 1s to compute heading and direction error
    projection_time_seconds: float = 3,
    trajectory_weight: float = 1,
    direction_weight: float = 1,
    heading_weight: float = 1,
    last_weight: float = 1,
) -> float:
    """
    Computes a composite error between an observed and an expected trajectory.

    The error is a weighted sum of:
    1. The Mean Squared Error (MSE) of the distances between corresponding points.
    2. The MSE of the distances between the trajectories after projecting them
       forward in time.

    Args:
        observed: The observed trajectory.
        expected: The expected (reference) trajectory.
        projection_time_seconds: The time delta (in seconds) for projection.
        projection_weight: The weight to apply to the projected error component.

    Returns:
        The composite error value.
    """

    if expected.df.empty or observed.df.empty:
        return 0.0

    # Resample the observed trajectory to align with the expected trajectory's timestamps
    resampled_observed_gdf = resample_trajectory(
        observed, expected.df.index, expected.crs
    )

    if resampled_observed_gdf.empty:
        return 0.0

    # 1. Calculate trajectory MSE
    trajectory_distances = resampled_observed_gdf.geometry.distance(
        expected.df.geometry
    )
    trajectory_rmse = np.sqrt(np.mean(trajectory_distances**2))

    # 2. Calculate projected MSE
    # Project both trajectories forward
    direction_expected_gdf = extrapolate_trajectory(
        expected.df, projection_time_seconds, direction_column="direction"
    )
    direction_observed_gdf = extrapolate_trajectory(
        resampled_observed_gdf, projection_time_seconds, direction_column="direction"
    )

    # Calculate distances between direction points
    direction_distances = direction_observed_gdf.geometry.distance(
        direction_expected_gdf.geometry
    )
    direction_rmse = np.sqrt(np.mean(direction_distances**2))

    heading_expected_gdf = extrapolate_trajectory(
        expected.df, projection_time_seconds, direction_column="heading"
    )
    heading_observed_gdf = extrapolate_trajectory(
        resampled_observed_gdf, projection_time_seconds, direction_column="heading"
    )

    # Calculate distances between heading points
    heading_distances = heading_observed_gdf.geometry.distance(
        heading_expected_gdf.geometry
    )
    heading_rmse = np.sqrt(np.mean(heading_distances**2))

    # Add penalty for distance between final points
    last_trajectory = trajectory_distances.iloc[-1]
    last_direction = direction_distances.iloc[-1]
    assert (
        direction_expected_gdf.index[-1] == direction_observed_gdf.index[-1]
    ), "last time should be equal"
    last_heading = heading_distances.iloc[-1]
    assert (
        heading_expected_gdf.index[-1] == heading_observed_gdf.index[-1]
    ), "last time should be equal"

    last_rmse = np.sqrt((last_trajectory**2 + last_direction**2 + last_heading**2) / 3)

    # Combine the errors
    sum_weights = trajectory_weight + direction_weight + heading_weight + last_weight
    total_error = (
        trajectory_weight * trajectory_rmse
        + direction_weight * direction_rmse
        + heading_weight * heading_rmse
        + last_weight * last_rmse
    ) / sum_weights

    return total_error


def simulate_ship_maneuver(
    ship: Ship,
    rudder_angle: float,
    tau_X: float,
    total_time: Optional[float] = None,
    dt: Optional[float] = None,
    make_copy: bool = True,
    timesteps: Optional[pd.DatetimeIndex] = None,
) -> tuple[mpd.Trajectory, Ship]:
    """
    Simulates a ship and returns the trajectory and final ship state.

    The simulation timestep 'dt' is dynamically adjusted for stability. If 'dt'
    is not provided, it defaults to the ship's estimated stable timestep.

    Args:
        ship: The ship object to simulate.
        rudder_angle: The rudder angle in radians.
        tau_X: The engine thrust in Newtons.
        total_time: The total simulation time in seconds.
        dt: The base time step in seconds (optional).
        make_copy: Whether to create a deep copy of the ship before simulation.
        timesteps: A pandas DatetimeIndex for non-uniform simulation steps.

    Returns:
        A tuple with the simulated trajectory and the final ship state.
    """
    if make_copy:
        ship = copy.deepcopy(ship)

    if timesteps is not None:
        if len(timesteps) < 2:
            return ship.trajectory, ship

        for i in range(len(timesteps) - 1):
            start_time = timesteps[i]
            end_time = timesteps[i + 1]
            time_to_next_step = (end_time - start_time).total_seconds()

            simulated_time_in_interval = 0.0
            while simulated_time_in_interval < time_to_next_step:
                estimated_dt = ship.estimate_dt()
                remaining_time = time_to_next_step - simulated_time_in_interval
                current_dt = min(estimated_dt, remaining_time)

                if current_dt <= 0:
                    break

                ship.update(dt=current_dt, rudder_angle=rudder_angle, tau_X=tau_X)
                simulated_time_in_interval += current_dt
    else:
        if total_time is None:
            raise ValueError("Either `total_time` or `timesteps` must be provided.")

        # If dt is not provided, use the ship's estimate as the base step
        base_dt = dt if dt is not None else ship.estimate_dt()

        simulated_time = 0.0
        while simulated_time < total_time:
            # Dynamically adjust dt for stability in every step
            estimated_dt = ship.estimate_dt()
            remaining_time = total_time - simulated_time
            current_dt = min(base_dt, estimated_dt, remaining_time)

            if current_dt <= 0:
                break

            ship.update(dt=current_dt, rudder_angle=rudder_angle, tau_X=tau_X)
            simulated_time += current_dt

    return ship.trajectory, ship


def evaluate_trajectory_imitation(
    true_trajectory: mpd.Trajectory,
    ship: Ship,
    rudder_angle: float,
    tau_X: float,
    make_copy: bool = True,
) -> float:
    """Simulates a ship and calculates the error against a reference trajectory."""
    total_time = (
        true_trajectory.get_end_time() - true_trajectory.get_start_time()
    ).total_seconds()
    if total_time <= 0:
        return 0.0

    imitated_traj, _ = simulate_ship_maneuver(
        ship=ship,
        rudder_angle=rudder_angle,
        tau_X=tau_X,
        timesteps=true_trajectory.df.index,  # Pass the trajectory's index
        make_copy=make_copy,
    )

    error = calculate_trajectory_error(imitated_traj, true_trajectory)

    return error


def find_optimal_controls_for_trajectory(
    true_trajectory: mpd.Trajectory,
    action_timestep: int = 30,
    get_current_for_trajectory: Optional[Callable[[mpd.Trajectory], tuple]] = None,
    tqdm_disable=False,
):
    """
    Determines the optimal rudder angle and engine thrust (tau_X) at each
    timestep to make a simulated ship follow a reference trajectory.

    This function works by:
    1. Initializing a ship model based on the start of the reference trajectory.
    2. Dividing the reference trajectory into segments based on the
       `action_timestep`.
    3. For each segment, using optimization to find the rudder angle and
       engine thrust that minimize the difference between the simulated path
       and the reference path.
    4. Updating the ship's state after each step, ensuring continuity.

    Args:
        true_trajectory: The MovingPandas trajectory to be imitated.
        action_timestep: The duration in seconds for each control decision
                         (rudder and engine setting).

    Returns:
        A list of dictionaries, where each dictionary contains the optimal
        'rudder_angle' and 'tau_X' for a timestep.
    """

    # Store the results of each optimization step
    control_solutions = []

    # 1. Set up the time intervals for the simulation
    start_time = true_trajectory.get_start_time()
    end_time = true_trajectory.get_end_time()

    time_steps = pd.date_range(
        start=start_time, end=end_time, freq=f"{action_timestep}s"
    )

    tau_X_scale = 100_000
    # Initial guesses for the optimization
    initial_guess = [
        0.0,
        1e5 / tau_X_scale,
    ]  # [rudder_angle_rad, tau_X_newtons / 100_000]

    # Define the bounds for the optimization
    # Rudder angle: -30 to 30 degrees, converted to radians
    # tau_X: A reasonable range for engine thrust
    bounds = [
        (-np.deg2rad(30), np.deg2rad(30)),
        (-1e5 / tau_X_scale, 3e5 / tau_X_scale),
    ]

    # 2. Iterate through each time step to find the best control inputs
    for i in tqdm(range(len(time_steps) - 1), disable=tqdm_disable):
        # Define the current trajectory segment to imitate
        segment_start_time = time_steps[i]
        segment_end_time = time_steps[i + 1]

        try:
            # Extract the portion of the trajectory for the current timestep
            traj_segment = true_trajectory.get_segment_between(
                segment_start_time, segment_end_time
            )
        except (ValueError, IndexError):
            # If the segment is invalid, skip to the next one
            continue

        # Instantiate ship
        # Re-initialize the ship from the reference trajectory
        ship = create_ship_from_trajectory(
            traj_segment, get_current_for_trajectory=get_current_for_trajectory
        )

        # 3. Define the objective function for the optimization
        #    This function will be minimized to find the best rudder and engine settings.
        def objective_func(args):
            rudder_angle, tau_X = args
            tau_X = tau_X * tau_X_scale
            result = evaluate_trajectory_imitation(
                tau_X=tau_X,
                rudder_angle=rudder_angle,
                true_trajectory=traj_segment,
                ship=ship,
                make_copy=True,  # Use a copy to not alter the ship's state during optimization
            )
            return result

        def callback(intermediate_result):
            rudder_angle, tau_X = intermediate_result.x

            logger.debug(
                "optimize result, segment %s, rudder angle: %s, tau_X: %s, error: %s",
                i,
                rudder_angle,
                tau_X * tau_X_scale,
                intermediate_result.fun,
            )

        # 4. Run the optimization to find the best parameters
        result = minimize(
            objective_func,
            initial_guess,
            bounds=bounds,
            method="L-BFGS-B",  # A good method for bounded problems
            options={"eps": 1e-5, "ftol": 1e-5},
            callback=callback,
        )

        best_rudder_angle, best_tau_X = result.x

        # resail the trajectory to store the logbook
        ship = create_ship_from_trajectory(
            traj_segment, get_current_for_trajectory=get_current_for_trajectory
        )
        evaluate_trajectory_imitation(
            tau_X=best_tau_X,
            rudder_angle=best_rudder_angle,
            true_trajectory=traj_segment,
            ship=ship,
            make_copy=False,  # Do not use a copy, so we save the logbook
        )

        trajectory = ship.trajectory
        trajectory.df["tau_X"] = best_tau_X * tau_X_scale
        trajectory.df["rudder_angle"] = best_rudder_angle
        trajectory.df["iteration"] = i
        trajectory.df["V_current"] = ship.V_current
        trajectory.df["beta_current"] = ship.beta_current

        # Store the optimal settings for this timestep
        control_solutions.append(
            {
                "timestamp": segment_start_time,
                "rudder_angle": best_rudder_angle,
                "tau_X": best_tau_X * tau_X_scale,
                "optimization_error": result.fun,
                "traj_full": true_trajectory,
                "traj_segment": traj_segment,
                "traj_imitation": trajectory,
                "iteration": i,
            }
        )

        # Update the initial guess for the next optimization step
        initial_guess = [best_rudder_angle, best_tau_X]

    return control_solutions
