"""
formula.py: A file that contains all the Physics formulas needed to interpolate telemetry.
"""

from math import fabs, sqrt
import numpy as np
from skaero.atmosphere import coesa


__author__  = 'u/Shahar603'
__status__ = "Development"



####### Physical Constants #######
AIR_ADIABATIC = 1.4 # Air's adiabatic constant
G = 6.67408 * 10 ** -11  # Universal Gravitational constant [m^3 kg^-1 s^-2]


####### Earth Constants ######
EARTH_RADIUS = 6.375 * 10 ** 6  # Earth's radius [meters]
EARTH_MASS = 5.972 * 10 ** 24  # Earth's mass [kg]
g0 = 9.80665  # Gravitational acceleration at the Earth surface [m/s^2]


##### Rocket Data #####
FAIRING_AREA = 21.2264  # Cross section area of the fairing [m^2]
Cd = 0.2  # Coefficient of drag at rest
MAX_THRUST = 7.6*10**6  # Maximum thrust of the rocket at sea level [N]
ENGINE_AREA = 0.68  # Cross section area of the rocket nozzle [m^2]
Isp0 = 288.5 # Rocket engine's Isp at sea level [s]
IspVac = 312 # Rocket engine's Isp in vacuum [s]
MFR = 298.73 # Engine's mass flow rate [kg/s]

p2 = 0.104365*10**6
dv = 405  # m/s

"""
def delta_mass(prev_data, data, thrust, stage_sep):
    dt = data['time'] - prev_data['time']
    return dt * (thrust - calc_pressure_thrust(data)) / (calc_isp(data, stage_sep) * g0)
"""

def mach_number(velocity, altitude):
    """This function calculates the Mach Number of the rocket at a given velocity and altitude.
    :param velocity: An object that contains data about the rocket's current altitude and velocity.
    :return: The current Mach Number
    """
    return velocity / c_at_altitude(altitude)



def c_at_altitude(altitude):
    """This function calculates the speed of sound(c) for a given altitude.
    :param data: An object that contains data about the rocket's altitude and velocity.
    :return: the speed of sound.
    """
    h, T, P, rho = coesa.table(altitude)

    if type(P) == complex or P < 1:
        return 0

    return (AIR_ADIABATIC * P/rho)**0.5


def drag_divergent(velocity):
    """
    :param velocity: Velocity of the rocket
    :return: The drag divergent mach number of the coefficient of drag
    """
    M = mach_number(velocity, 0)

    if M < 0.6 or M > 4:
        return 1
    elif M < 1:
        return 1 + 215.8 * (M - 0.6) ** 6
    elif M < 2:
        return 2.0881 * (M - 1) ** 3 - 3.7938 * (M - 1) ** 2 + 1.4618 * (M - 1) + 1.883917

    return max(1.0, 0.297 * (M - 2) ** 3 - 0.7937 * (M - 2) ** 2 - 0.1115 * (M - 2) + 1.64006)




def get_atmos_data(altitude):
    """
    This function calculates data about the atmosphere
    :param altitude: Distance from the surface of the Earth [m]
    :return:
    """
    h, T, P, rho = coesa.table(altitude)

    if np.isnan(P) or type(P) == complex:
        P = 0
        rho = 0

    return h, T, P, rho




def calc_thrust(drag, acceleration, theta, mass):
    """
    This function calculates the thrust of the rocket
    :param drag: The force of drag on the rocket [N]
    :param acceleration: The acceleration of the rocket (without gravity) [m/s^2]
    :param theta: The velocity angle of the rocket [in any unit]
    :param mass: The total mass of the rocket
    :return: Thrust of the rocket [N]
    """
    return (mass * acceleration) - drag * theta/fabs(theta)


def gravitational_acceleration(mass, distance):
    """
    This function calculates the gravitational acceleration of a object
    :param mass: The mass of the object
    :param distance: The distance from the surface of the object
    :return: The gravitational acceleration
    """
    return G * mass / distance**2


def earth_gravity(altitude):
    """
    This function calculates the gravitational acceleration at a given altitude
    above the surface of the earth
    :param altitude: The distance of the object from the surface of the earth [m]
    :return: The gravitational acceleration [N]
    """
    return gravitational_acceleration(EARTH_MASS, EARTH_RADIUS + altitude)


def earth_potential_energy(altitude):
    """
    This function calculates the potential energy of an object
    :param altitude: The distance of the object from the surface of the Earth
    :return: The potential energy
    """
    return -G * EARTH_MASS / (EARTH_RADIUS + altitude)


def calc_drag(stage):
    """

    :param data:
    :param stage:
    :return:
    """
    h, T, p, rho = get_atmos_data(stage.altitude)

    return 0.5 * stage.coefficient_of_drag({'velocity': stage.velocity}) * rho * stage.area * stage.velocity ** 2